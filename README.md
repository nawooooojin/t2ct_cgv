# t2ct_cgv

Team CGV code for the MICCAI 2026 VLM3D Challenge.

| Track | Hidden test result |
|---|---|
| CT Volume Generation | 1st. FVD 0.2799, FID 2.5D 54.40, CT-CLIP 52.48 |
| MR Volume Generation | 2nd of 4. SSIM 0.4891, PSNR 13.49, MSE 0.0466, FID 2.5D 263.5 |

## Method

- **WARP-CT**: Report2CT (latent diffusion, 3D U-Net, three frozen text encoders) with two changes.
  The MAISI VAE is replaced by the frozen Wan2.1 VAE (latent 16 x 64^3), and REPA is added during
  training: the U-Net middle block is aligned to frozen SPECTRE ViT-L tokens with the U-REPA loss
  (`L = L_velocity + 0.5 (L_REPA + 3 L_ML)`).
- **WARP-MR**: the same framework, with RadFinder text conditioning (token sequence to
  cross-attention, pooled vector to AdaGN), a (sequence, plane) class label, and per-volume
  percentile intensity normalization.

Pretrained models (all frozen): [Wan2.1 VAE](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers),
[SPECTRE-Large](https://huggingface.co/cclaess/SPECTRE-Large) (`spectre_backbone_vit_large_patch16_128_no_vla.pt`, training only),
MedEmbed-large-v0.1 / ClinicalBERT / BiomedVLP-CXR-BERT-specialized (CT), [RadFinder](https://huggingface.co/lmb-freiburg/radfinder) (MR).
The U-Nets are trained from scratch.

## Setup

```bash
conda create -n t2ct python=3.11 -y && conda activate t2ct
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install --no-deps monai==1.4.0
git clone https://github.com/cclaess/SPECTRE third_party/spectre && git -C third_party/spectre checkout c05af47

# MR text encoding only (RadFinder needs transformers >= 5)
conda create -n t2ct-radfinder python=3.11 -y && conda activate t2ct-radfinder
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-radfinder.txt
```

## Data and preprocessing

- [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE): `ids/ctrate_train.txt` (46,393 `train_fixed`
  volumes, out-of-FOV / unreadable ones removed) and `ids/ctrate_valid.txt` (1,304, one per validation patient).
- [MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE): 1,200 studies (8,492 series) sampled from the
  train split, one study per patient, seed 42 (`scripts/preprocess/select_mrrate_subset.py`).

Volumes are reoriented to RAS and resampled to 512 x 512 x 253, then encoded to Wan latents, SPECTRE
tokens (16^3) and text features. Each script processes a whole id list; run it once per split (for CT-RATE,
`ROOT` is `train_fixed` or `valid_fixed` and `CSV` is the matching report file).

```bash
python scripts/preprocess/encode_wan_latents.py --ids IDS --data-root ROOT --layout {ctrate,mrrate} [--intensity unit] --out-dir data/<task>/latents
python scripts/preprocess/encode_spectre.py     --ids IDS --data-root ROOT --layout {ctrate,mrrate} [--intensity unit] --ckpt SPECTRE_CKPT --out-dir data/<task>/spectre_ssl_16
python scripts/preprocess/encode_text_ctgen.py  --ids IDS --reports-csv CSV --out-dir data/ctgen/text   # CT
python scripts/preprocess/encode_text_mrgen.py  --ids IDS --reports-csv CSV --out-dir data/mrgen/text   # MR, in t2ct-radfinder
python scripts/preprocess/build_datalist.py --task <task> --root data/<task> --ids-train IDS_TRAIN --ids-valid IDS_VALID [--raw-dir data/mrgen/raw]
```

## Training

```bash
python src/train.py experiment=ctgen
python src/train.py experiment=mrgen
python scripts/extract_ema.py <run>/checkpoints/last.ckpt weights/<task>_ema.ckpt
```

Settings are in `configs/experiment/*.yaml` and `configs/model/*.yaml`. WARP-CT: 150 epochs, batch 24
(2 GPUs), Adam lr 3e-4, bf16, EMA 0.999. WARP-MR: 115 epochs, batch 32.

## Inference

Input is the challenge `prompts.json` (`[{"input_image_name", "report"}]`), output is one float32 `.nii.gz`
per prompt (CT: HU, 512 x 512 x 253, 0.75/0.75/1.3 mm; MR: [0, 1]).

```bash
python scripts/generate.py --task ctgen --ckpt weights/ctgen_ema.ckpt --prompts prompts.json --out-dir pred_ct
python scripts/preprocess/encode_text_mrgen.py --prompts prompts.json --out-dir work/text   # in t2ct-radfinder
python scripts/generate.py --task mrgen --ckpt weights/mrgen_ema.ckpt --prompts prompts.json --text-dir work/text \
    --spacing-table data/mrgen/spacing_table.json --axcodes-table data/mrgen/axcodes_table.json --out-dir pred_mr
```

CFG scale 5; 50 sampling steps for CT, 100 for MR.

References: Report2CT (arXiv:2509.14780), MAISI (WACV 2025), Wan (arXiv:2503.20314), U-REPA (NeurIPS 2025),
SPECTRE (CVPR 2026), RadFinder (arXiv:2603.02026).
