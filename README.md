# t2ct_cgv

Team CGV code for the MICCAI 2026 VLM3D Challenge.

| Track | Submission ID | Hidden test result |
|---|---|---|
| CT Volume Generation | 26550529 | FVD 0.3109, FID 2.5D 53.31, CT-CLIP 55.08 |
| MR Volume Generation | 94215494 | SSIM 0.4891, PSNR 13.49, MSE 0.0466, FID 2.5D 263.5 |

Weights for both submissions: [huggingface.co/jasonna33/t2ct_cgv](https://huggingface.co/jasonna33/t2ct_cgv).

## Method

- **WARP-CT**: latent diffusion with a 3D U-Net (MONAI MAISI layout) in the frozen Wan2.1 VAE latent
  space (16 x 64^3), conditioned on RadFinder text features: the layer-averaged token sequence of the
  findings and impression goes to cross-attention, and a pooled vector per section to AdaGN in every
  ResBlock. REPA during training: the U-Net middle block is aligned to frozen SPECTRE ViT-L tokens with
  the U-REPA loss (`L = L_velocity + 0.5 (L_REPA + 3 L_ML)`).
- **WARP-MR**: the same model, with a (sequence, plane) class label and per-volume percentile
  intensity normalization.

Pretrained models (all frozen): [Wan2.1 VAE](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers),
[SPECTRE-Large](https://huggingface.co/cclaess/SPECTRE-Large) (`spectre_backbone_vit_large_patch16_128_no_vla.pt`, training only),
[RadFinder](https://huggingface.co/lmb-freiburg/radfinder). The U-Nets are trained from scratch.

## Setup

```bash
conda create -n t2ct python=3.11 -y && conda activate t2ct
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install --no-deps monai==1.4.0
git clone https://github.com/cclaess/SPECTRE third_party/spectre && git -C third_party/spectre checkout c05af47

# text encoding only (RadFinder needs transformers >= 5)
conda create -n t2ct-radfinder python=3.11 -y && conda activate t2ct-radfinder
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-radfinder.txt
```

## Data and preprocessing

- [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE): `ids/ctrate_train.txt` (46,393 `train_fixed`
  volumes, out-of-FOV / unreadable ones removed) and `ids/ctrate_valid.txt` (1,304, one per validation patient).
- [MR-RATE](https://huggingface.co/datasets/Forithmus/MR-RATE): 1,200 studies (8,492 series) sampled from the
  train split, one study per patient, seed 42 (`scripts/preprocess/select_mrrate_subset.py`).

Volumes are reoriented to RAS and resampled to 512 x 512 x 253, then encoded to Wan latents, SPECTRE
tokens (16^3) and RadFinder text features. Each script processes a whole id list; run it once per split
(for CT-RATE, `ROOT` is `train_fixed` or `valid_fixed` and `CSV` is the matching report file).

```bash
python scripts/preprocess/encode_wan_latents.py --ids IDS --data-root ROOT --layout {ctrate,mrrate} [--intensity unit] --out-dir data/<task>/latents
python scripts/preprocess/encode_spectre.py     --ids IDS --data-root ROOT --layout {ctrate,mrrate} [--intensity unit] --ckpt SPECTRE_CKPT --out-dir data/<task>/spectre_ssl_16
python scripts/preprocess/encode_text_<task>.py --ids IDS --reports-csv CSV --out-dir data/<task>/text   # in t2ct-radfinder
python scripts/preprocess/build_datalist.py --task <task> --root data/<task> --ids-train IDS_TRAIN --ids-valid IDS_VALID [--raw-dir data/mrgen/raw]
```

## Training

```bash
python src/train.py experiment=ctgen
python src/train.py experiment=mrgen
python scripts/extract_ema.py <run>/checkpoints/last.ckpt weights/warp_ct_ema.ckpt
python scripts/extract_ema.py <run>/checkpoints/epoch_114.ckpt weights/warp_mr_ema.ckpt
```

Settings are in `configs/experiment/*.yaml` and `configs/model/*.yaml`. WARP-CT: batch 24 (2 GPUs x 12),
Adam lr 3e-4, bf16 mixed, gradient clipping 5, EMA 0.9999, 150 epochs. WARP-MR: batch 8 (4 GPUs x 2),
Adam lr 2e-4, fp16 mixed, gradient clipping 5, EMA 0.9999, LR schedule for 150 epochs; the submission
uses the weights after 115 epochs (`epoch_114.ckpt`).

## Inference

Input is the challenge `prompts.json` (`[{"input_image_name", "report"}]`), output is one float32 `.nii.gz`
per prompt (CT: HU, 512 x 512 x 253, 0.75/0.75/1.3 mm; MR: [0, 1]). Weights and the MR lookup tables
are in the Hugging Face repo above.

```bash
python scripts/preprocess/encode_text_ctgen.py --prompts prompts.json --out-dir work/ct_text   # in t2ct-radfinder
python scripts/generate.py --task ctgen --ckpt weights/warp_ct_ema.ckpt --prompts prompts.json --text-dir work/ct_text --out-dir pred_ct

python scripts/preprocess/encode_text_mrgen.py --prompts prompts.json --out-dir work/mr_text   # in t2ct-radfinder
python scripts/generate.py --task mrgen --ckpt weights/warp_mr_ema.ckpt --prompts prompts.json --text-dir work/mr_text \
    --spacing-table weights/spacing_table.json --axcodes-table weights/axcodes_table.json --out-dir pred_mr
```

CFG scale 5, 100 sampling steps.

References: MAISI (WACV 2025), Wan (arXiv:2503.20314), U-REPA (NeurIPS 2025), SPECTRE (CVPR 2026),
RadFinder (arXiv:2603.02026).

## Team

Team CGV

| Name | Affiliation |
|---|---|
| Hyunjun Park | Korea Advanced Institute of Science and Technology (KAIST) |
| Woojin Na | Korea Advanced Institute of Science and Technology (KAIST) |
| Jinah Park | Korea Advanced Institute of Science and Technology (KAIST) |

## Notes

- Evaluation: official VLM3D metrics (CT: FVD, FID 2.5D, CT-CLIP score; MR: SSIM, PSNR, MSE, FID 2.5D).
- Limitations: report-only generation, so lesion location and size are not reliably reproduced; trained and tested on CT-RATE / MR-RATE only.
- License: code under CC BY-NC 4.0 (`LICENSE`); weights under CC BY-NC-SA 4.0; pretrained models and datasets keep their own licenses.
