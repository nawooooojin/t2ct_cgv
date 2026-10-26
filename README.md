# Team CGV at VLM3D 2026: Report-to-Volume Generation

Code for our two submissions to the MICCAI 2026 VLM3D Challenge.

| Track | Submission | Method |
|---|---|---|
| CT Volume Generation (VLM3D-CT-VOLUME) | `1b8cdaee` | Report2CT + Wan2.1 VAE + representation alignment |
| MR Volume Generation (VLM3D-MR-VOLUME) | `94215494` | Same framework, RadFinder text conditioning + sequence/plane condition |

Team: Woojin Na, Hyunjun Park, Jinah Park (CGV Lab, KAIST)

This repository contains the training and inference code only. Evaluation code is not included
(we used the official VLM3D evaluation). Model weights are not released (see [Weights](#weights)).

## Method

### CT-Gen

We start from Report2CT (VLM3D 2025 winner), a latent diffusion model on the MAISI VAE with a 3D
U-Net, voxel-spacing conditioning and three frozen text encoders (BiomedVLP-CXR-BERT, MedEmbed,
ClinicalBERT). Findings and impression are encoded separately, mean-pooled, concatenated per
section (2560-d) and given to the U-Net through cross-attention. We made two changes.

1. **VAE**: MAISI VAE is replaced by the Wan2.1 video VAE (frozen, `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`).
   The CT volume (resampled to 512 x 512 x 253, HU clipped to [-1000, 1000] and scaled to [-1, 1]) is
   encoded as a grayscale video. The latent is (16, 64, 64, 64) instead of MAISI's (4, 120, 120, 64),
   so the spatial latent grid is about 3.5x smaller. Only `in/out_channels` of the U-Net change.
2. **Representation alignment (REPA)**: following U-REPA, the U-Net middle-block feature (8^3) is
   projected by a 3-layer MLP (1x1x1 conv), trilinearly upsampled to 16^3, and aligned with frozen
   SPECTRE ViT-L tokens of the ground-truth CT (32^3 token grid average-pooled to 16^3, spatially
   normalized). The teacher is used only during training.

```
L = L_velocity + lambda * (L_REPA + w * L_ML),   lambda = 0.5, w = 3
L_REPA = - mean_n cos(y*[n], h(h_t)[n])                       token-wise
L_ML   = mean || Sim(y*) - Sim(h(h_t)) ||_F^2                 pairwise cosine-similarity matrices
```

Training uses rectified flow (velocity prediction, MSE), per-sample text dropout of 0.15 for
classifier-free guidance, Adam with linear warm-up and polynomial decay, and EMA of the weights.
Inference: 50 Euler steps, CFG scale 5, spacing condition 0.75 / 0.75 / 1.3 mm, output 512 x 512 x 253.

### MR-Gen

The MR model uses the same latent space (Wan2.1 VAE), U-Net backbone, rectified-flow training and
REPA objective with the same SPECTRE teacher. The differences are:

- **Text conditioning**: the report is encoded with RadFinder (Qwen3-Embedding-0.6B + LoRA, frozen).
  Local path: per-token hidden states of all 29 layers, RMS-normalized per layer and averaged,
  given to cross-attention as a token sequence (no truncation, padding masked). Global path:
  [final-layer EOS, mean of the token sequence] per section (4096-d in total), added to the timestep
  embedding through a zero-initialized linear layer. The U-Net ResBlocks use AdaGN
  (scale/shift after GroupNorm).
- **Sequence and plane condition**: one report is shared by several series of a study, so the target
  (sequence, plane) pair (T1w/T2w/FLAIR/SWI x axial/sagittal/coronal/oblique) is given as a class label.
- **Intensity**: each volume is percentile-normalized (0.5 / 99.5) to [-1, 1] instead of HU windowing.
- At inference the spacing condition is the median spacing of the (sequence, plane) class in the
  training set, and the output is written in the most common orientation of that class.

## Pretrained models

| Model | Use | Source | License |
|---|---|---|---|
| Wan2.1 VAE | latent space (frozen) | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | Apache-2.0 |
| SPECTRE ViT-L (SSL, `spectre_backbone_vit_large_patch16_128_no_vla.pt`) | REPA teacher (training only) | [cclaess/SPECTRE](https://huggingface.co/cclaess/SPECTRE) | weights CC-BY-NC-SA |
| MedEmbed-large-v0.1, ClinicalBERT, BiomedVLP-CXR-BERT-specialized | CT text encoders (frozen) | Hugging Face | see model cards |
| RadFinder | MR text encoder (frozen) | [lmb-freiburg/radfinder](https://huggingface.co/lmb-freiburg/radfinder) | see model card |

The U-Nets are trained from scratch.

## Installation

Tested with Python 3.11, PyTorch 2.7 (CUDA 12.8).

```bash
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps monai==1.4.0
git clone https://github.com/cclaess/SPECTRE third_party/spectre   # REPA teacher (commit c05af47)
```

RadFinder needs `transformers>=5`, which conflicts with the main environment, so MR text encoding
runs in a second environment:

```bash
python -m venv .venv-radfinder && .venv-radfinder/bin/pip install -r requirements-radfinder.txt
```

## Data

**CT-RATE** (challenge data): https://huggingface.co/datasets/ibrahimhamamci/CT-RATE. We used the
`train_fixed` volumes and the English reports (`Findings_EN`, `Impressions_EN`). 46,393 training
volumes were used after removing volumes whose chest is out of the field of view or that could not
be read. Validation volumes were used only for local evaluation.

**MR-RATE** (challenge data, gated): https://huggingface.co/datasets/Forithmus/MR-RATE. We sampled
1,200 studies (one per patient, seed 42) from the `train` split, which gives 8,492 series
(T1w 35%, T2w 29%, FLAIR 26%, SWI 10%). 150 more studies were held out for local checks.

### Preprocessing

All features are precomputed. `data/<task>/` ends up with `latents/`, `spectre_ssl_16/`, `text/`,
`cond/` and `datalist.json`.

```bash
# CT-Gen
python scripts/preprocess/encode_wan_latents.py --ids data/ctgen/ids_train.txt \
    --data-root $CTRATE/dataset/train_fixed --layout ctrate --out-dir data/ctgen/latents
python scripts/preprocess/encode_spectre.py --ids data/ctgen/ids_train.txt \
    --data-root $CTRATE/dataset/train_fixed --layout ctrate \
    --ckpt $SPECTRE_CKPT --out-dir data/ctgen/spectre_ssl_16
python scripts/preprocess/encode_text_ctgen.py --ids data/ctgen/ids_train.txt \
    --reports-csv $CTRATE/dataset/radiology_text_reports/train_reports.csv --out-dir data/ctgen/text
# (same three commands for the validation ids)
python scripts/preprocess/build_datalist.py --task ctgen --root data/ctgen \
    --ids-train data/ctgen/ids_train.txt --ids-valid data/ctgen/ids_valid.txt

# MR-Gen
python scripts/preprocess/select_mrrate_subset.py select --out-dir data/mrgen
python scripts/preprocess/select_mrrate_subset.py download --out-dir data/mrgen --split train
python scripts/preprocess/select_mrrate_subset.py download --out-dir data/mrgen --split valid
python scripts/preprocess/encode_wan_latents.py --ids data/mrgen/ids_train.json \
    --data-root data/mrgen/raw --layout mrrate --intensity unit --out-dir data/mrgen/latents
python scripts/preprocess/encode_spectre.py --ids data/mrgen/ids_train.json \
    --data-root data/mrgen/raw --layout mrrate --intensity unit \
    --ckpt $SPECTRE_CKPT --out-dir data/mrgen/spectre_ssl_16
.venv-radfinder/bin/python scripts/preprocess/encode_text_mrgen.py --ids data/mrgen/ids_train.json \
    --reports-csv data/mrgen/reports_subset.csv --out-dir data/mrgen/text
# (same for ids_valid.json)
python scripts/preprocess/build_datalist.py --task mrgen --root data/mrgen \
    --ids-train data/mrgen/ids_train.json --ids-valid data/mrgen/ids_valid.json --raw-dir data/mrgen/raw
```

Volumes are reoriented to RAS and resampled to 512 x 512 x 253 for both the VAE and the teacher, so a
latent voxel and a teacher token cover the same region. For SPECTRE the volume is padded to 256 slices
with air (crop size 128 x 128 x 64).

## Training

```bash
python src/train.py experiment=ctgen    # 2 GPUs, batch 12/GPU, lr 3e-4, 150 epochs, bf16
python src/train.py experiment=mrgen    # 4 GPUs, batch 8/GPU, fp16
```

Configs are in `configs/` (Hydra). Main settings:

| | CT-Gen | MR-Gen |
|---|---|---|
| Data | CT-RATE train, 46,393 volumes | MR-RATE, 1,200 studies / 8,492 series |
| Latent | (16, 64, 64, 64) | (16, 64, 64, 64) |
| U-Net | MAISI U-Net, channels 64-128-256-512, cross-attention 2560 | same + AdaGN, cross-attention 1024, global text path |
| Text | 3 encoders, pooled, findings / impression | RadFinder tokens (local) + pooled (global) |
| REPA | SPECTRE 16^3, lambda 0.5, w 3, 1,024 sampled tokens for L_ML | same, all 4,096 tokens for L_ML |
| Optimizer | Adam, lr 3e-4, warm-up 1,000 steps, polynomial decay | Adam, lr 4e-4, warm-up 1,000 steps, polynomial decay |
| Batch | 24 (2 x 12) | 32 (4 x 8) |
| Epochs | 150 | 115 (submitted checkpoint) |
| EMA | 0.999 | 0.9999 |
| CFG dropout | 0.15 per sample | 0.15 per sample |

To get the EMA weights used for inference:

```bash
python scripts/extract_ema.py outputs/ctgen/<run>/checkpoints/last.ckpt weights/ctgen_ema.ckpt
```

## Inference

Input follows the challenge format: a JSON list of `{"input_image_name", "report"}`. One `.nii.gz`
(float32) is written per entry.

```bash
# CT-Gen: 512 x 512 x 253 HU volume, spacing 0.75 / 0.75 / 1.3 mm
python scripts/generate.py --task ctgen --ckpt weights/ctgen_ema.ckpt \
    --prompts prompts.json --out-dir outputs/ctgen_pred

# MR-Gen: encode the reports with RadFinder first
.venv-radfinder/bin/python scripts/preprocess/encode_text_mrgen.py --prompts prompts.json --out-dir work/text
python scripts/generate.py --task mrgen --ckpt weights/mrgen_ema.ckpt --prompts prompts.json \
    --text-dir work/text --spacing-table data/mrgen/spacing_table.json \
    --axcodes-table data/mrgen/axcodes_table.json --out-dir outputs/mrgen_pred
```

For MR-Gen, `input_image_name` must follow the MR-RATE series naming (`<study>_<sequence>-raw-<plane>`),
which is how the target sequence and plane are read. Outputs are in [0, 1].

Sampling: rectified flow with Euler steps (50 for the CT submission, 100 for MR), CFG scale 5.

## Results

Official hidden test set.

| Track | Metrics | Rank |
|---|---|---|
| CT-Gen | FVD (CT-Net) 0.2799, FID 2.5D 54.40, CT-CLIP score 52.48 | 1st |
| MR-Gen | SSIM 0.4891, PSNR 13.49, MSE 0.0466, FID 2.5D 263.5 | 2nd of 4 (best SSIM) |

Local ablation for CT-Gen (5k CT-RATE training subset, 300 validation cases, 300 epochs):

| Model | FID 2.5D | CT-CLIP T2I | CT-CLIP I2I | FVD (CT-CLIP features) |
|---|---|---|---|---|
| Report2CT (MAISI) | 54.34 | 61.80 | 44.72 | 0.386 |
| Report2CT (Wan) | 54.79 | 65.17 | 46.99 | 0.369 |
| + REPA (ours) | 41.61 | 63.76 | 46.69 | 0.336 |

The CT-Net weights for FVD were not available to us, so locally we computed FVD with CT-CLIP features.

## Limitations

- The model generates from the report only. Findings that need precise location or size (small
  nodules, diffuse fine findings) are not reliably reproduced, and FID gets worse for reports with
  many findings.
- REPA uses a CT foundation model (SPECTRE) as teacher; for MR it is applied as is, without an MR
  teacher, and we did not test whether it helps for MR.
- MR outputs use a class-wise median spacing and orientation, so volumes whose real geometry differs
  from the class median are generated at the wrong grid.
- Only CT-RATE / MR-RATE were used for training and evaluation; other institutions, scanners and
  populations were not tested.
- Generated volumes are for research only and must not be used for diagnosis.

## References

- Report2CT: Amirrajab et al., Radiology Report-Conditional 3D CT Generation with Multi-Encoder Latent Diffusion Model, arXiv:2509.14780
- MAISI: Guo et al., WACV 2025
- Wan: Team Wan et al., Wan: Open and Advanced Large-Scale Video Generative Models, arXiv:2503.20314
- U-REPA: Tian et al., U-REPA: Aligning Diffusion U-Nets to ViTs, NeurIPS 2025
- REPA: Yu et al., Representation Alignment for Generation, ICLR 2025
- SPECTRE: Claessens et al., Scaling Self-Supervised and Cross-Modal Pretraining for Volumetric CT Transformers, CVPR 2026
- RadFinder: lmb-freiburg/radfinder, arXiv:2603.02026
- CT-RATE: Hamamci et al.; MR-RATE: Forithmus

## Weights

Trained weights are not released. The SPECTRE weights used as the training teacher are CC-BY-NC-SA,
and the models are trained on CT-RATE / MR-RATE under their data use terms. Questions:
jasonna33@kaist.ac.kr.

## License

Code: MIT (see `LICENSE`). `src/models/rflow.py` is adapted from the MONAI MAISI bundle (Apache-2.0).
Pretrained models and datasets keep their own licenses and terms of use.

## Acknowledgement

This work was supported by Institute for Information & communications Technology Promotion (IITP)
grant funded by the Korea government (MSIT) (No. 00223446, Development of object-oriented synthetic
data generation and evaluation methods).
