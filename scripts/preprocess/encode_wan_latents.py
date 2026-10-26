"""Encode volumes into Wan2.1 latents.

Each volume is reoriented to RAS and resampled to 512 x 512 x 253 (X, Y, Z), then encoded to a
(16, 64, 64, 64) latent and saved as ``<out-dir>/<id>_emb.nii.gz`` with shape (H, W, D, C). The
NIfTI affine keeps the resampled voxel spacing, which build_datalist.py reads as the spacing
condition.

    # CT-RATE (HU)
    python scripts/preprocess/encode_wan_latents.py --ids data/ctgen/ids_train.txt \
        --data-root /path/CT-RATE/dataset/train_fixed --layout ctrate --out-dir data/ctgen/latents
    # MR-RATE (per-volume percentile normalisation)
    python scripts/preprocess/encode_wan_latents.py --ids data/mrgen/ids_train.json \
        --data-root data/mrgen/raw --layout mrrate --intensity unit --out-dir data/mrgen/latents
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import monai.transforms as mt
import nibabel as nib
import numpy as np
import rootutils
import torch
from einops import rearrange
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import mr_rate  # noqa: E402
from src.encoders.wan_vae import DEFAULT_WAN_HF_ID, WanVAE  # noqa: E402
from scripts.preprocess.common import ID_TO_PATH, read_ids  # noqa: E402

IN_PLANE, DEPTH = 512, 253  # 253 = 1 mod 4 -> lossless 4x temporal compression (64 latent frames)


def resample_transform(in_plane: int = IN_PLANE, depth: int = DEPTH) -> mt.Compose:
    return mt.Compose([
        mt.LoadImaged(keys="image"),
        mt.EnsureChannelFirstd(keys="image"),
        mt.Orientationd(keys="image", axcodes="RAS"),
        mt.EnsureTyped(keys="image", dtype=torch.float32),
        mt.Resized(keys="image", spatial_size=(in_plane, in_plane, depth), mode="trilinear"),
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help=".txt (one id per line) or .json ({'ids': [...]})")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--layout", choices=sorted(ID_TO_PATH), default="ctrate")
    ap.add_argument("--intensity", choices=["hu", "unit"], default="hu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hf-id", default=DEFAULT_WAN_HF_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    ids = read_ids(args.ids)[args.shard :: args.num_shards]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    to_path = ID_TO_PATH[args.layout]
    transform = resample_transform()
    vae = WanVAE(hf_id=args.hf_id, device=args.device)

    for vol_id in tqdm(ids, desc="wan latents"):
        out_path = out_dir / f"{vol_id}_emb.nii.gz"
        if out_path.exists():
            continue
        src = to_path(vol_id, args.data_root)
        if not src.is_file():
            print(f"missing: {src}", file=sys.stderr)
            continue
        img = transform({"image": str(src)})["image"]  # (1, 512, 512, 253)
        x = img.numpy().squeeze()
        if args.intensity == "unit":
            x = mr_rate.normalize_percentile(x)
        vol_thw = torch.from_numpy(x).permute(2, 0, 1).contiguous()  # (253, 512, 512)
        z = vae.encode(vol_thw, intensity=args.intensity)  # (16, 64, 64, 64)
        hwdc = rearrange(z.numpy(), "c d h w -> h w d c")  # (64, 64, 64, 16)
        nib.save(nib.Nifti1Image(np.float32(hwdc), affine=img.meta["affine"].numpy()), str(out_path))


if __name__ == "__main__":
    main()
