"""Precompute frozen SPECTRE teacher tokens for REPA.

Same geometry as the Wan latents (RAS, 512 x 512 x 253), padded with air to 256 slices so the
volume tiles into 128 x 128 x 64 crops. The 32^3 token grid is average-pooled to 16^3 and saved
as ``<out-dir>/<id>.npy`` with shape (4096, 1080), float16.

For MR (``--intensity unit``) the percentile-normalised volume ([-1, 1]) is multiplied by 1000 so
that it falls in the [-1000, 1000] window SPECTRE applies internally.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rootutils
import torch
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import mr_rate  # noqa: E402
from src.encoders.spectre import SpectreTeacher, pool_grid  # noqa: E402
from scripts.preprocess.common import ID_TO_PATH, read_ids  # noqa: E402
from scripts.preprocess.encode_wan_latents import resample_transform  # noqa: E402

DEPTH_PAD = 256
AIR_HU = -1000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--layout", choices=sorted(ID_TO_PATH), default="ctrate")
    ap.add_argument("--intensity", choices=["hu", "unit"], default="hu")
    ap.add_argument("--ckpt", required=True, help="spectre_backbone_vit_large_patch16_128_no_vla.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ids = read_ids(args.ids)[args.shard :: args.num_shards]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    to_path = ID_TO_PATH[args.layout]
    transform = resample_transform()
    teacher = SpectreTeacher(args.ckpt, device=args.device)

    for vol_id in tqdm(ids, desc="spectre"):
        out_path = out_dir / f"{vol_id}.npy"
        if out_path.exists():
            continue
        src = to_path(vol_id, args.data_root)
        if not src.is_file():
            print(f"missing: {src}", file=sys.stderr)
            continue
        vol = transform({"image": str(src)})["image"].as_tensor()  # (1, 512, 512, 253)
        if args.intensity == "unit":
            vol = torch.from_numpy(mr_rate.normalize_percentile(vol.numpy()) * 1000.0)
        vol = torch.nn.functional.pad(vol, (0, DEPTH_PAD - vol.shape[-1]), value=AIR_HU)  # (1, 512, 512, 256)
        dense = teacher.encode(vol)  # (32, 32, 32, 1080)
        grid = pool_grid(dense, (args.grid,) * 3)  # (16, 16, 16, 1080)
        np.save(out_path, grid.reshape(-1, grid.shape[-1]).to(torch.float16).cpu().numpy())


if __name__ == "__main__":
    main()
