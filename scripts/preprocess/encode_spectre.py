"""Encode CT volumes to SPECTRE-Large tokens: ``<out-dir>/<id>.npy``, float16 ``(4096, 1080)``.

CT -> resample (1, 512, 512, 253) raw HU -> depth end-pad to 256 with -1000 HU -> 64 crops
-> native grid (32, 32, 32, 1080) -> 16^3 average pool -> (4096, 1080), token index
(h * 16 + w) * 16 + d. TF32 is on by default (agrees with fp32 to cosine ~0.99998).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.preprocess.common import add_volume_args, resample_ct, select_ids  # noqa: E402
from src.data.ct_rate import volume_path  # noqa: E402
from src.encoders.spectre import SpectreBackbone  # noqa: E402

DEPTH_PAD = 256  # multiple of the SPECTRE crop depth (64)
AIR_HU = -1000.0
OUT_GRID = (16, 16, 16)


def main() -> None:
    """Encode every selected id that has no token file yet."""
    p = argparse.ArgumentParser(description=__doc__)
    add_volume_args(p)
    p.add_argument("--ckpt", required=True, help="spectre_backbone_vit_large_patch16_128_no_vla.pt")
    p.add_argument("--max-crops-per-forward", type=int, default=16)
    p.add_argument("--no-tf32", dest="tf32", action="store_false", help="disable TF32 matmuls")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    ids = select_ids(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = SpectreBackbone(args.ckpt, device=args.device)

    done = skipped = failed = 0
    for vol_id in tqdm(ids, desc="spectre"):
        out_path = out_dir / f"{vol_id}.npy"
        if out_path.exists():
            skipped += 1
            continue
        src = volume_path(args.data_root, vol_id)
        if not src.is_file():
            print(f"[miss] {vol_id}: {src}", file=sys.stderr)
            failed += 1
            continue
        try:
            vol, _ = resample_ct(src)  # (1, 512, 512, 253) raw HU
            vol = F.pad(vol, (0, DEPTH_PAD - vol.shape[-1]), value=AIR_HU)  # (1, 512, 512, 256)
            dense = backbone.encode_dense(vol, args.max_crops_per_forward)  # (32, 32, 32, 1080)
            grid = SpectreBackbone.pool_grid(dense, OUT_GRID)  # (16, 16, 16, 1080)
            flat = grid.reshape(-1, grid.shape[-1])  # (4096, 1080)
            np.save(out_path, flat.to(torch.float16).cpu().numpy())
            done += 1
        except Exception as exc:  # noqa: BLE001 - one bad volume must not stop the run
            print(f"[fail] {vol_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
    print(f"done={done} skipped={skipped} failed={failed} -> {out_dir}")


if __name__ == "__main__":
    main()
