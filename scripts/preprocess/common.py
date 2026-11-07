"""Shared helpers for the preprocessing scripts: CT resampling, CLI arguments, id selection, atomic npz.

MONAI is imported inside ``resample_ct`` so the text script can import this module in the
RadFinder env.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Final

import numpy as np
import torch

from src.data.ct_rate import read_ids

IN_PLANE: Final[int] = 512
DEPTH: Final[int] = 253  # = 1 mod 4, so the Wan temporal x4 compression maps 253 -> 64 -> 253


def resample_ct(path: str | Path) -> tuple[torch.Tensor, np.ndarray]:
    """Load a CT, reorient to RAS and resample to the fixed 512 x 512 x 253 grid (no intensity change).

    Args:
        path: NIfTI volume in HU (CT-RATE ``*_fixed`` volumes already carry HU).

    Returns:
        Raw HU volume ``(1, 512, 512, 253)`` float32 (RAS, (X, Y, Z)) and the ``(4, 4)`` affine of
        that grid.
    """
    from monai import transforms as mt  # noqa: PLC0415

    tf = mt.Compose(
        [
            mt.LoadImaged(keys="image"),
            mt.EnsureChannelFirstd(keys="image"),
            mt.Orientationd(keys="image", axcodes="RAS"),
            mt.EnsureTyped(keys="image", dtype=torch.float32),
            mt.Resized(keys="image", spatial_size=(IN_PLANE, IN_PLANE, DEPTH), mode="trilinear"),
        ]
    )
    img = tf({"image": str(path)})["image"]  # MetaTensor (1, 512, 512, 253)
    affine = np.asarray(img.meta["affine"], dtype=np.float64)  # (4, 4)
    vol = img.as_tensor() if hasattr(img, "as_tensor") else torch.as_tensor(img)
    return vol.float(), affine


def add_volume_args(p: argparse.ArgumentParser) -> None:
    """Add the arguments shared by the volume encoders (ids, raw data, output, sharding)."""
    p.add_argument("--ids", required=True, help="id list (.txt one per line, or .json)")
    p.add_argument("--data-root", required=True, help="CT-RATE train_fixed/ or valid_fixed/ directory")
    p.add_argument("--layout", choices=("ctrate",), default="ctrate", help="raw-volume directory layout")
    p.add_argument("--intensity", choices=("hu",), default="hu", help="input intensity (raw HU)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-shards", type=int, default=1, help="split the ids into N interleaved shards")
    p.add_argument("--shard", type=int, default=0, help="this process's shard index in [0, N)")
    p.add_argument("--limit", type=int, default=None, help="process at most N ids (quick test)")


def select_ids(args: argparse.Namespace) -> list[str]:
    """This shard's ids: ``read_ids(args.ids)[shard::num_shards][:limit]``."""
    ids = read_ids(args.ids)[args.shard :: args.num_shards]
    return ids if args.limit is None else ids[: args.limit]


def save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed npz via ``<name>.tmp.npz`` + ``os.replace`` (never half-written).

    The temp name ends in ``.npz`` because ``np.savez`` appends it to any other suffix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
