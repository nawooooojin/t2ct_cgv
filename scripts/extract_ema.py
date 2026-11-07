"""Write a standalone EMA checkpoint from a Lightning training checkpoint.

``EMACallback`` stores its shadow in ``ckpt["callbacks"]["EMACallback"]["shadow"]``. Parameters
are taken from the shadow; buffers (``scale_factor``, a data statistic) stay raw.

Usage: python scripts/extract_ema.py <run>/checkpoints/last.ckpt weights/warp_ct_ema.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    """Overlay the EMA shadow on the raw state_dict and save ``{"state_dict": ...}``."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="Lightning .ckpt saved with EMACallback")
    ap.add_argument("dst", help="output .ckpt")
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    shadow = ckpt.get("callbacks", {}).get("EMACallback", {}).get("shadow")
    if shadow is None:
        raise SystemExit(f"no EMA shadow in {args.src} (trained without EMACallback?)")
    state = {k: v.detach().cpu() for k, v in ckpt["state_dict"].items()}
    unknown = set(shadow) - set(state)
    if unknown:
        raise SystemExit(f"EMA keys absent from state_dict: {sorted(unknown)}")
    state.update({k: v.detach().cpu() for k, v in shadow.items()})

    out: dict = {"state_dict": state}
    if ckpt.get("hyper_parameters") is not None:
        out["hyper_parameters"] = ckpt["hyper_parameters"]
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"wrote {dst}: {len(shadow)} EMA tensors, {len(state) - len(shadow)} raw buffers")


if __name__ == "__main__":
    main()
