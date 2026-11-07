"""Encode CT volumes to Wan2.1 VAE latents: ``<out-dir>/<id>.nii.gz``, float32 ``(64, 64, 64, 16)``.

CT (X, Y, Z) -> resample (512, 512, 253) -> (T=Z, H=X, W=Y) -> WanVAE.encode (16, 64, 64, 64)
-> "c d h w -> h w d c" (64, 64, 64, 16), saved with the affine of the resampled grid (so the
header zooms are the resampled voxel size, read later as the spacing condition).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from einops import rearrange  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.preprocess.common import add_volume_args, resample_ct, select_ids  # noqa: E402
from src.data.ct_rate import volume_path  # noqa: E402
from src.encoders.wan_vae import DEFAULT_WAN_HF_ID, WanVAE  # noqa: E402


def main() -> None:
    """Encode every selected id that has no latent yet."""
    p = argparse.ArgumentParser(description=__doc__)
    add_volume_args(p)
    p.add_argument("--hf-id", default=DEFAULT_WAN_HF_ID, help="Wan VAE Hugging Face repo")
    args = p.parse_args()

    ids = select_ids(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vae = WanVAE(hf_id=args.hf_id, device=args.device)

    done = skipped = failed = 0
    for vol_id in tqdm(ids, desc="wan latents"):
        out_path = out_dir / f"{vol_id}.nii.gz"
        if out_path.exists():
            skipped += 1
            continue
        src = volume_path(args.data_root, vol_id)
        if not src.is_file():
            print(f"[miss] {vol_id}: {src}", file=sys.stderr)
            failed += 1
            continue
        try:
            vol, affine = resample_ct(src)  # (1, 512, 512, 253), (4, 4)
            ct_thw = vol[0].permute(2, 0, 1).contiguous()  # (253, 512, 512)
            z = vae.encode(ct_thw)  # (16, 64, 64, 64) = (C, D_lat, H_lat, W_lat)
            hwdc = rearrange(z.numpy(), "c d h w -> h w d c")  # (64, 64, 64, 16)
            nib.save(nib.Nifti1Image(np.float32(hwdc), affine=affine), str(out_path))
            done += 1
        except Exception as exc:  # noqa: BLE001 - one bad volume must not stop the run
            print(f"[fail] {vol_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
    print(f"done={done} skipped={skipped} failed={failed} -> {out_dir}")


if __name__ == "__main__":
    main()
