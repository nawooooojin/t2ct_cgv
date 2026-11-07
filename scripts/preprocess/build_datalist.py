"""Build ``<root>/datalist.json`` from the precomputed latents, text sidecars and SPECTRE tokens.

    python scripts/preprocess/build_datalist.py --task ctgen --root data/ctgen \
        --ids-train ids/ctrate_train.txt --ids-valid ids/ctrate_valid.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.ct_rate import read_ids  # noqa: E402


def build_split(ids: list[str], root: Path) -> tuple[list[dict], list[str]]:
    """Datalist entries for the ids whose three inputs all exist, and the skipped ids."""
    entries: list[dict] = []
    skipped: list[str] = []
    for vol_id in ids:
        image = root / "latents" / f"{vol_id}.nii.gz"
        text = root / "text" / f"{vol_id}.npz"
        spectre = root / "spectre_ssl_16" / f"{vol_id}.npy"
        if not (image.is_file() and text.is_file() and spectre.is_file()):
            skipped.append(vol_id)
            continue
        spacing = [float(s) for s in nib.load(str(image)).header.get_zooms()[:3]]  # mm, not multiplied
        entries.append(
            {
                "id": vol_id,
                "image": str(image.resolve()),
                "text": str(text.resolve()),
                "spectre": str(spectre.resolve()),
                "spacing": spacing,
            }
        )
    return entries, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", choices=("ctgen",), default="ctgen")
    p.add_argument("--root", required=True, help="data/<task> with latents/, text/, spectre_ssl_16/")
    p.add_argument("--ids-train", required=True)
    p.add_argument("--ids-valid", required=True)
    args = p.parse_args()

    root = Path(args.root)
    datalist: dict[str, list[dict]] = {}
    for split, ids_path in (("training", args.ids_train), ("validation", args.ids_valid)):
        entries, skipped = build_split(read_ids(ids_path), root)
        datalist[split] = entries
        print(f"{split}: {len(entries)} included, {len(skipped)} skipped" + (f" (e.g. {skipped[:5]})" if skipped else ""))
    if not datalist["training"]:
        print("ERROR: no training entries; run the preprocessing scripts first", file=sys.stderr)
        sys.exit(1)
    out = root / "datalist.json"
    out.write_text(json.dumps(datalist, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
