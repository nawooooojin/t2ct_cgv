"""Build the training datalist from the precomputed features.

For every id with a latent, writes ``<cond-dir>/<id>_cond.json`` (resampled spacing from the latent
header, plus the CT text embeddings for CT-Gen) and a datalist entry.

    python scripts/preprocess/build_datalist.py --task ctgen --root data/ctgen \
        --ids-train data/ctgen/ids_train.txt --ids-valid data/ctgen/ids_valid.txt
    python scripts/preprocess/build_datalist.py --task mrgen --root data/mrgen \
        --ids-train data/mrgen/ids_train.json --ids-valid data/mrgen/ids_valid.json --raw-dir data/mrgen/raw

Expected layout under --root: latents/<id>_emb.nii.gz, spectre_ssl_16/<id>.npy and
text/<id>_text.json (CT) or text/<id>_rf_navg.npz (MR).

For MR-Gen two lookup tables are also written, both keyed by (sequence, plane) class label:
spacing_table.json (median spacing, used as the spacing condition at inference) and, with
--raw-dir, axcodes_table.json (most common array orientation of the real volumes, used when
saving generated volumes).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import mr_rate  # noqa: E402
from scripts.preprocess.common import read_ids  # noqa: E402


def build_split(ids, root: Path, cond_dir: Path, task: str, spacings: list) -> list[dict]:
    entries = []
    for vol_id in tqdm(ids, leave=False):
        image = root / "latents" / f"{vol_id}_emb.nii.gz"
        spectre = root / "spectre_ssl_16" / f"{vol_id}.npy"
        text = root / "text" / (f"{vol_id}_text.json" if task == "ctgen" else f"{vol_id}_rf_navg.npz")
        if not (image.is_file() and spectre.is_file() and text.is_file()):
            continue
        entry = {"image": str(image.resolve()), "spectre": str(spectre.resolve())}
        if task == "mrgen":
            try:
                entry["class_label"] = mr_rate.class_label(vol_id)
            except ValueError:
                continue
            entry["text"] = str(text.resolve())

        spacing = [float(s) for s in nib.load(str(image)).header.get_zooms()[:3]]
        cond = {"spacing": spacing}
        if task == "ctgen":
            cond.update(json.loads(text.read_text()))
        cond_path = cond_dir / f"{vol_id}_cond.json"
        cond_path.write_text(json.dumps(cond))
        entry["cond"] = str(cond_path.resolve())
        entries.append(entry)
        spacings.append((vol_id, spacing))
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ctgen", "mrgen"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ids-train", required=True)
    ap.add_argument("--ids-valid", required=True)
    ap.add_argument("--raw-dir", default=None, help="MR-RATE raw volumes (for axcodes_table.json)")
    args = ap.parse_args()

    root = Path(args.root)
    cond_dir = root / "cond"
    cond_dir.mkdir(parents=True, exist_ok=True)
    spacings: list = []
    datalist = {
        "training": build_split(read_ids(args.ids_train), root, cond_dir, args.task, spacings),
        "validation": build_split(read_ids(args.ids_valid), root, cond_dir, args.task, spacings),
    }
    (root / "datalist.json").write_text(json.dumps(datalist, indent=1))
    print(f"datalist: {len(datalist['training'])} train / {len(datalist['validation'])} valid")

    if args.task == "mrgen":
        table = mr_rate.median_spacing_by_label(spacings)
        (root / "spacing_table.json").write_text(json.dumps(table, indent=1))
        if args.raw_dir:
            records = []
            for f in sorted(Path(args.raw_dir).glob("*/img/*.nii.gz")):
                records.append((f.name[: -len(".nii.gz")], nib.aff2axcodes(nib.load(str(f)).affine)))
            axcodes = mr_rate.axcodes_mode_by_label(records)
            (root / "axcodes_table.json").write_text(json.dumps({k: list(v) for k, v in axcodes.items()}, indent=1))


if __name__ == "__main__":
    main()
