"""Select and download the MR-RATE training subset (requires access to the gated HF dataset).

`select`: from the MR-RATE `train` split, shuffle patients (seed 42) and take one random study per
patient: 1,200 studies for training and 150 for local validation. Only raw (non-derived,
non-localizer) T1w / T2w / FLAIR / SWI series in axial / sagittal / coronal / oblique planes are kept.

`download`: fetch the selected study zips and extract the image volumes to ``<out-dir>/raw``.

    python scripts/preprocess/select_mrrate_subset.py select --out-dir data/mrgen
    python scripts/preprocess/select_mrrate_subset.py download --out-dir data/mrgen --split train
    python scripts/preprocess/select_mrrate_subset.py download --out-dir data/mrgen --split valid
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import pandas as pd
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import mr_rate  # noqa: E402

REPO_ID = "Forithmus/MR-RATE"
N_BATCHES = 28


def _csvs(local: Path, sub: str, suffix: str) -> pd.DataFrame:
    files = [local / sub / f"batch{b:02d}_{suffix}.csv" for b in range(N_BATCHES)]
    return pd.concat([pd.read_csv(f) for f in files if f.is_file()], ignore_index=True)


def cmd_select(args) -> None:
    from huggingface_hub import snapshot_download

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    local = Path(snapshot_download(REPO_ID, repo_type="dataset",
                                   allow_patterns=["splits.csv", "metadata/*.csv", "reports/*.csv"]))
    splits = pd.read_csv(local / "splits.csv")
    by_patient = splits[splits["split"] == "train"].groupby("patient_uid")["study_uid"].apply(list)
    patients = sorted(by_patient.index)
    rng = random.Random(args.seed)
    rng.shuffle(patients)
    chosen = patients[: args.n_train + args.n_valid]
    train = sorted(str(rng.choice(by_patient[p])) for p in chosen[: args.n_train])
    valid = sorted(str(rng.choice(by_patient[p])) for p in chosen[args.n_train:])

    meta = _csvs(local, "metadata", "metadata")
    meta = meta[
        meta["classified_modality"].astype(str).str.lower().isin(mr_rate.MODALITIES)
        & ~meta["is_derived"].astype(str).str.lower().eq("true")
        & ~meta["is_localizer"].astype(str).str.lower().eq("true")
    ].copy()
    meta["plane"] = meta["acquisition_plane"].astype(str).str.lower().str[:3]
    meta = meta[meta["plane"].isin(mr_rate.PLANES)]

    def ids_for(studies):
        rows = meta[meta["study_uid"].isin(studies)]
        return sorted(f"{r.study_uid}_{r.classified_modality.lower()}-raw-{r.plane}" for r in rows.itertuples())

    (out / "subset.json").write_text(json.dumps({"seed": args.seed, "train_studies": train, "valid_studies": valid}))
    (out / "ids_train.json").write_text(json.dumps({"ids": ids_for(train)}))
    (out / "ids_valid.json").write_text(json.dumps({"ids": ids_for(valid)}))
    reports = _csvs(local, "reports", "reports")
    reports[reports["study_uid"].isin(set(train) | set(valid))].to_csv(out / "reports_subset.csv", index=False)
    print(f"{len(train)} train / {len(valid)} valid studies")


def cmd_download(args) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    out = Path(args.out_dir)
    studies = json.loads((out / "subset.json").read_text())[f"{args.split}_studies"]
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    zip_paths: dict[str, str] = {}
    api = HfApi()
    for b in range(N_BATCHES):
        prefix = f"mri/batch{b:02d}"
        for f in api.list_repo_tree(REPO_ID, path_in_repo=prefix, repo_type="dataset"):
            name = Path(f.path).name
            if name.endswith(".zip"):
                zip_paths[name[:-4]] = f"{prefix}/{name}"

    for i, study in enumerate(studies, 1):
        if (raw / study / "img").is_dir():
            continue
        local_zip = hf_hub_download(REPO_ID, zip_paths[study], repo_type="dataset")
        with zipfile.ZipFile(local_zip) as z:
            for name in z.namelist():
                fname = Path(name).name
                if "/img/" in name and any(f"_{m}-raw-" in fname for m in mr_rate.MODALITIES):
                    z.extract(name, raw)
        if i % 50 == 0:
            print(f"{i}/{len(studies)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--n-train", type=int, default=1200)
    s.add_argument("--n-valid", type=int, default=150)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_select)
    d = sub.add_parser("download")
    d.add_argument("--out-dir", required=True)
    d.add_argument("--split", choices=["train", "valid"], required=True)
    d.set_defaults(func=cmd_download)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
