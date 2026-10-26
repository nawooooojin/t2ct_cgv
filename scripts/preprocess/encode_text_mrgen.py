"""MR-Gen text features with RadFinder (run in the RadFinder environment, transformers >= 5).

Writes ``<out-dir>/<id>_rf_navg.npz`` with ``{findings,impression}_rf_navg`` (L, 1024) and
``{findings,impression}_rf_last`` (1024,), float16. The report is split into findings /
impression with `mr_rate.split_report`, the same way the challenge prompt is split at test time.

Training data (one report per study, written for every series id of the study):
    python scripts/preprocess/encode_text_mrgen.py --ids data/mrgen/ids_train.json \
        --reports-csv data/mrgen/reports_subset.csv --out-dir data/mrgen/text
Challenge prompts (``[{"input_image_name", "report"}]``):
    python scripts/preprocess/encode_text_mrgen.py --prompts /input/prompts.json --out-dir work/text
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import mr_rate  # noqa: E402
from src.encoders.radfinder import RadFinderTextEncoder  # noqa: E402
from scripts.preprocess.common import read_ids  # noqa: E402


def case_stem(name: str) -> str:
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem


def encode(rf: RadFinderTextEncoder, report: str) -> dict[str, np.ndarray]:
    findings, impression = mr_rate.split_report(report)
    out = {}
    for section, text in (("findings", findings), ("impression", impression)):
        navg, last = rf.encode_navg(text)
        out[f"{section}_rf_navg"] = navg.numpy().astype(np.float16)
        out[f"{section}_rf_last"] = last.numpy().astype(np.float16)
    return out


def save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="series ids (training data)")
    ap.add_argument("--reports-csv", help="MR-RATE reports CSV (study_uid, report)")
    ap.add_argument("--prompts", help="challenge prompts JSON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rf = RadFinderTextEncoder(device=args.device)

    if args.prompts:
        for e in tqdm(json.loads(Path(args.prompts).read_text()), desc="prompts"):
            save(out_dir / f"{case_stem(str(e['input_image_name']))}_rf_navg.npz", encode(rf, str(e["report"])))
        return

    reports = pd.read_csv(args.reports_csv).set_index("study_uid")["report"].fillna("")
    by_study: dict[str, list[str]] = {}
    for vol_id in read_ids(args.ids):
        by_study.setdefault(mr_rate.study_of(vol_id), []).append(vol_id)
    for study, vol_ids in tqdm(by_study.items(), desc="studies"):
        todo = [v for v in vol_ids if not (out_dir / f"{v}_rf_navg.npz").exists()]
        if not todo or study not in reports.index:
            continue
        arrays = encode(rf, str(reports.loc[study]))
        for v in todo:
            save(out_dir / f"{v}_rf_navg.npz", arrays)


if __name__ == "__main__":
    main()
