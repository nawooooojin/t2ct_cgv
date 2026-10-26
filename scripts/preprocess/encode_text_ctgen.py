"""CT-Gen text features: pooled 3-encoder embeddings of findings and impression.

Writes ``<out-dir>/<id>_text.json`` = {"findings_embeddings": [2560], "impression_embeddings": [2560]}
from the CT-RATE report CSV (columns VolumeName, Findings_EN, Impressions_EN).

    python scripts/preprocess/encode_text_ctgen.py --ids ids/ctrate_train.txt \
        --reports-csv /path/CT-RATE/dataset/radiology_text_reports/train_reports.csv \
        --out-dir data/ctgen/text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.encoders.report_text import ReportTextEncoder  # noqa: E402
from scripts.preprocess.common import read_ids  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--reports-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    df = pd.read_csv(args.reports_csv)
    df.index = df["VolumeName"].str.replace(".nii.gz", "", regex=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = ReportTextEncoder(device=args.device)

    for vol_id in tqdm(read_ids(args.ids), desc="text"):
        out = out_dir / f"{vol_id}_text.json"
        if out.exists() or vol_id not in df.index:
            continue
        row = df.loc[vol_id]
        f, i = enc.encode_pair(str(row.get("Findings_EN", "") or ""), str(row.get("Impressions_EN", "") or ""))
        out.write_text(json.dumps({"findings_embeddings": f.tolist(), "impression_embeddings": i.tolist()}))


if __name__ == "__main__":
    main()
