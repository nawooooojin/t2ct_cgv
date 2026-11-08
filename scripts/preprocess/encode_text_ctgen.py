"""WARP-CT text features with RadFinder (run in the RadFinder environment, transformers >= 5).

Writes ``<out-dir>/<id>_rf_navg.npz`` in the same format as encode_text_mrgen.py.

Training data, from the CT-RATE report CSV (columns VolumeName, Findings_EN, Impressions_EN):
    python scripts/preprocess/encode_text_ctgen.py --ids ids/ctrate_train.txt \
        --reports-csv /path/CT-RATE/dataset/radiology_text_reports/train_reports.csv --out-dir data/ctgen/text
Challenge prompts (``[{"input_image_name", "report"}]``, split with `ct_rate.split_report`):
    python scripts/preprocess/encode_text_ctgen.py --prompts /input/prompts.json --out-dir work/text
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import ct_rate  # noqa: E402
from src.encoders.radfinder import RadFinderTextEncoder  # noqa: E402
from scripts.preprocess.common import read_ids  # noqa: E402
from scripts.preprocess.encode_text_mrgen import case_stem, encode_sections, save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", help="volume ids (training data)")
    ap.add_argument("--reports-csv", help="CT-RATE reports CSV")
    ap.add_argument("--prompts", help="challenge prompts JSON")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not args.prompts and not (args.ids and args.reports_csv):
        ap.error("give --prompts, or --ids and --reports-csv")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rf = RadFinderTextEncoder(device=args.device)

    if args.prompts:
        for e in tqdm(json.loads(Path(args.prompts).read_text()), desc="prompts"):
            arrays = encode_sections(rf, *ct_rate.split_report(str(e["report"])))
            save(out_dir / f"{case_stem(str(e['input_image_name']))}_rf_navg.npz", arrays)
        return

    df = pd.read_csv(args.reports_csv)
    df.index = df["VolumeName"].str.replace(".nii.gz", "", regex=False)
    for vol_id in tqdm(read_ids(args.ids), desc="text"):
        out = out_dir / f"{vol_id}_rf_navg.npz"
        if out.exists():
            continue
        if vol_id not in df.index:
            print(f"missing report: {vol_id}", file=sys.stderr)
            continue
        row = df.loc[vol_id]
        # str() as for the released model: a blank cell is encoded as the text "nan"
        save(out, encode_sections(rf, str(row["Findings_EN"]), str(row["Impressions_EN"])))


if __name__ == "__main__":
    main()
