"""Encode reports to RadFinder text sidecars ``<out-dir>/<id>.npz`` (run in the t2ct-radfinder env).

Sidecar = four float16 arrays: ``{findings,impression}_rf_navg`` ``(L, 1024)`` (29-layer
RMS-normalise-then-mean tokens, real length, EOS last) and ``{findings,impression}_rf_last``
``(1024,)`` (final-layer EOS, RMS-normalised). A blank section is encoded as ``""`` (L = 1).

Two inputs:
  --ids IDS --reports-csv CSV   training data; volumes of one scan share one report (encoded once)
  --prompts prompts.json        challenge prompts [{"input_image_name", "report"}]; the report is
                                split into findings / impression, output named by the image stem
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (no rootutils in this env)

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from scripts.preprocess.common import save_npz_atomic  # noqa: E402
from src.data.ct_rate import load_reports, prompt_stem, read_ids, scan_key, split_report  # noqa: E402
from src.encoders.radfinder import RadFinderTextEncoder  # noqa: E402


def encode_report(rf: RadFinderTextEncoder, findings: str, impression: str) -> dict[str, np.ndarray]:
    """Both sections of one report -> the four sidecar arrays (float16)."""
    out: dict[str, np.ndarray] = {}
    for section, text in (("findings", findings), ("impression", impression)):
        navg, last = rf.encode_navg(text)  # (L, 1024), (1024,) float32
        out[f"{section}_rf_navg"] = navg.numpy().astype(np.float16)
        out[f"{section}_rf_last"] = last.numpy().astype(np.float16)
    return out


def jobs_from_ids(args: argparse.Namespace) -> list[tuple[str, str, str, list[Path]]]:
    """``(key, findings, impression, out_paths)`` per distinct scan of ``--ids``."""
    reports = load_reports(args.reports_csv)
    out_dir = Path(args.out_dir)
    by_scan: dict[str, list[Path]] = {}
    for vol_id in read_ids(args.ids):
        by_scan.setdefault(scan_key(vol_id), []).append(out_dir / f"{vol_id}.npz")
    jobs = []
    for key, paths in by_scan.items():
        if key not in reports:
            print(f"[warn] no report row for {key}; skipped {len(paths)} volume(s)", file=sys.stderr)
            continue
        findings, impression = reports[key]
        jobs.append((key, findings, impression, paths))
    return jobs


def jobs_from_prompts(args: argparse.Namespace) -> list[tuple[str, str, str, list[Path]]]:
    """``(name, findings, impression, [out_path])`` per challenge prompt."""
    prompts = json.loads(Path(args.prompts).read_text())
    out_dir = Path(args.out_dir)
    jobs = []
    for item in prompts:
        name = item["input_image_name"]
        findings, impression = split_report(item["report"])
        jobs.append((name, findings, impression, [out_dir / f"{prompt_stem(name)}.npz"]))
    return jobs


def main() -> None:
    """Encode every job whose sidecars are not all written yet."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--ids", help="volume id list (.txt or .json); needs --reports-csv")
    inp.add_argument("--prompts", help="challenge prompts.json")
    p.add_argument("--reports-csv", help="CT-RATE report CSV (VolumeName, Findings_EN, Impressions_EN)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="encode at most N reports (quick test)")
    args = p.parse_args()
    if args.ids and not args.reports_csv:
        p.error("--ids needs --reports-csv")

    jobs = jobs_from_ids(args) if args.ids else jobs_from_prompts(args)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rf = RadFinderTextEncoder(device=args.device)

    done = skipped = 0
    for key, findings, impression, paths in tqdm(jobs, desc="radfinder"):
        if all(path.is_file() for path in paths):
            skipped += 1
            continue
        for section, text in (("findings", findings), ("impression", impression)):
            if not text:
                print(f"[blank] {section} of {key} -> encoded as ''", file=sys.stderr)
        arrays = encode_report(rf, findings, impression)
        for path in paths:
            if not path.is_file():
                save_npz_atomic(path, arrays)
        done += 1
    print(f"done={done} skipped={skipped} -> {args.out_dir}")


if __name__ == "__main__":
    main()
