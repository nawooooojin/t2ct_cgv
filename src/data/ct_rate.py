"""CT-RATE helpers: id lists, the volume path rule, report CSVs and challenge prompts (stdlib only at import)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

_IMPRESSION = re.compile(r"\bimpressions?\s*:", re.IGNORECASE)
_FINDINGS_LABEL = re.compile(r"^\s*findings\s*:", re.IGNORECASE)
_SUFFIXES: tuple[str, ...] = (".nii.gz", ".nii", ".mha", ".nrrd")


def read_ids(path: str | Path) -> list[str]:
    """Read volume ids from a ``.txt`` (one per line) or ``.json`` (``{"ids": [...]}`` or a list) file."""
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        ids = data["ids"] if isinstance(data, dict) else data
        return [str(i).strip() for i in ids if str(i).strip()]
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def volume_path(split_dir: str | Path, scan_id: str) -> Path:
    """``valid_1_a_1`` -> ``<split_dir>/valid_1/valid_1_a/valid_1_a_1.nii.gz``."""
    series = scan_id.rsplit("_", 1)[0]  # valid_1_a
    patient = series.rsplit("_", 1)[0]  # valid_1
    return Path(split_dir) / patient / series / f"{scan_id}.nii.gz"


def scan_key(vol_id: str) -> str:
    """``train_10000_a_1`` -> ``train_10000_a`` (reconstructions of one scan share a report)."""
    return "_".join(vol_id.split("_")[:3])


def report_text(cell: object) -> str:
    """A report CSV cell as text; a blank (NaN/None) cell is ``""``, never ``"nan"``."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return ""
    return str(cell)


def load_reports(
    csv_path: str | Path,
    id_col: str = "VolumeName",
    findings_col: str = "Findings_EN",
    impression_col: str = "Impressions_EN",
) -> dict[str, tuple[str, str]]:
    """Read a CT-RATE report CSV into ``{scan_key: (findings, impression)}``."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    return {
        scan_key(str(row[id_col]).removesuffix(".nii.gz")): (
            report_text(row[findings_col]),
            report_text(row[impression_col]),
        )
        for _, row in df.iterrows()
    }


def split_report(report: str) -> tuple[str, str]:
    """Split a challenge prompt into ``(findings, impression)`` at the first ``Impression(s):`` marker.

    A leading ``Findings:`` label is stripped. Without an impression marker the whole text is findings.
    """
    m = _IMPRESSION.search(report)
    if m is None:
        return _FINDINGS_LABEL.sub("", report.strip(), count=1).strip(), ""
    findings = _FINDINGS_LABEL.sub("", report[: m.start()], count=1).strip()
    return findings, report[m.end() :].strip()


def prompt_stem(input_image_name: str) -> str:
    """Strip a volume suffix (``.nii.gz``, ``.nii``, ``.mha``, ``.nrrd``) from a prompt's file name."""
    for suffix in _SUFFIXES:
        if input_image_name.endswith(suffix):
            return input_image_name[: -len(suffix)]
    return input_image_name
