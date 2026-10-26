"""MR-RATE helpers: volume ids, (sequence, plane) class labels, intensity, reports, lookup tables.

A volume id is ``<study_uid>_<series_id>`` with ``series_id = {modality}-{role}-{plane}[-N]``,
e.g. ``UWCMTFCZ47_t1w-raw-axi``. ``study_uid`` has no underscore, so the first ``_`` splits it.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

MODALITIES: tuple[str, ...] = ("t1w", "t2w", "flair", "swi")  # scored sequences; order fixes the labels
PLANES: tuple[str, ...] = ("axi", "sag", "cor", "obl")
# (modality, plane) -> 0..15, modality varying slowest. CT uses the fixed label 1 in its own model.
CLASS_LABELS: dict[tuple[str, str], int] = {pair: i for i, pair in enumerate(product(MODALITIES, PLANES))}

_ID_RE = re.compile(r"^(?P<study>[^_]+)_(?P<mod>[a-z0-9]+)-(?P<role>[a-z]+)-(?P<plane>[a-z]+)(?:-\d+)?$")


def parse_volume_id(vol_id: str) -> tuple[str, str, str, str]:
    """``"22B7CXEZ6T_t1w-raw-axi-2"`` -> ``("22B7CXEZ6T", "t1w", "raw", "axi")``."""
    m = _ID_RE.match(vol_id)
    if not m:
        raise ValueError(f"vol_id {vol_id!r} does not match <study>_<mod>-<role>-<plane>[-N]")
    return m["study"], m["mod"].lower(), m["role"].lower(), m["plane"].lower()


def study_of(vol_id: str) -> str:
    return parse_volume_id(vol_id)[0]


def class_label(vol_id: str) -> int:
    """(modality, plane) class id; ValueError for an unscored modality or unknown plane."""
    _, mod, _, plane = parse_volume_id(vol_id)
    try:
        return CLASS_LABELS[(mod, plane)]
    except KeyError as exc:
        raise ValueError(f"vol_id {vol_id!r} -> ({mod!r}, {plane!r}) is not a known class") from exc


def id_to_nifti_path(vol_id: str, root: str | Path) -> Path:
    """``<root>/<study_uid>/img/<vol_id>.nii.gz`` (extracted MR-RATE layout)."""
    return Path(root) / study_of(vol_id) / "img" / f"{vol_id}.nii.gz"


def normalize_percentile(vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Per-volume percentile clip to [0, 1] (the challenge scorer's statistic), rescaled to [-1, 1].

    A (near-)constant volume returns zeros.
    """
    vol = vol.astype(np.float32)
    lo, hi = np.percentile(vol, lo_pct), np.percentile(vol, hi_pct)
    if hi - lo < 1e-6:
        return np.zeros_like(vol)
    unit = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (2.0 * unit - 1.0).astype(np.float32)


_FINDINGS_RE = re.compile(r"(?im)^\s*findings\s*:\s*")
_IMPRESSION_RE = re.compile(r"(?im)^\s*impression\s*:\s*")


def split_report(report: str) -> tuple[str, str]:
    """Report -> (findings, impression) at line-start ``Findings:`` / ``Impression:`` headers.

    Without a ``Findings:`` header the whole report is the findings and the impression is empty.
    """
    report = str(report)
    fm = _FINDINGS_RE.search(report)
    if not fm:
        return report.strip(), ""
    im = _IMPRESSION_RE.search(report, fm.end())
    if im:
        return report[fm.end() : im.start()].strip(), report[im.end() :].strip()
    return report[fm.end() :].strip(), ""


def median_spacing_by_label(records: list[tuple[str, list[float]]]) -> dict[int, list[float]]:
    """``(vol_id, [x, y, z])`` pairs -> median spacing per class label (inference spacing table)."""
    by_label: dict[int, list[list[float]]] = {}
    for vol_id, spacing in records:
        try:
            lbl = class_label(vol_id)
        except ValueError:
            continue
        by_label.setdefault(lbl, []).append(spacing)
    return {lbl: list(np.median(np.asarray(sps), axis=0)) for lbl, sps in by_label.items()}


def axcodes_mode_by_label(records: list[tuple[str, tuple[str, str, str]]]) -> dict[int, tuple[str, str, str]]:
    """``(vol_id, axcodes)`` pairs -> most common axcodes per class label (output orientation table)."""
    votes: dict[int, Counter] = {}
    for vol_id, axcodes in records:
        try:
            lbl = class_label(vol_id)
        except ValueError:
            continue
        votes.setdefault(lbl, Counter())[tuple(axcodes)] += 1
    return {lbl: c.most_common(1)[0][0] for lbl, c in votes.items()}
