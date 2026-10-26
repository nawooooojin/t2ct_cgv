"""Small helpers shared by the preprocessing scripts."""

from __future__ import annotations

import json
from pathlib import Path

from src.data import mr_rate


def read_ids(path: str | Path) -> list[str]:
    """Volume ids from a .txt (one per line) or a .json (``{"ids": [...]}`` or a list)."""
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        ids = data["ids"] if isinstance(data, dict) else data
    else:
        ids = path.read_text().splitlines()
    return [str(i).strip() for i in ids if str(i).strip()]


def ctrate_path(vol_id: str, root: str | Path) -> Path:
    """``train_10000_a_1`` -> ``<root>/train_10000/train_10000_a/train_10000_a_1.nii.gz``."""
    p = vol_id.split("_")
    return Path(root) / "_".join(p[:2]) / "_".join(p[:3]) / f"{vol_id}.nii.gz"


ID_TO_PATH = {"ctrate": ctrate_path, "mrrate": mr_rate.id_to_nifti_path}
