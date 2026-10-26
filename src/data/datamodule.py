"""Training data: precomputed Wan latents, SPECTRE teacher tokens and text features.

Datalist (``scripts/preprocess/build_datalist.py``): ``{"training": [...], "validation": [...]}``,
each entry ``{"image": <id>_emb.nii.gz, "spectre": <id>.npy, "cond": <id>_cond.json}`` plus, for
MR-Gen, ``"text": <id>_rf_navg.npz`` and ``"class_label"``.

Batch keys
    image    (B, 16, 64, 64, 64)  Wan latent
    spacing  (B, 3)               resampled voxel spacing (mm) x `spacing_multiplier`
    spectre  (B, 4096, 1080)      teacher tokens (16^3 grid)
    CT-Gen : context_f / context_i       (B, 2560)  pooled findings / impression embeddings
    MR-Gen : context_{f,i}_seq           (B, T, 1024) RadFinder token sequence, right-padded
             context_{f,i}_mask          (B, T)       True = real token
             context_{f,i}               (B, 2048)    global vector (computed before padding)
             class_label                 (B,)         (sequence, plane) class
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from monai.data import DataLoader, Dataset
from monai.data.utils import list_data_collate
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

from src.data.rf_navg import global_vector, pad_sequences

_SECTIONS = {"f": "findings", "i": "impression"}


class _LoadFeatures:
    """Replaces the `cond` / `spectre` / `text` paths of an entry with tensors."""

    def __init__(self, task: str, spacing_multiplier: float) -> None:
        self.task = task
        self.spacing_multiplier = spacing_multiplier

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        cond = json.loads(Path(data.pop("cond")).read_text())
        data["spacing"] = torch.tensor(cond["spacing"], dtype=torch.float32) * self.spacing_multiplier
        data["spectre"] = torch.from_numpy(np.load(data["spectre"])).float()
        if self.task == "ctgen":
            data["context_f"] = torch.tensor(cond["findings_embeddings"], dtype=torch.float32)
            data["context_i"] = torch.tensor(cond["impression_embeddings"], dtype=torch.float32)
        else:
            npz = np.load(data.pop("text"))
            for short, section in _SECTIONS.items():
                seq = torch.from_numpy(npz[f"{section}_rf_navg"].astype(np.float32))  # (L, 1024)
                eos = torch.from_numpy(npz[f"{section}_rf_last"].astype(np.float32))  # (1024,)
                data[f"context_{short}_seq"] = seq
                data[f"context_{short}"] = global_vector(eos, seq, torch.ones(len(seq), dtype=torch.bool))
        return data


def mrgen_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pads the ragged text sequences and builds their masks; everything else is stacked."""
    rest = [dict(b) for b in batch]
    seqs = {k: [b.pop(k) for b in rest] for k in ("context_f_seq", "context_i_seq")}
    out = list_data_collate(rest)
    for key, s in seqs.items():
        out[key], out[key.replace("_seq", "_mask")] = pad_sequences(s)
    return out


class VolumeDataModule(LightningDataModule):
    def __init__(
        self,
        datalist_path: str,
        task: str = "ctgen",
        batch_size: int = 2,
        num_workers: int = 8,
        spacing_multiplier: float = 100.0,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        if task not in ("ctgen", "mrgen"):
            raise ValueError(f"task must be 'ctgen' or 'mrgen', got {task!r}")
        self.datalist_path = Path(datalist_path)
        self.task = task
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.spacing_multiplier = spacing_multiplier
        self.pin_memory = pin_memory
        self.train_ds: Dataset | None = None
        self.val_ds: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        datalist = json.loads(self.datalist_path.read_text())
        transform = Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),  # (64, 64, 64, 16) -> (16, 64, 64, 64)
            _LoadFeatures(self.task, self.spacing_multiplier),
        ])
        self.train_ds = Dataset(datalist["training"], transform=transform)
        self.val_ds = Dataset(datalist["validation"], transform=transform)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        kw = {"collate_fn": mrgen_collate} if self.task == "mrgen" else {}
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, **kw)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)
