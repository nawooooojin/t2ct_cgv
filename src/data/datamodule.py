"""CT-RATE Wan-latent datamodule: latent + RadFinder navg sidecar + SPECTRE teacher tokens."""

from __future__ import annotations

import json
from typing import Any

import nibabel as nib
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.rf_navg import load_text_sidecar, pad_sequences


class CTRateLatentDataset(Dataset):
    """One datalist split (§ datalist.json entries) as per-sample tensors.

    Args:
        entries: dicts with ``id``, ``image``, ``text``, ``spectre``, ``spacing`` (mm, ``[3]``).
        spacing_multiplier: factor applied to the spacing fed to the UNet.
        load_spectre: also load the ``(4096, 1080)`` SPECTRE tokens for REPA.
    """

    def __init__(
        self, entries: list[dict], spacing_multiplier: float = 100.0, load_spectre: bool = True
    ) -> None:
        self.entries = entries
        self.spacing_multiplier = spacing_multiplier
        self.load_spectre = load_spectre

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict[str, Any]:
        """Load one sample.

        Returns:
            ``id``; ``image`` ``(16, 64, 64, 64)`` (C, H, W, D); ``spacing`` ``(3,)`` (mm x multiplier);
            ``seq_f`` ``(L_f, 1024)``; ``seq_i`` ``(L_i, 1024)``; ``pooled_f``/``pooled_i`` ``(2048,)``;
            ``spectre`` ``(4096, 1080)`` if ``load_spectre``. All float32.
        """
        e = self.entries[i]
        arr = np.asarray(nib.load(e["image"]).dataobj, dtype=np.float32)  # (64, 64, 64, 16) HWDC
        sample: dict[str, Any] = {
            "id": e["id"],
            "image": torch.from_numpy(np.ascontiguousarray(np.moveaxis(arr, -1, 0))),  # (16, 64, 64, 64)
            "spacing": torch.tensor(e["spacing"], dtype=torch.float32) * self.spacing_multiplier,  # (3,)
            **load_text_sidecar(e["text"]),
        }
        if self.load_spectre:
            sample["spectre"] = torch.from_numpy(np.load(e["spectre"])).float()  # (4096, 1080)
        return sample


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack fixed-size fields; pad each text section to its own batch max, then concat findings | impression.

    Returns:
        ``id`` list; ``image`` ``(B, 16, 64, 64, 64)``; ``spacing`` ``(B, 3)``;
        ``context`` ``(B, T_f + T_i, 1024)``; ``context_mask`` ``(B, T_f + T_i)`` bool;
        ``pooled_f``/``pooled_i`` ``(B, 2048)``; ``spectre`` ``(B, 4096, 1080)`` when present.
    """
    seq_f, mask_f = pad_sequences([b["seq_f"] for b in batch])  # (B, T_f, 1024), (B, T_f)
    seq_i, mask_i = pad_sequences([b["seq_i"] for b in batch])  # (B, T_i, 1024), (B, T_i)
    out: dict[str, Any] = {
        "id": [b["id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),  # (B, 16, 64, 64, 64)
        "spacing": torch.stack([b["spacing"] for b in batch]),  # (B, 3)
        "context": torch.cat([seq_f, seq_i], dim=1),  # (B, T_f + T_i, 1024)
        "context_mask": torch.cat([mask_f, mask_i], dim=1),  # (B, T_f + T_i)
        "pooled_f": torch.stack([b["pooled_f"] for b in batch]),  # (B, 2048)
        "pooled_i": torch.stack([b["pooled_i"] for b in batch]),  # (B, 2048)
    }
    if "spectre" in batch[0]:
        out["spectre"] = torch.stack([b["spectre"] for b in batch])  # (B, 4096, 1080)
    return out


class CTRateLatentDataModule(LightningDataModule):
    """Lightning datamodule over ``datalist.json`` (``training`` / ``validation`` lists).

    Args:
        datalist_path: path of the datalist written by ``scripts/preprocess/build_datalist.py``.
        batch_size: per-GPU batch size.
        num_workers: DataLoader workers.
        spacing_multiplier: factor applied to the voxel spacing (mm) fed to the UNet.
        load_spectre: load SPECTRE tokens (needed for the REPA loss).
        pin_memory: DataLoader ``pin_memory``.
        prefetch_factor: DataLoader ``prefetch_factor`` (ignored when ``num_workers == 0``).
    """

    def __init__(
        self,
        datalist_path: str,
        batch_size: int = 12,
        num_workers: int = 16,
        spacing_multiplier: float = 100.0,
        load_spectre: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_ds: CTRateLatentDataset | None = None
        self.val_ds: CTRateLatentDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Read the datalist and build both datasets (once)."""
        if self.train_ds is not None:
            return
        hp = self.hparams
        with open(hp.datalist_path) as f:
            datalist = json.load(f)
        self.train_ds = CTRateLatentDataset(datalist["training"], hp.spacing_multiplier, hp.load_spectre)
        self.val_ds = CTRateLatentDataset(datalist["validation"], hp.spacing_multiplier, hp.load_spectre)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            ds,
            batch_size=hp.batch_size,
            shuffle=shuffle,
            num_workers=hp.num_workers,
            pin_memory=hp.pin_memory,
            prefetch_factor=hp.prefetch_factor if hp.num_workers > 0 else None,
            collate_fn=collate,
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled loader over the training split."""
        self.setup("fit")
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Unshuffled loader over the validation split."""
        self.setup("validate")
        return self._loader(self.val_ds, shuffle=False)
