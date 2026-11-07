"""Frozen SPECTRE-Large image backbone: dense 3D token grid used as the REPA target.

SPECTRE (cclaess/SPECTRE) is cloned into ``third_party/spectre`` (see README). Crops are
``(128, 128, 64)`` voxels with ``(16, 16, 8)`` patches -> 8x8x8 = 512 tokens per crop plus one
CLS token, which is dropped. Input is a RAS volume in raw HU; ``window_scan`` applies the HU
window [-1000, 1000] -> [0, 1] itself. Every axis must be a whole multiple of the crop size
(``window_scan`` would centre-crop the remainder away), so the caller pads depth 253 -> 256.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import torch
import torch.nn.functional as F
from torch import nn

SPECTRE_SRC: Final[Path] = Path(__file__).resolve().parents[2] / "third_party" / "spectre" / "src"
PRESET: Final[str] = "spectre-large"
CROP_SIZE: Final[tuple[int, int, int]] = (128, 128, 64)
PATCH_SIZE: Final[tuple[int, int, int]] = (16, 16, 8)
HU_RANGE: Final[tuple[float, float]] = (-1000.0, 1000.0)


def _install_hf_shim() -> None:
    """Provide ``huggingface_hub.load_state_dict_from_file`` for older hub versions.

    SPECTRE imports the symbol at module level but only calls it for URL checkpoints; local
    ``.pt`` loads never reach it.
    """
    import huggingface_hub  # noqa: PLC0415

    if hasattr(huggingface_hub, "load_state_dict_from_file"):
        return

    def load_state_dict_from_file(
        checkpoint_file: str | Path, map_location: str | None = None, weights_only: bool = True
    ) -> dict:
        path = str(checkpoint_file)
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file  # noqa: PLC0415

            return load_file(path, device=str(map_location or "cpu"))
        return torch.load(path, map_location=map_location or "cpu", weights_only=weights_only)

    huggingface_hub.load_state_dict_from_file = load_state_dict_from_file


def import_spectre() -> ModuleType:
    """Put ``third_party/spectre/src`` first on ``sys.path`` and return the ``spectre`` package."""
    _install_hf_shim()
    pkg = str(SPECTRE_SRC)
    while pkg in sys.path:
        sys.path.remove(pkg)
    sys.path.insert(0, pkg)
    import spectre  # noqa: PLC0415

    return spectre


class SpectreBackbone(nn.Module):
    """Lazily built, frozen SPECTRE-Large backbone producing a dense token grid.

    Args:
        ckpt_path: backbone checkpoint (``spectre_backbone_vit_large_patch16_128_no_vla.pt``).
        device: torch device string.
    """

    crop_size: tuple[int, int, int] = CROP_SIZE
    patch_size: tuple[int, int, int] = PATCH_SIZE

    def __init__(self, ckpt_path: str | Path, device: str = "cuda") -> None:
        super().__init__()
        self._ckpt_path = Path(ckpt_path)
        self._device = device
        self._model: nn.Module | None = None
        self._window_scan = None

    def _ensure_built(self) -> None:
        """Build the backbone from the preset and load the checkpoint (strict)."""
        if self._model is not None:
            return
        if not self._ckpt_path.is_file():
            raise FileNotFoundError(f"SPECTRE backbone checkpoint missing: {self._ckpt_path}")
        spectre = import_spectre()
        preset = spectre.presets.get_preset(PRESET)
        model = spectre.SpectreImageFeatureExtractor(
            backbone_name=preset.backbone,
            backbone_kwargs=preset.backbone_kwargs,
            backbone_checkpoint_path_or_url=str(self._ckpt_path),
        )
        model.eval().requires_grad_(False).to(self._device)
        self._model = model
        self._window_scan = spectre.window_scan

    def window(self, vol_hu: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """HU-window and tile a volume into backbone crops.

        Args:
            vol_hu: raw HU ``(1, H, W, D)``, e.g. ``(1, 512, 512, 256)``.

        Returns:
            Crops ``(N, 1, 128, 128, 64)`` in [0, 1] and the crop grid ``(n_h, n_w, n_d)``,
            e.g. ``(64, 1, 128, 128, 64)``, ``(4, 4, 4)``; crop ``n = (h * n_w + w) * n_d + d``.

        Raises:
            ValueError: if the shape is not ``(1, H, W, D)`` or an axis is not a multiple of the crop.
        """
        self._ensure_built()
        if vol_hu.ndim != 4 or vol_hu.shape[0] != 1:
            raise ValueError(f"expected (1, H, W, D), got {tuple(vol_hu.shape)}")
        bad = [int(s) for s, c in zip(vol_hu.shape[1:], self.crop_size) if int(s) % c]
        if bad:
            raise ValueError(
                f"every axis must be a multiple of {self.crop_size}, got {tuple(vol_hu.shape[1:])}; "
                "pad the volume first (e.g. depth 253 -> 256)"
            )
        return self._window_scan(vol_hu, self.crop_size, hu_range=HU_RANGE)

    @torch.no_grad()
    def encode_dense(self, vol_hu: torch.Tensor, max_crops_per_forward: int = 16) -> torch.Tensor:
        """Encode a whole volume to its native dense token grid (final ``forward_features``).

        Args:
            vol_hu: raw HU ``(1, 512, 512, 256)``.
            max_crops_per_forward: crops per backbone pass (memory cap).

        Returns:
            Token grid ``(32, 32, 32, 1080)`` = ``(H/16, W/16, D/8, F)``.
        """
        crops, grid = self.window(vol_hu)  # (N, 1, 128, 128, 64), (n_h, n_w, n_d)
        model = self._model
        device = next(model.parameters()).device
        crops = crops.to(device, non_blocking=True)
        chunks = [
            model.backbone.forward_features(chunk)  # (n, 1 + 512, F), CLS first
            for chunk in torch.split(crops, max_crops_per_forward, dim=0)
        ]
        tokens = torch.cat(chunks, dim=0)  # (N, 1 + 512, F)
        n_prefix = int(model.backbone.num_prefix_tokens)
        return self._crops_to_grid(tokens[:, n_prefix:], grid)

    @classmethod
    def _crops_to_grid(cls, patch_tokens: torch.Tensor, grid: tuple[int, int, int]) -> torch.Tensor:
        """Reassemble per-crop patch tokens into one volume grid (depth fastest in both orders).

        Args:
            patch_tokens: ``(N, 512, F)``, CLS removed.
            grid: crop grid ``(n_h, n_w, n_d)``.

        Returns:
            ``(n_h * 8, n_w * 8, n_d * 8, F)``.
        """
        n_h, n_w, n_d = grid
        p_h, p_w, p_d = (c // p for c, p in zip(cls.crop_size, cls.patch_size))
        feat = patch_tokens.shape[-1]
        if patch_tokens.shape[0] != n_h * n_w * n_d:
            raise ValueError(f"{patch_tokens.shape[0]} crops != grid {grid} product")
        if patch_tokens.shape[1] != p_h * p_w * p_d:
            raise ValueError(f"{patch_tokens.shape[1]} tokens per crop != {p_h * p_w * p_d} (CLS kept?)")
        x = patch_tokens.reshape(n_h, n_w, n_d, p_h, p_w, p_d, feat)  # (4, 4, 4, 8, 8, 8, F)
        x = x.permute(0, 3, 1, 4, 2, 5, 6)  # (4, 8, 4, 8, 4, 8, F)
        return x.reshape(n_h * p_h, n_w * p_w, n_d * p_d, feat)  # (32, 32, 32, F)

    @staticmethod
    def pool_grid(dense: torch.Tensor, out_grid: tuple[int, int, int]) -> torch.Tensor:
        """Average-pool a token grid ``(Gh, Gw, Gd, F)`` to ``out_grid``, e.g. ``(16, 16, 16, 1080)``."""
        g = dense.shape[:3]
        if any(s % o for s, o in zip(g, out_grid)):
            raise ValueError(f"grid {tuple(g)} is not a whole multiple of {out_grid}")
        x = dense.permute(3, 0, 1, 2).unsqueeze(0)  # (1, F, Gh, Gw, Gd)
        x = F.adaptive_avg_pool3d(x, out_grid)  # (1, F, oh, ow, od)
        return x.squeeze(0).permute(1, 2, 3, 0)  # (oh, ow, od, F)
