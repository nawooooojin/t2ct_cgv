"""Frozen SPECTRE ViT-L (Claessens et al., CVPR 2026) as the REPA teacher.

The volume (512 x 512 x 256 after padding) is split into 4 x 4 x 4 crops of 128 x 128 x 64,
each crop gives 8 x 8 x 8 patch tokens (patch 16 x 16 x 8, CLS dropped), and the tokens are put
back on a global 32^3 grid (1080-d). We average-pool this to 16^3 for training.

Requires the SPECTRE repository (https://github.com/cclaess/SPECTRE) on the Python path
(``third_party/spectre/src``) and the SSL backbone weights
``spectre_backbone_vit_large_patch16_128_no_vla.pt`` from https://huggingface.co/cclaess/SPECTRE.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SPECTRE_SRC = Path(__file__).resolve().parents[2] / "third_party" / "spectre" / "src"
CROP_SIZE = (128, 128, 64)
PATCH_SIZE = (16, 16, 8)
HU_RANGE = (-1000.0, 1000.0)
DENSE_DIM = 1080


def _import_spectre():
    import huggingface_hub

    if not hasattr(huggingface_hub, "load_state_dict_from_file"):  # needed by spectre.utils, unused here
        def _load(path, map_location=None, weights_only=True):
            return torch.load(str(path), map_location=map_location or "cpu", weights_only=weights_only)

        huggingface_hub.load_state_dict_from_file = _load
    if str(SPECTRE_SRC) not in sys.path:
        sys.path.insert(0, str(SPECTRE_SRC))
    import spectre

    return spectre


class SpectreTeacher:
    def __init__(self, ckpt_path: str, device: str = "cuda:0") -> None:
        spectre = _import_spectre()
        preset = spectre.presets.get_preset("spectre-large")
        model = spectre.SpectreImageFeatureExtractor(
            backbone_name=preset.backbone,
            backbone_kwargs=preset.backbone_kwargs,
            backbone_checkpoint_path_or_url=str(ckpt_path),
        )
        self.model = model.eval().requires_grad_(False).to(device)
        self.window_scan = spectre.window_scan
        self.device = device

    @torch.no_grad()
    def encode(self, vol: torch.Tensor, max_crops_per_forward: int = 16) -> torch.Tensor:
        """``(1, H, W, D)`` raw HU (every axis a multiple of the crop size) -> ``(H/16, W/16, D/8, 1080)``."""
        if any(int(s) % c for s, c in zip(vol.shape[1:], CROP_SIZE)):
            raise ValueError(f"pad the volume to a multiple of {CROP_SIZE} first, got {tuple(vol.shape[1:])}")
        crops, grid = self.window_scan(vol, CROP_SIZE, hu_range=HU_RANGE)  # (N, 1, 128, 128, 64)
        crops = crops.to(self.device)
        tokens = torch.cat(
            [self.model.backbone.forward_features(c) for c in torch.split(crops, max_crops_per_forward)], dim=0
        )  # (N, 1 + 512, 1080)
        tokens = tokens[:, int(self.model.backbone.num_prefix_tokens):]  # drop CLS
        n_h, n_w, n_d = grid
        p_h, p_w, p_d = (c // p for c, p in zip(CROP_SIZE, PATCH_SIZE))
        x = tokens.reshape(n_h, n_w, n_d, p_h, p_w, p_d, DENSE_DIM).permute(0, 3, 1, 4, 2, 5, 6)
        return x.reshape(n_h * p_h, n_w * p_w, n_d * p_d, DENSE_DIM)


def pool_grid(dense: torch.Tensor, out_grid: tuple[int, int, int]) -> torch.Tensor:
    """Average-pool a ``(H, W, D, C)`` token grid to ``(*out_grid, C)``."""
    x = F.adaptive_avg_pool3d(dense.permute(3, 0, 1, 2).unsqueeze(0), out_grid)
    return x.squeeze(0).permute(1, 2, 3, 0)
