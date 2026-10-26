"""RadFinder text features for MR-Gen (`<id>_rf_navg.npz`, written by encode_text_mrgen.py).

Per section the file holds ``<section>_rf_navg`` ``(L, 1024)`` (hidden states RMS-normalised per
layer, then averaged over the 29 layers) and ``<section>_rf_last`` ``(1024,)`` (RMS-normalised
final-layer EOS). The U-Net gets the sequence through cross-attention and
``concat(RMSNorm(EOS), RMSNorm(masked mean of the sequence))`` ``(2048,)`` through its global path.
"""

from __future__ import annotations

import torch

NORM_EPS = 1e-6  # inside the sqrt, so an all-zero (CFG-dropped) row stays zero


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    """Unit RMS over the last axis."""
    return x / x.pow(2).mean(dim=-1, keepdim=True).add(NORM_EPS).sqrt()


def aggregate_layers(stack: torch.Tensor) -> torch.Tensor:
    """``(n_layers, L, D)`` raw hidden states -> ``(L, D)``: normalise each layer, then average."""
    return rms_normalize(stack).mean(dim=0)


def masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``(..., T, D)`` over the real (``mask`` True) positions, in float32."""
    n = mask.sum(dim=-1, keepdim=True)
    if bool((n == 0).any()):
        raise ValueError("masked_mean: a sequence has no real tokens")
    w = mask.unsqueeze(-1).to(torch.float32)
    return (seq.to(torch.float32) * w).sum(dim=-2) / n.to(torch.float32)


def global_vector(eos: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``concat(RMSNorm(eos), RMSNorm(masked_mean(seq)))`` -> ``(..., 2 * D)``."""
    return torch.cat([rms_normalize(eos), rms_normalize(masked_mean(seq, mask))], dim=-1)


def pad_sequences(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad ``(L_i, D)`` sequences to the longest -> ``(B, T, D)`` and a ``(B, T)`` bool mask."""
    t_max = max(s.shape[0] for s in seqs)
    padded = torch.zeros(len(seqs), t_max, seqs[0].shape[1], dtype=torch.float32)
    mask = torch.zeros(len(seqs), t_max, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s
        mask[i, : s.shape[0]] = True
    return padded, mask
