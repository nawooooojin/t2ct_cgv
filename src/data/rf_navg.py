"""RadFinder layer-averaged ("navg") text features: reductions and the sidecar loader.

A sidecar ``<id>.npz`` holds, per section (``findings``, ``impression``):

    <section>_rf_navg  (L, 1024) fp16   29-layer RMS-normalised-then-mean token sequence
    <section>_rf_last  (1024,)   fp16   final-layer EOS row, RMS-normalised

The UNet consumes the sequence through cross-attention and, per section,
``concat(RMSNorm(EOS), RMSNorm(masked mean of the sequence))`` ``(2048,)`` through AdaGN.
Pure torch + numpy so it imports in both the training and the text-encoding env.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

NORM_EPS: float = 1e-6
HIDDEN_DIM: int = 1024
SECTIONS: dict[str, str] = {"f": "findings", "i": "impression"}


def rms_normalize(x: torch.Tensor) -> torch.Tensor:
    """Scale each row to unit RMS over the last axis (all-zero rows stay zero).

    Args:
        x: ``(..., D)``.

    Returns:
        Same shape as ``x``.
    """
    return x / x.pow(2).mean(dim=-1, keepdim=True).add(NORM_EPS).sqrt()


def aggregate_layers(stack: torch.Tensor) -> torch.Tensor:
    """RMS-normalise every ``(layer, token)`` row, then average over layers (not renormalised).

    Args:
        stack: ``(n_layers, L, D)`` raw hidden states, e.g. ``(29, L, 1024)``.

    Returns:
        ``(L, D)``.
    """
    return rms_normalize(stack).mean(dim=0)


def masked_mean(seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a padded sequence over its real positions only (float32 accumulation).

    Args:
        seq: ``(..., T, D)``.
        mask: ``(..., T)`` bool, ``True`` = real token.

    Returns:
        ``(..., D)`` float32.

    Raises:
        ValueError: if any row has no real token.
    """
    n = mask.sum(dim=-1, keepdim=True)  # (..., 1)
    if bool((n == 0).any()):
        raise ValueError("masked_mean: a sequence has no real tokens")
    w = mask.unsqueeze(-1).to(torch.float32)  # (..., T, 1)
    return (seq.to(torch.float32) * w).sum(dim=-2) / n.to(torch.float32)  # (..., D)


def global_vector(eos: torch.Tensor, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pooled section vector ``concat(RMSNorm(eos), RMSNorm(masked_mean(seq)))``.

    Args:
        eos: ``(..., D)`` the sidecar's ``rf_last``.
        seq: ``(..., T, D)`` the ``rf_navg`` sequence.
        mask: ``(..., T)`` bool, ``True`` = real token.

    Returns:
        ``(..., 2 * D)``.
    """
    return torch.cat([rms_normalize(eos), rms_normalize(masked_mean(seq, mask))], dim=-1)


def pad_sequences(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad variable-length sequences to the longest one and build the mask.

    Args:
        seqs: per-sample ``(L_b, D)`` tensors.

    Returns:
        ``(padded, mask)``: ``(B, T_max, D)`` float32 and ``(B, T_max)`` bool (``True`` = real token).
    """
    t_max = max(s.shape[0] for s in seqs)
    padded = torch.zeros(len(seqs), t_max, seqs[0].shape[1], dtype=torch.float32)  # (B, T_max, D)
    mask = torch.zeros(len(seqs), t_max, dtype=torch.bool)  # (B, T_max)
    for b, s in enumerate(seqs):
        padded[b, : s.shape[0]] = s
        mask[b, : s.shape[0]] = True
    return padded, mask


def load_text_sidecar(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a sidecar and finish the pooled vectors on the unpadded sequences.

    Args:
        path: ``<id>.npz`` with the four ``{findings,impression}_rf_{navg,last}`` keys.

    Returns:
        ``{"seq_f": (L_f, 1024), "seq_i": (L_i, 1024), "pooled_f": (2048,), "pooled_i": (2048,)}``, float32.
    """
    out: dict[str, torch.Tensor] = {}
    with np.load(path) as npz:
        for short, section in SECTIONS.items():
            seq = torch.from_numpy(npz[f"{section}_rf_navg"].astype(np.float32))  # (L, 1024)
            eos = torch.from_numpy(npz[f"{section}_rf_last"].astype(np.float32))  # (1024,)
            mask = torch.ones(seq.shape[0], dtype=torch.bool)  # (L,)
            out[f"seq_{short}"] = seq
            out[f"pooled_{short}"] = global_vector(eos, seq, mask)  # (2048,)
    return out
