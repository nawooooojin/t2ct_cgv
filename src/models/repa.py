"""Representation alignment (REPA) head, following U-REPA (Tian et al., NeurIPS 2025).

A middle-block U-Net feature is projected (3-layer MLP as 1x1x1 convs), trilinearly upsampled to
the teacher token grid, and aligned with the frozen teacher tokens:

    L_align = hard_weight * L_REPA + rel_weight * L_ML
    L_REPA  = - E_{b,n} cos( y*[n], h(h_t)[n] )                 (token-wise)
    L_ML    =   E_b || Sim(y*) - Sim(h(h_t)) ||_F^2  (mean)       (manifold / relational)

The teacher tokens are spatially normalised first (iREPA): the per-channel mean over tokens is
removed and the variance normalised, which removes the global component shared by all tokens.
The training module adds ``lambda * L_align`` to the velocity loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def spatial_norm(tokens: torch.Tensor, gamma: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """``(x - gamma * E_T[x]) / sqrt(Var_T[x] + eps)`` over the token axis of ``(B, T, D)``."""
    mean = tokens.mean(dim=-2, keepdim=True)
    var = tokens.var(dim=-2, unbiased=False, keepdim=True)
    return (tokens - gamma * mean) / torch.sqrt(var + eps)


class RepaAligner(nn.Module):
    """Projects a U-Net feature onto the teacher grid and scores the alignment.

    Args:
        student_channels: channels of the tapped U-Net feature (512 for `middle_block`).
        teacher_dim: teacher token width (1080 for SPECTRE ViT-L).
        teacher_grid: teacher token grid ``(H, W, D)`` (16^3: SPECTRE 32^3 tokens average-pooled).
        student_grid: spatial grid of the tapped feature (8^3 for a 64^3 latent).
        spatial_norm_teacher: apply spatial normalisation to the teacher tokens.
        gamma: mean coefficient of the spatial normalisation.
        hard_weight: weight of the token-wise cosine term.
        rel_weight: weight of the relational term (U-REPA's ``w``).
        rel_tokens: number of randomly sampled tokens for the relational term per step;
            ``None`` uses all teacher tokens (full Gram matrix).
    """

    def __init__(
        self,
        student_channels: int = 512,
        teacher_dim: int = 1080,
        teacher_grid: tuple[int, int, int] = (16, 16, 16),
        student_grid: tuple[int, int, int] = (8, 8, 8),
        spatial_norm_teacher: bool = True,
        gamma: float = 1.0,
        hard_weight: float = 1.0,
        rel_weight: float = 3.0,
        rel_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.teacher_grid = tuple(teacher_grid)
        self.student_grid = tuple(student_grid)
        self.spatial_norm_teacher = spatial_norm_teacher
        self.gamma = gamma
        self.hard_weight = hard_weight
        self.rel_weight = rel_weight
        self.rel_tokens = rel_tokens
        self.projector = nn.Sequential(
            nn.Conv3d(student_channels, teacher_dim, 1),
            nn.SiLU(),
            nn.Conv3d(teacher_dim, teacher_dim, 1),
            nn.SiLU(),
            nn.Conv3d(teacher_dim, teacher_dim, 1),
        )

    def project(self, student_feat: torch.Tensor) -> torch.Tensor:
        """``(B, C, h, w, d)`` -> ``(B, T, teacher_dim)``: MLP first, then upsample."""
        z = self.projector(student_feat)  # (B, D_t, h, w, d)
        if tuple(z.shape[2:]) != self.teacher_grid:
            z = F.interpolate(z, size=self.teacher_grid, mode="trilinear", align_corners=False)
        return z.flatten(2).transpose(1, 2)  # (B, T, D_t)

    def _rel_index(self, device: torch.device, generator: torch.Generator | None) -> torch.Tensor:
        n = self.teacher_grid[0] * self.teacher_grid[1] * self.teacher_grid[2]
        idx = torch.arange(n, device=device)
        if self.rel_tokens is not None and n > self.rel_tokens:
            idx = idx[torch.randperm(n, generator=generator, device=device)[: self.rel_tokens]]
        return idx

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_tokens: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return ``(weighted alignment loss, detached logs)``.

        Args:
            student_feat: ``(B, C, h, w, d)`` tapped U-Net feature (must keep its autograd graph).
            teacher_tokens: ``(B, T, teacher_dim)`` precomputed frozen teacher tokens.
            generator: RNG used for relational-token sampling.
        """
        teacher = teacher_tokens.float()
        if self.spatial_norm_teacher:
            teacher = spatial_norm(teacher, gamma=self.gamma)
        student = self.project(student_feat)  # (B, T, D_t)

        loss = student.new_zeros(())
        logs: dict[str, torch.Tensor] = {}
        if self.hard_weight:
            s_n = F.normalize(student.float(), dim=-1)
            t_n = F.normalize(teacher, dim=-1)
            cos = (s_n * t_n).sum(-1).mean()
            loss = loss + self.hard_weight * (-cos)
            logs["repa_cos"] = cos.detach()
        if self.rel_weight:
            idx = self._rel_index(student.device, generator)
            s = F.normalize(student[:, idx].float(), dim=-1)
            t = F.normalize(teacher[:, idx], dim=-1)
            rel = (s @ s.transpose(1, 2) - t @ t.transpose(1, 2)).pow(2).mean()
            loss = loss + self.rel_weight * rel
            logs["repa_rel"] = rel.detach()
        return loss, logs
