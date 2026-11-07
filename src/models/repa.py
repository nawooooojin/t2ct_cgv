"""U-REPA representation alignment of the U-Net middle block to frozen SPECTRE tokens.

The caller applies the outer weight: ``L = L_velocity + repa_weight * (hard_weight * L_cos + rel_weight * L_rel)``,
where ``L_cos`` is the negative mean token-wise cosine (REPA) and ``L_rel`` the mean squared difference of the
token Gram matrices (U-REPA). The teacher tokens are spatially normalised (iREPA); the student is projected at its
native 8^3 grid and then trilinearly upsampled to the teacher's 16^3 grid.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: U-Net block whose output is aligned, ``(B, 512, 8, 8, 8)``.
REPA_TAP: str = "middle_block"
#: Batch key holding the teacher tokens ``(B, 4096, 1080)``.
REPA_TEACHER_KEY: str = "spectre"


def spatial_norm(tokens: Tensor, gamma: float = 1.0, eps: float = 1e-6) -> Tensor:
    """iREPA spatial normalisation over the token axis: ``(x - gamma * E[x]) / sqrt(Var[x] + eps)``.

    Args:
        tokens: ``(B, T, D)``.

    Returns:
        ``(B, T, D)``.
    """
    mean = tokens.mean(dim=-2, keepdim=True)  # (B, 1, D)
    var = tokens.var(dim=-2, unbiased=False, keepdim=True)  # (B, 1, D)
    return (tokens - gamma * mean) / torch.sqrt(var + eps)


class RepaAligner(nn.Module):
    """Projects a U-Net feature onto the teacher grid and scores the alignment.

    Only the configuration used by the model is supported (global-scope U-REPA L2 relation, trilinear upsampling,
    no global head). ``student_grid`` and ``crop_tokens`` are stored for config compatibility only.

    Args:
        student_channels: channels of the tapped feature (512).
        teacher_dim: teacher token width (1080 for SPECTRE ViT-L).
        teacher_grid: teacher token grid ``(16, 16, 16)``.
        projector: ``"mlp"`` (three 1x1x1 convs with SiLU) or ``"conv"`` (one 3x3x3 conv).
        upsample: ``"trilinear"``.
        student_grid: native grid of the tapped feature.
        spatial_norm_teacher: apply ``spatial_norm`` to the teacher tokens.
        gamma: spatial-norm mean coefficient.
        hard_weight: weight of the token-wise cosine term.
        rel_weight: weight of the relational term.
        rel_loss: ``"urepa_l2"``.
        rel_scope: ``"global"`` (all teacher tokens).
        crop_tokens: unused in the global scope.
        global_weight: must be 0.0.
    """

    def __init__(
        self,
        student_channels: int = 512,
        teacher_dim: int = 1080,
        teacher_grid: tuple[int, int, int] = (16, 16, 16),
        projector: str = "mlp",
        upsample: str = "trilinear",
        student_grid: tuple[int, int, int] = (8, 8, 8),
        spatial_norm_teacher: bool = True,
        gamma: float = 1.0,
        hard_weight: float = 1.0,
        rel_weight: float = 3.0,
        rel_loss: str = "urepa_l2",
        rel_scope: str = "global",
        crop_tokens: tuple[int, int, int] = (4, 4, 4),
        global_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if upsample != "trilinear":
            raise ValueError(f"upsample must be 'trilinear', got {upsample!r}")
        if rel_loss != "urepa_l2":
            raise ValueError(f"rel_loss must be 'urepa_l2', got {rel_loss!r}")
        if rel_scope != "global":
            raise ValueError(f"rel_scope must be 'global', got {rel_scope!r}")
        if global_weight != 0.0:
            raise ValueError(f"global_weight must be 0.0, got {global_weight!r}")
        if hard_weight == 0.0 and rel_weight == 0.0:
            raise ValueError("hard_weight and rel_weight are both 0: the aligner would be a no-op")

        self.teacher_grid = tuple(teacher_grid)
        self.student_grid = tuple(student_grid)
        self.crop_tokens = tuple(crop_tokens)
        self.spatial_norm_teacher = spatial_norm_teacher
        self.gamma = gamma
        self.hard_weight = hard_weight
        self.rel_weight = rel_weight

        if projector == "mlp":
            # U-REPA build_mlp (Linear-SiLU-Linear-SiLU-Linear) as 1x1x1 convs
            self.projector: nn.Module = nn.Sequential(
                nn.Conv3d(student_channels, teacher_dim, 1),
                nn.SiLU(),
                nn.Conv3d(teacher_dim, teacher_dim, 1),
                nn.SiLU(),
                nn.Conv3d(teacher_dim, teacher_dim, 1),
            )
        elif projector == "conv":
            # iREPA: one 3x3x3 conv
            self.projector = nn.Conv3d(student_channels, teacher_dim, kernel_size=3, padding=1)
        else:
            raise ValueError(f"projector must be 'mlp' or 'conv', got {projector!r}")

    def project(self, student_feat: Tensor) -> Tensor:
        """Project at the native grid, then upsample to the teacher grid.

        Args:
            student_feat: ``(B, 512, 8, 8, 8)``.

        Returns:
            ``(B, 4096, 1080)`` (``T = prod(teacher_grid)``).
        """
        z = self.projector(student_feat)  # (B, 1080, 8, 8, 8)
        if tuple(z.shape[2:]) != self.teacher_grid:
            z = F.interpolate(z, size=self.teacher_grid, mode="trilinear", align_corners=False)  # (B, 1080, 16, 16, 16)
        return z.flatten(2).transpose(1, 2)  # (B, 4096, 1080)

    def forward(self, student_feat: Tensor, teacher_tokens: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """Weighted alignment loss and detached logging scalars.

        Args:
            student_feat: ``(B, 512, 8, 8, 8)``, still in the autograd graph.
            teacher_tokens: ``(B, 4096, 1080)`` frozen SPECTRE tokens.

        Returns:
            ``(loss, logs)``; ``logs["repa_cos"]`` is the mean cosine (higher is better).
        """
        n_tokens = self.teacher_grid[0] * self.teacher_grid[1] * self.teacher_grid[2]
        if teacher_tokens.shape[1] != n_tokens:
            raise ValueError(
                f"teacher has {teacher_tokens.shape[1]} tokens but teacher_grid {self.teacher_grid} implies {n_tokens}"
            )
        teacher = teacher_tokens.float()
        if self.spatial_norm_teacher:
            teacher = spatial_norm(teacher, gamma=self.gamma)  # (B, T, D)

        student = self.project(student_feat)  # (B, T, D)
        logs: dict[str, Tensor] = {}
        loss = student.new_zeros((), dtype=torch.float32)  # accumulation order: 0 + w_h*(-cos) + w_r*rel

        if self.hard_weight:
            student_n = F.normalize(student.float(), dim=-1)
            teacher_n = F.normalize(teacher, dim=-1)
            cos = (student_n * teacher_n).sum(-1).mean()  # REPA token-wise cosine
            loss = loss + self.hard_weight * (-cos)
            logs["repa_cos"] = cos.detach()
            if teacher_n.shape[0] > 1:
                # Diagnostic: cosine against another volume's teacher; the gap is the volume-specific part.
                shuffled = (student_n * torch.roll(teacher_n, 1, dims=0)).sum(-1).mean()
                logs["repa_cos_shuffled"] = shuffled.detach()
                logs["repa_cos_gap"] = (cos - shuffled).detach()

        if self.rel_weight:
            s = F.normalize(student.float(), dim=-1)
            t = F.normalize(teacher.float(), dim=-1)
            gram_s = s @ s.transpose(1, 2)  # (B, T, T)
            gram_t = t @ t.transpose(1, 2)  # (B, T, T)
            rel = (gram_s - gram_t).pow(2).mean()  # U-REPA relational L2
            loss = loss + self.rel_weight * rel
            logs["repa_rel"] = rel.detach()

        return loss, logs


class RepaTap:
    """Hooks ``unet.middle_block`` and scores its output against the batch's teacher tokens.

    A plain object (not an ``nn.Module``), so it adds no ``state_dict`` keys; the aligner stays registered on the
    LightningModule as ``repa``.

    Args:
        unet: the denoiser.
        aligner: the ``RepaAligner``.
    """

    def __init__(self, unet: nn.Module, aligner: RepaAligner) -> None:
        self.unet = unet
        self.aligner = aligner
        self.tap_feat: Tensor | None = None
        self._handle: Any = None

    def register(self) -> None:
        """Attach the forward hook (idempotent). The captured feature is NOT detached."""
        if self._handle is not None:
            return

        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            self.tap_feat = output[0] if isinstance(output, tuple) else output

        self._handle = getattr(self.unet, REPA_TAP).register_forward_hook(hook)

    def loss(self, batch: dict, training: bool) -> tuple[Tensor | None, dict[str, Tensor]]:
        """Consume the captured feature (always cleared) and score it.

        Args:
            batch: batch dict with ``"spectre"`` ``(B, 4096, 1080)``.
            training: ``False`` returns ``(None, {})`` (no REPA in validation).

        Returns:
            ``(loss, logs)`` or ``(None, {})``.
        """
        feat, self.tap_feat = self.tap_feat, None
        if not training or feat is None:
            return None, {}
        if REPA_TEACHER_KEY not in batch:
            raise KeyError(f"REPA is enabled but the batch has no {REPA_TEACHER_KEY!r} key (set data.load_spectre=true).")
        return self.aligner(feat.float(), batch[REPA_TEACHER_KEY].float())


__all__ = ["REPA_TAP", "REPA_TEACHER_KEY", "RepaAligner", "RepaTap", "spatial_norm"]
