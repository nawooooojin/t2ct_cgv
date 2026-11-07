"""Training LightningModule of WARP-CT: rectified-flow latent diffusion + U-REPA.

The UNet (``ReportUNet3D``), the ``RFlowScheduler`` and the ``RepaAligner`` are built by Hydra
and passed in; this module owns the training loop only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn

from src.models.repa import RepaTap

#: Class id of the CT modality fed to the UNet's class embedding.
MODALITY_CLASS_CT: int = 1


def apply_cfg_drop(
    context: torch.Tensor,
    text_pooled_f: torch.Tensor,
    text_pooled_i: torch.Tensor,
    drop_prob: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classifier-free-guidance dropout: zero a sample's whole report condition with prob ``drop_prob``.

    The cross-attention context and both pooled vectors of a dropped sample are zeroed together;
    the context mask is left unchanged.

    Args:
        context: ``(B, T, 1024)`` cross-attention context.
        text_pooled_f: ``(B, 2048)`` pooled findings vector.
        text_pooled_i: ``(B, 2048)`` pooled impression vector.
        drop_prob: per-sample drop probability; ``<= 0`` or ``training=False`` is a no-op.
        training: dropout applies only in training.

    Returns:
        ``(context, text_pooled_f, text_pooled_i)`` with dropped samples zeroed.
    """
    if not (training and drop_prob > 0.0):
        return context, text_pooled_f, text_pooled_i
    drop = torch.rand(context.shape[0], device=context.device) < drop_prob  # (B,)
    context = context.masked_fill(drop.view(-1, *([1] * (context.dim() - 1))), 0.0)  # (B, T, 1024)
    zero_at = drop.view(-1, *([1] * (text_pooled_i.dim() - 1)))  # (B, 1)
    text_pooled_f = text_pooled_f.masked_fill(zero_at, 0.0)  # (B, 2048)
    text_pooled_i = text_pooled_i.masked_fill(zero_at, 0.0)  # (B, 2048)
    return context, text_pooled_f, text_pooled_i


def _grad_norm(params: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Global L2 norm over ``params``' gradients; zero when none has one."""
    grads = [p.grad.detach() for p in params if p.grad is not None]
    if not grads:
        return torch.zeros((), device=device)
    return torch.norm(torch.stack([g.norm(2) for g in grads]), 2)


class T2CTModule(LightningModule):
    """Report-conditioned text-to-CT latent diffusion with an optional REPA alignment term."""

    def __init__(
        self,
        unet: nn.Module,
        noise_scheduler: Any,
        repa: nn.Module | None = None,
        lr: float = 1e-4,
        cfg_drop_prob: float = 0.15,
        lr_scheduler_power: float = 2.0,
        warmup_steps: int = 0,
        repa_weight: float = 0.5,
        scale_factor: float | None = None,
    ) -> None:
        """Wire the training loop around externally built components.

        Args:
            unet: denoising UNet (``ReportUNet3D``).
            noise_scheduler: ``RFlowScheduler`` (``sample_timesteps`` / ``add_noise``).
            repa: optional ``RepaAligner``; its middle-block tap needs ``batch["spectre"]``
                ``(B, 4096, 1080)``.
            lr: Adam learning rate.
            cfg_drop_prob: per-sample probability of dropping the report condition (CFG).
            lr_scheduler_power: exponent of the ``PolynomialLR`` decay.
            warmup_steps: linear warmup steps before the decay (``0`` = plain ``PolynomialLR``).
            repa_weight: ``lambda`` in ``L = L_velocity + lambda * L_align``.
            scale_factor: fixed latent multiplier; ``None`` estimates ``1 / std`` once from the
                first batch (all-gather mean over ranks; re-estimated on resume).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["unet", "noise_scheduler", "repa"])
        self.unet = unet
        # RFlowScheduler does not call nn.Module.__init__; store it without submodule registration.
        object.__setattr__(self, "noise_scheduler", noise_scheduler)
        self.lr = lr
        self.cfg_drop_prob = cfg_drop_prob
        self.lr_scheduler_power = lr_scheduler_power
        self.warmup_steps = warmup_steps
        self.repa = repa
        self.repa_weight = repa_weight
        # Plain object (hook + captured feature): adds no state_dict keys.
        self._repa_tap: RepaTap | None = RepaTap(unet, repa) if repa is not None else None

        self.register_buffer("scale_factor", torch.tensor(1.0))
        self._scale_factor_initialized: bool = False
        if scale_factor is not None:
            self.scale_factor = torch.tensor(float(scale_factor))
            self._scale_factor_initialized = True

    def _maybe_init_scale_factor(self, images: torch.Tensor) -> None:
        """Set ``scale_factor`` once from the first batch's latent std.

        Args:
            images: Wan latents ``(B, 16, 64, 64, 64)``.
        """
        if self._scale_factor_initialized:
            return
        sf = 1.0 / images.detach().float().std().clamp(min=1e-8)
        trainer = self._trainer  # None outside a Trainer (the `trainer` property would raise)
        if trainer is not None and getattr(trainer, "world_size", 1) > 1:
            sf = self.all_gather(sf).float().mean()  # (world_size,) -> ()
        self.scale_factor = sf.to(self.scale_factor.dtype).to(self.scale_factor.device)
        self._scale_factor_initialized = True

    def _shared_forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        """One denoising step: velocity loss plus the REPA bracket.

        Args:
            batch: ``image`` ``(B, 16, 64, 64, 64)``, ``spacing`` ``(B, 3)`` (mm x 100),
                ``context`` ``(B, T, 1024)``, ``context_mask`` ``(B, T)`` bool,
                ``pooled_f`` / ``pooled_i`` ``(B, 2048)``, ``spectre`` ``(B, 4096, 1080)`` (REPA).

        Returns:
            ``(loss_diff, repa_loss, repa_logs)``; ``repa_loss`` is ``None`` outside training or
            without REPA.
        """
        images = batch["image"]  # (B, 16, 64, 64, 64)
        self._maybe_init_scale_factor(images)
        images = images * self.scale_factor

        context = batch["context"]  # (B, T, 1024)
        context_mask = batch["context_mask"]  # (B, T) bool, True = real token
        context, text_pooled_f, text_pooled_i = apply_cfg_drop(
            context, batch["pooled_f"], batch["pooled_i"], self.cfg_drop_prob, self.training
        )
        spacing_tensor = batch["spacing"].to(images.device)  # (B, 3), already x 100
        class_labels = torch.full(
            (images.shape[0],), MODALITY_CLASS_CT, dtype=torch.long, device=images.device
        )  # (B,)

        noise = torch.randn_like(images)  # (B, 16, 64, 64, 64)
        timesteps = self.noise_scheduler.sample_timesteps(images)  # (B,) in (0, 1000)
        noisy = self.noise_scheduler.add_noise(
            original_samples=images, noise=noise, timesteps=timesteps
        )  # (B, 16, 64, 64, 64)
        model_output = self.unet(
            x=noisy,
            timesteps=timesteps,
            spacing_tensor=spacing_tensor,
            context=context,
            class_labels=class_labels,
            context_mask=context_mask,
            text_pooled_f=text_pooled_f,
            text_pooled_i=text_pooled_i,
        )  # (B, 16, 64, 64, 64)

        repa_loss, repa_logs = (
            self._repa_tap.loss(batch, self.training) if self._repa_tap is not None else (None, {})
        )
        # Rectified flow predicts velocity: target = x0 - noise.
        loss_diff = F.mse_loss(model_output.float(), (images - noise).float())
        return loss_diff, repa_loss, repa_logs

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Total training loss ``L_v + repa_weight * L_repa``; logs ``train/*``."""
        loss_diff, repa_loss, repa_logs = self._shared_forward(batch)
        loss = loss_diff
        common = {"on_step": True, "on_epoch": True, "sync_dist": True}
        if repa_loss is not None:
            loss = loss_diff + self.repa_weight * repa_loss
            self.log("train/loss_diff", loss_diff, **common)
            self.log("train/loss_repa", repa_loss, **common)
            for name, value in repa_logs.items():
                self.log(f"train/{name}", value, **common)
        self.log("train/loss", loss, prog_bar=True, **common)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Velocity loss on a validation batch (no REPA); logs ``val/loss``."""
        loss, _, _ = self._shared_forward(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Log the pre-clip global gradient norm (diagnostic only)."""
        self.log(
            "train/grad_norm",
            _grad_norm(list(self.parameters()), self.device),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    def setup(self, stage: str) -> None:
        """Pick the memory-efficient SDPA backend and register the REPA hook."""
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if self._repa_tap is not None:
            self._repa_tap.register()

    def configure_optimizers(self) -> dict[str, Any]:
        """Adam + per-step linear warmup then ``PolynomialLR`` decay."""
        params = list(self.unet.parameters())
        if self.repa is not None:
            params += list(self.repa.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, total_iters=self.warmup_steps
            )
            decay = torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=max(total_steps - self.warmup_steps, 1),
                power=self.lr_scheduler_power,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup, decay], milestones=[self.warmup_steps]
            )
        else:
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=total_steps, power=self.lr_scheduler_power
            )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


__all__ = ["MODALITY_CLASS_CT", "T2CTModule", "apply_cfg_drop"]
