"""Lightning training module: rectified-flow latent diffusion with optional REPA.

The training loop follows Report2CT (MAISI `diff_model_train` script): latents are scaled by a
fixed `scale_factor` (1 / std of the first batch), noised with the rectified-flow scheduler, and
the U-Net is trained to predict the velocity (x0 - noise) with an MSE loss. Classifier-free
guidance is learned by dropping the text condition with probability `cfg_drop_prob`.

Text conditioning modes
    pooled   (CT-Gen): context = [findings, impression] pooled vectors, ``(B, 2, 2560)``.
    sequence (MR-Gen): context = concat(findings tokens, impression tokens), ``(B, T, 1024)``
                       with a padding mask; the pooled vectors ``(B, 2048)`` per section go to
                       the U-Net's global path (`text_pooled_cond=True`).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn

from src.models.rflow import RFlowScheduler


class Report2VolumeModule(LightningModule):
    """Args:
        unet: denoising U-Net (built by Hydra).
        noise_scheduler: `RFlowScheduler`.
        lr: Adam learning rate.
        cfg_drop_prob: probability of dropping the text condition (per sample).
        text_condition: ``"pooled"`` or ``"sequence"`` (see module docstring).
        text_pooled_cond: also feed the pooled vectors to the U-Net's global path.
        modality_class_label: class id used when the batch has no ``class_label`` (CT = 1).
        lr_scheduler_power: power of the polynomial LR decay.
        warmup_steps: linear warm-up steps before the decay.
        repa: optional `RepaAligner`; requires ``spectre`` teacher tokens in the batch.
        repa_weight: lambda in ``L = L_velocity + lambda * L_align``.
        repa_tap: U-Net submodule whose output is aligned.
    """

    def __init__(
        self,
        unet: nn.Module,
        noise_scheduler: RFlowScheduler,
        lr: float = 2e-4,
        cfg_drop_prob: float = 0.15,
        text_condition: str = "pooled",
        text_pooled_cond: bool = False,
        modality_class_label: int = 1,
        lr_scheduler_power: float = 2.0,
        warmup_steps: int = 0,
        repa: nn.Module | None = None,
        repa_weight: float = 0.5,
        repa_tap: str = "middle_block",
    ) -> None:
        super().__init__()
        if text_condition not in ("pooled", "sequence"):
            raise ValueError(f"text_condition must be 'pooled' or 'sequence', got {text_condition!r}")
        self.save_hyperparameters(ignore=["unet", "noise_scheduler", "repa"])
        self.unet = unet
        # RFlowScheduler subclasses nn.Module without calling its __init__; keep it unregistered.
        object.__setattr__(self, "noise_scheduler", noise_scheduler)
        self.lr = lr
        self.cfg_drop_prob = cfg_drop_prob
        self.text_condition = text_condition
        self.text_pooled_cond = text_pooled_cond
        self.modality_class_label = modality_class_label
        self.lr_scheduler_power = lr_scheduler_power
        self.warmup_steps = warmup_steps

        self.repa = repa
        self.repa_weight = repa_weight
        self.repa_tap = repa_tap
        self._repa_feat: torch.Tensor | None = None
        self._repa_handle: Any = None
        self._repa_generator: torch.Generator | None = None

        self.register_buffer("scale_factor", torch.tensor(1.0))
        self._scale_factor_initialized = False

    # ---------------------------------------------------------------------------------------
    def setup(self, stage: str) -> None:
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if self.repa is not None and self._repa_handle is None:
            module: nn.Module = self.unet
            for part in self.repa_tap.split("."):
                module = module[int(part)] if part.isdigit() else getattr(module, part)

            def hook(_m, _inp, out):
                # keep the graph: the alignment loss has to reach the U-Net
                self._repa_feat = out[0] if isinstance(out, tuple) else out

            self._repa_handle = module.register_forward_hook(hook)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._scale_factor_initialized = True

    def _maybe_init_scale_factor(self, latents: torch.Tensor) -> None:
        if self._scale_factor_initialized:
            return
        sf = 1.0 / latents.detach().float().std().clamp(min=1e-8)
        if self.trainer is not None and getattr(self.trainer, "world_size", 1) > 1:
            sf = self.all_gather(sf).float().mean()
        self.scale_factor = sf.to(self.scale_factor)
        self._scale_factor_initialized = True

    def _text_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """Return ``(context, context_mask, pooled)`` before CFG dropout."""
        if self.text_condition == "pooled":
            f, i = batch["context_f"], batch["context_i"]
            f = f.unsqueeze(1) if f.dim() == 2 else f
            i = i.unsqueeze(1) if i.dim() == 2 else i
            context, mask = torch.cat([f, i], dim=1), None  # (B, 2, 2560)
        else:
            context = torch.cat([batch["context_f_seq"], batch["context_i_seq"]], dim=1)  # (B, T, 1024)
            mask = torch.cat([batch["context_f_mask"], batch["context_i_mask"]], dim=1)  # (B, T)
        pooled = {}
        if self.text_pooled_cond:
            pooled = {"text_pooled_f": batch["context_f"], "text_pooled_i": batch["context_i"]}  # (B, 2048) each
        return context, mask, pooled

    def _shared_forward(self, batch: dict) -> torch.Tensor:
        latents = batch["image"]  # (B, 16, 64, 64, 64)
        self._maybe_init_scale_factor(latents)
        latents = latents * self.scale_factor

        context, context_mask, pooled = self._text_inputs(batch)
        if self.training and self.cfg_drop_prob > 0:
            drop = torch.rand(context.shape[0], device=context.device) < self.cfg_drop_prob  # (B,)
            context = context.masked_fill(drop.view(-1, *([1] * (context.dim() - 1))), 0.0)
            pooled = {k: v.masked_fill(drop.view(-1, *([1] * (v.dim() - 1))), 0.0) for k, v in pooled.items()}

        if "class_label" in batch:  # MR: (sequence, plane) class per sample
            class_labels = batch["class_label"].to(latents.device).long()
        else:
            class_labels = torch.full((latents.shape[0],), self.modality_class_label, dtype=torch.long,
                                      device=latents.device)

        noise = torch.randn_like(latents)
        timesteps = self.noise_scheduler.sample_timesteps(latents)
        noisy = self.noise_scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)

        inputs = dict(x=noisy, timesteps=timesteps, spacing_tensor=batch["spacing"].to(latents.device),
                      context=context, class_labels=class_labels)
        if context_mask is not None:
            inputs["context_mask"] = context_mask
        inputs.update(pooled)
        pred = self.unet(**inputs)

        self._last_repa = self._repa_loss(batch)
        return F.mse_loss(pred.float(), (latents - noise).float())  # velocity target

    def _repa_loss(self, batch: dict) -> tuple[torch.Tensor | None, dict]:
        feat, self._repa_feat = self._repa_feat, None
        if not self.training or self.repa is None or self.repa_weight == 0 or feat is None:
            return None, {}
        if self._repa_generator is None:
            self._repa_generator = torch.Generator(device=feat.device).manual_seed(1234)
        return self.repa(feat.float(), batch["spectre"].float(), generator=self._repa_generator)

    # ---------------------------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = loss_diff = self._shared_forward(batch)
        repa_loss, repa_logs = self._last_repa
        if repa_loss is not None:
            loss = loss_diff + self.repa_weight * repa_loss
            self.log("train/loss_diff", loss_diff, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/loss_repa", repa_loss, on_step=True, on_epoch=True, sync_dist=True)
            for k, v in repa_logs.items():
                self.log(f"train/{k}", v, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._shared_forward(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        params = list(self.unet.parameters())
        if self.repa is not None:
            params += list(self.repa.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        if self.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=self.warmup_steps)
            decay = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=max(total - self.warmup_steps, 1), power=self.lr_scheduler_power
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, decay],
                                                              milestones=[self.warmup_steps])
        else:
            scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total,
                                                              power=self.lr_scheduler_power)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
