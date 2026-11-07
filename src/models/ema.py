"""Online exponential moving average of the model parameters, stored in every checkpoint.

The shadow lives in ``ckpt["callbacks"]["EMACallback"]``; ``scripts/extract_ema.py`` turns it into a standalone
``{"state_dict": ...}`` checkpoint. Buffers (``scale_factor``) are not averaged: it is a data statistic set once on the
first batch, and averaging it from its 1.0 init would generate at the wrong latent scale.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class EMACallback(Callback):
    """Track an fp32 EMA of every floating parameter (``unet.*``, ``repa.*``).

    Args:
        decay: target decay; the effective decay ramps as ``min(decay, (1 + n) / (10 + n))``.
    """

    def __init__(self, decay: float = 0.9999) -> None:
        super().__init__()
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] | None = None
        self._num_updates = 0
        # global_step at the last blend (not persisted; re-read in on_train_start after a resume).
        self._last_global_step: int | None = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Seed the shadow from the current weights, or move a resumed shadow to the device."""
        self._last_global_step = getattr(trainer, "global_step", None)
        if self._shadow is None:
            self._shadow = {
                name: p.detach().to(torch.float32).clone()
                for name, p in pl_module.named_parameters()
                if p.is_floating_point()
            }
            log.info(
                "EMA: tracking %d tensors (%.2f M params), decay=%g",
                len(self._shadow),
                sum(t.numel() for t in self._shadow.values()) / 1e6,
                self.decay,
            )
        else:
            self._shadow = {
                name: t.to(device=pl_module.device, dtype=torch.float32) for name, t in self._shadow.items()
            }
            log.info("EMA: resumed at %d updates", self._num_updates)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Blend the weights into the shadow once per optimizer step (skips gradient-accumulation micro-batches)."""
        assert self._shadow is not None
        step = getattr(trainer, "global_step", None)
        if step is not None:
            if step == self._last_global_step:
                return
            self._last_global_step = step
        d = min(self.decay, (1 + self._num_updates) / (10 + self._num_updates))
        for name, p in pl_module.named_parameters():
            shadow = self._shadow.get(name)
            if shadow is not None:
                shadow.lerp_(p.detach().to(torch.float32), 1.0 - d)  # shadow <- d*shadow + (1-d)*p
        self._num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        """Return ``{"shadow": {name: cpu fp32}, "num_updates": int}``."""
        assert self._shadow is not None
        return {
            "shadow": {k: v.detach().cpu() for k, v in self._shadow.items()},
            "num_updates": self._num_updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the shadow on resume; ``on_train_start`` moves it to the device."""
        self._shadow = dict(state_dict["shadow"])
        self._num_updates = state_dict["num_updates"]


__all__ = ["EMACallback"]
