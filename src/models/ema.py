"""Exponential moving average of the trainable parameters, stored in the checkpoint callback state.

Use ``scripts/extract_ema.py`` to turn a training checkpoint into an EMA weight file.
"""

from __future__ import annotations

from typing import Any

import torch
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    def __init__(self, decay: float = 0.9999) -> None:
        super().__init__()
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] | None = None
        self._num_updates = 0
        self._last_step: int | None = None

    def on_train_start(self, trainer, pl_module) -> None:
        self._last_step = trainer.global_step
        if self._shadow is None:
            self._shadow = {n: p.detach().float().clone() for n, p in pl_module.named_parameters()
                            if p.is_floating_point()}
        else:
            self._shadow = {n: t.to(device=pl_module.device, dtype=torch.float32) for n, t in self._shadow.items()}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step == self._last_step:  # no optimizer step yet (gradient accumulation)
            return
        self._last_step = trainer.global_step
        d = min(self.decay, (1 + self._num_updates) / (10 + self._num_updates))
        for n, p in pl_module.named_parameters():
            s = self._shadow.get(n)
            if s is not None:
                s.lerp_(p.detach().float(), 1.0 - d)
        self._num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {"shadow": {k: v.detach().cpu() for k, v in self._shadow.items()}, "num_updates": self._num_updates}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._shadow = dict(state_dict["shadow"])
        self._num_updates = state_dict["num_updates"]
