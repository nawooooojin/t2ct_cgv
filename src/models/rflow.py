# DUPLICATION INTENTIONAL: vendored from the MONAI MAISI bundle, scripts/rectified_flow.py
# (Project-MONAI/model-zoo, models/maisi_ct_generative), itself adapted from Open-Sora
# opensora/schedulers/rf/rectified_flow.py (hpcaitech/Open-Sora). Both Apache-2.0.
# monai==1.4.0 ships no RFlowScheduler, so the bundle file is copied here with its numerics unchanged
# (only type hints, docstrings and the debug print were touched).
"""Rectified-flow noise scheduler (training timestep sampling, noising and Euler sampling steps)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai.networks.schedulers import Scheduler
from torch import Tensor
from torch.distributions import LogisticNormal


def timestep_transform(
    t: Tensor,
    input_img_size: Tensor,
    base_img_size: int = 32 * 32 * 32,
    scale: float = 1.0,
    num_train_timesteps: int = 1000,
    spatial_dim: int = 3,
) -> Tensor:
    """Shift timesteps toward the noisy end for inputs larger than ``base_img_size`` voxels.

    Args:
        t: timesteps in ``[0, num_train_timesteps]``, any shape.
        input_img_size: number of latent voxels (scalar tensor).

    Returns:
        Transformed timesteps, same shape as ``t``.
    """
    t = t / num_train_timesteps
    ratio_space = (input_img_size / base_img_size).pow(1.0 / spatial_dim)

    ratio = ratio_space * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_train_timesteps
    return new_t


class RFlowScheduler(Scheduler):
    """Rectified-flow scheduler: ``x_t = (1 - t/T) x_0 + (t/T) noise``, velocity target ``x_0 - noise``.

    Note: ``Scheduler.__init__`` is not called (upstream behaviour), so this object is not registered as a submodule.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
        use_discrete_timesteps: bool = False,
        sample_method: str = "uniform",
        loc: float = 0.0,
        scale: float = 1.0,
        use_timestep_transform: bool = False,
        transform_scale: float = 1.0,
        steps_offset: int = 0,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        assert sample_method in ["uniform", "logit-normal"]
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        self.steps_offset = steps_offset

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Noise clean latents to timestep ``t``.

        Args:
            original_samples: ``(B, C, H, W, D)``.
            noise: ``(B, C, H, W, D)``.
            timesteps: ``(B,)``.

        Returns:
            ``(B, C, H, W, D)``.
        """
        timepoints = timesteps.float() / self.num_train_timesteps
        timepoints = 1 - timepoints  # [1, 1/1000]

        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 1, 1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        input_img_size: Tensor | int | None = None,
        base_img_size: int = 32 * 32 * 32,
    ) -> None:
        """Set the inference timesteps (stored in ``self.timesteps``, float16 values).

        Args:
            num_inference_steps: number of sampling steps.
            device: target device.
            input_img_size: H*W*D of the latent, used when ``use_timestep_transform``.
            base_img_size: reference H*W*D.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.num_train_timesteps`:"
                f" {self.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps
        timesteps = [
            (1.0 - i / self.num_inference_steps) * self.num_train_timesteps for i in range(self.num_inference_steps)
        ]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    input_img_size=input_img_size,
                    base_img_size=base_img_size,
                    num_train_timesteps=self.num_train_timesteps,
                )
                for t in timesteps
            ]
        timesteps = np.array(timesteps).astype(np.float16)
        if self.use_discrete_timesteps:
            timesteps = timesteps.astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.steps_offset

    def sample_timesteps(self, x_start: Tensor) -> Tensor:
        """Draw one training timestep per sample.

        Args:
            x_start: ``(B, C, H, W, D)``.

        Returns:
            ``(B,)`` float timesteps in ``(0, 1000)``.
        """
        if self.sample_method == "uniform":
            t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_train_timesteps
        elif self.sample_method == "logit-normal":
            t = self.sample_t(x_start) * self.num_train_timesteps

        if self.use_discrete_timesteps:
            t = t.long()

        if self.use_timestep_transform:
            input_img_size = torch.prod(torch.tensor(x_start.shape[-3:]))
            base_img_size = 32 * 32 * 32
            t = timestep_transform(
                t,
                input_img_size=input_img_size,
                base_img_size=base_img_size,
                num_train_timesteps=self.num_train_timesteps,
            )

        return t

    def step(
        self, model_output: Tensor, timestep: float, sample: Tensor, next_timestep: float | None = None
    ) -> tuple[Tensor, Any]:
        """One Euler step ``z + v * (t - t_next) / T``.

        Args:
            model_output: predicted velocity ``(B, C, H, W, D)``.
            timestep: current timestep.
            sample: current latent ``(B, C, H, W, D)``.
            next_timestep: next timestep (``None``: uniform ``1 / num_inference_steps``).

        Returns:
            ``(next_sample, None)``.
        """
        v_pred = model_output
        if next_timestep is None:
            dt = 1.0 / self.num_inference_steps
        else:
            dt = timestep - next_timestep
            dt = dt / self.num_train_timesteps
        z = sample + v_pred * dt

        return z, None


__all__ = ["RFlowScheduler", "timestep_transform"]
