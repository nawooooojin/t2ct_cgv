# Rectified-flow scheduler used by the MAISI latent diffusion model.
# Copied from the MONAI MAISI bundle (scripts/rectified_flow.py, Apache-2.0), which in turn adapts
# https://github.com/hpcaitech/Open-Sora/blob/main/opensora/schedulers/rf/rectified_flow.py
# Only change: the debug print in `set_timesteps` was removed.
from typing import Any

import numpy as np
import torch
from monai.networks.schedulers import Scheduler
from torch.distributions import LogisticNormal


def timestep_transform(
    t, input_img_size, base_img_size=32 * 32 * 32, scale=1.0, num_train_timesteps=1000, spatial_dim=3
):
    t = t / num_train_timesteps
    ratio_space = (input_img_size / base_img_size).pow(1.0 / spatial_dim)

    ratio = ratio_space * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_train_timesteps
    return new_t


class RFlowScheduler(Scheduler):
    def __init__(
        self,
        num_train_timesteps=1000,
        num_inference_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
        steps_offset: int = 0,
    ):
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

    def add_noise(
        self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        """Linear interpolation between data (t=0) and noise (t=num_train_timesteps)."""
        timepoints = timesteps.float() / self.num_train_timesteps
        timepoints = 1 - timepoints  # [1, 1/1000]
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])
        return timepoints * original_samples + (1 - timepoints) * noise

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device | None = None,
        input_img_size: int | None = None,
        base_img_size: int = 32 * 32 * 32,
    ) -> None:
        """Set the inference timesteps (optionally resolution-shifted by `timestep_transform`)."""
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than "
                f"`self.num_train_timesteps`: {self.num_train_timesteps}."
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

    def sample_timesteps(self, x_start):
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
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, next_timestep=None
    ) -> tuple[torch.Tensor, Any]:
        """One Euler step along the predicted velocity."""
        v_pred = model_output
        if next_timestep is None:
            dt = 1.0 / self.num_inference_steps
        else:
            dt = timestep - next_timestep
            dt = dt / self.num_train_timesteps
        z = sample + v_pred * dt

        return z, None
