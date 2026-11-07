"""Frozen Wan2.1 VAE: CT volume in HU <-> normalised Wan latent.

HU are clipped to [-1000, 1000] and scaled to [-1, 1], the grey slice is repeated to the VAE's
3 input channels, and the posterior mode is normalised per channel with the VAE's
``latents_mean`` / ``latents_std`` (diffusers leaves this step to the pipeline). Decoding inverts
it and averages the 3 output channels. fp32 only (bf16 degrades the reconstruction).
"""

from __future__ import annotations

import logging
from typing import Final

import torch
from einops import rearrange

log = logging.getLogger(__name__)

DEFAULT_WAN_HF_ID: Final[str] = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_INPUT_CHANNELS: Final[int] = 3
HU_CLIP: Final[float] = 1000.0


class WanVAE:
    """Lazily built, frozen Wan2.1 VAE (``diffusers.AutoencoderKLWan``).

    Args:
        hf_id: Hugging Face repo holding the VAE.
        device: torch device string.
        subfolder: subfolder of the VAE weights in ``hf_id``.
    """

    def __init__(
        self,
        hf_id: str = DEFAULT_WAN_HF_ID,
        device: str = "cuda",
        subfolder: str = "vae",
    ) -> None:
        self.hf_id = hf_id
        self.device = device
        self.subfolder = subfolder
        self._vae: torch.nn.Module | None = None
        self._mean: torch.Tensor | None = None  # (1, 16, 1, 1, 1)
        self._std: torch.Tensor | None = None  # (1, 16, 1, 1, 1)

    def _ensure_built(self) -> None:
        """Load the VAE (fp32, eval, frozen) and its per-channel latent statistics."""
        if self._vae is not None:
            return
        from diffusers import AutoencoderKLWan  # noqa: PLC0415

        vae = AutoencoderKLWan.from_pretrained(
            self.hf_id, subfolder=self.subfolder, torch_dtype=torch.float32
        )
        vae.requires_grad_(False).eval().to(self.device)
        z_dim = len(vae.config.latents_mean)  # 16
        self._mean = torch.tensor(
            vae.config.latents_mean, dtype=torch.float32, device=self.device
        ).view(1, z_dim, 1, 1, 1)
        self._std = torch.tensor(
            vae.config.latents_std, dtype=torch.float32, device=self.device
        ).view(1, z_dim, 1, 1, 1)
        self._vae = vae
        log.info("Loaded Wan VAE %s (z_dim=%d) on %s", self.hf_id, z_dim, self.device)

    @torch.inference_mode()
    def encode(self, ct_thw: torch.Tensor) -> torch.Tensor:
        """Encode one CT volume to a normalised Wan latent.

        Args:
            ct_thw: HU volume ``(T, H, W)``, e.g. ``(253, 512, 512)``; H, W divisible by 8.

        Returns:
            Latent ``(16, 1 + (T - 1) // 4, H // 8, W // 8)`` = ``(16, 64, 64, 64)``, CPU float32.
        """
        self._ensure_built()
        x = ct_thw.to(self.device, torch.float32)
        x = torch.clamp(x, -HU_CLIP, HU_CLIP) / HU_CLIP  # (T, H, W) in [-1, 1]
        x = x[:, None].repeat(1, WAN_INPUT_CHANNELS, 1, 1)  # (T, 3, H, W)
        x = rearrange(x, "t c h w -> c t h w")[None]  # (1, 3, T, H, W)
        z = self._vae.encode(x).latent_dist.mode()  # (1, 16, T_lat, H/8, W/8)
        z = (z - self._mean) / self._std  # per-channel normalisation
        return z[0].cpu()  # (16, T_lat, H/8, W/8)

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a normalised Wan latent back to HU.

        Args:
            z: latent ``(16, T_lat, h, w)``, e.g. ``(16, 64, 64, 64)``.

        Returns:
            HU volume ``(1 + (T_lat - 1) * 4, 8h, 8w)`` = ``(253, 512, 512)``, CPU float32,
            in [-1000, 1000].
        """
        self._ensure_built()
        z = z.to(self.device, torch.float32)[None]  # (1, 16, T_lat, h, w)
        z = z * self._std + self._mean  # back to the raw VAE latent
        x = self._vae.decode(z).sample  # (1, 3, T, 8h, 8w)
        x = x[0].mean(0)  # (T, 8h, 8w)
        x = torch.clamp(x, -1.0, 1.0) * HU_CLIP
        return x.cpu()
