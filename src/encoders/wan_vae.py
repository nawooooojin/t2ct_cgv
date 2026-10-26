"""Frozen Wan2.1 VAE (diffusers `AutoencoderKLWan`) used as the 3D latent space.

A volume is treated as a grayscale video: depth -> time, each slice repeated to 3 channels.
8x8 spatial and 4x temporal compression with 16 latent channels, so a 512 x 512 x 253 volume
maps to a (16, 64, 64, 64) latent. Latents are normalised per channel with the VAE's
`latents_mean` / `latents_std` (diffusers leaves this to the pipeline).

Intensity
    "hu"  : CT, HU clipped to [-1000, 1000] and divided by 1000 -> [-1, 1]
    "unit": input already in [-1, 1] (MR, per-volume percentile normalisation)
"""

from __future__ import annotations

import torch
from einops import rearrange

DEFAULT_WAN_HF_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HU_CLIP = 1000.0


class WanVAE:
    def __init__(self, hf_id: str = DEFAULT_WAN_HF_ID, device: str = "cuda:0") -> None:
        from diffusers import AutoencoderKLWan

        vae = AutoencoderKLWan.from_pretrained(hf_id, subfolder="vae", torch_dtype=torch.float32)
        self.vae = vae.requires_grad_(False).eval().to(device)
        self.device = device
        z = len(vae.config.latents_mean)
        self.mean = torch.tensor(vae.config.latents_mean, device=device).view(1, z, 1, 1, 1)
        self.std = torch.tensor(vae.config.latents_std, device=device).view(1, z, 1, 1, 1)

    @torch.inference_mode()
    def encode(self, vol_thw: torch.Tensor, intensity: str = "hu") -> torch.Tensor:
        """``(T, H, W)`` volume -> normalised latent ``(16, 1 + (T-1)//4, H//8, W//8)`` (CPU)."""
        x = vol_thw.to(self.device, torch.float32)
        x = torch.clamp(x, -HU_CLIP, HU_CLIP) / HU_CLIP if intensity == "hu" else torch.clamp(x, -1.0, 1.0)
        x = rearrange(x[:, None].repeat(1, 3, 1, 1), "t c h w -> c t h w")[None]  # (1, 3, T, H, W)
        z = self.vae.encode(x).latent_dist.mode()
        return ((z - self.mean) / self.std)[0].cpu()

    @torch.inference_mode()
    def decode(self, z: torch.Tensor, intensity: str = "hu") -> torch.Tensor:
        """Normalised latent ``(16, T_lat, h, w)`` -> volume ``(T, H, W)`` (HU or [-1, 1], CPU)."""
        z = z.to(self.device, torch.float32)[None] * self.std + self.mean
        x = self.vae.decode(z).sample[0].mean(0)  # (T, H, W), channel mean
        x = torch.clamp(x, -1.0, 1.0)
        return (x * HU_CLIP if intensity == "hu" else x).cpu()
