"""Generate CT volumes from challenge prompts with a trained WARP-CT checkpoint.

Per prompt: RadFinder sidecar (``encode_text_ctgen.py --prompts``) -> CFG rectified-flow sampling
in the Wan latent space ``(1, 16, 64, 64, 64)`` -> Wan VAE decode ``(253, 512, 512)`` HU ->
float32 ``.nii.gz`` of 512 x 512 x 253 at the given spacing. The decoded content is RAS; the two
in-plane axes are flipped to LPS content (CT-RATE convention), direction left at identity.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import rootutils

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.ct_rate import prompt_stem  # noqa: E402
from src.data.rf_navg import load_text_sidecar  # noqa: E402
from src.encoders.wan_vae import DEFAULT_WAN_HF_ID, WanVAE  # noqa: E402
from src.models.module import MODALITY_CLASS_CT  # noqa: E402

log = logging.getLogger("generate")
LATENT_SHAPE = (16, 64, 64, 64)  # (C, H, W, D)
SPACING_MULTIPLIER = 100.0


def load_unet(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, object, float]:
    """Build the UNet + scheduler from ``configs/model/ctgen.yaml`` and load the checkpoint.

    The EMA shadow is overlaid on the raw weights when the checkpoint carries one (a file from
    ``extract_ema.py`` already holds EMA weights in ``state_dict``).

    Returns:
        ``(unet, scheduler, scale_factor)``.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = dict(ckpt["state_dict"])
    shadow = ckpt.get("callbacks", {}).get("EMACallback", {}).get("shadow")
    if shadow:
        sd.update(shadow)
        log.info("EMA shadow overlaid (%d tensors)", len(shadow))
    scale_factor = float(sd["scale_factor"])
    log.info("scale_factor %.6f", scale_factor)

    cfg = OmegaConf.load(Path(ROOT) / "configs" / "model" / "ctgen.yaml")
    cfg.unet.use_flash_attention = False
    unet = hydra.utils.instantiate(cfg.unet)
    unet.load_state_dict({k[len("unet.") :]: v for k, v in sd.items() if k.startswith("unet.")}, strict=True)
    unet = unet.to(device).eval().requires_grad_(False)
    scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    return unet, scheduler, scale_factor


@torch.no_grad()
def sample_latent(
    unet: torch.nn.Module,
    scheduler: object,
    sidecar: dict[str, torch.Tensor],
    spacing_mm: list[float],
    cfg_scale: float,
    steps: int,
    device: torch.device,
) -> torch.Tensor:
    """CFG rectified-flow sampling for one prompt.

    The unconditional branch zeroes the context and both pooled vectors (mask unchanged), as the
    training-time CFG dropout does.

    Returns:
        Latent ``(1, 16, 64, 64, 64)`` = (B, C, H, W, D), still multiplied by ``scale_factor``.
    """
    ctx = torch.cat([sidecar["seq_f"], sidecar["seq_i"]], 0)[None].to(device)  # (1, L_f + L_i, 1024)
    mask = torch.ones(ctx.shape[:2], dtype=torch.bool, device=device)  # (1, L_f + L_i)
    pf = sidecar["pooled_f"][None].to(device)  # (1, 2048)
    pi = sidecar["pooled_i"][None].to(device)  # (1, 2048)
    spacing = (torch.tensor(spacing_mm, dtype=torch.float32) * SPACING_MULTIPLIER)[None].to(device)  # (1, 3)
    cls = torch.tensor([MODALITY_CLASS_CT], device=device)  # (1,)

    scheduler.set_timesteps(steps, device=device, input_img_size=torch.tensor(float(64 * 64 * 64)))
    all_t = scheduler.timesteps
    all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype, device=all_t.device)))
    z = torch.randn(1, *LATENT_SHAPE, device=device, dtype=torch.float32)  # (1, 16, 64, 64, 64)

    use_cfg = cfg_scale != 1.0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for t, next_t in zip(all_t, all_next_t):
            t_s, next_s = float(t.item()), float(next_t.item())
            t_tensor = torch.tensor([t_s], device=device, dtype=torch.float32)  # (1,)
            if use_cfg:
                pred = unet(
                    x=torch.cat([z, z]),  # (2, 16, 64, 64, 64)
                    timesteps=t_tensor.repeat(2),
                    context=torch.cat([torch.zeros_like(ctx), ctx]),  # (2, L, 1024) [uncond, cond]
                    class_labels=cls.repeat(2),
                    spacing_tensor=spacing.repeat(2, 1),
                    context_mask=mask.repeat(2, 1),
                    text_pooled_f=torch.cat([torch.zeros_like(pf), pf]),
                    text_pooled_i=torch.cat([torch.zeros_like(pi), pi]),
                )
                u, c = pred.chunk(2)
                v = u + cfg_scale * (c - u)
            else:
                v = unet(
                    x=z, timesteps=t_tensor, context=ctx, class_labels=cls, spacing_tensor=spacing,
                    context_mask=mask, text_pooled_f=pf, text_pooled_i=pi,
                )
            z, _ = scheduler.step(v, t_s, z, next_s)
    return z


def save_nifti(vol_zxy: np.ndarray, spacing_mm: list[float], out_path: Path) -> None:
    """Save a decoded ``(Z, X, Y)`` RAS HU volume as float32 with LPS content (X, Y reversed)."""
    arr = vol_zxy.transpose(0, 2, 1)  # (Z, Y, X) for SimpleITK
    arr = arr[:, ::-1, ::-1].copy()  # RAS content -> LPS content
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing([float(s) for s in spacing_mm])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))


def main() -> None:
    """Generate one volume per prompt (existing outputs are skipped)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("ctgen",), default="ctgen")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompts", required=True, help='prompts.json [{"input_image_name", "report"}]')
    p.add_argument("--text-dir", required=True, help="sidecars from encode_text_ctgen.py --prompts")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cfg-scale", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--spacing", type=float, nargs=3, default=[0.75, 0.75, 1.3], help="mm (x, y, z)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hf-id", default=DEFAULT_WAN_HF_ID, help="Wan VAE Hugging Face repo")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    device = torch.device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    unet, scheduler, scale_factor = load_unet(args.ckpt, device)
    vae = WanVAE(hf_id=args.hf_id, device=args.device)

    prompts = json.loads(Path(args.prompts).read_text())
    out_dir = Path(args.out_dir)
    text_dir = Path(args.text_dir)
    for item in tqdm(prompts, desc="generate"):
        name = item["input_image_name"]
        out_path = out_dir / name
        if out_path.exists():
            continue
        sidecar = load_text_sidecar(text_dir / f"{prompt_stem(name)}.npz")
        z = sample_latent(unet, scheduler, sidecar, args.spacing, args.cfg_scale, args.steps, device)
        z = (z / scale_factor).float()  # (1, 16, 64, 64, 64) = (B, C, H, W, D)
        z_wan = z[0].permute(0, 3, 1, 2).contiguous()  # (16, 64, 64, 64) = (C, T_lat=D, H, W)
        vol = vae.decode(z_wan)  # (253, 512, 512) = (Z, X, Y), HU
        save_nifti(vol.numpy(), args.spacing, out_path)
        log.info("saved %s", out_path)


if __name__ == "__main__":
    main()
