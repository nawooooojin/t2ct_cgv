"""Generate volumes from reports (challenge format).

Input : prompts JSON, ``[{"input_image_name": ..., "report": ...}, ...]``
Output: ``<out-dir>/<input_image_name stem>.nii.gz`` (float32), one per prompt

CT-Gen (text is encoded on the fly with the three Report2CT encoders):
    python scripts/generate.py --task ctgen --ckpt weights/ctgen_ema.ckpt \
        --prompts prompts.json --out-dir outputs/ctgen_pred
MR-Gen (encode the prompts with RadFinder first, see encode_text_mrgen.py --prompts):
    python scripts/generate.py --task mrgen --ckpt weights/mrgen_ema.ckpt \
        --prompts prompts.json --text-dir work/text --out-dir outputs/mrgen_pred \
        --spacing-table data/mrgen/spacing_table.json --axcodes-table data/mrgen/axcodes_table.json

Sampling: rectified flow, Euler steps (CT-Gen submission: 50, MR-Gen: 100), classifier-free
guidance 5 (unconditional branch = zeroed text). CT volumes are 512 x 512 x 253 HU at spacing
0.75 / 0.75 / 1.3 mm; MR volumes are [0, 1] with the median spacing of their (sequence, plane) class.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import rootutils
import SimpleITK as sitk
import torch
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import mr_rate  # noqa: E402
from src.data.rf_navg import global_vector  # noqa: E402
from src.encoders.wan_vae import WanVAE  # noqa: E402
from src.models.rflow import RFlowScheduler  # noqa: E402
from src.models.unet import build_unet  # noqa: E402

LATENT_SHAPE = (16, 64, 64, 64)  # U-Net layout (C, H, W, D)
SPACING_MULTIPLIER = 100.0
CT_CLASS_LABEL = 1

_FINDINGS = re.compile(r"\bFindings\s*:", re.IGNORECASE)
_IMPRESSION = re.compile(r"\bImpression[s]?\s*:", re.IGNORECASE)


def split_ct_report(report: str) -> tuple[str, str]:
    """``"<preamble> Findings: ... Impression: ..."`` -> (findings, impression)."""
    text = report.strip()
    m_f = _FINDINGS.search(text)
    m_i = _IMPRESSION.search(text, m_f.end() if m_f else 0)
    if m_f is None and m_i is None:
        return text, ""
    findings = text[(m_f.end() if m_f else 0):(m_i.start() if m_i else len(text))].strip()
    return findings, text[m_i.end():].strip() if m_i else ""


def case_stem(name: str) -> str:
    return name[: -len(".nii.gz")] if name.endswith(".nii.gz") else Path(name).stem


def load_unet(ckpt_path: str, task: str, device: torch.device):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    unet_sd = {k[len("unet."):]: v for k, v in sd.items() if k.startswith("unet.")}
    xattn_dim = next(v.shape[1] for k, v in unet_sd.items() if k.endswith("attn2.to_k.weight"))
    pooled_dim = unet_sd["text_pooled_proj.weight"].shape[1] if "text_pooled_proj.weight" in unet_sd else None
    unet = build_unet(task, cross_attention_dim=xattn_dim, text_pooled_dim=pooled_dim)
    unet.load_state_dict(unet_sd, strict=True)
    return unet.to(device).eval().requires_grad_(False), float(sd["scale_factor"])


@torch.no_grad()
def denoise(unet, scale_factor, cond: dict, class_label: int, spacing_mm, n_steps: int, cfg: float,
            device: torch.device) -> torch.Tensor:
    """Rectified-flow sampling with classifier-free guidance -> latent ``(16, 64, 64, 64)`` (Wan layout)."""
    sched = RFlowScheduler(num_train_timesteps=1000, scale=1.4, use_timestep_transform=True,
                           use_discrete_timesteps=False, sample_method="uniform")
    sched.set_timesteps(n_steps, device=device, input_img_size=torch.tensor(float(64 ** 3)))
    ts = sched.timesteps
    next_ts = torch.cat([ts[1:], torch.zeros(1, dtype=ts.dtype, device=ts.device)])

    z = torch.randn(1, *LATENT_SHAPE, device=device)
    labels = torch.tensor([class_label, class_label], device=device)
    spacing = (torch.tensor(spacing_mm, dtype=torch.float32) * SPACING_MULTIPLIER).to(device)[None].repeat(2, 1)
    # batch the unconditional (zeroed text) and conditional branches together
    kw = {k: torch.cat([torch.zeros_like(v), v]) for k, v in cond.items() if k != "context_mask"}
    if "context_mask" in cond:
        kw["context_mask"] = cond["context_mask"].repeat(2, 1)

    for t, t_next in zip(ts, next_ts):
        t, t_next = float(t), float(t_next)
        pred = unet(x=torch.cat([z, z]), timesteps=torch.tensor([t, t], device=device),
                    spacing_tensor=spacing, class_labels=labels, **kw)
        uncond, cond_pred = pred.chunk(2)
        z = sched.step(uncond + cfg * (cond_pred - uncond), t, z, t_next)[0]
    z = z / scale_factor
    return z.squeeze(0).permute(0, 3, 1, 2).contiguous().float()  # (C, H, W, D) -> (C, T, H, W)


def save_ct(hu_zxy: np.ndarray, spacing_mm, out_path: Path) -> None:
    """Decoded (Z, X, Y) RAS volume -> LPS array as in CT-RATE, integer HU stored as float32."""
    arr = np.ascontiguousarray(hu_zxy.transpose(0, 2, 1))[:, ::-1, ::-1]  # (Z, Y, X), RAS -> LPS
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr).astype(np.int16).astype(np.float32))
    img.SetSpacing([float(s) for s in spacing_mm])
    sitk.WriteImage(img, str(out_path))


def save_mr(vol_zxy: np.ndarray, spacing_mm, axcodes, out_path: Path) -> None:
    """Decoded (Z, X, Y) RAS volume in [-1, 1] -> [0, 1] array reoriented to `axcodes`."""
    vol = vol_zxy.transpose(1, 2, 0)  # (X, Y, Z)
    ornt = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")),
                                           nib.orientations.axcodes2ornt(tuple(axcodes)))
    vol = nib.orientations.apply_orientation(vol, ornt)
    vol = np.clip((vol.astype(np.float32) + 1.0) / 2.0, 0.0, 1.0)
    signs = [-1.0 if a in ("L", "P", "I") else 1.0 for a in axcodes]
    affine = np.diag([signs[0] * spacing_mm[0], signs[1] * spacing_mm[1], signs[2] * spacing_mm[2], 1.0])
    nib.save(nib.Nifti1Image(vol, affine), str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ctgen", "mrgen"], required=True)
    ap.add_argument("--ckpt", required=True, help="EMA weights (scripts/extract_ema.py)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: 50 (ctgen) / 100 (mrgen)")
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--spacing", type=float, nargs=3, default=[0.75, 0.75, 1.3], help="ctgen output spacing (mm)")
    ap.add_argument("--text-dir", help="mrgen: RadFinder features of the prompts")
    ap.add_argument("--spacing-table", help="mrgen: spacing_table.json")
    ap.add_argument("--axcodes-table", help="mrgen: axcodes_table.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    steps = args.steps or (50 if args.task == "ctgen" else 100)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    torch.backends.cuda.enable_flash_sdp(False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts).read_text())

    unet, scale_factor = load_unet(args.ckpt, args.task, device)
    if args.task == "ctgen":
        from src.encoders.report_text import ReportTextEncoder

        text_encoder = ReportTextEncoder(device=device)
    else:
        spacing_table = {int(k): v for k, v in json.loads(Path(args.spacing_table).read_text()).items()}
        axcodes_table = {int(k): v for k, v in json.loads(Path(args.axcodes_table).read_text()).items()}
    vae = WanVAE(device=args.device)

    for e in tqdm(prompts, desc="generate"):
        stem = case_stem(str(e["input_image_name"]))
        out_path = out_dir / f"{stem}.nii.gz"
        if out_path.exists():
            continue
        if args.task == "ctgen":
            f, i = text_encoder.encode_pair(*split_ct_report(str(e["report"])))
            cond = {"context": torch.stack([f, i])[None].to(device)}  # (1, 2, 2560)
            label, spacing = CT_CLASS_LABEL, list(args.spacing)
        else:
            npz = np.load(Path(args.text_dir) / f"{stem}_rf_navg.npz")
            seqs, pooled = [], {}
            for short, section in (("f", "findings"), ("i", "impression")):
                seq = torch.from_numpy(npz[f"{section}_rf_navg"].astype(np.float32))  # (L, 1024)
                eos = torch.from_numpy(npz[f"{section}_rf_last"].astype(np.float32))  # (1024,)
                pooled[f"text_pooled_{short}"] = global_vector(eos, seq, torch.ones(len(seq), dtype=torch.bool))[None]
                seqs.append(seq)
            context = torch.cat(seqs)[None]  # (1, L_f + L_i, 1024)
            cond = {"context": context, "context_mask": torch.ones(context.shape[:2], dtype=torch.bool), **pooled}
            cond = {k: v.to(device) for k, v in cond.items()}
            label = mr_rate.class_label(stem)
            spacing = spacing_table[label]

        with torch.autocast(device.type, dtype=torch.bfloat16):
            z = denoise(unet, scale_factor, cond, label, spacing, steps, args.cfg, device)
        vol = vae.decode(z, intensity="hu" if args.task == "ctgen" else "unit").numpy()  # (Z, X, Y)
        if args.task == "ctgen":
            save_ct(vol, spacing, out_path)
        else:
            save_mr(vol, spacing, axcodes_table[label], out_path)


if __name__ == "__main__":
    main()
