"""Write the EMA weights of a Lightning training checkpoint to a standalone file.

Parameters are taken from the EMA shadow, buffers (e.g. `scale_factor`) from the checkpoint.

    python scripts/extract_ema.py outputs/ctgen/<run>/checkpoints/last.ckpt weights/ctgen_ema.ckpt
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = dict(ckpt["state_dict"])
    shadow = ckpt["callbacks"]["EMACallback"]["shadow"]
    for k, v in shadow.items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    sd = {k: v for k, v in sd.items() if not k.startswith("repa.")}  # training-only head
    torch.save({"state_dict": sd, "hyper_parameters": ckpt.get("hyper_parameters", {})}, args.out)
    print(f"saved {args.out} ({len(sd)} tensors)")


if __name__ == "__main__":
    main()
