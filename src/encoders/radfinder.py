"""RadFinder text tower (MR-Gen): Qwen3-Embedding-0.6B + LoRA from `lmb-freiburg/radfinder`.

Needs transformers >= 5 (separate environment, see README). For one text section it returns the
layer-aggregated token sequence ``(L, 1024)`` and the RMS-normalised final-layer EOS ``(1024,)``
(see `src/data/rf_navg.py`). No truncation is applied.
"""

from __future__ import annotations

import glob
import os
import sys

import torch

from src.data.rf_navg import aggregate_layers, rms_normalize

RADFINDER_REPO = "lmb-freiburg/radfinder"
RADFINDER_REVISION = "fa74969fcbb7857b7c8792e09b20463f43329050"  # the revision the MR-Gen model was trained with
QWEN3_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"


class RadFinderTextEncoder:
    def __init__(self, device: str = "cuda:0") -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        # the remote code imports the `radfinder` package shipped inside the model repo
        snap = snapshot_download(RADFINDER_REPO, revision=RADFINDER_REVISION,
                                 cache_dir=os.environ.get("HF_HUB_CACHE"))
        if snap not in sys.path:
            sys.path.insert(0, snap)
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER)
        model = AutoModel.from_pretrained(RADFINDER_REPO, revision=RADFINDER_REVISION, trust_remote_code=True)
        self.model = model.eval().requires_grad_(False).to(device)
        self.device = device

    @torch.no_grad()
    def encode_navg(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Text -> (``(L, 1024)`` layer-averaged tokens, ``(1024,)`` normalised EOS), CPU float32."""
        enc = self.tokenizer(text, return_tensors="pt", truncation=False)
        out = self.model.model.backbone_text(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
            output_hidden_states=True,
        )
        stack = torch.stack([h[0] for h in out.hidden_states]).cpu().float()  # (29, L, 1024)
        return aggregate_layers(stack), rms_normalize(stack[-1][-1])
