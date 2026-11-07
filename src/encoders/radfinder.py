"""Frozen RadFinder text tower (Qwen3-Embedding-0.6B + LoRA) -> layer-averaged token features.

Needs ``transformers >= 5`` (the ``t2ct-radfinder`` env). Importing this module is free; the model
is built on the first ``encode_*`` call.

``hidden_states`` holds 29 entries: the token embedding, 27 raw residual-stream states and the
final post-RMSNorm state. Their per-token norms differ by orders of magnitude, so every
(layer, token) row is RMS-normalised before the layer average (``aggregate_layers``).
"""

from __future__ import annotations

import sys
from typing import Final

import torch
from torch import nn

from src.data.rf_navg import aggregate_layers, rms_normalize

RADFINDER_REPO: Final[str] = "lmb-freiburg/radfinder"
QWEN3_TOKENIZER: Final[str] = "Qwen/Qwen3-Embedding-0.6B"  # RadFinder config `text_tokenizer_name`
TEXT_DIM: Final[int] = 1024


class RadFinderTextEncoder(nn.Module):
    """Lazily built, frozen RadFinder text tower.

    Args:
        device: torch device string.
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device_ = torch.device(device)
        self._model: nn.Module | None = None
        self._tokenizer = None

    def _ensure_built(self) -> None:
        """Download the snapshot, make its ``radfinder`` package importable, load tokenizer + model."""
        if self._model is not None:
            return
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        snap = snapshot_download(RADFINDER_REPO)
        # The remote modeling code imports the snapshot's own `radfinder` package.
        while snap in sys.path:
            sys.path.remove(snap)
        sys.path.insert(0, snap)
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(QWEN3_TOKENIZER)
        model = AutoModel.from_pretrained(RADFINDER_REPO, trust_remote_code=True)
        model.eval().requires_grad_(False).to(self.device_)
        self._model = model

    @torch.no_grad()
    def encode_layer_stack(self, text: str) -> torch.Tensor:
        """Encode one section to every hidden state, unnormalised, without truncation.

        Args:
            text: findings or impression text; ``""`` is valid (EOS only, ``L = 1``).

        Returns:
            ``(29, L, 1024)`` CPU float32; ``L`` includes the appended ``<|endoftext|>``.
        """
        self._ensure_built()
        enc = self._tokenizer(text, return_tensors="pt", truncation=False)
        out = self._model.model.backbone_text(
            input_ids=enc["input_ids"].to(self.device_),
            attention_mask=enc["attention_mask"].to(self.device_),
            output_hidden_states=True,
        )
        stack = torch.stack([h[0] for h in out.hidden_states])  # (29, L, 1024)
        return stack.cpu().float()

    @torch.no_grad()
    def encode_navg(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one section to its two sidecar arrays from a single forward.

        Args:
            text: findings or impression text.

        Returns:
            ``navg`` ``(L, 1024)``: per-layer RMS-normalised, averaged over the 29 layers, not
            renormalised; ``last`` ``(1024,)``: the final-layer EOS row, RMS-normalised. CPU float32.
        """
        stack = self.encode_layer_stack(text)  # (29, L, 1024)
        return aggregate_layers(stack), rms_normalize(stack[-1][-1])  # (L, 1024), (1024,)
