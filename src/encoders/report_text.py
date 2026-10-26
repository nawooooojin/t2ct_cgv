"""Report2CT text encoder (CT-Gen): three frozen biomedical encoders, mean-pooled and concatenated.

MedEmbed-large (1024) + ClinicalBERT (768) + BiomedVLP-CXR-BERT-specialized (768) = 2560-d per
section. Findings and impression are encoded separately.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

MODEL_IDS = [
    "abhinand/MedEmbed-large-v0.1",
    "medicalai/ClinicalBERT",
    "microsoft/BiomedVLP-CXR-BERT-specialized",
]
MAX_SEQ_LEN = 512


class ReportTextEncoder(nn.Module):
    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__()
        self.device_ = torch.device(device)
        self.tokenizers, self.models = [], []
        for name in MODEL_IDS:
            self.tokenizers.append(AutoTokenizer.from_pretrained(name, trust_remote_code=True))
            model = AutoModel.from_pretrained(name, trust_remote_code=True).eval().to(self.device_)
            self.models.append(model.requires_grad_(False))

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """One text -> ``(2560,)`` (CPU, float32)."""
        pooled = []
        for tok, model in zip(self.tokenizers, self.models):
            inputs = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
            inputs = {k: v.to(self.device_) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state  # (1, L, D)
            m = inputs["attention_mask"].unsqueeze(-1).float()
            pooled.append((hidden * m).sum(1) / m.sum(1).clamp(min=1e-9))  # (1, D)
        return torch.cat(pooled, dim=-1).squeeze(0).cpu().float()

    def encode_pair(self, findings: str, impression: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(findings), self.encode(impression)
