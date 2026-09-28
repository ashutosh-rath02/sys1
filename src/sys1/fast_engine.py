"""v3: a genuinely fast engine -- small encoder + one-neuron decision
head (see training/train_fast_encoder.py), not a causal chat LM read via
letter-logits. No chat template, no generation: one batched forward pass
per judgment. Same judge() interface as Engine (engine.py), so Choice/
Noul/Score and the game code work with either engine unchanged.
"""
from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .engine import RawJudgment


class _CrossEncoderScorer(nn.Module):
    def __init__(self, backbone_dir: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_dir)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)


class FastEngine:
    """Loads a trained models/sys1-fast-encoder/-style checkpoint."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = _CrossEncoderScorer(model_dir)

        head_path = Path(model_dir) / "head.pt"
        if not head_path.is_file():
            head_path = hf_hub_download(repo_id=model_dir, filename="head.pt")
        self.model.head.load_state_dict(torch.load(head_path, weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def judge(self, instructions: str, primitive: str, items: list[str], state: dict | None = None) -> RawJudgment:
        context = f"{instructions}\nContext: {state}" if state else instructions
        batch = self.tokenizer(
            [context] * len(items), items, return_tensors="pt", padding=True, truncation=True
        )
        scores = self.model(batch["input_ids"], batch["attention_mask"])
        probs = torch.softmax(scores, dim=0).tolist()

        prob_by_label = dict(zip(items, probs))
        top_label = items[max(range(len(items)), key=lambda i: probs[i])]
        return RawJudgment(labels=items, probabilities=prob_by_label, top_label=top_label)
