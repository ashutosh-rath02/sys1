"""v4: dual-encoder engine (see training/train_dual_encoder.py). Context
and options are embedded independently -- the score is a dot product --
so an option's embedding is the same every time its text is the same.
That's what makes caching valid: DualEngine caches every action
embedding it computes, so a repeated candidate set (Snake's four
directions, a fixed ticket-routing option list) only ever gets encoded
once, no matter how many judgments are made against it.

Same judge() interface as Engine/FastEngine.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .engine import RawJudgment


class DualEngine:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.backbone = AutoModel.from_pretrained(model_dir)
        self.backbone.eval()

        heads_path = Path(model_dir) / "heads.pt"
        if not heads_path.is_file():
            from huggingface_hub import hf_hub_download

            heads_path = hf_hub_download(repo_id=model_dir, filename="heads.pt")
        checkpoint = torch.load(heads_path, weights_only=True)

        hidden = self.backbone.config.hidden_size
        embed_dim = checkpoint["embed_dim"]
        self.state_head = nn.Linear(hidden, embed_dim)
        self.action_head = nn.Linear(hidden, embed_dim)
        self.state_head.load_state_dict(checkpoint["state_head"])
        self.action_head.load_state_dict(checkpoint["action_head"])
        self.state_head.eval()
        self.action_head.eval()
        self.temperature = checkpoint["temperature"]

        self._action_cache: dict[str, torch.Tensor] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _pool(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def _encode_state(self, text: str) -> torch.Tensor:
        batch = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        pooled = self._pool(batch["input_ids"], batch["attention_mask"])
        return F.normalize(self.state_head(pooled), dim=-1)[0]

    @torch.no_grad()
    def _encode_actions_uncached(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        pooled = self._pool(batch["input_ids"], batch["attention_mask"])
        return F.normalize(self.action_head(pooled), dim=-1)

    def _encode_actions(self, items: list[str]) -> torch.Tensor:
        """Encode each item, reusing any embedding already in the cache
        and only running the model on the ones that are actually new."""
        missing = [item for item in items if item not in self._action_cache]
        if missing:
            embs = self._encode_actions_uncached(missing)
            for item, emb in zip(missing, embs):
                self._action_cache[item] = emb
        self.cache_hits += len(items) - len(missing)
        self.cache_misses += len(missing)
        return torch.stack([self._action_cache[item] for item in items])

    def judge(self, instructions: str, primitive: str, items: list[str], state: dict | None = None) -> RawJudgment:
        context = f"{instructions}\nContext: {state}" if state else instructions
        state_emb = self._encode_state(context)
        action_embs = self._encode_actions(items)

        scores = (action_embs @ state_emb) / self.temperature
        probs = torch.softmax(scores, dim=0).tolist()

        prob_by_label = dict(zip(items, probs))
        top_label = items[max(range(len(items)), key=lambda i: probs[i])]
        return RawJudgment(labels=items, probabilities=prob_by_label, top_label=top_label)
