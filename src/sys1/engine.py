"""Core engine: loads a small instruct model and turns its next-token logits,
over a constrained set of candidate answers, into a calibrated-shaped
probability distribution.

This is the mechanism, not a trained "judgment model" yet (that's v1, in
training/). v0 works with any off-the-shelf instruct model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _adapter_base_model(model_name: str) -> str | None:
    """If model_name is a local PEFT adapter directory, return the base
    model it was trained on; otherwise None."""
    config_path = Path(model_name) / "adapter_config.json"
    if not config_path.is_file():
        return None
    return json.loads(config_path.read_text(encoding="utf-8"))["base_model_name_or_path"]


@dataclass
class RawJudgment:
    labels: list[str]
    probabilities: dict[str, float]
    top_label: str


class Engine:
    """Loads a base model once and answers constrained judgment prompts."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16

        base_model = _adapter_base_model(model_name)
        if base_model:
            from peft import PeftModel

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
            self.model = PeftModel.from_pretrained(base, model_name).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype
            ).to(self.device)
        self.model.eval()

    def _build_prompt(self, instructions: str, state: dict | None, answer_line: str) -> str:
        state_block = f"\nContext:\n{state}\n" if state else "\n"
        user_msg = f"{instructions}{state_block}{answer_line}"
        messages = [{"role": "user", "content": user_msg}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _candidate_token_id(self, label: str) -> int | None:
        """Best-effort single-token id for a candidate label, trying a
        leading-space variant first since that's how labels usually appear
        mid-generation after 'Answer: '."""
        for variant in (label, " " + label):
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) >= 1:
                return ids[0]
        return None

    @torch.no_grad()
    def judge(self, instructions: str, labels: list[str], state: dict | None = None) -> RawJudgment:
        answer_line = "Respond with only the single letter of your answer, nothing else.\nAnswer:"
        prompt = self._build_prompt(instructions, state, answer_line)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1, :]

        letters = [chr(ord("A") + i) for i in range(len(labels))]
        candidate_ids = [self._candidate_token_id(letter) for letter in letters]
        if any(cid is None for cid in candidate_ids):
            raise RuntimeError("Could not resolve candidate letter tokens for this tokenizer.")

        candidate_logits = torch.tensor([logits[cid].item() for cid in candidate_ids])
        probs = torch.softmax(candidate_logits, dim=0).tolist()

        prob_by_label = {label: p for label, p in zip(labels, probs)}
        top_label = labels[max(range(len(labels)), key=lambda i: probs[i])]
        return RawJudgment(labels=labels, probabilities=prob_by_label, top_label=top_label)


@lru_cache(maxsize=1)
def default_engine() -> Engine:
    return Engine()
