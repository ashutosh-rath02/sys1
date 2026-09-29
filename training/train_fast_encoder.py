"""Train a genuinely fast v3 architecture: a small encoder (not a causal
chat LM) plus a one-neuron decision head, scored per-option and trained
with a listwise softmax over each example's real candidate set. One
batched forward pass per judgment, no chat template, no generation --
this is what actually gets to millisecond-level decisions on CPU (a
causal LM read via letter-logits, v1/v2's approach, cannot: it pays for
tokenizing and attending over a full natural-language prompt every call).

Training batches examples in groups that share the same number of
options -- e.g. all the noul (2-option) examples batch together, all the
4-option choice examples batch together -- since that's what can share a
single padded tensor. This is a real minibatch, not one example at a
time: on the same hardware it trains several times faster, and it's a
prerequisite for a GPU run to actually help at all.

    python training/train_fast_encoder.py data/seed_examples.jsonl data/real_examples.jsonl

Saves to models/sys1-fast-encoder/ (backbone + tokenizer + head weights).
"""
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"


def row_to_example(row: dict) -> dict:
    primitive = row["primitive"]
    instructions = row["instructions"]
    state = row.get("state")
    answer = row["answer"]

    if primitive == "noul":
        items = ["Yes", "No"]
        answer = "Yes" if answer == "yes" else "No"
    else:
        items = row["options"] if primitive == "choice" else row["levels"]

    context = f"{instructions}\nContext: {state}" if state else instructions
    return {"context": context, "items": items, "answer_index": items.index(answer)}


class CrossEncoderScorer(nn.Module):
    """One scalar score per (context, option) pair via mean-pooled backbone output."""

    def __init__(self, backbone_name: str = BACKBONE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)  # (batch,)


def make_batches(examples: list[dict], batch_size: int, rng: random.Random) -> list[list[dict]]:
    """Group examples with the same option count K together (so they share
    one padded tensor), shuffle within and across those groups, then chunk
    into batches of at most batch_size."""
    by_k: dict[int, list[dict]] = defaultdict(list)
    for ex in examples:
        by_k[len(ex["items"])].append(ex)

    batches = []
    for group in by_k.values():
        rng.shuffle(group)
        for i in range(0, len(group), batch_size):
            batches.append(group[i : i + batch_size])
    rng.shuffle(batches)
    return batches


def score_batch(model: CrossEncoderScorer, tokenizer, batch: list[dict], device: str) -> torch.Tensor:
    """All examples in `batch` share the same K (see make_batches). Returns
    (B, K) scores."""
    k = len(batch[0]["items"])
    contexts, options = [], []
    for ex in batch:
        contexts.extend([ex["context"]] * k)
        options.extend(ex["items"])
    tok = tokenizer(contexts, options, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    scores = model(tok["input_ids"], tok["attention_mask"])
    return scores.view(len(batch), k)


@torch.no_grad()
def evaluate(model, tokenizer, examples: list[dict], device: str, batch_size: int) -> tuple[float, float]:
    model.eval()
    n_correct = 0
    brier_total = 0.0
    rng = random.Random(0)
    for batch in make_batches(examples, batch_size, rng):
        scores = score_batch(model, tokenizer, batch, device)
        probs = torch.softmax(scores, dim=1)
        preds = probs.argmax(dim=1)
        answer_idx = torch.tensor([ex["answer_index"] for ex in batch], device=device)
        n_correct += (preds == answer_idx).sum().item()
        one_hot = F.one_hot(answer_idx, probs.size(1)).float()
        brier_total += ((probs - one_hot) ** 2).sum().item()
    model.train()
    n = len(examples)
    return n_correct / n, brier_total / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("models/sys1-fast-encoder"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    rows = []
    for path in args.input_files:
        with path.open(encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    examples = [row_to_example(r) for r in rows]

    random.Random(args.seed).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    print(f"train={len(train_examples)} val={len(val_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    model = CrossEncoderScorer(BACKBONE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    acc, brier = evaluate(model, tokenizer, val_examples, device, args.batch_size)
    print(f"before training: accuracy={acc:.2%} brier={brier:.4f}")

    best_brier = brier
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = -1

    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        batches = make_batches(train_examples, args.batch_size, rng)
        total_loss = 0.0
        for batch in batches:
            scores = score_batch(model, tokenizer, batch, device)
            answer_idx = torch.tensor([ex["answer_index"] for ex in batch], device=device)
            loss = F.cross_entropy(scores, answer_idx)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        acc, brier = evaluate(model, tokenizer, val_examples, device, args.batch_size)
        print(f"epoch {epoch}: mean_loss={total_loss / len(batches):.4f} val_accuracy={acc:.2%} val_brier={brier:.4f}")
        if brier < best_brier:
            best_brier = brier
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    print(f"using best checkpoint: epoch {best_epoch} (val_brier={best_brier:.4f})")
    model.load_state_dict(best_state)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.out_dir)
    model.backbone.save_pretrained(args.out_dir)
    torch.save(model.head.state_dict(), args.out_dir / "head.pt")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
