"""v4: a dual-encoder architecture -- separate state and action encoders
compared via dot product -- scaled to what we can actually train (a
shared small backbone with two projection heads, trained with an
InfoNCE-style contrastive loss).

Why this is a genuinely different architecture from v3's cross-encoder,
not just a rename: v3 encodes (context, option) TOGETHER every call, so
an option's representation depends on the context and can never be
reused. Here, context and options are encoded SEPARATELY -- the score is
just a dot product -- so an option's embedding is the same every time
its text is the same. Fixed, reused candidate sets (Snake's four
directions, a ticket-routing option list) can have their action
embeddings computed once and cached.

Training batches examples that share the same number of options K
together (see train_fast_encoder.py's docstring for why) -- a real
minibatch, not one example at a time.

    python training/train_dual_encoder.py data/seed_examples.jsonl data/real_examples.jsonl data/benchmark_domain_examples.jsonl

Saves to models/sys1-dual-encoder/ (backbone + tokenizer + two heads).
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
EMBED_DIM = 256


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


class DualEncoder(nn.Module):
    """One shared backbone, two small projection heads -> a shared
    embedding space. State and action embeddings are computed
    independently of each other; only their dot product ever combines
    them, which is exactly what makes caching an action embedding valid."""

    def __init__(self, backbone_name: str = BACKBONE, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.state_head = nn.Linear(hidden, embed_dim)
        self.action_head = nn.Linear(hidden, embed_dim)

    def _pool(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode_state(self, input_ids, attention_mask) -> torch.Tensor:
        return F.normalize(self.state_head(self._pool(input_ids, attention_mask)), dim=-1)

    def encode_action(self, input_ids, attention_mask) -> torch.Tensor:
        return F.normalize(self.action_head(self._pool(input_ids, attention_mask)), dim=-1)


def make_batches(examples: list[dict], batch_size: int, rng: random.Random) -> list[list[dict]]:
    """Group examples with the same option count K together (so their
    action tensors share one padded shape), shuffle, then chunk."""
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


def score_batch(model: DualEncoder, tokenizer, batch: list[dict], temperature: float, device: str) -> torch.Tensor:
    """All examples in `batch` share the same K. Returns (B, K) scores."""
    b, k = len(batch), len(batch[0]["items"])

    ctx_tok = tokenizer([ex["context"] for ex in batch], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    state_emb = model.encode_state(ctx_tok["input_ids"], ctx_tok["attention_mask"])  # (B, d)

    flat_items = [item for ex in batch for item in ex["items"]]
    item_tok = tokenizer(flat_items, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    action_emb = model.encode_action(item_tok["input_ids"], item_tok["attention_mask"]).view(b, k, -1)  # (B, K, d)

    return torch.einsum("bd,bkd->bk", state_emb, action_emb) / temperature


@torch.no_grad()
def evaluate(model, tokenizer, examples: list[dict], temperature: float, device: str, batch_size: int) -> tuple[float, float]:
    model.eval()
    n_correct = 0
    brier_total = 0.0
    rng = random.Random(0)
    for batch in make_batches(examples, batch_size, rng):
        scores = score_batch(model, tokenizer, batch, temperature, device)
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
    parser.add_argument("--out-dir", type=Path, default=Path("models/sys1-dual-encoder"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
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
    model = DualEncoder(BACKBONE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    acc, brier = evaluate(model, tokenizer, val_examples, args.temperature, device, args.batch_size)
    print(f"before training: accuracy={acc:.2%} brier={brier:.4f}")

    best_brier = brier
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = -1

    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        batches = make_batches(train_examples, args.batch_size, rng)
        total_loss = 0.0
        for batch in batches:
            scores = score_batch(model, tokenizer, batch, args.temperature, device)
            answer_idx = torch.tensor([ex["answer_index"] for ex in batch], device=device)
            loss = F.cross_entropy(scores, answer_idx)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        acc, brier = evaluate(model, tokenizer, val_examples, args.temperature, device, args.batch_size)
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
    torch.save(
        {"state_head": model.state_head.state_dict(), "action_head": model.action_head.state_dict(),
         "embed_dim": EMBED_DIM, "temperature": args.temperature},
        args.out_dir / "heads.pt",
    )
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
