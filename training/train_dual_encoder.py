"""v4: a dual-encoder architecture, same spirit as Contrastive-LM/CLM's
state-encoder / action-encoder split, scaled to what we can actually
train (no 90M-example pretrain -- a shared small backbone with two
projection heads, trained with an InfoNCE-style contrastive loss).

Why this is a genuinely different architecture from v3's cross-encoder,
not just a rename: v3 encodes (context, option) TOGETHER every call, so
an option's representation depends on the context and can never be
reused. Here, context and options are encoded SEPARATELY -- the score is
just a dot product -- so an option's embedding is the same every time
its text is the same. Fixed, reused candidate sets (Snake's four
directions, a ticket-routing option list) can have their action
embeddings computed once and cached, matching CLM's own claimed latency
win from this exact property.

    python training/train_dual_encoder.py data/seed_examples.jsonl data/real_examples.jsonl data/benchmark_domain_examples.jsonl

Saves to models/sys1-dual-encoder/ (backbone + tokenizer + two heads).
"""
from __future__ import annotations

import argparse
import json
import random
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


def score_example(model: DualEncoder, tokenizer, context: str, items: list[str], temperature: float) -> torch.Tensor:
    ctx_batch = tokenizer([context], return_tensors="pt", padding=True, truncation=True)
    state_emb = model.encode_state(ctx_batch["input_ids"], ctx_batch["attention_mask"])  # (1, d)

    item_batch = tokenizer(items, return_tensors="pt", padding=True, truncation=True)
    action_embs = model.encode_action(item_batch["input_ids"], item_batch["attention_mask"])  # (K, d)

    return (state_emb @ action_embs.T).squeeze(0) / temperature  # (K,)


@torch.no_grad()
def evaluate(model, tokenizer, examples: list[dict], temperature: float) -> tuple[float, float]:
    model.eval()
    n_correct = 0
    brier_total = 0.0
    for ex in examples:
        scores = score_example(model, tokenizer, ex["context"], ex["items"], temperature)
        probs = torch.softmax(scores, dim=0)
        pred = probs.argmax().item()
        n_correct += pred == ex["answer_index"]
        one_hot = torch.zeros_like(probs)
        one_hot[ex["answer_index"]] = 1.0
        brier_total += ((probs - one_hot) ** 2).sum().item()
    model.train()
    n = len(examples)
    return n_correct / n, brier_total / n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("models/sys1-dual-encoder"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
    model = DualEncoder(BACKBONE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    acc, brier = evaluate(model, tokenizer, val_examples, args.temperature)
    print(f"before training: accuracy={acc:.2%} brier={brier:.4f}")

    best_brier = brier
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = -1

    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(train_examples)
        total_loss = 0.0
        for ex in train_examples:
            scores = score_example(model, tokenizer, ex["context"], ex["items"], args.temperature)
            loss = F.cross_entropy(scores.unsqueeze(0), torch.tensor([ex["answer_index"]]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        acc, brier = evaluate(model, tokenizer, val_examples, args.temperature)
        print(f"epoch {epoch}: mean_loss={total_loss / len(train_examples):.4f} val_accuracy={acc:.2%} val_brier={brier:.4f}")
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
