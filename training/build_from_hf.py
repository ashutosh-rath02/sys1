"""Build a real (not synthetic) training set from public Hugging Face
datasets, converted into the data/schema.md judgment format:

- Choice (ticket category) <- Ataur77/ecommerce-customer-support
- Score  (ticket urgency)  <- Ataur77/ecommerce-customer-support
- Noul   (is frustrated)   <- dair-ai/emotion (anger vs. joy/love)

    python training/build_from_hf.py --per-primitive 150

Writes data/real_examples.jsonl. Combine with data/seed_examples.jsonl
before running training/prepare_dataset.py.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

CHOICE_OPTIONS = [
    "Account & Shipping",
    "Payments",
    "Order Tracking",
    "Technical Issues",
    "Promotions",
    "Product Issues",
    "Refunds",
    "Order Issues",
]
URGENCY_LEVELS = ["Low", "Medium", "High", "Immediate"]


def build_choice_and_score(n_each: int, seed: int) -> list[dict]:
    ds = load_dataset("Ataur77/ecommerce-customer-support", split="train")
    rows = list(ds)
    random.Random(seed).shuffle(rows)

    out = []
    n_choice = n_score = 0
    for row in rows:
        complaint, category, urgency = row["Complaint"], row["Category"], row["Urgency"]
        if n_choice < n_each and category in CHOICE_OPTIONS:
            out.append(
                {
                    "primitive": "choice",
                    "instructions": "Classify what this customer complaint is about.",
                    "state": {"complaint": complaint},
                    "options": CHOICE_OPTIONS,
                    "answer": category,
                }
            )
            n_choice += 1
        if n_score < n_each and urgency in URGENCY_LEVELS:
            out.append(
                {
                    "primitive": "score",
                    "instructions": "How urgent is this customer complaint?",
                    "state": {"complaint": complaint},
                    "levels": URGENCY_LEVELS,
                    "answer": urgency,
                }
            )
            n_score += 1
        if n_choice >= n_each and n_score >= n_each:
            break
    return out


def build_noul(n_each: int, seed: int) -> list[dict]:
    ds = load_dataset("dair-ai/emotion", split="train")
    label_names = ds.features["label"].names
    anger_id = label_names.index("anger")
    positive_ids = {label_names.index("joy"), label_names.index("love")}

    rows = list(ds)
    random.Random(seed).shuffle(rows)

    out = []
    n_yes = n_no = 0
    for row in rows:
        text, label = row["text"], row["label"]
        if label == anger_id and n_yes < n_each:
            out.append(
                {
                    "primitive": "noul",
                    "instructions": "Does this message express frustration or anger?",
                    "state": {"message": text},
                    "answer": "yes",
                }
            )
            n_yes += 1
        elif label in positive_ids and n_no < n_each:
            out.append(
                {
                    "primitive": "noul",
                    "instructions": "Does this message express frustration or anger?",
                    "state": {"message": text},
                    "answer": "no",
                }
            )
            n_no += 1
        if n_yes >= n_each and n_no >= n_each:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-primitive", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/real_examples.jsonl"))
    args = parser.parse_args()

    rows = []
    rows += build_choice_and_score(args.per_primitive, args.seed)
    rows += build_noul(args.per_primitive, args.seed)
    random.Random(args.seed).shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    by_primitive: dict[str, int] = {}
    for row in rows:
        by_primitive[row["primitive"]] = by_primitive.get(row["primitive"], 0) + 1
    print(f"wrote {len(rows)} examples to {args.out}: {by_primitive}")


if __name__ == "__main__":
    main()
