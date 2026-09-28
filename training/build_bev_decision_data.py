"""Pull TRAIN-only data from avbiswas/bev-decision -- 125k+ typed-decision
examples across 8 genuinely diverse domains (software engineering,
browser interaction, spatial/logical reasoning, sentiment, scientific
paper understanding, etc). Converts to our data/schema.md format.

Uses the dataset's own "train" split; its "test" split is left untouched
for held-out evaluation, same policy as build_benchmark_domain_data.py.

    python training/build_bev_decision_data.py --per-domain-limit 500

Writes data/bev_decision_examples.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path("data/bev_decision_examples.jsonl")


def row_questions_to_examples(state: str, questions: dict) -> list[dict]:
    rows = []
    for q in questions.values():
        instructions, qtype, label = q["instructions"], q["type"], q["label"]
        if qtype == "choice":
            rows.append(
                {
                    "primitive": "choice",
                    "instructions": instructions,
                    "state": state,
                    "options": list(q["criteria"].keys()),
                    "answer": label,
                }
            )
        elif qtype == "noul":
            rows.append(
                {
                    "primitive": "noul",
                    "instructions": instructions,
                    "state": state,
                    "answer": "yes" if label else "no",
                }
            )
        elif qtype == "score":
            levels = list(q["criteria"])
            rows.append(
                {
                    "primitive": "score",
                    "instructions": instructions,
                    "state": state,
                    "levels": levels,
                    "answer": levels[int(label)],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-domain-limit", type=int, default=500, help="max source rows per domain")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = load_dataset("avbiswas/bev-decision", split="train")
    ds = ds.shuffle(seed=args.seed)

    per_domain_count: dict[str, int] = defaultdict(int)
    rows = []
    for row in ds:
        domain = row["domain"]
        if per_domain_count[domain] >= args.per_domain_limit:
            continue
        per_domain_count[domain] += 1
        questions = json.loads(row["questions_json"])
        rows.extend(row_questions_to_examples(row["state"], questions))
        if all(c >= args.per_domain_limit for c in per_domain_count.values()) and len(per_domain_count) == 8:
            break

    random.Random(args.seed).shuffle(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    by_primitive: dict[str, int] = defaultdict(int)
    for row in rows:
        by_primitive[row["primitive"]] += 1
    print(f"source rows per domain: {dict(per_domain_count)}")
    print(f"wrote {len(rows)} examples to {OUT_PATH}: {dict(by_primitive)}")


if __name__ == "__main__":
    main()
