"""Pull TRAIN-only data from the same public sources used to benchmark
Jev/Laya (Luni/laya-jev-benchmark), converted into our data/schema.md
judgment format, so sys1 can actually be trained on these domains
instead of only being tested on them cold.

Deliberately avoids the benchmark's held-out eval data:
- typed-decisions: uses the dataset's own "train" split (the benchmark
  evaluates on "test").
- phishing: uses the non-"core" splits (cross_domain_legitimate_v5,
  infrastructure_phishing_expanded, real_phishing_validation) -- "core"
  is the 2,000-email split eval/benchmark_phishing.py scores against.

    python training/build_benchmark_domain_data.py

Writes data/benchmark_domain_examples.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path("data/benchmark_domain_examples.jsonl")


def build_typed_decisions_examples() -> list[dict]:
    # All four workflows, not just agent-trace. Training on one and
    # evaluating across four made our score look 16 points better than it
    # was (62.6% in-distribution vs 46.6% across the set), and Laya's
    # published number comes from a checkpoint fitted to all four -- so
    # one workflow was never the comparable thing to train on either.
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    rows = []
    for row in ds:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        gold_all = json.loads(row["gold"])
        for key, q in questions.items():
            if key not in gold_all:
                continue
            # criteria is absent on noul questions in some workflows
            instructions, criteria, qtype = q["instructions"], q.get("criteria"), q["type"]
            label = gold_all[key]["label"]
            if qtype == "choice":
                rows.append(
                    {
                        "primitive": "choice",
                        "instructions": instructions,
                        "state": state,
                        "options": list(criteria.keys()),
                        "answer": label,
                    }
                )
            elif qtype == "noul":
                rows.append(
                    {
                        "primitive": "noul",
                        "instructions": instructions,
                        "state": state,
                        "answer": "yes" if label == "true" else "no",
                    }
                )
            elif qtype == "score":
                levels = list(criteria)
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


def build_phishing_examples() -> list[dict]:
    rows = []
    for split in ("cross_domain_legitimate_v5", "infrastructure_phishing_expanded", "real_phishing_validation"):
        ds = load_dataset("AreLit/PhishNChips", "emails", split=split)
        for row in ds:
            email = json.loads(row["email_content"])
            rows.append(
                {
                    "primitive": "noul",
                    "instructions": "Is this email a phishing attempt?",
                    "state": email,
                    "answer": "yes" if row["phish_label"] == 1 else "no",
                }
            )
    return rows


def main() -> None:
    rows = build_typed_decisions_examples() + build_phishing_examples()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    by_primitive: dict[str, int] = {}
    for row in rows:
        by_primitive[row["primitive"]] = by_primitive.get(row["primitive"], 0) + 1
    print(f"wrote {len(rows)} examples to {OUT_PATH}: {by_primitive}")


if __name__ == "__main__":
    main()
