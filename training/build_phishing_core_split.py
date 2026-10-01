"""Split PhishNChips `core` in half: train on one half, evaluate on the
other.

Why this exists: training phishing on the non-core splits produced a model
scoring 100.0% accuracy at 1.000 confidence on that training data and
58.0% with 17.6% recall on `core`. It separated
`real_phishing_validation` from `cross_domain_legitimate_v5` on some
superficial cue rather than learning phishing. Training on data drawn from
the same distribution we score against is the direct test of whether the
model can learn this task at all, or whether the earlier number was never
about model capacity.

The halves are a deterministic shuffle so train and eval can never drift
apart: eval/run_all.py takes the second half, this takes the first.

    python training/build_phishing_core_split.py

Writes data/phishing_core_train.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path("data/phishing_core_train.jsonl")
SPLIT_SEED = 0
INSTRUCTIONS = "Is this email a phishing attempt?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    ds = load_dataset("AreLit/PhishNChips", "emails", split="core").shuffle(seed=SPLIT_SEED)
    half = len(ds) // 2
    train = ds.select(range(half))  # eval/run_all.py uses range(half, len(ds))

    rows = []
    for row in train:
        rows.append(
            {
                "primitive": "noul",
                "instructions": INSTRUCTIONS,
                "state": json.loads(row["email_content"]),
                "answer": "yes" if row["phish_label"] == 1 else "no",
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    n_phish = sum(1 for r in rows if r["answer"] == "yes")
    print(f"wrote {len(rows)} examples to {args.out} ({n_phish} phishing, {len(rows) - n_phish} legitimate)")
    print(f"eval half is the remaining {len(ds) - half} examples, untouched")


if __name__ == "__main__":
    main()
