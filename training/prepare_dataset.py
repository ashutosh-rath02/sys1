"""Turn data/*.jsonl (see data/schema.md) into chat-formatted train/val
splits ready for LoRA fine-tuning in training/train_lora.ipynb.

    python training/prepare_dataset.py data/seed_examples.jsonl

Each output row is a single chat exchange: the same prompt shape
src/sys1/engine.py builds at inference time, with the assistant's reply
being just the correct letter — so the fine-tuned model learns to answer
in exactly the format the logit-scoring engine reads from.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

_LETTERS = [chr(ord("A") + i) for i in range(26)]


def _lettered_block(items: list[str]) -> tuple[str, list[str]]:
    letters = _LETTERS[: len(items)]
    block = "\n".join(f"{letter}) {item}" for letter, item in zip(letters, items))
    return block, letters


def build_example(row: dict) -> dict:
    primitive = row["primitive"]
    instructions = row["instructions"]
    state = row.get("state")
    answer = row["answer"]

    if primitive == "noul":
        options, letters = ["Yes", "No"], ["A", "B"]
        answer_letter = "A" if answer == "yes" else "B"
    else:
        key = "options" if primitive == "choice" else "levels"
        options = row[key]
        block, letters = _lettered_block(options)
        answer_letter = letters[options.index(answer)]

    block, letters = _lettered_block(options)
    state_block = f"\nContext:\n{state}\n" if state else "\n"
    user_msg = (
        f"{instructions}\nOptions:\n{block}{state_block}"
        "Respond with only the single letter of your answer, nothing else.\nAnswer:"
    )
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer_letter},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("data/prepared"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for path in args.input_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(build_example(json.loads(line)))

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train_rows), ("val", val_rows)):
        out_path = args.out_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in split:
                f.write(json.dumps(row) + "\n")
        print(f"wrote {len(split)} examples to {out_path}")


if __name__ == "__main__":
    main()
