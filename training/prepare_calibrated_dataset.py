"""Build training examples for training/train_calibrated.ipynb: unlike
prepare_dataset.py (plain next-token SFT over the whole vocabulary), this
keeps the exact candidate-letter set and the correct answer's index, so
training can apply a loss directly on the same softmax-over-candidates
distribution src/sys1/engine.py reads at inference time.

    python training/prepare_calibrated_dataset.py \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        data/seed_examples.jsonl data/real_examples.jsonl

Only needs the tokenizer (for the chat template), not the model weights.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1.engine import DEFAULT_MODEL  # noqa: E402
from sys1.promptutil import row_to_prompt_inputs  # noqa: E402


def build_example(row: dict, tokenizer) -> dict:
    user_message, letters, answer_index = row_to_prompt_inputs(row)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}], tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "letters": letters, "answer_index": answer_index}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+", type=Path)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("data/prepared_calibrated"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    rows = []
    for path in args.input_files:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(build_example(json.loads(line), tokenizer))

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
