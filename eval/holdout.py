"""Reproduce training/prepare_dataset.py's exact train/val split, but keep
the raw judgment-schema rows (not the chat-converted ones) so eval/calibration.py
can be run on genuinely held-out data.

    python eval/holdout.py data/seed_examples.jsonl data/real_examples.jsonl
"""
import argparse
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("input_files", nargs="+", type=Path)
parser.add_argument("--out", type=Path, default=Path("data/prepared/val_raw.jsonl"))
parser.add_argument("--val-fraction", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

rows = []
for path in args.input_files:
    with path.open(encoding="utf-8") as f:
        rows += [json.loads(line) for line in f if line.strip()]

random.Random(args.seed).shuffle(rows)
n_val = max(1, int(len(rows) * args.val_fraction))
val_rows = rows[:n_val]

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", encoding="utf-8") as f:
    for row in val_rows:
        f.write(json.dumps(row) + "\n")
print(f"wrote {len(val_rows)} held-out raw examples to {args.out}")
