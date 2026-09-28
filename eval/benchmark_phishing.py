"""Run sys1 through the same phishing-detection benchmark used to
compare Jev and Laya: AreLit/PhishNChips (via Luni/laya-jev-benchmark),
2,000 emails, binary phish/legitimate.

    python eval/benchmark_phishing.py --model models/sys1-fast-encoder --engine fast --limit 200
    python eval/benchmark_phishing.py --model models/sys1-calibrated-out --engine causal --limit 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset  # noqa: E402

from sys1 import Noul  # noqa: E402
from sys1.engine import Engine  # noqa: E402
from sys1.fast_engine import FastEngine  # noqa: E402
from sys1.dual_engine import DualEngine  # noqa: E402

INSTRUCTIONS = "Is this email a phishing attempt?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=["causal", "fast", "dual"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = load_dataset("AreLit/PhishNChips", "emails", split="core")
    if args.limit:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.limit, len(ds))))

    if args.engine == "fast":
        engine = FastEngine(args.model)
    elif args.engine == "dual":
        engine = DualEngine(args.model)
    else:
        engine = Engine(model_name=args.model)
    noul = Noul()

    n = n_correct = 0
    brier_total = 0.0
    latency_total = 0.0
    tp = fp = tn = fn = 0

    for i, row in enumerate(ds):
        email = json.loads(row["email_content"])
        gold = row["phish_label"]  # 1 = phishing, 0 = legitimate

        start = time.perf_counter()
        result = noul.ask(INSTRUCTIONS, email, engine)
        latency_total += time.perf_counter() - start

        p_phish = result["probability_yes"]
        pred = 1 if p_phish >= 0.5 else 0
        n += 1
        n_correct += pred == gold
        brier_total += (p_phish - gold) ** 2
        tp += pred == 1 and gold == 1
        fp += pred == 1 and gold == 0
        tn += pred == 0 and gold == 0
        fn += pred == 0 and gold == 1

        if (i + 1) % 20 == 0:
            print(f"...{i + 1}/{len(ds)} done")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n=== {args.engine} engine ({args.model}) on PhishNChips ({n} emails) ===")
    print(f"accuracy={n_correct / n:.1%}  brier={brier_total / n:.4f}  avg_latency={latency_total / n * 1000:.1f}ms")
    print(f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    print(f"confusion: tp={tp} fp={fp} tn={tn} fn={fn}")


if __name__ == "__main__":
    main()
