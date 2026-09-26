"""Run sys1 through the same public benchmark used to compare Jev and
Laya: LocalLLaMA/typed-decisions (agent-trace-observability workflow),
via Luni/laya-jev-benchmark. Five typed sub-decisions per case (choice,
choice, noul, score, score), each with a gold *distribution* from a
teacher ensemble, not just a hard label -- scored the same way the
published Jev-vs-Laya comparisons were: argmax accuracy AND Brier/soft
accuracy against that teacher distribution.

    python eval/benchmark_typed_decisions.py --model models/sys1-fast-encoder --engine fast
    python eval/benchmark_typed_decisions.py --model models/sys1-calibrated-out --engine causal --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset  # noqa: E402

from sys1 import Choice, Noul, Score  # noqa: E402
from sys1.engine import Engine  # noqa: E402
from sys1.fast_engine import FastEngine  # noqa: E402

SUBDECISIONS = ["outcome", "action", "needs_review", "risk", "urgency"]


def run_one(row: dict, questions: dict, engine) -> dict:
    state = json.loads(row["state"])
    results = {}

    for key in SUBDECISIONS:
        q = questions[key]
        instructions, criteria, qtype = q["instructions"], q["criteria"], q["type"]
        gold_probs = json.loads(row[f"{key}__probabilities"])
        gold_label = row[f"{key}__label"]

        start = time.perf_counter()
        if qtype in ("choice",):
            options = list(criteria.keys())
            r = Choice(options).ask(instructions, state, engine)
            pred_label, pred_probs = r["choice"], r["probabilities"]
        elif qtype == "noul":
            r = Noul().ask(instructions, state, engine)
            pred_probs = {"true": r["probability_yes"], "false": 1 - r["probability_yes"]}
            pred_label = "true" if r["probability_yes"] >= 0.5 else "false"
        elif qtype == "score":
            levels = list(criteria)  # already ordered low->high, no keys
            r = Score(levels).ask(instructions, state, engine)
            index_to_level = {str(i): levels[i] for i in range(len(levels))}
            pred_probs = {idx: r["probabilities"][level] for idx, level in index_to_level.items()}
            pred_label = next(idx for idx, level in index_to_level.items() if level == r["level"])
        else:
            raise ValueError(f"unknown question type: {qtype}")
        elapsed = time.perf_counter() - start

        correct = pred_label == gold_label
        # Brier against the teacher's full soft distribution, not a one-hot label.
        all_keys = set(gold_probs) | set(pred_probs)
        brier = sum((pred_probs.get(k, 0.0) - gold_probs.get(k, 0.0)) ** 2 for k in all_keys)

        results[key] = {"correct": correct, "brier": brier, "latency_s": elapsed}

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=["causal", "fast"], required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ds = load_dataset("LocalLLaMA/typed-decisions", "agent_trace_observability", split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    engine = FastEngine(args.model) if args.engine == "fast" else Engine(model_name=args.model)

    totals = {key: {"n": 0, "correct": 0, "brier": 0.0, "latency": 0.0} for key in SUBDECISIONS}

    for i, row in enumerate(ds):
        questions = json.loads(row["questions"])
        results = run_one(row, questions, engine)
        for key, r in results.items():
            t = totals[key]
            t["n"] += 1
            t["correct"] += r["correct"]
            t["brier"] += r["brier"]
            t["latency"] += r["latency_s"]
        if (i + 1) % 10 == 0:
            print(f"...{i + 1}/{len(ds)} cases done")

    print(f"\n=== {args.engine} engine ({args.model}) on typed-decisions ({len(ds)} cases) ===")
    grand_n = grand_correct = 0
    grand_brier = 0.0
    for key in SUBDECISIONS:
        t = totals[key]
        acc = t["correct"] / t["n"]
        brier = t["brier"] / t["n"]
        lat_ms = (t["latency"] / t["n"]) * 1000
        print(f"{key:>14}: accuracy={acc:.1%}  brier={brier:.4f}  avg_latency={lat_ms:.1f}ms")
        grand_n += t["n"]
        grand_correct += t["correct"]
        grand_brier += t["brier"]
    print(f"{'OVERALL':>14}: accuracy={grand_correct / grand_n:.1%}  brier={grand_brier / grand_n:.4f}")


if __name__ == "__main__":
    main()
