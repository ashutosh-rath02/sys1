"""Evaluate accuracy and calibration (Brier score) of the current engine
against a held-out set of judgment-shaped examples (data/schema.md format).

    python eval/calibration.py data/seed_examples.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1 import Choice, Noul, Score  # noqa: E402
from sys1.engine import Engine  # noqa: E402
from sys1.fast_engine import FastEngine  # noqa: E402
from sys1.dual_engine import DualEngine  # noqa: E402


def evaluate(rows: list[dict], engine: Engine) -> None:
    n_correct = 0
    brier_total = 0.0

    for row in rows:
        primitive = row["primitive"]
        instructions = row["instructions"]
        state = row.get("state")
        answer = row["answer"]

        if primitive == "choice":
            result = Choice(row["options"]).ask(instructions, state, engine)
            predicted, probs = result["choice"], result["probabilities"]
        elif primitive == "score":
            result = Score(row["levels"]).ask(instructions, state, engine)
            predicted, probs = result["level"], result["probabilities"]
        elif primitive == "noul":
            result = Noul().ask(instructions, state, engine)
            p_yes = result["probability_yes"]
            predicted = "yes" if p_yes >= 0.5 else "no"
            probs = {"yes": p_yes, "no": 1 - p_yes}
        else:
            raise ValueError(f"unknown primitive: {primitive}")

        correct = predicted == answer
        n_correct += correct
        # Brier score for the multi-class case: sum over all classes of
        # (predicted_prob - actual)^2, actual=1 for the true class else 0.
        brier_total += sum(
            (p - (1.0 if label == answer else 0.0)) ** 2 for label, p in probs.items()
        )

        print(f"[{'OK ' if correct else 'ERR'}] predicted={predicted!r} answer={answer!r} probs={probs}")

    n = len(rows)
    print(f"\naccuracy: {n_correct}/{n} = {n_correct / n:.2%}")
    print(f"mean Brier score: {brier_total / n:.4f} (lower is better, 0 = perfect)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--model", default=None, help="override the base model name")
    parser.add_argument("--engine", choices=["causal", "fast", "dual"], default="causal")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.engine == "fast":
        engine = FastEngine(args.model)
    elif args.engine == "dual":
        engine = DualEngine(args.model)
    else:
        engine = Engine(model_name=args.model) if args.model else Engine()
    evaluate(rows, engine)


if __name__ == "__main__":
    main()
