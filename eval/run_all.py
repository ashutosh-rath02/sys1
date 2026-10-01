"""One command, one scoreboard, across every benchmark we track.

    python eval/run_all.py --model models/sys1-fast-encoder-v2 --engine fast

Runs the same held-out data every time so numbers from different runs are
comparable. Reports accuracy, Brier, ECE and latency per benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402

from metrics import ScoreAccumulator  # noqa: E402
from sys1 import Choice, Noul, Score  # noqa: E402
from sys1.dual_engine import DualEngine  # noqa: E402
from sys1.engine import Engine  # noqa: E402
from sys1.calibration import Calibrator  # noqa: E402
from sys1.fast_engine import FastEngine  # noqa: E402

SUBDECISIONS = ["outcome", "action", "needs_review", "risk", "urgency"]
PHISHING_INSTRUCTIONS = "Is this email a phishing attempt?"

# Note on typed-decisions Brier: eval/benchmark_typed_decisions.py scores
# Brier against that benchmark's soft teacher distribution (its annotators
# disagree ~50% of the time, so the gold "answer" is a spread, not a
# point). Here we score against the hard label instead, like the other two
# benchmarks, so the three are comparable to each other and to ECE. The
# same model therefore reads a higher Brier here than there -- different
# metric, not a regression.


def load_engine(model: str, engine_type: str):
    if engine_type == "fast":
        return FastEngine(model)
    if engine_type == "dual":
        return DualEngine(model)
    return Engine(model_name=model)


def ask_primitive(
    primitive: str, items: list[str], instructions: str, state, engine, calibrator=None,
    domain: str | None = None,
) -> tuple[dict, str]:
    """Returns (probabilities, predicted_label) for any primitive type.

    A calibrator, when given, rescales confidence without changing which
    answer wins -- so the predicted label is read after scaling purely for
    consistency, never because scaling could flip it."""
    if primitive == "noul":
        result = Noul().ask(instructions, state, engine)
        p_yes = result["probability_yes"]
        probs = {"yes": p_yes, "no": 1 - p_yes}
    elif primitive == "score":
        probs = Score(items).ask(instructions, state, engine)["probabilities"]
    else:
        probs = Choice(items).ask(instructions, state, engine)["probabilities"]

    if calibrator is not None:
        probs = calibrator.apply(probs, primitive, domain)
    return probs, max(probs, key=probs.get)


def bench_own_domain(engine, path: Path, calibrator=None) -> ScoreAccumulator:
    acc = ScoreAccumulator()
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for row in rows:
        primitive = row["primitive"]
        items = row.get("options") or row.get("levels") or []
        start = time.perf_counter()
        probs, _ = ask_primitive(primitive, items, row["instructions"], row.get("state"), engine, calibrator,
                                 domain="own_domain")
        acc.add(probs, row["answer"], time.perf_counter() - start)
    return acc


def bench_typed_decisions(engine, limit: int, calibrator=None) -> ScoreAccumulator:
    acc = ScoreAccumulator()
    ds = load_dataset("LocalLLaMA/typed-decisions", "agent_trace_observability", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    for row in ds:
        state = json.loads(row["state"])
        questions = json.loads(row["questions"])
        for key in SUBDECISIONS:
            q = questions[key]
            qtype, criteria = q["type"], q["criteria"]
            gold = row[f"{key}__label"]

            start = time.perf_counter()
            if qtype == "noul":
                probs, _ = ask_primitive("noul", [], q["instructions"], state, engine, calibrator,
                                         domain="typed_decisions")
                probs = {"true": probs["yes"], "false": probs["no"]}
            elif qtype == "score":
                levels = list(criteria)
                raw, _ = ask_primitive("score", levels, q["instructions"], state, engine, calibrator,
                                       domain="typed_decisions")
                probs = {str(i): raw[level] for i, level in enumerate(levels)}
            else:
                probs, _ = ask_primitive("choice", list(criteria.keys()), q["instructions"], state, engine,
                                         calibrator, domain="typed_decisions")
            acc.add(probs, gold, time.perf_counter() - start)
    return acc


def bench_phishing(engine, limit: int, seed: int = 0, calibrator=None) -> ScoreAccumulator:
    acc = ScoreAccumulator()
    ds = load_dataset("AreLit/PhishNChips", "emails", split="core")
    if limit:
        ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))

    tp = fn = 0
    for row in ds:
        email = json.loads(row["email_content"])
        gold = "yes" if row["phish_label"] == 1 else "no"
        start = time.perf_counter()
        probs, predicted = ask_primitive("noul", [], PHISHING_INSTRUCTIONS, email, engine, calibrator,
                                         domain="phishing")
        acc.add(probs, gold, time.perf_counter() - start)
        if gold == "yes":
            tp += predicted == "yes"
            fn += predicted == "no"
    acc.recall = tp / (tp + fn) if (tp + fn) else 0.0  # type: ignore[attr-defined]
    return acc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=["causal", "fast", "dual"], required=True)
    parser.add_argument("--own-domain-file", type=Path, default=Path("data/prepared/val_raw.jsonl"))
    parser.add_argument("--typed-limit", type=int, default=100)
    parser.add_argument("--phishing-limit", type=int, default=200)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None,
                        help="calibration.json from eval/fit_calibration.py")
    args = parser.parse_args()

    engine = load_engine(args.model, args.engine)
    calibrator = Calibrator.load(args.calibration) if args.calibration else None
    if calibrator:
        print(f"calibration: {calibrator.temperatures}")

    results = {}
    print("running own-domain...", flush=True)
    results["own_domain"] = bench_own_domain(engine, args.own_domain_file, calibrator)
    print("running typed-decisions...", flush=True)
    results["typed_decisions"] = bench_typed_decisions(engine, args.typed_limit, calibrator)
    print("running phishing...", flush=True)
    results["phishing"] = bench_phishing(engine, args.phishing_limit, calibrator=calibrator)

    print(f"\n=== {args.engine} engine · {args.model} ===")
    print(f"{'benchmark':<18} {'n':>5} {'acc':>8} {'brier':>8} {'ECE':>8} {'ms':>8}")
    for name, acc in results.items():
        s = acc.summary()
        print(
            f"{name:<18} {s['n']:>5} {s['accuracy']:>7.1%} {s['brier']:>8.3f} "
            f"{s['ece']:>8.3f} {s['latency_ms']:>8.1f}"
        )
    recall = getattr(results["phishing"], "recall", None)
    if recall is not None:
        print(f"{'phishing recall':<18} {recall:>13.1%}")

    if args.json_out:
        payload = {name: acc.summary() for name, acc in results.items()}
        payload["phishing"]["recall"] = recall
        payload["_meta"] = {"model": args.model, "engine": args.engine}
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
