"""Fit one temperature per primitive, on train-side data only.

    python eval/fit_calibration.py --model models/sys1-fast-encoder-v2 --engine fast \
        --out models/sys1-fast-encoder-v2/calibration.json

Defaults to data/benchmark_domain_examples.jsonl and
data/bev_decision_examples.jsonl because both are built exclusively from
their datasets' TRAIN splits -- nothing in either file appears in any
benchmark we score against, so fitting here and reporting there is clean.
Passing the own-domain eval file would be leakage and is not the default
for that reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_all import ask_primitive, load_engine  # noqa: E402

from sys1.calibration import Calibrator, _nll, fit_temperature  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", choices=["causal", "fast", "dual"], required=True)
    parser.add_argument(
        "--input-files",
        nargs="+",
        type=Path,
        default=[
            Path("data/benchmark_domain_examples.jsonl"),
            Path("data/bev_decision_examples.jsonl"),
        ],
    )
    parser.add_argument("--per-primitive-limit", type=int, default=400)
    parser.add_argument("--domain-key", default=None,
                        help="save as \"<domain>:<primitive>\" instead of a global per-primitive fit")
    parser.add_argument("--filter-instructions", default=None,
                        help="only fit on rows whose instructions match this exactly")
    parser.add_argument("--merge-into", type=Path, default=None,
                        help="load this calibration file and add to it rather than starting fresh")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    engine = load_engine(args.model, args.engine)

    by_primitive: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    for path in args.input_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if args.filter_instructions and row["instructions"] != args.filter_instructions:
                continue
            primitive = row["primitive"]
            if len(by_primitive[primitive]) >= args.per_primitive_limit:
                continue
            items = row.get("options") or row.get("levels") or []
            probs, _ = ask_primitive(primitive, items, row["instructions"], row.get("state"), engine)
            answer = row["answer"]
            if primitive == "noul":
                answer = "yes" if answer in ("yes", True, "true") else "no"
            by_primitive[primitive].append((probs, answer))
        if all(len(v) >= args.per_primitive_limit for v in by_primitive.values()) and len(by_primitive) >= 3:
            break

    temperatures = {}
    if args.merge_into and args.merge_into.is_file():
        temperatures = Calibrator.load(args.merge_into).temperatures
        print(f"merging into existing: {temperatures}\n")
    for primitive, records in by_primitive.items():
        before = _nll(records, 1.0)
        temperature = fit_temperature(records)
        after = _nll(records, temperature)
        key = f"{args.domain_key}:{primitive}" if args.domain_key else primitive
        temperatures[key] = temperature
        print(
            f"{key:>22}: n={len(records):<5} T={temperature:.3f}  "
            f"NLL {before:.4f} -> {after:.4f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Calibrator(temperatures).save(args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
