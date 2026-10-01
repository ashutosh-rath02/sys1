"""How far do code-computed signals get on phishing, with no model at all?

If mechanical URL/sender features alone score well, that reframes the
benchmark: it is largely a pattern-matching task, and the honest
architecture is to compute those in code and spend the model only on what
genuinely needs semantics. Fit on the train half, scored on the eval half
-- the same split everything else uses.

    python eval/phishing_signal_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets import load_dataset  # noqa: E402

from metrics import ScoreAccumulator  # noqa: E402
from sys1.decompose import fit_combiner  # noqa: E402
from sys1.phishing_signals import FEATURE_NAMES, extract  # noqa: E402

SPLIT_SEED = 0


def rows_for(ds) -> list[tuple[list[float], int]]:
    out = []
    for row in ds:
        email = json.loads(row["email_content"])
        feats = extract(email)
        out.append(([feats[name] for name in FEATURE_NAMES], int(row["phish_label"])))
    return out


def main() -> None:
    ds = load_dataset("AreLit/PhishNChips", "emails", split="core").shuffle(seed=SPLIT_SEED)
    half = len(ds) // 2
    train = rows_for(ds.select(range(half)))
    test = rows_for(ds.select(range(half, len(ds))))

    print(f"train={len(train)} test={len(test)}\n")

    # Single-signal rules first -- if one feature alone carries the task,
    # that is worth knowing before fitting anything.
    print("single-signal accuracy on the eval half:")
    for i, name in enumerate(FEATURE_NAMES):
        agree = sum(1 for f, y in test if int(f[i]) == y) / len(test)
        print(f"  {name:<22} {agree:>6.1%}  (positive rate {sum(f[i] for f, _ in test) / len(test):.1%})")

    combiner = fit_combiner(train, FEATURE_NAMES, epochs=800, lr=1.0, l2=1e-2)
    print("\nfitted weights:")
    for name, w in zip(combiner.feature_names, combiner.weights):
        print(f"  {name:<22} {w:>+7.2f}")
    print(f"  {'bias':<22} {combiner.bias:>+7.2f}")

    acc = ScoreAccumulator()
    tp = fn = 0
    for feats, label in test:
        p = combiner.predict_proba(feats)
        gold = "yes" if label == 1 else "no"
        acc.add({"yes": p, "no": 1 - p}, gold)
        if label == 1:
            tp += p >= 0.5
            fn += p < 0.5

    s = acc.summary()
    print(
        f"\ncode-only combiner on eval half: accuracy={s['accuracy']:.1%} "
        f"brier={s['brier']:.3f} ECE={s['ece']:.3f} recall={tp / (tp + fn):.1%}"
    )


if __name__ == "__main__":
    main()
