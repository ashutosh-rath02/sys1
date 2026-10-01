"""Temperature scaling: the cheapest real calibration fix there is.

A single scalar per primitive, fit on held-out data, divides the scores
before the softmax. It cannot change which answer wins (dividing every
score by the same positive number preserves their order), so accuracy is
untouched and only confidence moves -- which is exactly the knob we want
when a model is right about as often as before but far too sure of
itself.

Works on probabilities rather than raw scores, so it applies to any
engine without touching them: softmax(log(p) / T) recovers exactly the
temperature-scaled distribution, since softmax ignores additive constants
and log(p) recovers the original scores up to one.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

_EPS = 1e-12


def apply_temperature(probs: dict[str, float], temperature: float) -> dict[str, float]:
    if temperature == 1.0:
        return dict(probs)
    scaled = {k: math.log(max(p, _EPS)) / temperature for k, p in probs.items()}
    top = max(scaled.values())
    exps = {k: math.exp(v - top) for k, v in scaled.items()}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def _nll(records: list[tuple[dict[str, float], str]], temperature: float) -> float:
    """Negative log-likelihood of the true labels under this temperature."""
    total = 0.0
    for probs, answer in records:
        scaled = apply_temperature(probs, temperature)
        total -= math.log(max(scaled.get(answer, 0.0), _EPS))
    return total / len(records) if records else 0.0


def fit_temperature(
    records: list[tuple[dict[str, float], str]],
    lo: float = 0.05,
    hi: float = 10.0,
    steps: int = 60,
) -> float:
    """Find the temperature minimising NLL. A coarse sweep then a local
    refine -- the objective is 1-D and smooth, so this is plenty and
    avoids a scipy dependency."""
    if not records:
        return 1.0

    def sweep(low: float, high: float, n: int) -> float:
        best_t, best_loss = 1.0, float("inf")
        for i in range(n + 1):
            t = low + (high - low) * i / n
            if t <= 0:
                continue
            loss = _nll(records, t)
            if loss < best_loss:
                best_t, best_loss = t, loss
        return best_t

    coarse = sweep(lo, hi, steps)
    window = (hi - lo) / steps
    return sweep(max(coarse - window, 0.01), coarse + window, 20)


class Calibrator:
    """Per-primitive temperatures, fit once and reused at inference."""

    def __init__(self, temperatures: dict[str, float] | None = None):
        self.temperatures = temperatures or {}

    def apply(self, probs: dict[str, float], primitive: str) -> dict[str, float]:
        return apply_temperature(probs, self.temperatures.get(primitive, 1.0))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.temperatures, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
