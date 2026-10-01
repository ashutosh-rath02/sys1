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


class DegenerateFitError(RuntimeError):
    """Raised when fitting data can't constrain a temperature at all."""


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
    best = sweep(max(coarse - window, 0.01), coarse + window, 20)

    # A fit that runs to either end of the range means the data couldn't
    # constrain it -- usually because the model is already perfect on this
    # slice (NLL ~ 0 pushes temperature toward 0 to sharpen further) or
    # hopeless on it. Either way the number says more about the fitting
    # data than about calibration, and applying it elsewhere does damage:
    # we hit exactly this fitting phishing on train-side data the model
    # scores 100% on, while it manages 58% on the eval split.
    if best <= lo * 1.5 or best >= hi * 0.95:
        raise DegenerateFitError(
            f"temperature fit hit the search boundary (T={best:.3f}); the fitting "
            f"data does not constrain it -- check whether the model is already "
            f"saturated on this slice"
        )
    return best


class Calibrator:
    """Temperatures fit once and reused at inference.

    Scoped per "<domain>:<primitive>" where a domain-specific fit exists,
    falling back to a global per-primitive temperature. Domain scoping
    matters more than it sounds: a model can be close to calibrated
    in-distribution and badly overconfident on a shifted domain, and one
    global temperature fit across both just splits the difference and
    fixes neither.
    """

    def __init__(self, temperatures: dict[str, float] | None = None):
        self.temperatures = temperatures or {}

    def apply(self, probs: dict[str, float], primitive: str, domain: str | None = None) -> dict[str, float]:
        temperature = None
        if domain:
            temperature = self.temperatures.get(f"{domain}:{primitive}")
        if temperature is None:
            temperature = self.temperatures.get(primitive, 1.0)
        return apply_temperature(probs, temperature)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.temperatures, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Calibrator":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
