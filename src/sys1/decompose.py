"""Ask several narrow questions instead of one broad one, then let code
combine the answers.

This is the composition idea the whole project rests on, applied to a
single judgment: rather than asking "is this email phishing?" and hoping
a 23M-parameter model has an opinion worth trusting, ask whether the
sender domain matches the claimed identity, whether the message
manufactures urgency, whether the link text matches its target -- each a
narrower judgment the model has a real chance of getting right -- and fit
the combination on labelled data.

Reported precedent: the same benchmark where a single question scores
62.6% reaches 95.0% when split into five sub-questions with weights fit
on ~1000 labelled examples. The model doesn't change; the decomposition
does the work.

The combiner is plain logistic regression over the sub-answer
probabilities, implemented here rather than pulled from scikit-learn to
keep inference dependency-free -- it's a handful of lines of gradient
descent on a convex objective, and it keeps the whole path CPU-cheap.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SubQuestion:
    """One narrow judgment. `key` names it for the combiner's weights."""

    key: str
    instructions: str
    primitive: str = "noul"
    items: list[str] = field(default_factory=list)

    def features(self, probs: dict[str, float]) -> list[float]:
        """Sub-answer -> feature values. A noul contributes its single
        yes-probability; a choice or score contributes one feature per
        option so the combiner can weight them independently."""
        if self.primitive == "noul":
            return [probs.get("yes", 0.0)]
        return [probs[item] for item in self.items]

    def feature_names(self) -> list[str]:
        if self.primitive == "noul":
            return [self.key]
        return [f"{self.key}={item}" for item in self.items]


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)  # avoids overflow for large negative x
    return e / (1.0 + e)


@dataclass
class LogisticCombiner:
    weights: list[float]
    bias: float
    feature_names: list[str]

    def predict_proba(self, features: list[float]) -> float:
        z = self.bias + sum(w * f for w, f in zip(self.weights, features))
        return _sigmoid(z)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"weights": self.weights, "bias": self.bias, "feature_names": self.feature_names},
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticCombiner":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["weights"], d["bias"], d["feature_names"])


def fit_combiner(
    rows: list[tuple[list[float], int]],
    feature_names: list[str],
    epochs: int = 400,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> LogisticCombiner:
    """Logistic regression by full-batch gradient descent.

    L2 is on by default because the fitting sets here are small (~1000
    examples, a handful of features) and an unregularised fit will happily
    drive a weight to infinity on a feature that happens to separate the
    training half.
    """
    if not rows:
        raise ValueError("no rows to fit on")
    n_features = len(rows[0][0])
    weights = [0.0] * n_features
    bias = 0.0
    n = len(rows)

    for _ in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for features, label in rows:
            error = _sigmoid(bias + sum(w * f for w, f in zip(weights, features))) - label
            for i, f in enumerate(features):
                grad_w[i] += error * f
            grad_b += error
        for i in range(n_features):
            weights[i] -= lr * (grad_w[i] / n + l2 * weights[i])
        bias -= lr * (grad_b / n)

    return LogisticCombiner(weights, bias, feature_names)


class DecomposedJudgment:
    """A broad yes/no judgment answered via several narrow ones."""

    def __init__(self, sub_questions: list[SubQuestion], combiner: LogisticCombiner | None = None):
        self.sub_questions = sub_questions
        self.combiner = combiner

    def feature_names(self) -> list[str]:
        return [name for q in self.sub_questions for name in q.feature_names()]

    def features_for(self, state, engine) -> list[float]:
        from .primitives import Choice, Noul, Score

        values: list[float] = []
        for q in self.sub_questions:
            if q.primitive == "noul":
                probs = {"yes": Noul().ask(q.instructions, state, engine)["probability_yes"]}
            elif q.primitive == "score":
                probs = Score(q.items).ask(q.instructions, state, engine)["probabilities"]
            else:
                probs = Choice(q.items).ask(q.instructions, state, engine)["probabilities"]
            values.extend(q.features(probs))
        return values

    def probability_yes(self, state, engine) -> float:
        if self.combiner is None:
            raise RuntimeError("combiner not fitted -- run eval/fit_decomposition.py first")
        return self.combiner.predict_proba(self.features_for(state, engine))
