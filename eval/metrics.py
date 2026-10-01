"""Shared scoring for every benchmark, so numbers from different runs are
actually comparable.

Three metrics, because they answer different questions:
- accuracy: did the argmax match the label
- Brier: squared error of the whole predicted distribution
- ECE: of the times the model said "80% sure", was it right 80% of the
  time? This is the one the funded models publish (Laya reports 0.466
  out-of-box, 0.081 after temperature fitting), so it's the one we need
  to compare against them at all.
"""
from __future__ import annotations

import math


class ScoreAccumulator:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.n = 0
        self.n_correct = 0
        self.brier_total = 0.0
        self._nll_total = 0.0
        self.latency_total = 0.0
        # (confidence, was_correct) per prediction, for ECE
        self.records: list[tuple[float, bool]] = []

    def add(self, probs: dict[str, float], answer: str, latency_s: float = 0.0) -> bool:
        predicted = max(probs, key=probs.get)
        correct = predicted == answer

        self.n += 1
        self.n_correct += correct
        self.latency_total += latency_s
        self.brier_total += sum(
            (p - (1.0 if label == answer else 0.0)) ** 2 for label, p in probs.items()
        )
        self._nll_total -= math.log(max(probs.get(answer, 0.0), 1e-12))
        self.records.append((probs[predicted], correct))
        return correct

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def brier(self) -> float:
        return self.brier_total / self.n if self.n else 0.0

    @property
    def ece(self) -> float:
        """Expected Calibration Error: average gap between stated
        confidence and observed accuracy, weighted by bin population."""
        if not self.records:
            return 0.0
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(self.n_bins)]
        for confidence, correct in self.records:
            idx = min(int(confidence * self.n_bins), self.n_bins - 1)
            bins[idx].append((confidence, correct))

        total_gap = 0.0
        for bucket in bins:
            if not bucket:
                continue
            avg_confidence = sum(c for c, _ in bucket) / len(bucket)
            avg_accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
            total_gap += (len(bucket) / len(self.records)) * abs(avg_confidence - avg_accuracy)
        return total_gap

    @property
    def avg_latency_ms(self) -> float:
        return (self.latency_total / self.n * 1000) if self.n else 0.0

    def accuracy_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """Normal-approximation interval on accuracy.

        Here because we spent a while reading differences as results that
        were inside the noise: at n=500 the half-width is about 4 points,
        which is wider than most changes we were celebrating. Every
        comparison should carry one of these.
        """
        if self.n == 0:
            return (0.0, 0.0)
        z = 1.96 if confidence == 0.95 else 2.576
        p = self.accuracy
        half = z * math.sqrt(max(p * (1 - p), 1e-12) / self.n)
        return (max(0.0, p - half), min(1.0, p + half))

    @property
    def nll(self) -> float:
        """Mean negative log-likelihood of the true label. Worth reporting
        because it's what temperature is fit on -- optimising one number
        and reporting another invites confusion."""
        if not self.records:
            return 0.0
        return self._nll_total / self.n

    def summary(self) -> dict:
        lo, hi = self.accuracy_ci()
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "accuracy_ci": [lo, hi],
            "brier": self.brier,
            "ece": self.ece,
            "nll": self.nll,
            "latency_ms": self.avg_latency_ms,
        }
