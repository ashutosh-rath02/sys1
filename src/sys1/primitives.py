"""The three judgment primitives: Choice, Noul, Score.

Each hands its items straight to Engine.judge, which builds the lettered
prompt (via promptutil, shared with training) and returns a probability
distribution over the original labels.
"""
from __future__ import annotations

from .engine import Engine, default_engine


class Choice:
    """Pick one of N options; returns the full probability distribution."""

    def __init__(self, options: list[str]):
        if len(options) < 2:
            raise ValueError("Choice needs at least two options")
        self.options = options

    def ask(self, instructions: str, state: dict | None = None, engine: Engine | None = None) -> dict:
        engine = engine or default_engine()
        judgment = engine.judge(instructions, "choice", self.options, state)
        return {"choice": judgment.top_label, "probabilities": judgment.probabilities}


class Noul:
    """Is a condition true; returns the probability of 'yes'."""

    def ask(self, instructions: str, state: dict | None = None, engine: Engine | None = None) -> dict:
        engine = engine or default_engine()
        judgment = engine.judge(instructions, "noul", ["Yes", "No"], state)
        return {"probability_yes": judgment.probabilities["Yes"]}


class Score:
    """A degree along an ordered set of levels, low to high."""

    def __init__(self, levels: list[str]):
        if len(levels) < 2:
            raise ValueError("Score needs at least two ordered levels")
        self.levels = levels

    def ask(self, instructions: str, state: dict | None = None, engine: Engine | None = None) -> dict:
        engine = engine or default_engine()
        judgment = engine.judge(instructions, "score", self.levels, state)
        expected_index = sum(
            i * judgment.probabilities[level] for i, level in enumerate(self.levels)
        )
        return {
            "level": judgment.top_label,
            "probabilities": judgment.probabilities,
            "expected_index": expected_index,
        }
