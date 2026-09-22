"""The three judgment primitives: Choice, Noul, Score.

Each wraps a set of possible answers into lettered options, asks the engine
for a next-token distribution over those letters, and maps the result back
onto the caller's own labels.
"""
from __future__ import annotations

from .engine import Engine, default_engine

_LETTERS = [chr(ord("A") + i) for i in range(26)]


def _lettered_block(items: list[str]) -> tuple[str, list[str]]:
    letters = _LETTERS[: len(items)]
    block = "\n".join(f"{letter}) {item}" for letter, item in zip(letters, items))
    return block, letters


class Choice:
    """Pick one of N options; returns the full probability distribution."""

    def __init__(self, options: list[str]):
        if len(options) < 2:
            raise ValueError("Choice needs at least two options")
        self.options = options

    def ask(self, instructions: str, state: dict | None = None, engine: Engine | None = None) -> dict:
        engine = engine or default_engine()
        block, letters = _lettered_block(self.options)
        full_instructions = f"{instructions}\nOptions:\n{block}"
        judgment = engine.judge(full_instructions, letters, state)
        letter_to_option = dict(zip(letters, self.options))
        probabilities = {letter_to_option[l]: p for l, p in judgment.probabilities.items()}
        return {
            "choice": letter_to_option[judgment.top_label],
            "probabilities": probabilities,
        }


class Noul:
    """Is a condition true; returns the probability of 'yes'."""

    def ask(self, instructions: str, state: dict | None = None, engine: Engine | None = None) -> dict:
        engine = engine or default_engine()
        full_instructions = f"{instructions}\nOptions:\nA) Yes\nB) No"
        judgment = engine.judge(full_instructions, ["A", "B"], state)
        return {"probability_yes": judgment.probabilities["A"]}


class Score:
    """A degree along an ordered set of levels, low to high."""

    def __init__(self, levels: list[str]):
        if len(levels) < 2:
            raise ValueError("Score needs at least two ordered levels")
        self.levels = levels

    def ask(self, instructions: str, state: dict | None = None, engine: Engine | None = None) -> dict:
        engine = engine or default_engine()
        block, letters = _lettered_block(self.levels)
        full_instructions = f"{instructions}\nLevels (low to high):\n{block}"
        judgment = engine.judge(full_instructions, letters, state)
        letter_to_level = dict(zip(letters, self.levels))
        probabilities = {letter_to_level[l]: p for l, p in judgment.probabilities.items()}
        expected_index = sum(
            i * judgment.probabilities[letter] for i, letter in enumerate(letters)
        )
        return {
            "level": letter_to_level[judgment.top_label],
            "probabilities": probabilities,
            "expected_index": expected_index,
        }
