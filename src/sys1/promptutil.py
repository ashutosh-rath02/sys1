"""Judgment-prompt construction, shared by inference (engine.py) and
training data prep (training/prepare_calibrated_dataset.py) so the two
can never drift into different formats again.

(They did once: prepare_dataset.py used an "Options:" header for every
primitive, while primitives.py's Score used "Levels (low to high):" for
inference. Every Score training example was in the wrong format as a
result. This module is the fix — one place, one format, per primitive.)
"""
from __future__ import annotations

_LETTERS = [chr(ord("A") + i) for i in range(26)]
ANSWER_LINE = "Respond with only the single letter of your answer, nothing else.\nAnswer:"


def lettered_block(items: list[str]) -> tuple[str, list[str]]:
    letters = _LETTERS[: len(items)]
    block = "\n".join(f"{letter}) {item}" for letter, item in zip(letters, items))
    return block, letters


def header_for(primitive: str) -> str:
    return "Levels (low to high):" if primitive == "score" else "Options:"


def build_user_message(instructions: str, primitive: str, items: list[str], state: dict | None) -> tuple[str, list[str]]:
    block, letters = lettered_block(items)
    state_block = f"\nContext:\n{state}\n" if state else "\n"
    header = header_for(primitive)
    message = f"{instructions}\n{header}\n{block}{state_block}{ANSWER_LINE}"
    return message, letters


def row_to_prompt_inputs(row: dict) -> tuple[str, list[str], int]:
    """Given a raw judgment-schema row (data/schema.md), return
    (user_message, letters, answer_index)."""
    primitive = row["primitive"]
    instructions = row["instructions"]
    state = row.get("state")
    answer = row["answer"]

    if primitive == "noul":
        items = ["Yes", "No"]
        answer = "Yes" if answer == "yes" else "No"
    else:
        items = row["options"] if primitive == "choice" else row["levels"]

    message, letters = build_user_message(instructions, primitive, items, state)
    return message, letters, items.index(answer)
