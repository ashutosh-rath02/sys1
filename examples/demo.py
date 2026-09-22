"""Run the v0 sys1 primitives against a small local model.

    python examples/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1 import Choice, Noul, Score  # noqa: E402


def main() -> None:
    print("Loading base model (first run downloads it)...")

    choice = Choice(["billing", "technical support", "sales", "spam"])
    result = choice.ask(
        "Classify what this support ticket is about.",
        state={"ticket": "My card was charged twice for the same order, please refund me."},
    )
    print("\nChoice — ticket routing")
    print(result)

    noul = Noul()
    result = noul.ask(
        "Does this message express frustration or anger?",
        state={"message": "This is the third time I've had to email you about this. Fix it."},
    )
    print("\nNoul — is the customer frustrated?")
    print(result)

    score = Score(["not urgent", "somewhat urgent", "urgent", "critical"])
    result = score.ask(
        "How urgent is this support ticket?",
        state={"ticket": "The production API has been returning 500s for 20 minutes."},
    )
    print("\nScore — ticket urgency")
    print(result)


if __name__ == "__main__":
    main()
