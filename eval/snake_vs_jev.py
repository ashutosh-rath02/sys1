"""Head-to-head: our sys1 engine vs. the real Jev API, playing the same
Snake game (identical starting seed) with identical instructions and state.

    python eval/snake_vs_jev.py --model models/sys1-calibrated-out

Costs real API calls against TYPESAFE_API_KEY -- keep --max-ticks modest.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1 import Choice  # noqa: E402
from sys1.engine import Engine  # noqa: E402
from sys1.games import SnakeGame  # noqa: E402
from sys1.jev_client import JevChoiceClient  # noqa: E402

INSTRUCTIONS = (
    "You control a snake in a grid game. Pick the move direction that "
    "keeps you alive (don't hit a wall or your own body) and, when safe, "
    "moves you closer to the food."
)
OPTIONS = ["up", "down", "left", "right"]


def play(chooser, max_ticks: int, seed: int) -> SnakeGame:
    random.seed(seed)
    game = SnakeGame(width=10, height=10)
    for _ in range(max_ticks):
        result = chooser(INSTRUCTIONS, OPTIONS, game.state_dict())
        alive, _ = game.step(result["choice"])
        if not alive:
            break
    return game


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local sys1 model or adapter path")
    parser.add_argument("--max-ticks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    engine = Engine(model_name=args.model)
    sys1_choice = Choice(OPTIONS)
    sys1_game = play(
        lambda instr, opts, state: sys1_choice.ask(instr, state, engine),
        args.max_ticks,
        args.seed,
    )
    print(f"sys1 ({args.model}): ticks={sys1_game.ticks} score={sys1_game.score} alive={sys1_game.alive}")

    jev = JevChoiceClient()
    jev_game = play(
        lambda instr, opts, state: jev.ask(instr, opts, state),
        args.max_ticks,
        args.seed,
    )
    print(f"jev (jev-latest):            ticks={jev_game.ticks} score={jev_game.score} alive={jev_game.alive}")


if __name__ == "__main__":
    main()
