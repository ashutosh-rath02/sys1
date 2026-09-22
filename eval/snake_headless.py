"""Headless test: how many ticks does the model survive at Snake?

Safety (wall/self-collision avoidance) is handled entirely by the game
engine -- the model is only ever asked to pick a direction from the
already-safe options, purely on food direction. See
src/sys1/games/snake.py:decide_move.

    python eval/snake_headless.py --model models/sys1-calibrated-out
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1 import Choice  # noqa: E402
from sys1.engine import Engine  # noqa: E402
from sys1.games import SnakeGame, decide_move  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-ticks", type=int, default=150)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--width", type=int, default=24, help="matches laya-mlx demo default")
    parser.add_argument("--height", type=int, default=16, help="matches laya-mlx demo default")
    parser.add_argument("--initial-length", type=int, default=6, help="matches laya-mlx demo default")
    args = parser.parse_args()

    engine = Engine(model_name=args.model)

    def choose(instructions, options, state):
        return Choice(options).ask(instructions, state, engine)

    for g in range(args.games):
        game = SnakeGame(width=args.width, height=args.height, initial_length=args.initial_length)
        n_judgments = 0
        for _ in range(args.max_ticks):
            direction, judgment = decide_move(game, choose)
            n_judgments += judgment is not None
            alive, _ = game.step(direction)
            if not alive:
                break
        print(f"game {g}: ticks={game.ticks} score={game.score} alive={game.alive} judgments_asked={n_judgments}")


if __name__ == "__main__":
    main()
