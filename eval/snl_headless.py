"""sys1 vs a random baseline at the "choose your die" Snakes & Ladders
variant -- does the model's die choice actually beat picking randomly?

    python eval/snl_headless.py --model models/sys1-calibrated-out --games 20
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1 import Choice  # noqa: E402
from sys1.engine import Engine  # noqa: E402
from sys1.games import SnakesAndLaddersGame  # noqa: E402


def random_choose(instructions, options, state):
    return {"choice": random.choice(options)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    engine = Engine(model_name=args.model)
    sys1_choice = Choice(["die_a", "die_b"])

    def sys1_choose(instructions, options, state):
        return sys1_choice.ask(instructions, state, engine)

    sys1_wins = 0
    for i in range(args.games):
        random.seed(args.seed + i)
        game = SnakesAndLaddersGame(players=2)
        # player 0 = sys1, player 1 = random baseline
        while game.winner is None and game.turn < args.max_turns:
            chooser = sys1_choose if game.current_player == 0 else random_choose
            game.play_turn(chooser)
        result = "sys1" if game.winner == 0 else "random" if game.winner == 1 else "no winner"
        print(f"game {i}: winner={result} turns={game.turn} positions={game.positions}")
        sys1_wins += game.winner == 0

    print(f"\nsys1 won {sys1_wins}/{args.games} games (random baseline expected: ~50%)")


if __name__ == "__main__":
    main()
