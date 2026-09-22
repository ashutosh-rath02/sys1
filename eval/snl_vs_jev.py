"""sys1 vs the real Jev API, playing the "choose your die" Snakes &
Ladders variant head-to-head, alternating turns on the same board.

    python eval/snl_vs_jev.py --model models/sys1-calibrated-out

Costs one real API call per Jev turn that has a genuine die choice
(dice ties are auto-resolved without asking anyone) -- keep --max-turns
modest to control usage.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sys1 import Choice  # noqa: E402
from sys1.engine import Engine  # noqa: E402
from sys1.games import SnakesAndLaddersGame  # noqa: E402
from sys1.jev_client import JevChoiceClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="local sys1 model or adapter path")
    parser.add_argument("--max-turns", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    engine = Engine(model_name=args.model)
    sys1_choice = Choice(["die_a", "die_b"])
    jev = JevChoiceClient()

    def sys1_choose(instructions, options, state):
        return sys1_choice.ask(instructions, state, engine)

    def jev_choose(instructions, options, state):
        return jev.ask(instructions, options, state)

    game = SnakesAndLaddersGame(players=2)  # player 0 = sys1, player 1 = jev
    while game.winner is None and game.turn < args.max_turns:
        chooser = sys1_choose if game.current_player == 0 else jev_choose
        game.play_turn(chooser)

    print(f"positions: sys1={game.positions[0]} jev={game.positions[1]}")
    print(f"turns played: {game.turn}")
    if game.winner == 0:
        print("winner: sys1")
    elif game.winner == 1:
        print("winner: jev")
    else:
        print(f"no winner within {args.max_turns} turns")

    print("\nlast 10 moves:")
    for entry in game.log[-10:]:
        who = "sys1" if entry["player"] == 0 else "jev"
        print(f"  turn {entry['turn']:>3} {who:>4}: dice={entry['dice']} chose={entry['chose']} {entry['from']} -> {entry['to']}")


if __name__ == "__main__":
    main()
