"""Snakes & Ladders with one real decision per turn: standard pure S&L is
100% luck (no judgment for a model to make), so this uses the well-known
"choose your die" house rule -- roll two dice, pick which one to play.
That's a real, bounded decision every turn: the board is still exactly
Snakes & Ladders, the game engine still owns every rule, the model just
picks die_a or die_b via the Choice primitive.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

BOARD_SIZE = 100

# A commonly used classic Snakes & Ladders layout.
LADDERS = {1: 38, 4: 14, 9: 31, 21: 42, 28: 84, 36: 44, 51: 67, 71: 91, 80: 100}
SNAKES = {16: 6, 47: 26, 49: 11, 56: 53, 62: 19, 64: 60, 87: 24, 93: 73, 95: 75, 98: 78}


def resolve_landing(square: int) -> tuple[int, str]:
    """Apply a ladder/snake at this square, if any. Returns (final_square, effect_description)."""
    if square in LADDERS:
        return LADDERS[square], f"ladder up to {LADDERS[square]}"
    if square in SNAKES:
        return SNAKES[square], f"snake down to {SNAKES[square]}"
    return square, "plain square"


def die_option(position: int, value: int) -> dict:
    """What playing this die value would do from the given position."""
    target = position + value
    if target > BOARD_SIZE:
        return {"value": value, "resulting_square": position, "landing_effect": "overshoots — no move"}
    final_square, effect = resolve_landing(target)
    return {"value": value, "resulting_square": final_square, "landing_effect": effect}


@dataclass
class SnakesAndLaddersGame:
    players: int = 2
    positions: list[int] = field(default_factory=list)
    current_player: int = 0
    turn: int = 0
    winner: int | None = None
    log: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.positions = [0] * self.players

    def roll_dice(self) -> tuple[int, int]:
        return random.randint(1, 6), random.randint(1, 6)

    def state_dict(self, dice: tuple[int, int]) -> dict:
        pos = self.positions[self.current_player]
        opponents = [p for i, p in enumerate(self.positions) if i != self.current_player]
        return {
            "board_size": BOARD_SIZE,
            "your_position": pos,
            "opponent_positions": opponents,
            "die_a": die_option(pos, dice[0]),
            "die_b": die_option(pos, dice[1]),
        }

    def apply_choice(self, dice: tuple[int, int], choice: str) -> None:
        pos = self.positions[self.current_player]
        value = dice[0] if choice == "die_a" else dice[1]
        target = pos + value
        final_square = pos if target > BOARD_SIZE else resolve_landing(target)[0]

        self.log.append(
            {
                "turn": self.turn,
                "player": self.current_player,
                "dice": dice,
                "chose": choice,
                "from": pos,
                "to": final_square,
            }
        )
        self.positions[self.current_player] = final_square
        if final_square == BOARD_SIZE:
            self.winner = self.current_player

        self.turn += 1
        self.current_player = (self.current_player + 1) % self.players

    def play_turn(self, choose) -> None:
        """choose(instructions, options, state) -> {'choice': 'die_a'|'die_b', ...}"""
        dice = self.roll_dice()
        if dice[0] == dice[1]:
            self.apply_choice(dice, "die_a")
            return
        state = self.state_dict(dice)
        instructions = (
            "You are playing Snakes & Ladders. You rolled two dice and must "
            "choose which one to play this turn. Pick whichever gets you "
            "closer to square 100, favors landing on a ladder, and avoids "
            "landing on a snake."
        )
        result = choose(instructions, ["die_a", "die_b"], state)
        self.apply_choice(dice, result["choice"])
