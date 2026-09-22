"""Classic arcade Snake: the game engine owns every rule (movement,
collision, food spawning); the model only answers one Choice judgment per
tick -- which direction to move -- from structured board state. Same
pattern as the jev-snake / snake-jev community demos: deterministic
engine, model makes the one decision that isn't deterministic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

DIRECTIONS = ["up", "down", "left", "right"]
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


@dataclass
class SnakeGame:
    # defaults match the laya-mlx snake demo (github.com/mizorewww/laya-mlx)
    # for a like-for-like comparison: 24x16 board, initial length 6.
    width: int = 24
    height: int = 16
    initial_length: int = 6
    body: list[tuple[int, int]] = field(default_factory=list)
    food: tuple[int, int] = (0, 0)
    direction: str = "right"
    alive: bool = True
    ticks: int = 0
    score: int = 0

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        cx, cy = self.width // 2, self.height // 2
        self.body = [(cx - i, cy) for i in range(self.initial_length)]
        self.direction = "right"
        self.alive = True
        self.ticks = 0
        self.score = 0
        self._spawn_food()

    def _spawn_food(self) -> None:
        occupied = set(self.body)
        free = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        self.food = random.choice(free) if free else self.body[0]

    def head(self) -> tuple[int, int]:
        return self.body[0]

    def is_safe(self, direction: str) -> bool:
        dx, dy = _DELTA[direction]
        hx, hy = self.head()
        nx, ny = hx + dx, hy + dy
        if not (0 <= nx < self.width and 0 <= ny < self.height):
            return False
        # the tail cell is vacated this tick unless we're about to eat, so
        # it's safe to move into even though it's currently "body"
        body_to_check = self.body[:-1] if (nx, ny) != self.food else self.body
        return (nx, ny) not in body_to_check

    def state_dict(self) -> dict:
        hx, hy = self.head()
        fx, fy = self.food
        return {
            "grid_size": [self.width, self.height],
            "head": [hx, hy],
            "food": [fx, fy],
            "food_direction": [
                "right" if fx > hx else "left" if fx < hx else None,
                "down" if fy > hy else "up" if fy < hy else None,
            ],
            "body_excluding_head": [list(seg) for seg in self.body[1:]],
            "current_direction": self.direction,
            "safe_moves": [d for d in DIRECTIONS if self.is_safe(d)],
            "score": self.score,
        }

    def step(self, direction: str) -> tuple[bool, bool]:
        """Returns (still_alive, ate_food)."""
        if not self.alive:
            return False, False
        if direction not in DIRECTIONS:
            direction = self.direction
        self.direction = direction
        dx, dy = _DELTA[direction]
        hx, hy = self.head()
        nx, ny = hx + dx, hy + dy
        self.ticks += 1

        if not (0 <= nx < self.width and 0 <= ny < self.height):
            self.alive = False
            return False, False

        ate = (nx, ny) == self.food
        new_body = [(nx, ny)] + self.body if ate else [(nx, ny)] + self.body[:-1]
        if (nx, ny) in new_body[1:]:
            self.alive = False
            return False, False

        self.body = new_body
        if ate:
            self.score += 1
            self._spawn_food()
        return True, ate


FOOD_DIRECTION_INSTRUCTIONS = (
    "You control a snake in a grid game. Every direction offered is already "
    "confirmed safe (no wall, no self-collision) -- that part is handled for "
    "you. Your only job: pick whichever safe direction moves you closer to "
    "the food, using food_direction relative to your head position."
)


def decide_move(game: "SnakeGame", choose) -> tuple[str, dict | None]:
    """The one judgment this game actually needs: given only the *safe*
    directions, which one heads toward the food? Safety itself is a rule
    the engine already knows and enforces here -- it's never something we
    ask a model to (re-)derive.

    choose(instructions, options, state) -> {"choice": ..., ...}
    Returns (chosen_direction, judgment_result_or_None). judgment_result is
    None when the move was forced (0 or 1 safe options) and no judgment was
    needed at all.
    """
    safe_moves = [d for d in DIRECTIONS if game.is_safe(d)]
    if len(safe_moves) <= 1:
        return (safe_moves[0] if safe_moves else game.direction), None

    state = game.state_dict()
    result = choose(FOOD_DIRECTION_INSTRUCTIONS, safe_moves, state)
    return result["choice"], result
