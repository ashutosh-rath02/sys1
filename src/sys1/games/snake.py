"""Classic arcade Snake, built around one rule: the model is called only
for a genuine judgment, never for anything code can already determine.

Memory lives in the game state (game.focus_axis), not in the model:
- Safety (wall/self-collision) is always fully deterministic.
- Once we've committed to closing the gap on one axis (x or y), the
  correct move along that axis is fully deterministic too -- there's
  no judgment in "should I keep going the direction I already know
  is right." That commitment persists across ticks in game.focus_axis
  until that axis is actually closed, which is exactly the "memory" a
  model without cross-tick state can't provide on its own.
- The model is asked exactly two kinds of real questions, each genuinely
  requiring judgment: which axis to prioritize when *both* are open (no
  structural reason to prefer one), and what to do when the
  deterministically-preferred move turns out to be unsafe.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

DIRECTIONS = ["up", "down", "left", "right"]
_DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


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
    focus_axis: str | None = None  # "x" | "y" | None -- the persistent plan

    def __post_init__(self):
        self.reset()

    def reset(self) -> None:
        cx, cy = self.width // 2, self.height // 2
        self.body = [(cx - i, cy) for i in range(self.initial_length)]
        self.direction = "right"
        self.alive = True
        self.ticks = 0
        self.score = 0
        self.focus_axis = None
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
        self.focus_axis = None  # the old plan's target no longer exists

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
            self._spawn_food()  # also clears focus_axis: new target
        return True, ate


AXIS_CHOICE_INSTRUCTIONS = (
    "You control a snake in a grid game. The food is offset from you on "
    "both axes. Pick which axis to close first: 'x' (chase the food's "
    "left/right offset) or 'y' (chase its up/down offset). Whichever you "
    "pick, you'll keep closing that gap every turn until it's shut."
)
FALLBACK_INSTRUCTIONS = (
    "You control a snake in a grid game. Your planned move (following "
    "food_direction) isn't safe this turn. Pick whichever of the offered "
    "safe directions is the least costly detour."
)


def decide_move(game: "SnakeGame", choose) -> tuple[str, dict | None]:
    """choose(instructions, options, state) -> {"choice": ..., ...}

    Returns (chosen_direction, judgment_result_or_None). judgment_result
    is None on every tick that was fully deterministic -- which is most
    of them, by design: the model is invoked only for the two situations
    that are genuinely ambiguous (see module docstring).
    """
    safe_moves = [d for d in DIRECTIONS if game.is_safe(d)]
    if len(safe_moves) <= 1:
        return (safe_moves[0] if safe_moves else game.direction), None

    hx, hy = game.head()
    fx, fy = game.food
    dx, dy = fx - hx, fy - hy

    if game.focus_axis == "x" and dx == 0:
        game.focus_axis = None
    if game.focus_axis == "y" and dy == 0:
        game.focus_axis = None

    judgment = None
    if game.focus_axis is None:
        if dx != 0 and dy == 0:
            game.focus_axis = "x"
        elif dy != 0 and dx == 0:
            game.focus_axis = "y"
        elif dx != 0 and dy != 0:
            result = choose(AXIS_CHOICE_INSTRUCTIONS, ["x", "y"], game.state_dict())
            game.focus_axis = result["choice"]
            judgment = result
        # dx == 0 and dy == 0 would mean we're already on the food, which
        # step() would have resolved as "ate" -- shouldn't reach here.

    preferred = None
    if game.focus_axis == "x":
        preferred = "right" if dx > 0 else "left"
    elif game.focus_axis == "y":
        preferred = "down" if dy > 0 else "up"

    if preferred and preferred in safe_moves:
        return preferred, judgment

    # the deterministic plan's move isn't safe -- this, and only this,
    # is a real judgment call among whatever safe options remain.
    result = choose(FALLBACK_INSTRUCTIONS, safe_moves, game.state_dict())
    return result["choice"], result
