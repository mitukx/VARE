"""Propose integer expressions whose answers recompute exactly.

The integer band is 1000-9999, disjoint from the train, dev, and confirm
bands. A problem is kept only when the exact evaluator accepts it and the
answer is not one of the expression's tokens. Difficulty moves toward a
solve rate in [0.2, 0.8].
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from minigrpo.expr import evaluate_expression
from minigrpo.splits import INSTRUCTION, SplitTask, build_split

PROPOSER_MASTER = 20261021
LOW_SOLVE_RATE = 0.2
HIGH_SOLVE_RATE = 0.8
_BAND = (1_000, 9_999)


@dataclass(frozen=True, slots=True)
class Problem:
    seed: int
    difficulty: int
    expression: str
    prompt: str
    value: int

    def as_task(self) -> SplitTask:
        return SplitTask(
            split="proposed",
            seed=self.seed,
            difficulty=self.difficulty,
            expression=self.expression,
            prompt=self.prompt,
            value=self.value,
        )


def keep_value(expression: str) -> int | None:
    try:
        value = evaluate_expression(expression)
    except ValueError:
        return None
    tokens = expression.replace("(", " ").replace(")", " ").split()
    if str(value) in tokens:
        return None
    return value


def adjust_difficulty(difficulty: int, solve_rate: float) -> int:
    if difficulty not in {1, 2, 3}:
        raise ValueError("difficulty must be 1, 2, or 3")
    if not 0.0 <= solve_rate <= 1.0:
        raise ValueError("solve rate must lie in [0, 1]")
    if solve_rate < LOW_SOLVE_RATE:
        return max(1, difficulty - 1)
    if solve_rate > HIGH_SOLVE_RATE:
        return min(3, difficulty + 1)
    return difficulty


def _format(draws: list[int], difficulty: int) -> str:
    if difficulty == 1:
        return f"{draws[0]} + {draws[1]}"
    if difficulty == 2:
        return f"{draws[0]} + {draws[1]} - {draws[2]}"
    if difficulty == 3:
        return f"({draws[0]} + {draws[1]}) * {draws[2] % 9 + 2} - {draws[3]}"
    raise ValueError("difficulty must be 1, 2, or 3")


def blocked_expressions() -> set[str]:
    blocked: set[str] = set()
    for split in ("train", "dev", "confirm"):
        blocked.update(task.expression for task in build_split(split))
    return blocked


def propose(round_index: int, difficulty: int, count: int, blocked: set[str] | None = None) -> list[Problem]:
    if round_index < 1 or count < 1:
        raise ValueError("round and count must be positive")
    if difficulty not in {1, 2, 3}:
        raise ValueError("difficulty must be 1, 2, or 3")
    taken = set(blocked_expressions() if blocked is None else blocked)
    rng = random.Random(f"selfimprove:{PROPOSER_MASTER}:{round_index}:{difficulty}:{count}")
    kept: list[Problem] = []
    attempt = 0
    while len(kept) < count:
        attempt += 1
        if attempt > count * 80:
            raise RuntimeError("proposer could not fill a recomputable batch")
        draws = [rng.randint(_BAND[0], _BAND[1]) for _ in range(difficulty + 1)]
        expression = _format(draws, difficulty)
        value = keep_value(expression)
        if value is None or expression in taken:
            continue
        taken.add(expression)
        kept.append(
            Problem(
                seed=5_000_000 + round_index * 1_000 + len(kept),
                difficulty=difficulty,
                expression=expression,
                prompt=INSTRUCTION.format(expression=expression),
                value=value,
            )
        )
    return kept
