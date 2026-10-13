"""Train and dev integer expressions from the existing seed split.

Master seed 20261014, difficulty 2, and the train/dev/confirm bands match the
integer seed-split generator. Confirm tasks are built only so a caller can
prove they were not used. Training and dev evaluation do not score them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from minigrpo.expr import evaluate_expression

MASTER_SEED = 20261014
DIFFICULTY = 2
TRAIN_COUNT = 128
DEV_COUNT = 32
CONFIRM_COUNT = 64
INSTRUCTION = (
    "Compute the integer value of this expression: {expression}\n"
    "Write the result on the final line in exactly this form: #### <integer>\n"
    "The final line must contain no other text."
)
_BANDS = {
    "train": (10_000, 29_999),
    "dev": (40_000, 59_999),
    "confirm": (70_000, 89_999),
}
_SEED_RANGES = {
    "train": (1_000_000, 1_999_999),
    "dev": (2_000_000, 2_999_999),
    "confirm": (3_000_000, 3_999_999),
}
_COUNTS = {"train": TRAIN_COUNT, "dev": DEV_COUNT, "confirm": CONFIRM_COUNT}


@dataclass(frozen=True, slots=True)
class SplitTask:
    split: str
    seed: int
    difficulty: int
    expression: str
    prompt: str
    value: int


def derive_seeds(split: str, *, master: int = MASTER_SEED, count: int | None = None) -> tuple[int, ...]:
    if split not in _SEED_RANGES:
        raise ValueError(f"unknown split: {split}")
    size = _COUNTS[split] if count is None else count
    if size < 1:
        raise ValueError("count must be positive")
    low, high = _SEED_RANGES[split]
    if size > high - low + 1:
        raise ValueError("count exceeds the seed range")
    rng = random.Random(f"{master}:{split}")
    return tuple(sorted(rng.sample(range(low, high + 1), size)))


def _expression(seed: int, split: str, difficulty: int, attempt: int) -> str:
    low, high = _BANDS[split]
    rng = random.Random(f"{MASTER_SEED}:{split}:{difficulty}:{seed}:{attempt}")
    draws = [rng.randint(low, high) for _ in range(difficulty + 1)]
    if difficulty == 1:
        return f"{draws[0]} + {draws[1]}"
    if difficulty == 2:
        return f"{draws[0]} + {draws[1]} - {draws[2]}"
    if difficulty == 3:
        scale = draws[2] % 9 + 2
        return f"({draws[0]} + {draws[1]}) * {scale} - {draws[3]}"
    raise ValueError("difficulty must be 1, 2, or 3")


def _leaks(expression: str, value: int) -> bool:
    tokens = expression.replace("(", " ").replace(")", " ").split()
    return str(value) in tokens


def task_from_seed(seed: int, split: str, *, difficulty: int = DIFFICULTY) -> SplitTask:
    if difficulty not in {1, 2, 3}:
        raise ValueError("difficulty must be 1, 2, or 3")
    for attempt in range(30):
        expression = _expression(seed, split, difficulty, attempt)
        value = evaluate_expression(expression)
        if _leaks(expression, value):
            continue
        return SplitTask(
            split=split,
            seed=seed,
            difficulty=difficulty,
            expression=expression,
            prompt=INSTRUCTION.format(expression=expression),
            value=value,
        )
    raise RuntimeError(f"could not draw a non-leaking expression for {split}:{seed}")


def build_split(split: str, *, difficulty: int = DIFFICULTY) -> list[SplitTask]:
    tasks = [task_from_seed(seed, split, difficulty=difficulty) for seed in derive_seeds(split)]
    expressions = [task.expression for task in tasks]
    if len(set(expressions)) != len(expressions):
        raise RuntimeError(f"{split} produced a repeated expression")
    return tasks


def train_tasks() -> list[SplitTask]:
    return build_split("train")


def dev_tasks() -> list[SplitTask]:
    return build_split("dev")
