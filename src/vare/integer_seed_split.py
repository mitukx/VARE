"""Deterministic train, dev, and confirm integer expressions.

The confirm seed list is committed. A protocol file records the SHA-256 of
that list. Splits use disjoint seeds and disjoint integer bands, so a
whitespace token 3-gram of an expression cannot appear in two splits.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from vare.heldout_exact_integer import evaluate_expression

MASTER_SEED = 20261014
DIFFICULTY = 2
TRAIN_COUNT = 128
DEV_COUNT = 32
CONFIRM_COUNT = 64
NGRAM_N = 3
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

    def to_dict(self) -> dict[str, object]:
        return {
            "difficulty": self.difficulty,
            "expression": self.expression,
            "prompt": self.prompt,
            "seed": self.seed,
            "split": self.split,
            "value_sha256": hashlib.sha256(str(self.value).encode("utf-8")).hexdigest(),
        }


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
    rng = random.Random(f"{master_tag(seed, split, difficulty)}:{attempt}")
    draws = [rng.randint(low, high) for _ in range(difficulty + 1)]
    if difficulty == 1:
        return f"{draws[0]} + {draws[1]}"
    if difficulty == 2:
        return f"{draws[0]} + {draws[1]} - {draws[2]}"
    if difficulty == 3:
        return f"({draws[0]} + {draws[1]}) * {draws[2] % 9 + 2} - {draws[3]}"
    raise ValueError("difficulty must be 1, 2, or 3")


def master_tag(seed: int, split: str, difficulty: int) -> str:
    return f"{MASTER_SEED}:{split}:{difficulty}:{seed}"


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


def token_ngrams(expression: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    tokens = expression.replace("(", " ").replace(")", " ").split()
    if n < 1 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(left: Iterable[str], right: Iterable[str], n: int = NGRAM_N) -> set[tuple[str, ...]]:
    right_grams: set[tuple[str, ...]] = set()
    for expression in right:
        right_grams.update(token_ngrams(expression, n))
    overlap: set[tuple[str, ...]] = set()
    for expression in left:
        overlap.update(token_ngrams(expression, n) & right_grams)
    return overlap


def canonical_split_document() -> dict[str, object]:
    document: dict[str, object] = {
        "confirm_count": CONFIRM_COUNT,
        "dev_count": DEV_COUNT,
        "difficulty": DIFFICULTY,
        "master_seed": MASTER_SEED,
        "ngram_n": NGRAM_N,
        "train_count": TRAIN_COUNT,
    }
    for split in ("train", "dev", "confirm"):
        document[f"{split}_seeds"] = list(derive_seeds(split))
    return document


def canonical_split_bytes() -> bytes:
    return (json.dumps(canonical_split_document(), indent=2, sort_keys=True) + "\n").encode("utf-8")


def split_sha256() -> str:
    return hashlib.sha256(canonical_split_bytes()).hexdigest()


def load_split_seeds(path: Path) -> Mapping[str, Sequence[int]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {name: tuple(int(seed) for seed in document[f"{name}_seeds"]) for name in ("train", "dev", "confirm")}
