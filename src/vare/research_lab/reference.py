"""Trusted numerical references. Candidate processes cannot import this module.

The worker is started without the repository on ``sys.path``. These functions
run only in the parent and are the behavioral contract, not a source template
the grader diffs against.
"""

from __future__ import annotations

import math
import random
from typing import Any


def max_abs_diff(left: Any, right: Any) -> float:
    if isinstance(left, bool) or isinstance(right, bool):
        return math.inf
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return math.inf
        return abs(float(left) - float(right))
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return math.inf
        worst = 0.0
        for item, other in zip(left, right):
            worst = max(worst, max_abs_diff(item, other))
        return worst
    return math.inf


def l2(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def mse_microbatch_grad(
    features: list[list[float]],
    targets: list[float],
    params: list[float],
) -> list[float]:
    """Gradient of mean(0.5 * squared error) inside one microbatch."""
    count = len(targets)
    if count == 0 or count != len(features):
        raise ValueError("microbatch is empty or ragged")
    width = len(params)
    grad = [0.0] * width
    for row, target in zip(features, targets):
        if len(row) != width:
            raise ValueError("feature width does not match params")
        pred = sum(value * params[j] for j, value in enumerate(row))
        resid = pred - float(target)
        for j, value in enumerate(row):
            grad[j] += value * resid
    scale = 1.0 / count
    return [value * scale for value in grad]


def accumulated_gradient(batches: list[dict[str, Any]], params: list[float]) -> list[float]:
    if not batches:
        raise ValueError("batches must be non-empty")
    acc = [0.0] * len(params)
    for batch in batches:
        grad = mse_microbatch_grad(batch["x"], batch["y"], params)
        acc = [left + right for left, right in zip(acc, grad)]
    scale = 1.0 / len(batches)
    return [value * scale for value in acc]


def dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot inputs differ in length")
    return sum(lhs * rhs for lhs, rhs in zip(left, right))


def score_matrix(
    rows: list[list[float]],
    cols: list[list[float]],
    passes: int,
) -> list[list[float]]:
    if passes < 1:
        raise ValueError("passes must be positive")
    return [[dot(row, col) for col in cols] for row in rows]


def descent_step(
    raw_x: list[float],
    raw_y: list[float],
    a: float,
    b: float,
    learning_rate: float,
) -> tuple[float, float]:
    count = len(raw_x)
    if count == 0 or count != len(raw_y):
        raise ValueError("fit series is empty or ragged")
    grad_a = 0.0
    grad_b = 0.0
    for x, y in zip(raw_x, raw_y):
        resid = (a * x + b) - y
        grad_a += resid * x
        grad_b += resid
    grad_a /= count
    grad_b /= count
    return a - learning_rate * grad_a, b - learning_rate * grad_b


def recomputed_mse(raw_x: list[float], raw_y: list[float], a: float, b: float) -> float:
    count = len(raw_x)
    total = 0.0
    for x, y in zip(raw_x, raw_y):
        resid = (a * x + b) - y
        total += resid * resid
    return total / count


def affine_series(
    seed: int,
    true_a: float,
    true_b: float,
    count: int = 48,
) -> tuple[list[float], list[float]]:
    rng = random.Random(seed)
    raw_x = [rng.uniform(-1.5, 1.5) for _ in range(count)]
    raw_y = [true_a * x + true_b for x in raw_x]
    return raw_x, raw_y


def grad_case(seed: int) -> tuple[list[dict[str, Any]], list[float]]:
    rng = random.Random(seed)
    width = 3
    params = [rng.uniform(-0.5, 0.5) for _ in range(width)]
    batches = []
    for _ in range(4):
        rows = []
        targets = []
        for _ in range(5):
            row = [rng.uniform(-1.0, 1.0) for _ in range(width)]
            target = sum(row) * 0.25 + rng.uniform(-0.1, 0.1)
            rows.append(row)
            targets.append(target)
        batches.append({"x": rows, "y": targets})
    return batches, params


def score_case(seed: int, rows: int, cols: int, width: int) -> tuple[list, list]:
    rng = random.Random(seed)
    left = [[rng.random() for _ in range(width)] for _ in range(rows)]
    right = [[rng.random() for _ in range(width)] for _ in range(cols)]
    return left, right


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)
