"""Reference implementation used only by harness calibration."""
from __future__ import annotations

import math


def _validate(scores):
    if not scores or any(len(row) != len(scores) for row in scores):
        raise ValueError("scores must be a non-empty square matrix")
    for row in scores:
        for value in row:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("scores must be finite")


def causal_attention_weights(scores):
    _validate(scores)
    size = len(scores)
    weights = []
    for row_index, row in enumerate(scores):
        allowed = [column for column in range(size) if column <= row_index]
        peak = max(row[column] for column in allowed)
        exponentials = [math.exp(row[column] - peak) for column in allowed]
        normalizer = sum(exponentials)
        output = [0.0] * size
        for column, value in zip(allowed, exponentials):
            output[column] = value / normalizer
        weights.append(output)
    return weights
