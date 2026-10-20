"""Causally masked attention weights.

``causal_attention_weights(scores)`` takes a non-empty square matrix of
finite floats, as a list of rows. Position ``i`` may attend only to
columns ``j <= i``.

The returned matrix has the same shape. Row ``i`` is the softmax, using
the natural exponential, of ``scores[i][0 : i + 1]``, and it is zero on
columns ``j > i``. Each row sums to 1. Large finite scores must not
overflow. A non-square matrix, an empty matrix, or a non-finite score
raises ``ValueError``.
"""
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
        allowed = list(range(row_index + 1))
        output = [0.0] * size
        if allowed:
            peak = max(row[column] for column in allowed)
            exponentials = [math.exp(row[column] - peak) for column in allowed]
            normalizer = sum(exponentials)
            for column, value in zip(allowed, exponentials):
                output[column] = value / normalizer
        weights.append(output)
    return weights
