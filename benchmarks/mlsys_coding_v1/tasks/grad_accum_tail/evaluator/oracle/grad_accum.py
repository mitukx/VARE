"""Reference implementation used only by harness calibration."""

from __future__ import annotations

import math


def _shape(gradient):
    if not gradient or not gradient[0]:
        raise ValueError("gradient must be non-empty")
    width = len(gradient[0])
    if any(len(row) != width for row in gradient):
        raise ValueError("jagged gradient")
    return len(gradient), width


def average_microbatch_grads(microbatches, accumulation_steps):
    if (
        not isinstance(accumulation_steps, int)
        or isinstance(accumulation_steps, bool)
        or accumulation_steps <= 0
    ):
        raise ValueError("accumulation_steps must be a positive integer")
    if not microbatches:
        raise ValueError("microbatches must be non-empty")
    rows, width = _shape(microbatches[0])
    accumulator = [[0.0 for _ in range(width)] for _ in range(rows)]
    for gradient in microbatches:
        if _shape(gradient) != (rows, width):
            raise ValueError("microbatch shapes differ")
        for row_index, row in enumerate(gradient):
            for column, value in enumerate(row):
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise ValueError("gradient entries must be finite")
                accumulator[row_index][column] += float(value)
    scale = float(len(microbatches))
    return [[value / scale for value in row] for row in accumulator]
