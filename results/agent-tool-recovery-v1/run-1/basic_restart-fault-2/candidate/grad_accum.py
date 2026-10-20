"""Mean of the microbatch gradients that were actually supplied.

``average_microbatch_grads(microbatches, accumulation_steps)`` takes a
non-empty list of gradients and a positive integer ``accumulation_steps``.
Each gradient is a non-empty list of equal-length rows of finite floats.
Every microbatch must share that shape.

The result is a new gradient of the same shape. Each entry is the
arithmetic mean of that entry across ``microbatches``. The mean uses
``len(microbatches)`` even when that length is shorter than the nominal
``accumulation_steps`` window. Inputs are not mutated. Empty input, a
non-positive window, a shape mismatch, or a non-finite entry raises
``ValueError``.
"""

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
    scale = float(min(len(microbatches), accumulation_steps))
    return [[value / scale for value in row] for row in accumulator]
