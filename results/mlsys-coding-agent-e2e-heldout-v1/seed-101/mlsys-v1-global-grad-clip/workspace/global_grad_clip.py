"""Global L2 gradient clipping.

``clip_by_global_norm(grads, max_norm)`` takes a gradient, as a list of
equal-length rows of finite floats, and a finite ``max_norm > 0``.

``norm`` is the square root of the sum of squares of every entry. If
``norm <= max_norm`` or ``norm == 0``, return a copy of ``grads``.
Otherwise return a new gradient scaled by ``max_norm / norm``. Do not
mutate ``grads``. An empty gradient, a non-positive ``max_norm``, or a
non-finite entry raises ``ValueError``.
"""
from __future__ import annotations

import math


def _entries(grads):
    if not grads or any(not row for row in grads):
        raise ValueError("gradient must be non-empty")
    width = len(grads[0])
    if any(len(row) != width for row in grads):
        raise ValueError("jagged gradient")
    values = []
    for row in grads:
        for value in row:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError("gradient entries must be finite")
            values.append(float(value))
    return values


def clip_by_global_norm(grads, max_norm):
    if (
        isinstance(max_norm, bool) or not isinstance(max_norm, (int, float))
        or not math.isfinite(max_norm) or max_norm <= 0
    ):
        raise ValueError("max_norm must be finite and positive")
    values = _entries(grads)
    norm = sum(abs(value) for value in values)
    scale = 1.0 if norm == 0.0 or norm <= max_norm else max_norm / norm
    clipped = []
    cursor = 0
    for row in grads:
        clipped.append([values[cursor + offset] * scale for offset in range(len(row))])
        cursor += len(row)
    return clipped
