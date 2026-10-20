"""Reference implementation used only by harness calibration."""
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
    norm = math.sqrt(sum(value * value for value in values))
    scale = 1.0 if norm == 0.0 or norm <= max_norm else max_norm / norm
    clipped = []
    cursor = 0
    for row in grads:
        clipped.append([values[cursor + offset] * scale for offset in range(len(row))])
        cursor += len(row)
    return clipped
