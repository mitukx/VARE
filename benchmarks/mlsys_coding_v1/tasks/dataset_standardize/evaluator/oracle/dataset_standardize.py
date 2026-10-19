"""Reference implementation used only by harness calibration."""
from __future__ import annotations

import math


def standardized_rows(rows):
    if not rows:
        raise ValueError("rows must be non-empty")
    values = []
    for value in rows:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("rows must be finite")
        values.append(float(value))
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    if variance == 0.0:
        return [0.0] * count
    scale = math.sqrt(variance)
    return [(value - mean) / scale for value in values]
