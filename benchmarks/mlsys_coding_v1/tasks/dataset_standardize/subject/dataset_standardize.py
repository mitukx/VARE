"""Dataset-wide standardization.

``standardized_rows(rows)`` takes a non-empty list of finite floats.
Let ``n = len(rows)``, ``mean = sum(rows) / n`` in input order, and
``var = sum((x - mean) ** 2 for x in rows) / n`` in input order.
If ``var == 0``, return ``n`` zeros. Otherwise return
``[(x - mean) / sqrt(var) for x in rows]``.

The result depends on the whole list. An empty list or a non-finite
entry raises ``ValueError``. The input is not mutated.
"""
from __future__ import annotations

import math


def _mean_and_variance(rows):
    count = len(rows)
    mean = sum(rows) / count
    variance = sum((value - mean) ** 2 for value in rows) / count
    return mean, variance


def standardized_rows(rows):
    if not rows:
        raise ValueError("rows must be non-empty")
    values = []
    for value in rows:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("rows must be finite")
        values.append(float(value))
    standardized = []
    for value in values:
        mean, variance = _mean_and_variance(values)
        if variance == 0.0:
            standardized.append(0.0)
        else:
            standardized.append((value - mean) / math.sqrt(variance))
    return standardized
