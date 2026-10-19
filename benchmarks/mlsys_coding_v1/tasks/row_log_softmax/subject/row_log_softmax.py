"""Row-wise log-softmax.

``row_log_softmax(rows)`` takes a non-empty sequence of non-empty rows of
finite floats and returns a new list of rows. For a row ``x`` the result is

    y_i = x_i - log(sum_j exp(x_j))

using the natural logarithm. Every output entry must be finite for every
finite input, including rows whose entries are large and positive or large
and negative. An empty argument, an empty row, or a non-finite entry raises
``ValueError``. Inputs are not mutated.
"""
from __future__ import annotations

import math


def row_log_softmax(rows):
    if not rows:
        raise ValueError("rows must be non-empty")
    result = []
    for row in rows:
        if not row:
            raise ValueError("row must be non-empty")
        total = 0.0
        for value in row:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("row entries must be finite")
            total += math.exp(value)
        if total <= 0.0:
            raise ValueError("softmax normalizer underflowed")
        log_total = math.log(total)
        result.append([value - log_total for value in row])
    return result
