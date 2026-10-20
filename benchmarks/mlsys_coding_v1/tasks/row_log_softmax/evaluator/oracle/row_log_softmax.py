"""Reference implementation used only by harness calibration."""

from __future__ import annotations

import math


def row_log_softmax(rows):
    if not rows:
        raise ValueError("rows must be non-empty")
    result = []
    for row in rows:
        if not row:
            raise ValueError("row must be non-empty")
        finite = []
        for value in row:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("row entries must be finite")
            finite.append(float(value))
        peak = max(finite)
        total = sum(math.exp(value - peak) for value in finite)
        log_total = peak + math.log(total)
        result.append([value - log_total for value in finite])
    return result
