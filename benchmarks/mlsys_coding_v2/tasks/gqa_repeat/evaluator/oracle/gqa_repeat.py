"""Reference implementation used only by harness calibration."""

from __future__ import annotations

import math


def _validate(keys, n_rep):
    if type(n_rep) is not int or n_rep < 1:
        raise ValueError("n_rep must be an integer >= 1")
    if not keys or any(not row for row in keys):
        raise ValueError("keys must be non-empty")
    width = len(keys[0])
    if any(len(row) != width for row in keys):
        raise ValueError("jagged keys")
    for row in keys:
        for value in row:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("keys must be finite")


def repeat_kv_heads(keys, n_rep):
    _validate(keys, n_rep)
    return [row[:] for row in keys for _ in range(n_rep)]
