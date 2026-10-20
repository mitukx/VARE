"""Hidden L2 checks for global-norm clipping."""
from __future__ import annotations

import math
import sys
from pathlib import Path


def _expected(grads, max_norm):
    values = [float(value) for row in grads for value in row]
    norm = math.sqrt(sum(value * value for value in values))
    scale = 1.0 if norm == 0.0 or norm <= max_norm else max_norm / norm
    clipped = []
    cursor = 0
    for row in grads:
        clipped.append([values[cursor + offset] * scale for offset in range(len(row))])
        cursor += len(row)
    return clipped


def _assert_close(actual, expected):
    for actual_row, expected_row in zip(actual, expected):
        for left, right in zip(actual_row, expected_row):
            if abs(left - right) > 1e-12:
                raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(workspace))
    from global_grad_clip import clip_by_global_norm

    grads = [[3.0, 4.0], [0.0, 12.0]]
    snapshot = [row[:] for row in grads]
    _assert_close(clip_by_global_norm(grads, 2.5), _expected(grads, 2.5))
    if grads != snapshot:
        raise AssertionError("input gradient was mutated")
    at_limit = [[3.0, 4.0]]
    _assert_close(clip_by_global_norm(at_limit, 5.0), [[3.0, 4.0]])
    _assert_close(clip_by_global_norm([[0.0, 0.0]], 1.0), [[0.0, 0.0]])
    try:
        clip_by_global_norm([[1.0, 2.0]], 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive max_norm must fail")


if __name__ == "__main__":
    main()
