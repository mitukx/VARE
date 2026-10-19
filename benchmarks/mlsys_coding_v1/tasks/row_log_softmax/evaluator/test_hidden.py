"""Hidden correctness checks for row-wise log-softmax."""
from __future__ import annotations

import math
import sys
from pathlib import Path


def _reference(row):
    peak = max(row)
    total = sum(math.exp(value - peak) for value in row)
    log_total = peak + math.log(total)
    return [value - log_total for value in row]


def _close(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("row length changed")
    for left, right in zip(actual, expected):
        if not math.isfinite(left) or abs(left - right) > 1e-9:
            raise AssertionError(f"{actual} != {expected}")


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(workspace))
    from row_log_softmax import row_log_softmax

    small = row_log_softmax([[0.0, math.log(3.0)], [1.0, 1.0, 1.0]])
    _close(small[0], _reference([0.0, math.log(3.0)]))
    _close(small[1], _reference([1.0, 1.0, 1.0]))

    large = row_log_softmax([[1000.0, 1000.0, 999.0], [-1000.0, -1001.0, -1000.5]])
    _close(large[0], _reference([1000.0, 1000.0, 999.0]))
    _close(large[1], _reference([-1000.0, -1001.0, -1000.5]))
    for row in large:
        total = sum(math.exp(value) for value in row)
        if abs(total - 1.0) > 1e-8:
            raise AssertionError(f"probabilities sum to {total}")

    try:
        row_log_softmax([[]])
    except ValueError:
        pass
    else:
        raise AssertionError("empty row must fail")
    try:
        row_log_softmax([[math.inf]])
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite input must fail")


if __name__ == "__main__":
    main()
