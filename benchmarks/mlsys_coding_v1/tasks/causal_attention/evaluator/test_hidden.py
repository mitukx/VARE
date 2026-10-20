"""Hidden causal-mask checks."""
from __future__ import annotations

import math
import sys
from pathlib import Path


def _softmax(values):
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    normalizer = sum(exponentials)
    return [value / normalizer for value in exponentials]


def _expected(scores):
    weights = []
    for row_index, row in enumerate(scores):
        allowed = _softmax(row[: row_index + 1])
        weights.append(allowed + [0.0] * (len(row) - row_index - 1))
    return weights


def _assert_close(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("shape changed")
    for actual_row, expected_row in zip(actual, expected):
        if len(actual_row) != len(expected_row):
            raise AssertionError("shape changed")
        for left, right in zip(actual_row, expected_row):
            if not math.isfinite(left) or abs(left - right) > 1e-9:
                raise AssertionError(f"{actual} != {expected}")
        if abs(sum(actual_row) - 1.0) > 1e-8:
            raise AssertionError(f"row does not sum to 1: {actual_row}")


def main() -> None:
    workspace = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(workspace))
    from causal_attention import causal_attention_weights

    cases = [
        [[0.0]],
        [[1000.0]],
        [[0.0, 5.0], [0.0, 40.0]],
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [-2.0, 1.0, 4.0]],
    ]
    for scores in cases:
        _assert_close(causal_attention_weights(scores), _expected(scores))
    try:
        causal_attention_weights([[1.0, 2.0]])
    except ValueError:
        pass
    else:
        raise AssertionError("non-square scores must fail")


if __name__ == "__main__":
    main()
