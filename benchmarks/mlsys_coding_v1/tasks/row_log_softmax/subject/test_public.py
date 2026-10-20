"""Smoke check. This file is not the grader."""

import math

from row_log_softmax import row_log_softmax


def test_balanced_row_is_finite():
    result = row_log_softmax([[0.0, 0.0]])
    assert len(result) == 1
    assert all(math.isfinite(value) for value in result[0])
    assert abs(sum(math.exp(value) for value in result[0]) - 1.0) < 1e-9


if __name__ == "__main__":
    test_balanced_row_is_finite()
