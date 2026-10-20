"""Smoke check. This file is not the grader."""
from range_sums import range_sums


def test_small_intervals():
    assert range_sums([1, 2, 3], [(0, 2), (1, 1), (0, 3)]) == [3, 0, 6]


if __name__ == "__main__":
    test_small_intervals()
