"""Smoke check. This file is not the grader."""

from dataset_standardize import standardized_rows


def test_two_points():
    result = standardized_rows([0.0, 2.0])
    assert abs(result[0] + 1.0) < 1e-12
    assert abs(result[1] - 1.0) < 1e-12


if __name__ == "__main__":
    test_two_points()
