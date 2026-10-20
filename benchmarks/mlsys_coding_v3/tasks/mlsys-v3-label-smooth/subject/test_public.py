"""Smoke check. This file is not the grader."""

import math

from label_smooth import smooth_targets


def test_single_class_is_certain():
    assert smooth_targets(0, 1, 0.0) == [1.0]


def test_smoothing_mass_is_shared_over_all_classes():
    actual = smooth_targets(1, 4, 0.2)
    expected = [0.05, 0.85, 0.05, 0.05]
    assert len(actual) == len(expected)
    assert all(math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
               for left, right in zip(actual, expected))


if __name__ == "__main__":
    test_single_class_is_certain()
    test_smoothing_mass_is_shared_over_all_classes()
