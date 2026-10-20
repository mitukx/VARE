"""Smoke check. This file is not the grader."""

from label_smooth import smooth_targets


def test_single_class_is_certain():
    assert smooth_targets(0, 1, 0.0) == [1.0]


def test_smoothing_mass_is_shared_over_all_classes():
    assert smooth_targets(1, 4, 0.2) == [0.05, 0.85, 0.05, 0.05]


if __name__ == "__main__":
    test_single_class_is_certain()
    test_smoothing_mass_is_shared_over_all_classes()
