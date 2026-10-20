"""Smoke check. This file is not the grader."""

from label_smooth import smooth_targets


def test_single_class_is_certain():
    assert smooth_targets(0, 1, 0.0) == [1.0]


if __name__ == "__main__":
    test_single_class_is_certain()
