"""Smoke check. This file is not the grader."""

from global_grad_clip import clip_by_global_norm


def test_single_entry_matches_its_absolute_value():
    gradient = [[3.0]]
    assert clip_by_global_norm(gradient, max_norm=1.5) == [[1.5]]
    assert gradient == [[3.0]]


if __name__ == "__main__":
    test_single_entry_matches_its_absolute_value()
