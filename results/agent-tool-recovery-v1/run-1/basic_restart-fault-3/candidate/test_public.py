"""Smoke check. This file is not the grader."""

from grad_accum import average_microbatch_grads


def test_full_window_matches_the_mean():
    first = [[1.0, 3.0]]
    second = [[3.0, 5.0]]
    averaged = average_microbatch_grads([first, second], accumulation_steps=2)
    assert averaged == [[2.0, 4.0]]
    assert first == [[1.0, 3.0]]


if __name__ == "__main__":
    test_full_window_matches_the_mean()
