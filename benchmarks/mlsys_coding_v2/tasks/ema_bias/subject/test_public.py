"""Smoke check. This file is not the grader."""

from ema_bias import bias_corrected_ema


def test_zero_decay_returns_the_observation():
    assert bias_corrected_ema(5.0, 7.0, 0.0, 3) == 7.0


if __name__ == "__main__":
    test_zero_decay_returns_the_observation()
