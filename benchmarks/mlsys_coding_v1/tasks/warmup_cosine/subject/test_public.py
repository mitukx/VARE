"""Smoke check. This file is not the grader."""
from warmup_cosine import learning_rate


def test_first_cosine_step_without_warmup():
    value = learning_rate(0, total_steps=8, warmup_steps=0, base_lr=0.1, min_lr=0.0)
    assert abs(value - 0.1) < 1e-12


if __name__ == "__main__":
    test_first_cosine_step_without_warmup()
