from __future__ import annotations

import math

import pytest

from vare.eval_stats import (
    mcnemar_exact,
    mcnemar_power,
    mean_unbiased_pass_at_k,
    paired_bootstrap_interval,
    unbiased_pass_at_k,
)


def test_unbiased_pass_at_k_matches_binomial_coefficients():
    assert unbiased_pass_at_k(4, 2, 1) == 0.5
    assert math.isclose(unbiased_pass_at_k(4, 2, 2), 1.0 - 1.0 / 6.0)
    assert unbiased_pass_at_k(4, 2, 4) == 1.0
    assert unbiased_pass_at_k(4, 0, 1) == 0.0
    assert mean_unbiased_pass_at_k([(4, 4), (4, 0)], 1) == 0.5
    with pytest.raises(ValueError):
        unbiased_pass_at_k(4, 2, 5)


def test_paired_bootstrap_interval_is_degenerate_for_a_constant_difference():
    interval = paired_bootstrap_interval([0.0, 0.0, 0.0], [0.25, 0.25, 0.25], samples=40, seed=3)
    assert interval["mean"] == 0.25
    assert interval["low"] == 0.25
    assert interval["high"] == 0.25
    flipped = paired_bootstrap_interval([1.0, 0.0], [0.0, 1.0], samples=200, seed=1)
    again = paired_bootstrap_interval([1.0, 0.0], [0.0, 1.0], samples=200, seed=1)
    assert flipped == again
    assert flipped["low"] <= flipped["mean"] <= flipped["high"]


def test_mcnemar_and_power_match_known_values():
    exact = mcnemar_exact(8, 0)
    assert exact["p_value"] == 2 / 256
    assert exact["statistic"] == 8
    tied = mcnemar_exact(3, 3)
    assert tied["p_value"] == 1.0
    assert mcnemar_exact(0, 0)["p_value"] == 1.0
    assert math.isclose(mcnemar_power(100, 0.2, 0.05), 0.8621993605389122)
    assert mcnemar_power(40, 0.1, 0.1) == 0.05
