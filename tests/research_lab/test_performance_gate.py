import pytest

from vare.research_lab.evaluation import paired_speed_gate


def test_gate_rejects_a_fixed_two_x_ratio_inside_aa_noise():
    baseline = [1.0, 1.0, 1.0, 1.0, 1.0]
    noisy_aa = [1.0, 1.0, 1.0, 0.2, 1.0]
    twice_as_fast = [0.5, 0.5, 0.5, 0.5, 0.5]
    decision = paired_speed_gate(baseline, noisy_aa, twice_as_fast)
    assert decision["speedup"] == pytest.approx(2.0)
    assert decision["performance_passed"] is False
    assert decision["gain_low_s"] <= decision["noise_floor_s"]


def test_gate_accepts_a_smaller_ratio_that_clears_aa_noise():
    baseline = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    quiet_aa = [1.001, 0.999, 1.0, 1.002, 0.998, 1.0]
    modest = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
    decision = paired_speed_gate(baseline, quiet_aa, modest)
    assert float(decision["speedup"]) < 2.0
    assert decision["performance_passed"] is True
    assert decision["gain_low_s"] > decision["noise_floor_s"]
    assert decision["performance_gate"] == "paired_ci_exceeds_aa_noise"


def test_identical_timing_fails_closed():
    samples = [0.2, 0.2, 0.21, 0.19, 0.2]
    decision = paired_speed_gate(samples, list(samples), list(samples))
    assert decision["performance_passed"] is False


def test_undefined_interval_raises():
    with pytest.raises(ValueError):
        paired_speed_gate([0.1], [0.1], [0.05])
