import math

import pytest

from vare.config import LagConfig
from vare.lag import DistributionShiftMonitor, LagController


def test_shift_monitor_scores_complete_batch_and_resets_reference_window():
    monitor = DistributionShiftMonitor(window=4)
    monitor.fit_reference(["a", "a", "b", "b"])
    assert monitor.observe_batch(["a", "a", "b", "b"]) == 0.0
    assert monitor.score() == 0.0

    monitor.fit_reference(["a", "a", "b", "b"])
    assert monitor.score() == 0.0
    assert monitor.observe_batch(["a", "a", "a", "a"]) == 0.5


def test_lag_controller_fails_closed():
    c = LagController(LagConfig(max_policy_lag=1, max_verifier_lag=1, max_shift=0.2))
    ok = c.assess(rollout_policy_version=4, active_policy_version=5, reward_verifier_version=8, active_verifier_version=9, shift_score=0.1)
    assert ok.admitted
    bad = c.assess(rollout_policy_version=2, active_policy_version=5, reward_verifier_version=8, active_verifier_version=9, shift_score=0.1)
    assert not bad.admitted
    assert "policy_stale" in bad.reasons


def test_lag_controller_rejects_future_policy_and_verifier_versions():
    c = LagController(LagConfig(max_policy_lag=1, max_verifier_lag=1, max_shift=0.2))
    future_policy = c.assess(
        rollout_policy_version=6,
        active_policy_version=5,
        reward_verifier_version=8,
        active_verifier_version=9,
        shift_score=0.1,
    )
    assert not future_policy.admitted
    assert future_policy.policy_lag == 0
    assert "policy_version_ahead" in future_policy.reasons

    future_verifier = c.assess(
        rollout_policy_version=5,
        active_policy_version=5,
        reward_verifier_version=10,
        active_verifier_version=9,
        shift_score=0.1,
    )
    assert not future_verifier.admitted
    assert future_verifier.verifier_lag == 0
    assert "verifier_version_ahead" in future_verifier.reasons


def test_lag_controller_rejects_negative_and_non_integer_versions():
    c = LagController(LagConfig(max_policy_lag=1, max_verifier_lag=1, max_shift=0.2))
    bad_cases = [
        ({"rollout_policy_version": -1}, "invalid_policy_version"),
        ({"active_policy_version": True}, "invalid_policy_version"),
        ({"rollout_policy_version": 4.0}, "invalid_policy_version"),
        ({"reward_verifier_version": -1}, "invalid_verifier_version"),
        ({"active_verifier_version": "9"}, "invalid_verifier_version"),
    ]
    base = dict(
        rollout_policy_version=4,
        active_policy_version=5,
        reward_verifier_version=8,
        active_verifier_version=9,
        shift_score=0.1,
    )
    for overrides, expected_reason in bad_cases:
        decision = c.assess(**{**base, **overrides})
        assert not decision.admitted
        assert expected_reason in decision.reasons


@pytest.mark.parametrize(
    "shift_score",
    [math.nan, math.inf, -math.inf, -0.01, 1.01, 1 << 4096],
    ids=["nan", "positive-infinity", "negative-infinity", "negative", "above-one", "oversized-int"],
)
def test_lag_controller_rejects_invalid_shift_scores(shift_score):
    c = LagController(LagConfig(max_shift=0.2))
    decision = c.assess(
        rollout_policy_version=5,
        active_policy_version=5,
        reward_verifier_version=9,
        active_verifier_version=9,
        shift_score=shift_score,
    )

    assert not decision.admitted
    assert "invalid_shift_score" in decision.reasons
    assert decision.freshness == 0.0
    assert math.isfinite(decision.freshness)


@pytest.mark.parametrize(
    ("shift_score", "admitted", "reason"),
    [
        (0.0, True, None),
        (0.2, True, None),
        (0.21, False, "distribution_shift"),
    ],
)
def test_lag_controller_preserves_valid_shift_score_behavior(shift_score, admitted, reason):
    c = LagController(LagConfig(max_shift=0.2))
    decision = c.assess(
        rollout_policy_version=5,
        active_policy_version=5,
        reward_verifier_version=9,
        active_verifier_version=9,
        shift_score=shift_score,
    )

    assert decision.admitted is admitted
    if reason is not None:
        assert reason in decision.reasons
    assert decision.freshness == pytest.approx(math.exp(-1.25 * shift_score))
