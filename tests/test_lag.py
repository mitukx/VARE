from vare.config import LagConfig
from vare.lag import LagController


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
