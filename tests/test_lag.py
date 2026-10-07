from vare.config import LagConfig
from vare.lag import LagController


def test_lag_controller_fails_closed():
    c = LagController(LagConfig(max_policy_lag=1, max_verifier_lag=1, max_shift=0.2))
    ok = c.assess(rollout_policy_version=4, active_policy_version=5, reward_verifier_version=8, active_verifier_version=9, shift_score=0.1)
    assert ok.admitted
    bad = c.assess(rollout_policy_version=2, active_policy_version=5, reward_verifier_version=8, active_verifier_version=9, shift_score=0.1)
    assert not bad.admitted
    assert "policy_stale" in bad.reasons
