from vare.config import LagConfig
from vare.controls import RVLOuterPlanner, ReplayHealthGate
from vare.integrations.rvl import RVLReplaySnapshot, RVLTokenReplayReader

from test_rvl_integration import _make_replay


def test_health_gate_freezes_on_verifier_lag():
    snap = RVLReplaySnapshot(
        groups=10,
        generations=20,
        statuses={"ready": 8, "pending_verification": 2},
        mean_reward=0.5,
        policy_versions=(1, 2),
        verifier_versions=(1,),
        max_policy_lag=1,
        max_verifier_lag=3,
    )
    d = ReplayHealthGate(LagConfig(max_verifier_lag=1)).decide(snap)
    assert d.freeze_training
    assert d.refresh_verifier
    assert "verifier_lag_exceeded" in d.reasons


def test_outer_planner_excludes_stale_from_curriculum(tmp_path):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    planner = RVLOuterPlanner(LagConfig(max_policy_lag=2, max_verifier_lag=2))
    plan = planner.plan(
        RVLTokenReplayReader(path),
        current_policy_version=6,
        current_verifier_version=5,
    )
    assert plan.fresh_experiences == 2
    assert plan.excluded_stale_experiences == 0
    assert plan.family_stats["math"]["failures"] == 1
    assert plan.family_weights["math"] > 1.0
