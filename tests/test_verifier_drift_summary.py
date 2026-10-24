import pytest

from scripts.benchmark_verifier_drift_scenarios import count_quarantined_valid_trajectories


@pytest.mark.parametrize("scenario_id", ["no_drift", "false_accept"])
def test_quarantined_valid_count_uses_the_actual_first_group(scenario_id):
    first_group = {
        "group_id": f"g-00-{scenario_id}",
        "members": [{"score": float(index % 2)} for index in range(8)],
        "independent_scores": [float(index % 2) for index in range(8)],
    }
    unrelated = {
        "group_id": f"g-02-{scenario_id}",
        "members": [{"score": 1.0} for _ in range(8)],
        "independent_scores": [0.0 for _ in range(8)],
    }

    assert count_quarantined_valid_trajectories(
        [first_group, unrelated], "vare_async", scenario_id) == 8
    assert count_quarantined_valid_trajectories(
        [first_group, unrelated], "sync", scenario_id) == 0


def test_quarantined_valid_count_fails_if_the_stale_group_is_missing():
    with pytest.raises(RuntimeError, match="expected one stale group"):
        count_quarantined_valid_trajectories([], "vare_async", "no_drift")
