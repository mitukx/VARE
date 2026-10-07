from vare.config import ReplayConfig
from vare.replay import PrioritizedReplay
from vare.types import Attempt, Experience, Task, Verification


def _exp(version, group):
    task = Task(f"t{version}", "x", metadata={"vare_rollout_group": group})
    return Experience(
        attempt=Attempt(task, "x", f"p{version}", version, version, metadata={"vare_rollout_group": group}),
        verification=Verification(1.0, True, 1.0, 1, "v", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )


def test_sample_current_drops_experience_that_became_stale():
    replay = PrioritizedReplay(ReplayConfig(), seed=0)
    old = _exp(0, "old")
    new = _exp(2, "new")
    replay.add(old, 1.0)
    replay.add(new, 1.0)
    batch = replay.sample_current(
        8,
        freshness_fn=lambda exp: None if exp.attempt.policy_version < 2 else 1.0,
        grouped=True,
    )
    assert batch == [new]
