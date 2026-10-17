import pytest

from vare.config import ReplayConfig
from vare.replay import PrioritizedReplay
from vare.types import Attempt, Experience, Task, Verification


def _exp(version, group):
    task = Task(f"t{version}", "x", metadata={"vare_rollout_group": group})
    return Experience(
        attempt=Attempt(
            task,
            "x",
            f"p{version}",
            version,
            version,
            metadata={"vare_rollout_group": group},
        ),
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


@pytest.mark.parametrize("grouped", [False, True])
def test_prune_current_preserves_groups_and_does_not_consume_rng(grouped):
    replay = PrioritizedReplay(ReplayConfig(), seed=9)
    old, sibling, fresh = _exp(0, "mixed"), _exp(1, "mixed"), _exp(1, "fresh")
    for exp in (old, sibling, fresh):
        replay.add(exp, 1.0)
    rng = replay._rng.getstate()
    count = replay.prune_current(
        freshness_fn=lambda exp: None if exp is old else 0.5, grouped=grouped
    )
    assert count == (2 if grouped else 1)
    assert {id(item.experience) for item in replay._heap} == (
        {id(fresh)} if grouped else {id(fresh), id(sibling)}
    )
    assert replay._rng.getstate() == rng
    assert fresh.priority == replay._priority(fresh, 0.5)


def test_prune_assessor_failure_leaves_heap_and_priorities_unchanged():
    replay = PrioritizedReplay(ReplayConfig(), seed=9)
    for group in ("a", "b"):
        replay.add(_exp(0, group), 1.0)
    before = [
        (id(item), item.priority, item.experience.priority) for item in replay._heap
    ]
    calls = 0

    def fail(exp):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("assessor failed")
        return None

    with pytest.raises(RuntimeError, match="assessor failed"):
        replay.prune_current(freshness_fn=fail)
    assert before == [
        (id(item), item.priority, item.experience.priority) for item in replay._heap
    ]
