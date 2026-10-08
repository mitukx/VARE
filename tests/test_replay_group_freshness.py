from collections import Counter

from vare.config import ReplayConfig
from vare.replay import PrioritizedReplay
from vare.types import Attempt, Experience, Task, Verification


def _experience(index: int, group: str) -> Experience:
    task = Task(f"task-{index}", "fixture")
    attempt = Attempt(
        task, f"response-{index}", "policy-v1", 1, 0,
        metadata={"vare_rollout_group": group, "vare_rollout_group_size": 4},
    )
    verification = Verification(0.75, True, 1.0, 1, "fixture", trusted=True)
    return Experience(attempt, verification, 0, 0, 0.0)


def test_one_stale_member_drops_its_whole_group_and_keeps_fresh_groups():
    replay = PrioritizedReplay(ReplayConfig(capacity=16), seed=0)
    for index in range(8):
        group = "group-a" if index < 4 else "group-b"
        replay.add(_experience(index, group), freshness=1.0)
    freshness = lambda item: None if item.attempt.task.id == "task-0" else 1.0

    selected = replay.sample_current(8, freshness_fn=freshness, grouped=True)

    counts = Counter(item.attempt.metadata["vare_rollout_group"] for item in selected)
    assert counts == {"group-b": 4}

    per_item = replay.sample_current(8, freshness_fn=freshness, grouped=False)
    assert len(per_item) == 7


def test_replay_capacity_cannot_send_a_partial_declared_group_to_training():
    replay = PrioritizedReplay(ReplayConfig(capacity=7), seed=0)
    for index in range(8):
        group = "group-a" if index < 4 else "group-b"
        replay.add(_experience(index, group), freshness=1.0)

    selected = replay.sample_current(7, freshness_fn=lambda _: 1.0, grouped=True)
    counts = Counter(item.attempt.metadata["vare_rollout_group"] for item in selected)

    # Per-item capacity eviction leaves four members of group-a and three of
    # group-b. The incomplete group is excluded instead of changing the trainer
    # comparison set.
    assert counts == {"group-a": 4}

    selected_direct = replay.sample_grouped(7)
    direct_counts = Counter(
        item.attempt.metadata["vare_rollout_group"] for item in selected_direct
    )
    assert direct_counts == {"group-a": 4}
