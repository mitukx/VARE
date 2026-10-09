from collections import Counter

from vare.config import ReplayConfig
from vare.replay import PrioritizedReplay
from vare.types import Attempt, Experience, Task, Verification


def _experience(index: int, group: str, *, group_size: int = 4) -> Experience:
    task = Task(f"task-{index}", "fixture")
    attempt = Attempt(
        task, f"response-{index}", "policy-v1", 1, 0,
        metadata={
            "vare_rollout_group": group,
            "vare_rollout_group_size": group_size,
        },
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

    # Legacy per-item insertion is unchanged; the incomplete group is excluded
    # instead of changing the trainer comparison set.
    assert counts == {"group-a": 4}

    selected_direct = replay.sample_grouped(7)
    direct_counts = Counter(
        item.attempt.metadata["vare_rollout_group"] for item in selected_direct
    )
    assert direct_counts == {"group-a": 4}


def _group(group_id: str, start: int) -> list[Experience]:
    return [_experience(start + offset, group_id) for offset in range(4)]


def _add_group(
    replay: PrioritizedReplay, group_id: str, start: int, priority: float
) -> bool:
    # A passed score of 0.75 yields base priority 0.5, so this freshness
    # produces the requested priority exactly.
    return replay.add_group(_group(group_id, start), [priority * 2.0] * 4)


def test_atomic_groups_prevent_partial_capacity_starvation():
    replay = PrioritizedReplay(ReplayConfig(capacity=7), seed=0)

    assert _add_group(replay, "group-a", 0, 1.0)
    assert _add_group(replay, "group-b", 10, 1.1)
    assert not _add_group(replay, "group-c", 20, 1.05)

    selected = replay.sample_grouped(7)
    assert len(replay) == 4
    assert {item.attempt.metadata["vare_rollout_group"] for item in selected} == {
        "group-b"
    }
    assert len(selected) == 4


def test_equal_priority_group_replaces_the_older_group():
    replay = PrioritizedReplay(ReplayConfig(capacity=7), seed=0)

    assert _add_group(replay, "group-a", 0, 1.0)
    assert _add_group(replay, "group-b", 10, 1.0)

    selected = replay.sample_grouped(7)
    assert len(replay) == 4
    assert {item.attempt.metadata["vare_rollout_group"] for item in selected} == {
        "group-b"
    }


def test_incomplete_or_malformed_group_is_rejected_without_insertion():
    replay = PrioritizedReplay(ReplayConfig(capacity=16), seed=0)
    incomplete = _group("incomplete", 0)[:3]
    assert not replay.add_group(incomplete, [2.0] * 3)

    mixed_ids = _group("mixed", 10)
    mixed_ids[-1].attempt.metadata["vare_rollout_group"] = "other"
    assert not replay.add_group(mixed_ids, [2.0] * 4)

    mixed_sizes = _group("mixed-size", 20)
    mixed_sizes[-1].attempt.metadata["vare_rollout_group_size"] = 3
    assert not replay.add_group(mixed_sizes, [2.0] * 4)

    duplicated = _group("duplicate", 30)
    duplicated[-1] = duplicated[0]
    assert not replay.add_group(duplicated, [2.0] * 4)
    assert not replay.add_group(_group("length", 40), [2.0] * 3)
    assert len(replay) == 0


def test_atomic_group_removes_existing_partial_declared_group():
    replay = PrioritizedReplay(ReplayConfig(capacity=12), seed=0)
    for exp in _group("partial", 0)[:2]:
        replay.add(exp, freshness=1.0)

    assert _add_group(replay, "complete", 10, 1.0)

    selected = replay.sample_grouped(12)
    assert len(replay) == 4
    assert {item.attempt.metadata["vare_rollout_group"] for item in selected} == {
        "complete"
    }


def test_atomic_group_rejects_capacity_smaller_than_group():
    replay = PrioritizedReplay(ReplayConfig(capacity=3), seed=0)

    assert not _add_group(replay, "too-large", 0, 1.0)
    assert len(replay) == 0
