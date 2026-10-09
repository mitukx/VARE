from vare.config import ReplayConfig
from vare.replay import PrioritizedReplay
from vare.types import Attempt, Experience, Task, Verification


def _experience(index: int, group_id: str, prompt: str) -> Experience:
    task = Task(id=f"task-{index}", prompt=prompt)
    attempt = Attempt(
        task=task,
        output=f"response-{index}",
        policy_id="policy-0",
        policy_version=0,
        created_step=index,
        metadata={
            "vare_rollout_group": group_id,
            "vare_rollout_group_size": 4,
        },
    )
    verification = Verification(0.75, True, 1.0, 1, "fixture", trusted=True)
    return Experience(attempt, verification, 0, 0, 0.0)


def test_atomic_admission_rejects_mixed_prompts_without_mutating_replay():
    replay = PrioritizedReplay(ReplayConfig(capacity=8), seed=0)
    valid = [_experience(i, "valid", "same prompt") for i in range(4)]
    mixed = [
        _experience(10 + i, "mixed", "prompt A" if i < 2 else "prompt B")
        for i in range(4)
    ]

    assert replay.add_group(valid, [1.0] * 4)
    assert not replay.add_group(mixed, [1.0] * 4)
    assert len(replay) == 4
    assert {item.attempt.metadata["vare_rollout_group"] for item in replay.sample_grouped(8)} == {
        "valid"
    }


def test_legacy_grouped_samplers_fail_closed_on_mixed_prompts():
    replay = PrioritizedReplay(ReplayConfig(capacity=8), seed=0)
    mixed = [
        _experience(i, "mixed", "prompt A" if i < 2 else "prompt B")
        for i in range(4)
    ]
    for item in mixed:
        replay.add(item, freshness=1.0)

    assert replay.sample_grouped(4) == []
    assert replay.sample_current(4, freshness_fn=lambda _: 1.0, grouped=True) == []


def test_same_prompt_group_may_use_distinct_task_ids():
    replay = PrioritizedReplay(ReplayConfig(capacity=8), seed=0)
    same_prompt = [_experience(i, "same-prompt", "same prompt") for i in range(4)]

    assert len({item.attempt.task.id for item in same_prompt}) == 4
    assert replay.add_group(same_prompt, [1.0] * 4)
    assert len(replay.sample_grouped(4)) == 4


def test_atomic_admission_rejects_mixed_policy_versions_and_ids():
    replay = PrioritizedReplay(ReplayConfig(capacity=8), seed=0)
    mixed_version = [_experience(i, "mixed-version", "same prompt") for i in range(2)]
    mixed_id = [_experience(i + 2, "mixed-id", "same prompt") for i in range(2)]
    for experience in mixed_version + mixed_id:
        experience.attempt.metadata["vare_rollout_group_size"] = 2
    mixed_version[1].attempt.policy_version = 1
    mixed_id[1].attempt.policy_id = "policy-other"

    assert not replay.add_group(mixed_version, [1.0, 1.0])
    assert not replay.add_group(mixed_id, [1.0, 1.0])
    assert len(replay) == 0


def test_atomic_admission_rejects_mixed_verifier_versions():
    replay = PrioritizedReplay(ReplayConfig(capacity=8), seed=0)
    mixed = [_experience(i, "mixed-verifier", "same prompt") for i in range(2)]
    for experience in mixed:
        experience.attempt.metadata["vare_rollout_group_size"] = 2
    mixed[1].verification.verifier_version = 2

    assert not replay.add_group(mixed, [1.0, 1.0])
    assert len(replay) == 0


def test_grouped_sampling_fails_closed_on_legacy_mixed_policy_versions():
    replay = PrioritizedReplay(ReplayConfig(capacity=8), seed=0)
    mixed = [_experience(i, "legacy-mixed", "same prompt") for i in range(2)]
    for experience in mixed:
        experience.attempt.metadata["vare_rollout_group_size"] = 2
    mixed[1].attempt.policy_version = 1
    for experience in mixed:
        replay.add(experience, freshness=1.0)

    assert replay.sample_grouped(2) == []
    assert replay.sample_current(2, freshness_fn=lambda _: 1.0, grouped=True) == []
