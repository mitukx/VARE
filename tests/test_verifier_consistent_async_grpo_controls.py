from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.benchmark_verifier_consistent_async_grpo import (
    install_naive_rvl_reward_path,
)
from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.types import Attempt, Experience, Task, Verification


def _verified(reward: float, index: int):
    return SimpleNamespace(
        generation=SimpleNamespace(
            prompt_id="one-prompt",
            response=str(index),
            logprob=-0.1,
        ),
        reward=reward,
        verifier_version=index % 2,
    )


def test_naive_control_uses_rvl_prompt_grouping_without_verifier_lineage():
    grpo = pytest.importorskip("rvl_systems.grpo")
    rewards = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    samples = [_verified(reward, index) for index, reward in enumerate(rewards)]
    experiences = [SimpleNamespace(
        verification=SimpleNamespace(verifier_version=index % 2)
    ) for index in range(len(samples))]
    hooks = SimpleNamespace()

    install_naive_rvl_reward_path(hooks)
    actual = hooks._group_relative_advantages(
        experiences, samples, eps=1e-6, clip=5.0
    )
    expected = [
        row.advantage
        for row in grpo.compute_group_advantages(samples, eps=1e-6, clip=5.0)
    ]

    assert actual == expected
    assert len(actual) == 8


def test_vare_group_guard_rejects_mixed_verifier_versions():
    experiences = []
    samples = []
    for index in range(8):
        version = index % 2
        task = Task("one-prompt", "prompt", "landing")
        attempt = Attempt(
            task=task,
            output=str(index),
            policy_id="policy-0",
            policy_version=0,
            created_step=index,
            metadata={
                "vare_rollout_group": "mixed-group",
                "vare_rollout_group_size": 8,
                "rvl_generation": {
                    "prompt_id": task.id,
                    "prompt": task.prompt,
                    "metadata": {
                        "model": "pinned-model",
                        "sampling_temperature": 0.8,
                        "logprob_distribution": "raw-logits",
                    },
                },
            },
        )
        verification = Verification(
            score=float(index % 2),
            passed=bool(index % 2),
            confidence=1.0,
            verifier_version=version,
            verifier_name="landing",
            metadata={"rvl_verification_config_sha256": f"verifier-{version}"},
        )
        experiences.append(Experience(attempt, verification, 0, 1, 0.0))
        samples.append(_verified(float(index % 2), index))

    with pytest.raises(ValueError, match="mixes prompt, policy, behavior, or evaluator provenance"):
        RVLGRPOHooks._group_relative_advantages(
            experiences, samples, eps=1e-6, clip=5.0
        )
