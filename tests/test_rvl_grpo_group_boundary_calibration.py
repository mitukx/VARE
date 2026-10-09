from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.types import Attempt, Experience, Task, Verification


@dataclass
class Generation:
    prompt_id: str
    prompt: str
    response: str
    logprob: float
    token_count: int
    latency_s: float
    metadata: dict = field(default_factory=dict)


@dataclass
class VerifiedGeneration:
    generation: Generation
    reward: float
    verifier_latency_s: float
    verifier_version: int
    metadata: dict = field(default_factory=dict)


class CapturingTrainer:
    def __init__(self, *, clip_advantage=5.0):
        self.config = SimpleNamespace(advantage_eps=1e-6, clip_advantage=clip_advantage)
        self.weight = 0
        self.optimizer_step = 0
        self.received_advantages = None

    def snapshot_training_state(self):
        return {"weight": self.weight, "optimizer_step": self.optimizer_step}

    def restore_training_state(self, state):
        self.weight = state["weight"]
        self.optimizer_step = state["optimizer_step"]

    def train_step(self, samples, *, advantages=None):
        self.received_advantages = advantages
        self.weight += 1
        self.optimizer_step += 1


class Backend:
    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        raise AssertionError("generation is not part of this trainer-boundary test")


def _experience(group_id, group_size, prompt_id, reward, response):
    task = Task(id=prompt_id, prompt=f"prompt for {prompt_id}", family="fixture")
    generation = {
        "prompt_id": prompt_id,
        "prompt": task.prompt,
        "response": response,
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {},
    }
    return Experience(
        attempt=Attempt(
            task=task,
            output=response,
            policy_id="policy-0",
            policy_version=0,
            created_step=0,
            metadata={
                "rvl_generation": generation,
                "vare_rollout_group": group_id,
                "vare_rollout_group_size": group_size,
            },
        ),
        verification=Verification(
            score=reward,
            passed=bool(reward),
            confidence=1.0,
            verifier_version=0,
            verifier_name="fixture",
            trusted=True,
        ),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )


def _hooks(trainer):
    return RVLGRPOHooks(
        backend=Backend(),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda _task, _response: 0.0,
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )


def test_nonconstant_rewards_stay_separate_for_same_prompt_id():
    trainer = CapturingTrainer()
    experiences = [
        _experience("group-a", 2, "same-task", 0.0, "a0"),
        _experience("group-a", 2, "same-task", 1.0, "a1"),
        _experience("group-b", 2, "same-task", 0.75, "b0"),
        _experience("group-b", 2, "same-task", 1.0, "b1"),
    ]

    asyncio.run(_hooks(trainer).train_candidate("policy-0", experiences))

    assert trainer.received_advantages == pytest.approx(
        [-0.999998, 0.999998, -0.999968002, 0.999968002], abs=1e-8
    )


def test_distinct_prompt_ids_match_promptwise_group_normalization():
    trainer = CapturingTrainer()
    experiences = [
        _experience("group-a", 2, "task-a", 0.0, "a0"),
        _experience("group-a", 2, "task-a", 1.0, "a1"),
        _experience("group-b", 2, "task-b", 0.75, "b0"),
        _experience("group-b", 2, "task-b", 1.0, "b1"),
    ]

    asyncio.run(_hooks(trainer).train_candidate("policy-0", experiences))

    assert trainer.received_advantages == pytest.approx(
        [-0.999998, 0.999998, -0.999968002, 0.999968002], abs=1e-8
    )


def test_group_relative_advantages_apply_clipping_symmetrically():
    trainer = CapturingTrainer(clip_advantage=0.5)
    experiences = [
        _experience("group-a", 4, "same-task", 0.0, "r0"),
        _experience("group-a", 4, "same-task", 0.0, "r1"),
        _experience("group-a", 4, "same-task", 0.0, "r2"),
        _experience("group-a", 4, "same-task", 1.0, "r3"),
    ]

    asyncio.run(_hooks(trainer).train_candidate("policy-0", experiences))

    assert trainer.received_advantages == [-0.5, -0.5, -0.5, 0.5]


def test_incomplete_declared_group_fails_before_optimizer_step():
    trainer = CapturingTrainer()
    experience = _experience("group-a", 2, "same-task", 0.0, "a0")

    with pytest.raises(ValueError, match="incomplete VARE rollout group"):
        asyncio.run(_hooks(trainer).train_candidate("policy-0", [experience]))

    assert trainer.optimizer_step == 0
