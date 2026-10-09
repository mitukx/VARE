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
    def __init__(self):
        self.config = SimpleNamespace(advantage_eps=1e-6, clip_advantage=5.0)
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


def _experience(group_id: str, reward: float, response: str) -> Experience:
    task = Task(id="same-task", prompt="same prompt", family="fixture")
    generation = {
        "prompt_id": "same-task",
        "prompt": "same prompt",
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
                "vare_rollout_group_size": 2,
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


def _hooks(trainer: CapturingTrainer) -> RVLGRPOHooks:
    return RVLGRPOHooks(
        backend=Backend(),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda _task, _response: 0.0,
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )


def test_rvl_adapter_passes_advantages_normalized_by_vare_group():
    trainer = CapturingTrainer()
    hooks = _hooks(trainer)
    experiences = [
        _experience("group-a", 0.0, "a0"),
        _experience("group-a", 0.0, "a1"),
        _experience("group-b", 1.0, "b0"),
        _experience("group-b", 1.0, "b1"),
    ]

    asyncio.run(hooks.train_candidate("policy-0", experiences))

    assert trainer.received_advantages == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)


def test_ungrouped_direct_rvl_experiences_keep_trainer_default():
    trainer = CapturingTrainer()
    hooks = _hooks(trainer)
    experience = _experience("temporary", 1.0, "answer")
    experience.attempt.metadata.pop("vare_rollout_group")
    experience.attempt.metadata.pop("vare_rollout_group_size")

    asyncio.run(hooks.train_candidate("policy-0", [experience]))

    assert trainer.received_advantages is None
