import asyncio
from dataclasses import dataclass, field
import math

import pytest

from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
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


class FakeTrainer:
    def __init__(self):
        self.weight = 0
        self.optimizer_step = 0

    def snapshot_training_state(self):
        return {"weight": self.weight, "optimizer_step": self.optimizer_step}

    def restore_training_state(self, state):
        self.weight = state["weight"]
        self.optimizer_step = state["optimizer_step"]

    def train_step(self, samples):
        assert samples and isinstance(samples[0], VerifiedGeneration)
        self.weight += 1
        self.optimizer_step += 1
        return {"loss": 0.0}


class PartiallyFailingTrainer(FakeTrainer):
    def train_step(self, samples):
        super().train_step(samples)
        raise RuntimeError("simulated failure after mutating policy state")


class PartiallyFailingRestoreTrainer(FakeTrainer):
    def __init__(self):
        super().__init__()
        self.fail_candidate_restore = False
        self.restore_targets = []

    def restore_training_state(self, state):
        self.restore_targets.append(state["weight"])
        # Simulate a restore that changes part of the live trainer before
        # failing. The adapter must then force a second restore to the incumbent.
        self.weight = state["weight"]
        self.optimizer_step = state["optimizer_step"]
        if state["weight"] > 0 and self.fail_candidate_restore:
            self.fail_candidate_restore = False
            raise RuntimeError("simulated partial candidate restore failure")


class FakeBackend:
    def __init__(self, trainer):
        self.trainer = trainer
        self.fail = False

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert n == 1
        if self.fail:
            raise RuntimeError("simulated evaluation generation failure")
        return [Generation(
            prompt_id=prompt_id,
            prompt=prompt,
            response="1" if self.trainer.weight > 0 else "0",
            logprob=-0.1,
            token_count=1,
            latency_s=0.001,
            metadata={
                "sampling_temperature": max(temperature, 0.1),
                "prompt_token_ids": [1],
                "response_token_ids": [2],
                "response_token_logprobs": [-0.1],
            },
        )]


def test_rvl_grpo_hooks_candidate_is_transactional():
    trainer = FakeTrainer()
    backend = FakeBackend(trainer)
    eval_tasks = [Task(id=f"e{i}", prompt="answer", family="math") for i in range(4)]
    hooks = RVLGRPOHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=eval_tasks,
        score_fn=lambda task, response: float(response == "1"),
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )
    task = Task(id="t", prompt="train", family="math")
    raw = {
        "prompt_id": "t",
        "prompt": "train",
        "response": "0",
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {
            "sampling_temperature": 0.8,
            "prompt_token_ids": [1],
            "response_token_ids": [2],
            "response_token_logprobs": [-0.1],
        },
    }
    exp = Experience(
        attempt=Attempt(task, "0", "policy-0", 0, 0, logprob=-0.1, metadata={"rvl_generation": raw}),
        verification=Verification(1.0, True, 1.0, 1, "oracle", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )

    async def run():
        candidate = await hooks.train_candidate("policy-0", [exp])
        assert trainer.weight == 0  # candidate is not silently installed
        backend.fail = True
        try:
            await hooks.evaluate(candidate)
        except RuntimeError as exc:
            assert "simulated evaluation generation failure" in str(exc)
        else:
            raise AssertionError("evaluation generation failure should propagate")
        finally:
            backend.fail = False
        assert hooks.active_policy() == ("policy-0", 0)
        assert hooks._loaded_id == "policy-0"
        assert trainer.weight == 0
        inc = await hooks.evaluate("policy-0")
        cand = await hooks.evaluate(candidate)
        assert inc.primary == 0.0
        assert cand.primary == 1.0
        assert trainer.weight == 0
        await hooks.promote(candidate)
        assert hooks.active_policy() == (candidate, 1)
        assert trainer.weight == 1

    asyncio.run(run())


@pytest.mark.parametrize("score", [2.0, math.nan, math.inf])
def test_rvl_trainer_boundary_rejects_mutated_reward_before_optimizer(score):
    trainer = FakeTrainer()
    hooks = RVLGRPOHooks(
        backend=FakeBackend(trainer),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda task, response: 0.0,
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )
    verification = Verification(0.75, True, 1.0, 1, "fixture", trusted=True)
    verification.score = score
    exp = Experience(
        attempt=Attempt(
            Task("train", "prompt"),
            "answer",
            "policy-0",
            0,
            0,
            metadata={
                "rvl_generation": {
                    "prompt_id": "train",
                    "prompt": "prompt",
                    "response": "answer",
                    "logprob": -0.1,
                    "token_count": 1,
                    "latency_s": 0.001,
                    "metadata": {},
                }
            },
        ),
        verification=verification,
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )
    with pytest.raises(ValueError, match="score"):
        asyncio.run(hooks.train_candidate("policy-0", [exp]))
    assert trainer.optimizer_step == 0


def test_rvl_grpo_hooks_restores_incumbent_after_partial_train_failure():
    trainer = PartiallyFailingTrainer()
    backend = FakeBackend(trainer)
    task = Task(id="t", prompt="train", family="math")
    raw = {
        "prompt_id": "t",
        "prompt": "train",
        "response": "0",
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {
            "sampling_temperature": 0.8,
            "prompt_token_ids": [1],
            "response_token_ids": [2],
            "response_token_logprobs": [-0.1],
        },
    }
    exp = Experience(
        attempt=Attempt(task, "0", "policy-0", 0, 0, logprob=-0.1, metadata={"rvl_generation": raw}),
        verification=Verification(1.0, True, 1.0, 1, "oracle", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )
    hooks = RVLGRPOHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=[task],
        score_fn=lambda task, response: float(response == "1"),
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )

    async def run():
        try:
            await hooks.train_candidate("policy-0", [exp])
        except RuntimeError as exc:
            assert "simulated failure" in str(exc)
        else:
            raise AssertionError("partial training failure should propagate")

        assert hooks.active_policy() == ("policy-0", 0)
        assert trainer.weight == 0
        assert trainer.optimizer_step == 0
        assert hooks._loaded_id == "policy-0"
        assert set(hooks._states) == {"policy-0"}
        assert hooks._counter == 1
        attempt = await hooks.rollout(task, "policy-0", 0, 1)
        assert attempt.output == "0"
        assert trainer.weight == 0
        assert trainer.optimizer_step == 0

    asyncio.run(run())


def test_rvl_grpo_hooks_forces_incumbent_restore_after_partial_candidate_restore_failure():
    trainer = PartiallyFailingRestoreTrainer()
    backend = FakeBackend(trainer)
    task = Task(id="t", prompt="train", family="math")
    raw = {
        "prompt_id": "t",
        "prompt": "train",
        "response": "0",
        "logprob": -0.1,
        "token_count": 1,
        "latency_s": 0.001,
        "metadata": {
            "sampling_temperature": 0.8,
            "prompt_token_ids": [1],
            "response_token_ids": [2],
            "response_token_logprobs": [-0.1],
        },
    }
    exp = Experience(
        attempt=Attempt(task, "0", "policy-0", 0, 0, logprob=-0.1, metadata={"rvl_generation": raw}),
        verification=Verification(1.0, True, 1.0, 1, "oracle", trusted=True),
        policy_lag=0,
        verifier_lag=0,
        shift_score=0.0,
    )
    hooks = RVLGRPOHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=[task],
        score_fn=lambda task, response: float(response == "1"),
        config=RVLGRPOConfig(seed=1),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )

    async def run():
        candidate = await hooks.train_candidate("policy-0", [exp])
        trainer.fail_candidate_restore = True

        try:
            await hooks.evaluate(candidate)
        except RuntimeError as exc:
            assert "simulated partial candidate restore failure" in str(exc)
        else:
            raise AssertionError("partial candidate restore failure should propagate")

        # The failed candidate restore changed live state to the candidate.
        # Recovery must issue a fresh incumbent restore despite that partial
        # mutation, then leave the adapter's identity marker in sync.
        assert trainer.restore_targets[-2:] == [1, 0]
        assert trainer.weight == 0
        assert trainer.optimizer_step == 0
        assert hooks._loaded_id == "policy-0"
        assert hooks.active_policy() == ("policy-0", 0)

        # A subsequent candidate evaluation remains possible after recovery.
        report = await hooks.evaluate(candidate)
        assert report.primary == 1.0
        assert trainer.weight == 0

    asyncio.run(run())
