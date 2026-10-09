import asyncio
from dataclasses import dataclass

import pytest

from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.types import Experience, Task, Verification


@dataclass
class Generation:
    prompt_id: str
    prompt: str
    response: str
    logprob: float
    token_count: int
    latency_s: float
    metadata: dict


@dataclass
class VerifiedGeneration:
    generation: Generation
    reward: float
    verifier_latency_s: float
    verifier_version: int
    metadata: dict


class Trainer:
    def __init__(self):
        self.step_calls = 0
        self.samples = []

    def snapshot_training_state(self):
        return {"step_calls": self.step_calls}

    def restore_training_state(self, state):
        self.step_calls = state["step_calls"]

    def train_step(self, samples):
        self.step_calls += 1
        self.samples = samples
        return {"loss": 0.0}


class Backend:
    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert (prompt_id, prompt) == ("task-a", "prompt a")
        return [Generation(
            prompt_id="task-b",
            prompt="prompt b",
            response="answer",
            logprob=-0.1,
            token_count=2,
            latency_s=0.001,
            metadata={
                "sampling_temperature": temperature,
                "prompt_token_ids": [202],
                "response_token_ids": [7],
                "response_token_logprobs": [-0.1],
            },
        )]


def _hooks(trainer):
    return RVLGRPOHooks(
        backend=Backend(),
        trainer=trainer,
        eval_tasks=[],
        score_fn=lambda task, response: float(task.id == "task-a" and response == "answer"),
        generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )


def test_mismatched_generation_never_reaches_grpo_train_step():
    task = Task("task-a", "prompt a", "fixture")
    trainer = Trainer()
    hooks = _hooks(trainer)

    async def run():
        with pytest.raises(ValueError, match="RVL generation task identity mismatch"):
            attempt = await hooks.rollout(task, "policy-0", 0, 1)
            reward = hooks.score_fn(attempt.task, attempt.output)
            experience = Experience(
                attempt=attempt,
                verification=Verification(reward, reward >= 0.5, 1.0, 0, "fixture", trusted=True),
                policy_lag=0,
                verifier_lag=0,
                shift_score=0.0,
            )
            await hooks.train_candidate("policy-0", [experience])

    asyncio.run(run())
    assert trainer.step_calls == 0
    assert trainer.samples == []
