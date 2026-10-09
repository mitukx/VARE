import asyncio
from dataclasses import dataclass

import pytest

from vare.integrations.rvl_grpo import RVLGRPOHooks
from vare.types import Task


@dataclass
class Generation:
    prompt_id: str
    prompt: str
    response: str
    logprob: float = -0.1
    token_count: int = 1
    latency_s: float = 0.001
    metadata: dict | None = None


class Trainer:
    def snapshot_training_state(self):
        return {"weight": 0}


class Backend:
    def __init__(self, prompt_id, prompt):
        self.prompt_id = prompt_id
        self.prompt = prompt

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert n == 1
        return [Generation(self.prompt_id, self.prompt, "answer", metadata={})]


def _hooks(prompt_id, prompt):
    return RVLGRPOHooks(
        backend=Backend(prompt_id, prompt),
        trainer=Trainer(),
        eval_tasks=[],
        score_fn=lambda task, response: 1.0,
        generation_factory=Generation,
        verified_generation_factory=lambda **kwargs: kwargs,
    )


@pytest.mark.parametrize(
    ("returned_id", "returned_prompt"),
    [("task-b", "prompt a"), ("task-a", "prompt b")],
)
def test_rvl_generation_for_a_different_task_is_rejected(returned_id, returned_prompt):
    task = Task("task-a", "prompt a", "math")
    hooks = _hooks(returned_id, returned_prompt)

    with pytest.raises(ValueError, match="RVL generation task identity mismatch"):
        asyncio.run(hooks.rollout(task, "policy-0", 0, 1))


def test_rvl_generation_for_dispatched_task_remains_valid():
    task = Task("task-a", "prompt a", "math")
    hooks = _hooks(task.id, task.prompt)

    attempt = asyncio.run(hooks.rollout(task, "policy-0", 0, 1))

    assert attempt.task is task
    assert attempt.metadata["rvl_generation"]["prompt_id"] == task.id
    assert attempt.metadata["rvl_generation"]["prompt"] == task.prompt
