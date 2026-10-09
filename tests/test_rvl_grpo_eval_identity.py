import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

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

    def restore_training_state(self, state):
        pass


class Backend:
    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        assert n == 1
        response = {"first prompt": "wrong", "second prompt": "right"}[prompt]
        return [Generation(prompt_id=prompt_id, prompt=prompt, response=response)]


def _hooks(tasks):
    return RVLGRPOHooks(
        backend=Backend(),
        trainer=Trainer(),
        eval_tasks=tasks,
        score_fn=lambda task, response: float(response == "right"),
        verified_generation_factory=lambda **kwargs: SimpleNamespace(**kwargs),
        generation_factory=Generation,
    )


def test_rvl_evaluator_rejects_duplicate_task_ids_before_scoring():
    tasks = [
        Task(id="duplicate", prompt="first prompt", family="qa"),
        Task(id="duplicate", prompt="second prompt", family="qa"),
    ]

    with pytest.raises(ValueError, match="evaluation task IDs must be unique"):
        _hooks(tasks)


def test_rvl_evaluator_keeps_distinct_tasks_in_primary_and_paired_scores():
    tasks = [
        Task(id="task-0", prompt="first prompt", family="qa"),
        Task(id="task-1", prompt="second prompt", family="qa"),
    ]
    report = asyncio.run(_hooks(tasks).evaluate("policy-0"))

    assert report.n == 2
    assert report.primary == 0.5
    assert report.metadata["per_task_scores"] == {"task-0": 0.0, "task-1": 1.0}
