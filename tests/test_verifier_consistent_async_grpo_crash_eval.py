from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.benchmark_verifier_consistent_grpo_sigkill import independent_eval
from vare.types import Task


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids):
        logits = torch.zeros((*input_ids.shape, 8)) + self.weight
        return SimpleNamespace(logits=logits)


class _Tokenizer:
    eos_token_id = 7

    def encode(self, _prompt):
        return [1]

    def decode(self, _tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "ground."


def test_crash_recovery_evaluator_scores_every_preregistered_task():
    model = _Model()
    tasks = [Task(f"heldout-{index}", "prompt", "landing") for index in range(4)]

    report, detail = independent_eval(
        model, _Tokenizer(), {"model": model.state_dict()}, tasks, version=1
    )

    assert report.n == 4
    assert report.primary == 1.0
    assert report.metadata["per_task_scores"] == {task.id: 1.0 for task in tasks}
    assert set(detail["outputs"]) == {task.id for task in tasks}
