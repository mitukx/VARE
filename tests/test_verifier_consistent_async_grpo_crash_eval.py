from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.benchmark_verifier_consistent_grpo_sigkill import independent_eval
from scripts.benchmark_verifier_consistent_grpo_sigkill import replay_ordered_experiences
from vare.types import Attempt, Experience, Task, Verification


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


def test_crash_recovery_preserves_frozen_priority_order_for_grpo_group():
    scores = [1, 1, 0, 0, 1, 0, 0, 0]
    experiences = []
    for member, score in enumerate(scores):
        task = Task(
            "task", "prompt", "landing",
            {"learning_rule": "terminal_action_on_policy_grpo_v1"},
        )
        attempt = Attempt(
            task, "", "policy-0", 0, member,
            metadata={
                "vare_rollout_group": "g-01-fresh",
                "vare_rollout_group_size": len(scores),
                "member_index": member,
            },
        )
        verification = Verification(
            score=float(score), passed=bool(score), confidence=1.0,
            verifier_version=1, verifier_name="landing", trusted=True,
            metadata={"verifier_config_sha256": "a" * 64},
        )
        experiences.append(Experience(attempt, verification, 0, 0, 0.0))

    ordered = replay_ordered_experiences(experiences, seed=82106)

    assert [item.attempt.metadata["member_index"] for item in ordered] == [
        2, 3, 5, 6, 7, 0, 1, 4,
    ]
