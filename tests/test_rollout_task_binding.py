import asyncio

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


class AlwaysVerifier:
    name = "always"
    version = 0
    trusted = True

    def __init__(self):
        self.seen_tasks = []

    async def verify(self, attempt):
        self.seen_tasks.append(attempt.task.id)
        return Verification(1.0, True, 1.0, 0, self.name, trusted=True)


class Hooks:
    def __init__(self, altered_field=None):
        self.altered_field = altered_field
        self.trained_task_ids = []

    def active_policy(self):
        return "p0", 0

    async def rollout(self, task, policy_id, policy_version, step):
        values = {"id": task.id, "prompt": task.prompt, "family": task.family}
        if self.altered_field:
            values[self.altered_field] = {
                "id": "different-id",
                "prompt": "different prompt",
                "family": "different-family",
            }[self.altered_field]
        returned_task = Task(values["id"], values["prompt"], values["family"])
        return Attempt(returned_task, "response", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        self.trained_task_ids.extend(exp.attempt.task.id for exp in experiences)
        return incumbent_id

    async def evaluate(self, policy_id):
        return EvaluationReport(policy_id, 1.0, n=1)

    async def promote(self, candidate_id):
        raise AssertionError("same policy should not be promoted")

    async def discard(self, candidate_id):
        raise AssertionError("same policy should not be discarded")


def _loop(hooks, verifier):
    return CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(verifier)]),
        config=EngineConfig(
            replay_batch_size=1,
            promotion=PromotionConfig(min_eval_examples=1),
        ),
    )


@pytest.mark.parametrize("altered_field", ["id", "prompt", "family"])
def test_rollout_for_a_different_task_is_rejected_before_verification_or_training(altered_field):
    hooks = Hooks(altered_field=altered_field)
    verifier = AlwaysVerifier()
    loop = _loop(hooks, verifier)

    with pytest.raises(ValueError, match="rollout task identity mismatch"):
        asyncio.run(loop.run_round([Task("task-a", "prompt a", "math")], round_index=0))

    assert verifier.seen_tasks == []
    assert hooks.trained_task_ids == []
    assert len(loop.replay) == 0


def test_rollout_for_the_dispatched_task_remains_admissible():
    hooks = Hooks()
    verifier = AlwaysVerifier()
    loop = _loop(hooks, verifier)

    result = asyncio.run(loop.run_round([Task("task-a", "prompt a", "math")], round_index=0))

    assert verifier.seen_tasks == ["task-a"]
    assert result.admitted_experiences == 1
    assert hooks.trained_task_ids == ["task-a"]
