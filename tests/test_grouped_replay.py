import asyncio
from collections import Counter

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


class Verifier:
    name = "v"
    version = 1
    trusted = True
    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 1, "v", trusted=True)


class Hooks:
    def __init__(self):
        self.groups = []
        self.train_group_counts = None
    def active_policy(self): return "p", 0
    async def rollout(self, task, policy_id, policy_version, step):
        self.groups.append(task.metadata["vare_rollout_group"])
        return Attempt(task, "x", policy_id, policy_version, step)
    async def train_candidate(self, incumbent_id, experiences):
        self.train_group_counts = Counter(e.attempt.metadata["vare_rollout_group"] for e in experiences)
        return incumbent_id
    async def evaluate(self, policy_id): return EvaluationReport(policy_id, 1.0, n=100)
    async def promote(self, candidate_id): pass
    async def discard(self, candidate_id): pass


def test_rollout_and_replay_preserve_grpo_groups_for_nonmultiple_budget():
    hooks = Hooks()
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(Verifier())]),
        config=EngineConfig(
            rollout_concurrency=8,
            replay_batch_size=7,
            samples_per_task=4,
            preserve_rollout_groups=True,
            promotion=PromotionConfig(min_eval_examples=1),
        ),
        seed=4,
    )
    tasks = [Task(str(i), "x") for i in range(8)]
    # Six requested rollouts require two complete groups of four. The engine
    # rounds the target up instead of sending a partial comparison group.
    asyncio.run(loop.run_round(tasks, round_index=0, rollout_count=6))
    rollout_counts = Counter(hooks.groups)
    assert sum(rollout_counts.values()) == 8
    assert set(rollout_counts.values()) == {4}
    assert hooks.train_group_counts is not None
    # Budget is 7, but whole groups of 4 are returned; no group is truncated.
    assert set(hooks.train_group_counts.values()) == {4}
    assert sum(hooks.train_group_counts.values()) == 8


@pytest.mark.parametrize("samples_per_task", [1, 4])
def test_rollout_group_ids_are_unique_when_round_index_is_reused(samples_per_task):
    hooks = Hooks()
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(Verifier())]),
        config=EngineConfig(
            rollout_concurrency=8,
            replay_batch_size=8,
            samples_per_task=samples_per_task,
            preserve_rollout_groups=True,
            promotion=PromotionConfig(min_eval_examples=1),
        ),
        seed=4,
    )
    tasks = [Task("task", "x")]

    asyncio.run(loop.run_round(tasks, round_index=3, rollout_count=1))
    first_run_groups = set(hooks.groups)
    hooks.groups.clear()

    asyncio.run(loop.run_round(tasks, round_index=3, rollout_count=1))
    second_run_groups = set(hooks.groups)

    assert first_run_groups
    assert second_run_groups
    assert first_run_groups.isdisjoint(second_run_groups)
