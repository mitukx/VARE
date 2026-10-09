from __future__ import annotations

import asyncio

from vare.config import EngineConfig, LagConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


class _Verifier:
    name = "fixture"
    version = 1
    trusted = True

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 1, self.name, trusted=True)


class _Hooks:
    def __init__(self, versions: list[int]):
        self.versions = versions
        self.trained_rows = None

    def active_policy(self):
        return "policy-current", 1

    async def rollout(self, task, policy_id, policy_version, step):
        return Attempt(task, f"sample-{step}", policy_id, self.versions[step], step)

    async def train_candidate(self, incumbent_id, experiences):
        self.trained_rows = [
            (exp.attempt.policy_id, exp.attempt.policy_version)
            for exp in experiences
        ]
        return incumbent_id

    async def evaluate(self, policy_id):
        return EvaluationReport(policy_id, 0.0, n=64)

    async def promote(self, candidate_id):
        raise AssertionError("no candidate is created")

    async def discard(self, candidate_id):
        raise AssertionError("no candidate is created")


def _run(versions):
    hooks = _Hooks(versions)
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(_Verifier())]),
        config=EngineConfig(
            rollout_concurrency=2,
            replay_batch_size=2,
            samples_per_task=2,
            lag=LagConfig(max_policy_lag=2, max_verifier_lag=2),
            promotion=PromotionConfig(min_eval_examples=1),
        ),
    )
    result = asyncio.run(loop.run_round([Task("task", "prompt")], round_index=0))
    return result, hooks


def test_engine_rejects_policy_heterogeneity_within_declared_grpo_group():
    result, hooks = _run([1, 0])

    # Rows pass individual lag admission, but the complete group is rejected
    # before replay and the learner hook is never called.
    assert result.admitted_experiences == 2
    assert result.dropped_experiences == 0
    assert hooks.trained_rows is None


def test_engine_retains_homogeneous_current_and_permitted_stale_groups():
    current_result, current_hooks = _run([1, 1])
    stale_result, stale_hooks = _run([0, 0])

    assert current_result.admitted_experiences == 2
    assert current_hooks.trained_rows == [
        ("policy-current", 1),
        ("policy-current", 1),
    ]
    assert stale_result.admitted_experiences == 2
    assert stale_hooks.trained_rows == [
        ("policy-current", 0),
        ("policy-current", 0),
    ]


def test_engine_continues_to_reject_ahead_of_incumbent_groups():
    result, hooks = _run([2, 2])

    assert result.admitted_experiences == 0
    assert result.dropped_experiences == 2
    assert hooks.trained_rows is None
