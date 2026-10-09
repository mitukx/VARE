import asyncio

import pytest

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


class AlwaysVerifier:
    name = "always"
    version = 1
    trusted = True

    async def verify(self, attempt):
        await asyncio.sleep(0)
        return Verification(1.0, True, 1.0, 1, self.name, trusted=True)


class Hooks:
    def __init__(self):
        self.steps = []
        self.training_called = False

    def active_policy(self):
        return "p0", 0

    async def rollout(self, task, policy_id, policy_version, step):
        await asyncio.sleep(0.001)
        self.steps.append(step)
        return Attempt(task, "ok", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        self.training_called = True
        return incumbent_id

    async def evaluate(self, policy_id):
        return EvaluationReport(policy_id, 1.0, n=100)

    async def promote(self, candidate_id):
        raise AssertionError("same policy should not be promoted")

    async def discard(self, candidate_id):
        raise AssertionError("same policy should not be discarded")


def test_concurrent_rollouts_have_unique_provenance_steps():
    hooks = Hooks()
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(AlwaysVerifier())]),
        config=EngineConfig(
            rollout_concurrency=16,
            replay_batch_size=4,
            promotion=PromotionConfig(min_eval_examples=1),
        ),
        seed=0,
    )
    tasks = [Task(str(i), "x") for i in range(32)]
    asyncio.run(loop.run_round(tasks, round_index=0, rollout_count=32))
    assert sorted(hooks.steps) == list(range(32))


def test_engine_revalidates_custom_verifier_result_before_replay_or_training():
    class BypassingVerifier:
        @property
        def active_version(self):
            return 1

        async def verify(self, attempt):
            result = Verification(0.75, True, 1.0, 1, "custom", trusted=True)
            result.score = 2.0
            return result

    hooks = Hooks()
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=BypassingVerifier(),
        config=EngineConfig(replay_batch_size=1),
        seed=0,
    )
    with pytest.raises(ValueError, match="score"):
        asyncio.run(loop.run_round([Task("t", "prompt")], round_index=0))
    assert not hooks.training_called
    assert len(loop.replay) == 0


def test_engine_rejects_evaluation_report_for_wrong_policy_identity():
    class CandidateHooks(Hooks):
        def __init__(self):
            super().__init__()
            self.promoted = []
            self.discarded = []

        async def train_candidate(self, incumbent_id, experiences):
            self.training_called = True
            return "p1"

        async def evaluate(self, policy_id):
            report_id = "p0" if policy_id == "p1" else policy_id
            return EvaluationReport(report_id, 0.9 if policy_id == "p1" else 0.5, n=10)

        async def promote(self, candidate_id):
            self.promoted.append(candidate_id)

        async def discard(self, candidate_id):
            self.discarded.append(candidate_id)

    hooks = CandidateHooks()
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(AlwaysVerifier())]),
        config=EngineConfig(
            replay_batch_size=1,
            promotion=PromotionConfig(min_primary_gain=0.01, min_eval_examples=1),
        ),
        seed=0,
    )
    result = asyncio.run(loop.run_round([Task("t", "prompt")], round_index=0))

    assert not result.decision.accepted
    assert result.decision.reasons == ("evaluation_policy_mismatch",)
    assert hooks.promoted == []
    assert hooks.discarded == ["p1"]
