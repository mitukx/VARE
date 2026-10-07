import asyncio

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

    def active_policy(self):
        return "p0", 0

    async def rollout(self, task, policy_id, policy_version, step):
        await asyncio.sleep(0.001)
        self.steps.append(step)
        return Attempt(task, "ok", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
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
