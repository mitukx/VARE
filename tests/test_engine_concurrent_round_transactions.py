import asyncio

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


class _Verifier:
    name = "fixture"
    version = 1
    trusted = True

    async def verify(self, attempt):
        return Verification(1.0, True, 1.0, 1, self.name, trusted=True)


class _VersionedHooks:
    def __init__(self):
        self.active_id = "p0"
        self.state = {"p0": 0.0}
        self.parents = {}
        self._training_count = 0
        self._second_candidate_trained = asyncio.Event()
        self._first_candidate_promoted = asyncio.Event()

    def active_policy(self):
        return self.active_id, len(self.state) - 1

    async def rollout(self, task, policy_id, policy_version, step):
        return Attempt(task, "response", policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id, experiences):
        self._training_count += 1
        candidate_id = f"c{self._training_count}"
        self.parents[candidate_id] = incumbent_id
        self.state[candidate_id] = self.state[incumbent_id] + 0.1
        if self._training_count == 2:
            self._second_candidate_trained.set()
        return candidate_id

    async def evaluate(self, policy_id):
        parent_id = self.parents.get(policy_id)
        if parent_id == "p0" and policy_id == "c1":
            # Let an overlapping round finish training before this candidate's
            # evaluation returns. A serialized implementation simply times out.
            try:
                await asyncio.wait_for(self._second_candidate_trained.wait(), timeout=0.05)
            except TimeoutError:
                pass
        elif parent_id == "p0" and policy_id == "c2":
            await self._first_candidate_promoted.wait()
        return EvaluationReport(policy_id, self.state[policy_id], n=1)

    async def promote(self, candidate_id):
        self.active_id = candidate_id
        if candidate_id == "c1":
            self._first_candidate_promoted.set()

    async def discard(self, candidate_id):
        self.state.pop(candidate_id, None)
        self.parents.pop(candidate_id, None)


def test_overlapping_rounds_do_not_promote_a_stale_branch_over_newer_incumbent():
    hooks = _VersionedHooks()
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(_Verifier())]),
        config=EngineConfig(
            rollout_concurrency=2,
            replay_batch_size=2,
            samples_per_task=1,
            promotion=PromotionConfig(min_primary_gain=0.01, min_eval_examples=1),
        ),
    )

    async def run_overlapping_rounds():
        return await asyncio.gather(
            loop.run_round([Task("task-a", "prompt")], round_index=0, rollout_count=1),
            loop.run_round([Task("task-b", "prompt")], round_index=1, rollout_count=1),
        )

    results = asyncio.run(run_overlapping_rounds())

    assert len(results) == 2
    assert hooks.parents["c2"] == "c1"
    assert hooks.active_id == "c2"
    assert hooks.state[hooks.active_id] == 0.2
