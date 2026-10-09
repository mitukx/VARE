"""Frozen CPU source-path reproduction for protocol v1.

Run this against VARE baseline commit
`ccb3f791f212e80197fddea4048d3491c3458900`, with that worktree's `src/` on
`PYTHONPATH` and Python 3.12. This intentionally records baseline behavior;
post-fix invariants are covered by pytest.
"""
from __future__ import annotations

import asyncio
import json

from vare.config import EngineConfig, LagConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.types import Attempt, EvaluationReport, Task, Verification
from vare.verifiers import VerifierEnsemble, VerifierMember


class _VersionedVerifier:
    name = "fixture"
    version = 1
    trusted = True

    async def verify(self, attempt: Attempt) -> Verification:
        version = int(attempt.metadata["fixture_verifier_version"])
        reward = float(attempt.metadata["fixture_reward"])
        return Verification(reward, reward >= 0.5, 1.0, version, self.name, trusted=True)


class _Hooks:
    def __init__(self, policy_versions: list[int], verifier_versions: list[int]):
        self.policy_versions = policy_versions
        self.verifier_versions = verifier_versions
        self.train_rows: list[dict[str, object]] | None = None

    def active_policy(self) -> tuple[str, int]:
        return "policy-current", 1

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        return Attempt(
            task=task,
            output=f"sample-{step}",
            policy_id=policy_id,
            policy_version=self.policy_versions[step],
            created_step=step,
            metadata={
                "fixture_verifier_version": self.verifier_versions[step],
                "fixture_reward": float(step),
            },
        )

    async def train_candidate(self, incumbent_id: str, experiences):
        self.train_rows = [
            {
                "policy_id": exp.attempt.policy_id,
                "policy_version": exp.attempt.policy_version,
                "verifier_version": exp.verification.verifier_version,
            }
            for exp in experiences
        ]
        return incumbent_id

    async def evaluate(self, policy_id: str) -> EvaluationReport:
        return EvaluationReport(policy_id, 0.0, n=64)

    async def promote(self, candidate_id: str) -> None:
        raise AssertionError("fixture does not create a candidate")

    async def discard(self, candidate_id: str) -> None:
        raise AssertionError("fixture does not create a candidate")


async def _run(policy_versions: list[int], verifier_versions: list[int]) -> dict[str, object]:
    hooks = _Hooks(policy_versions, verifier_versions)
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=VerifierEnsemble([VerifierMember(_VersionedVerifier())]),
        config=EngineConfig(
            rollout_concurrency=2,
            replay_batch_size=2,
            samples_per_task=2,
            lag=LagConfig(max_policy_lag=2, max_verifier_lag=2),
            promotion=PromotionConfig(min_eval_examples=1),
        ),
        seed=0,
    )
    result = await loop.run_round([Task("audit-task", "same prompt")], round_index=0)
    return {
        "requested_policy": ["policy-current", 1],
        "attempt_policy_versions": policy_versions,
        "attempt_verifier_versions": verifier_versions,
        "admitted_experiences": result.admitted_experiences,
        "train_hook_called": hooks.train_rows is not None,
        "train_rows": hooks.train_rows,
    }


async def main() -> None:
    cases = {
        "mixed_policy": await _run([1, 0], [1, 1]),
        "homogeneous_current": await _run([1, 1], [1, 1]),
        "homogeneous_permitted_stale": await _run([0, 0], [1, 1]),
        "future_policy": await _run([2, 2], [1, 1]),
        "mixed_verifier": await _run([1, 1], [1, 0]),
    }
    print(json.dumps(cases, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
