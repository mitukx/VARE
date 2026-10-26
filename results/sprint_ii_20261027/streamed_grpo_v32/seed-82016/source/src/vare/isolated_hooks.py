"""Trusted evaluator substitution for existing VARE LoopHooks.

The wrapped learner never supplies promotion evaluation reports. A separate
Docker-isolated scorer reads operator-owned checkpoint bytes and frozen tasks.
The operator must still ensure the learner cannot access Docker control APIs.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Sequence

from .attestation import EvidenceTrustRoot
from .container_isolation import DockerIsolationRunner, IsolationLimits
from .interfaces import LoopHooks
from .lag import LagDecision
from .types import Attempt, EvaluationReport, Experience, Task


class IsolatedEvaluationHooks:
    """Reuse the learner unchanged; replace promotion evidence generation."""

    def __init__(
        self,
        underlying: LoopHooks,
        *,
        runner: DockerIsolationRunner,
        evaluator_limits: IsolationLimits,
        trust: EvidenceTrustRoot,
        checkpoint_store: Path,
        heldout_set: Path,
        key_file: Path,
    ) -> None:
        if not trust.strict_checkpoint_binding:
            raise ValueError("isolated evaluation requires strict checkpoint-bound attestation")
        self.underlying = underlying
        self.runner = runner
        self.evaluator_limits = evaluator_limits
        self.trust = trust
        self.checkpoint_store = checkpoint_store
        self.heldout_set = heldout_set
        self.key_file = key_file

    def active_policy(self) -> tuple[str, int]:
        return self.underlying.active_policy()

    async def rollout(self, task: Task, policy_id: str, policy_version: int, step: int) -> Attempt:
        return await self.underlying.rollout(task, policy_id, policy_version, step)

    async def train_candidate(self, incumbent_id: str, experiences: Sequence[Experience]) -> str:
        return await self.underlying.train_candidate(incumbent_id, experiences)

    async def train_candidate_with_freshness(
        self,
        incumbent_id: str,
        experiences: Sequence[Experience],
        freshness_assessor: Callable[[Experience], LagDecision],
    ) -> str:
        train_with_freshness = getattr(
            self.underlying, "train_candidate_with_freshness", None
        )
        if callable(train_with_freshness):
            return await train_with_freshness(
                incumbent_id,
                experiences,
                freshness_assessor,
            )
        return await self.underlying.train_candidate(incumbent_id, experiences)

    async def evaluate(self, policy_id: str) -> EvaluationReport:
        return await self.runner.run_evaluator(
            self.evaluator_limits,
            policy_id=policy_id,
            checkpoint_store=self.checkpoint_store,
            heldout_set=self.heldout_set,
            key_file=self.key_file,
            trust=self.trust,
        )

    async def promote(self, candidate_id: str) -> None:
        await self.underlying.promote(candidate_id)

    async def discard(self, candidate_id: str) -> None:
        await self.underlying.discard(candidate_id)
