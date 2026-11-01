from __future__ import annotations

import asyncio
from pathlib import Path

from ..types import Attempt, Verification
from .runner import ExecutableEvaluator
from .spec import EngineeringTaskSpec


class ExecutableEnvironmentVerifier:
    """Adapter from repository evaluator results to VARE Verification objects.

    The rollout/agent must place an isolated candidate workspace path in
    `attempt.metadata["workspace_path"]`. Evaluation commands come from a trusted
    task specification, not from model output.
    """

    name = "executable-environment"
    trusted = True

    def __init__(self, spec: EngineeringTaskSpec, *, version: int = 0, baseline_metrics: dict[str, float] | None = None) -> None:
        self.spec = spec
        self.version = version
        self.baseline_metrics = dict(baseline_metrics or {})
        self.evaluator = ExecutableEvaluator(spec)

    async def verify(self, attempt: Attempt) -> Verification:
        workspace = attempt.metadata.get("workspace_path")
        if not workspace:
            raise ValueError("attempt is missing workspace_path")
        result = await asyncio.to_thread(
            self.evaluator.evaluate,
            Path(str(workspace)),
            baseline_metrics=self.baseline_metrics,
            provenance={"policy_id": attempt.policy_id, "policy_version": attempt.policy_version},
        )
        return Verification(
            score=result.score,
            passed=result.passed,
            confidence=1.0,
            verifier_version=self.version,
            verifier_name=self.name,
            trusted=True,
            metadata={"environment_result": result.to_dict()},
        )
