from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    prompt: str
    family: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Attempt:
    task: Task
    output: str
    policy_id: str
    policy_version: int
    created_step: int
    logprob: float | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Verification:
    score: float
    passed: bool
    confidence: float
    verifier_version: int
    verifier_name: str
    disagreement: float = 0.0
    trusted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def snapshot_verification(value: Verification, *, context: str = "verification") -> Verification:
    """Validate an untrusted verifier result and detach its top-level mutable state."""
    if not isinstance(value, Verification):
        raise TypeError(f"{context} must be a Verification instance")
    for field_name in ("score", "confidence", "disagreement"):
        number = getattr(value, field_name)
        if isinstance(number, bool) or not isinstance(number, Real):
            raise TypeError(f"{context}.{field_name} must be a real number")
        if not math.isfinite(float(number)) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{context}.{field_name} must be finite and in [0,1]")
    if type(value.passed) is not bool:
        raise TypeError(f"{context}.passed must be bool")
    if type(value.trusted) is not bool:
        raise TypeError(f"{context}.trusted must be bool")
    if type(value.verifier_version) is not int or value.verifier_version < 0:
        raise ValueError(f"{context}.verifier_version must be a nonnegative integer")
    if not isinstance(value.verifier_name, str) or not value.verifier_name.strip():
        raise ValueError(f"{context}.verifier_name must be a nonempty string")
    if not isinstance(value.metadata, dict):
        raise TypeError(f"{context}.metadata must be a dictionary")
    return Verification(
        score=float(value.score),
        passed=value.passed,
        confidence=float(value.confidence),
        verifier_version=value.verifier_version,
        verifier_name=value.verifier_name,
        disagreement=float(value.disagreement),
        trusted=value.trusted,
        metadata=dict(value.metadata),
    )


@dataclass(slots=True)
class Experience:
    attempt: Attempt
    verification: Verification
    policy_lag: int
    verifier_lag: int
    shift_score: float
    priority: float = 1.0
    failure_label: str | None = None


@dataclass(frozen=True, slots=True)
class FailureCluster:
    label: str
    count: int
    task_families: tuple[str, ...]
    mean_score: float


@dataclass(slots=True)
class EvaluationReport:
    policy_id: str
    primary: float
    slices: dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
    verifier_disagreement: float = 0.0
    n: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    accepted: bool
    reasons: tuple[str, ...]
    primary_gain: float
    worst_slice_regression: float
    paired_gain: float | None = None
    paired_lcb: float | None = None
    paired_n: int = 0


@dataclass(slots=True)
class RoundResult:
    round_index: int
    incumbent_id: str
    candidate_id: str
    promoted_id: str
    decision: PromotionDecision
    incumbent_eval: EvaluationReport
    candidate_eval: EvaluationReport
    failures: tuple[FailureCluster, ...]
    admitted_experiences: int
    dropped_experiences: int
