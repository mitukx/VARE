from __future__ import annotations

from dataclasses import dataclass, field
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
