"""Held-out proxy-verifier false-acceptance audit for post-training promotion.

This is an *audit of labels*, not proof of reward-model calibration or a
time-uniform guarantee under adaptive/repeated evaluation. Observations must
come from independent held-out tasks scored by a trusted reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RewardAuditObservation:
    task_id: str
    family: str
    proxy_pass: bool
    trusted_pass: bool


@dataclass(frozen=True, slots=True)
class RewardAuditSlice:
    proxy_positives: int
    false_accepts: int
    empirical_false_accept_rate: float | None
    upper_bound: float | None


@dataclass(frozen=True, slots=True)
class RewardAuditResult:
    accepted: bool
    reasons: tuple[str, ...]
    overall: RewardAuditSlice
    by_family: dict[str, RewardAuditSlice]


def _slice(rows: Sequence[RewardAuditObservation], delta: float) -> RewardAuditSlice:
    positive = [r for r in rows if r.proxy_pass]
    n = len(positive)
    k = sum(not r.trusted_pass for r in positive)
    if n == 0:
        return RewardAuditSlice(0, 0, None, None)
    rate = k / n
    # One-sided Hoeffding bound for i.i.d. held-out proxy-positive examples.
    # Bonferroni below protects the overall and each observed family jointly.
    upper = min(1.0, rate + math.sqrt(math.log(1.0 / delta) / (2 * n)))
    return RewardAuditSlice(n, k, rate, upper)


def audit_reward_labels(
    observations: Sequence[RewardAuditObservation],
    *,
    max_false_accept_ucb: float = 0.25,
    min_proxy_positives: int = 32,
    alpha: float = 0.05,
) -> RewardAuditResult:
    """Bound P(trusted_fail | proxy_pass), overall and in each task family.

    The statistical bound assumes independent, representative examples from a
    fixed evaluation distribution and a frozen proxy. Repeated/adaptive looks
    require additional correction. This cannot attest evaluator independence.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if not 0 <= max_false_accept_ucb <= 1:
        raise ValueError("max_false_accept_ucb must be in [0,1]")
    if type(min_proxy_positives) is not int or min_proxy_positives < 1:
        raise ValueError("min_proxy_positives must be a positive integer")
    if not observations:
        raise ValueError("audit requires observations")
    seen: set[str] = set()
    groups: dict[str, list[RewardAuditObservation]] = defaultdict(list)
    for row in observations:
        if (
            not isinstance(row, RewardAuditObservation)
            or not isinstance(row.task_id, str) or not row.task_id
            or not isinstance(row.family, str) or not row.family
            or type(row.proxy_pass) is not bool
            or type(row.trusted_pass) is not bool
            or row.task_id in seen
        ):
            raise ValueError("invalid or duplicate reward audit observation")
        seen.add(row.task_id)
        groups[row.family].append(row)
    delta = alpha / (len(groups) + 1)
    overall = _slice(observations, delta)
    by_family = {k: _slice(v, delta) for k, v in sorted(groups.items())}
    reasons: list[str] = []
    for label, s in [("overall", overall), *by_family.items()]:
        if s.proxy_positives < min_proxy_positives:
            reasons.append(f"insufficient_proxy_positives:{label}")
        elif s.upper_bound is not None and s.upper_bound > max_false_accept_ucb:
            reasons.append(f"verifier_false_accept_risk:{label}")
    return RewardAuditResult(not reasons, tuple(reasons), overall, by_family)
