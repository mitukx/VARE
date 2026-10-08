from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable

from .config import LagConfig


@dataclass(frozen=True, slots=True)
class LagDecision:
    admitted: bool
    freshness: float
    policy_lag: int
    verifier_lag: int
    shift_score: float
    reasons: tuple[str, ...]


class DistributionShiftMonitor:
    """Tracks total-variation shift in task-family distributions."""

    def __init__(self, window: int = 256) -> None:
        self._reference: Counter[str] = Counter()
        self._recent: deque[str] = deque(maxlen=window)

    def fit_reference(self, families: Iterable[str]) -> None:
        self._reference = Counter(families)

    def observe(self, family: str) -> None:
        self._recent.append(family)

    @staticmethod
    def _tv(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        sa, sb = sum(a.values()), sum(b.values())
        keys = set(a) | set(b)
        return 0.5 * sum(abs(a[k] / sa - b[k] / sb) for k in keys)

    def score(self) -> float:
        return self._tv(self._reference, Counter(self._recent))


class LagController:
    def __init__(self, config: LagConfig) -> None:
        self.config = config

    def assess(
        self,
        *,
        rollout_policy_version: int,
        active_policy_version: int,
        reward_verifier_version: int,
        active_verifier_version: int,
        shift_score: float,
    ) -> LagDecision:
        policy_valid = all(
            type(version) is int and version >= 0
            for version in (rollout_policy_version, active_policy_version)
        )
        verifier_valid = all(
            type(version) is int and version >= 0
            for version in (reward_verifier_version, active_verifier_version)
        )
        policy_lag = (
            max(0, active_policy_version - rollout_policy_version) if policy_valid else 0
        )
        verifier_lag = (
            max(0, active_verifier_version - reward_verifier_version) if verifier_valid else 0
        )
        reasons: list[str] = []
        if not policy_valid:
            reasons.append("invalid_policy_version")
        elif rollout_policy_version > active_policy_version:
            reasons.append("policy_version_ahead")
        if not verifier_valid:
            reasons.append("invalid_verifier_version")
        elif reward_verifier_version > active_verifier_version:
            reasons.append("verifier_version_ahead")
        if policy_lag > self.config.max_policy_lag:
            reasons.append("policy_stale")
        if verifier_lag > self.config.max_verifier_lag:
            reasons.append("verifier_stale")
        if shift_score > self.config.max_shift:
            reasons.append("distribution_shift")
        freshness = math.exp(
            -self.config.policy_decay * policy_lag
            -self.config.verifier_decay * verifier_lag
            -self.config.shift_decay * max(0.0, shift_score)
        )
        return LagDecision(
            admitted=not reasons,
            freshness=max(0.0, min(1.0, freshness)),
            policy_lag=policy_lag,
            verifier_lag=verifier_lag,
            shift_score=shift_score,
            reasons=tuple(reasons),
        )
