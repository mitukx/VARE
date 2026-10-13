from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from .config import LagConfig
from .integrations.rvl import RVLReplaySnapshot, RVLTokenReplayReader
from .reward_evidence import RewardEvidenceManifest


@dataclass(frozen=True, slots=True)
class ReplayHealthConfig:
    max_pending_fraction: float = 0.40
    max_quarantined_fraction: float = 0.05
    require_ready_data: bool = True


@dataclass(frozen=True, slots=True)
class OuterControlDecision:
    freeze_training: bool
    throttle_rollouts: bool
    refresh_verifier: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "freeze_training": self.freeze_training,
            "throttle_rollouts": self.throttle_rollouts,
            "refresh_verifier": self.refresh_verifier,
            "reasons": list(self.reasons),
        }


class ReplayHealthGate:
    """Fail-closed outer-loop health gate for an RVL replay snapshot."""

    def __init__(self, lag: LagConfig, health: ReplayHealthConfig | None = None) -> None:
        self.lag = lag
        self.health = health or ReplayHealthConfig()

    def decide(self, snapshot: RVLReplaySnapshot) -> OuterControlDecision:
        reasons: list[str] = []
        total = max(1, snapshot.groups)
        pending = snapshot.statuses.get("pending_verification", 0) + snapshot.statuses.get("verifying", 0)
        quarantined = snapshot.statuses.get("quarantined", 0)
        ready = snapshot.statuses.get("ready", 0)
        refresh = False
        freeze = False
        throttle = False
        if snapshot.max_verifier_lag > self.lag.max_verifier_lag:
            reasons.append("verifier_lag_exceeded")
            refresh = True
            freeze = True
        if snapshot.max_policy_lag > self.lag.max_policy_lag:
            reasons.append("policy_lag_exceeded")
        if pending / total > self.health.max_pending_fraction:
            reasons.append("verification_backlog")
            throttle = True
        if quarantined / total > self.health.max_quarantined_fraction:
            reasons.append("verification_quarantine")
            refresh = True
            freeze = True
        if self.health.require_ready_data and ready == 0:
            reasons.append("no_ready_verified_data")
            freeze = True
        return OuterControlDecision(
            freeze_training=freeze,
            throttle_rollouts=throttle,
            refresh_verifier=refresh,
            reasons=tuple(reasons),
        )


@dataclass(frozen=True, slots=True)
class RVLOuterPlan:
    control: OuterControlDecision
    family_weights: dict[str, float]
    family_stats: dict[str, dict[str, float | int]]
    fresh_experiences: int
    excluded_stale_experiences: int
    excluded_untrusted_experiences: int = 0
    excluded_unattested_experiences: int = 0
    authenticated_curriculum_experiences: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "control": self.control.to_dict(),
            "family_weights": self.family_weights,
            "family_stats": self.family_stats,
            "fresh_experiences": self.fresh_experiences,
            "excluded_stale_experiences": self.excluded_stale_experiences,
            "excluded_untrusted_experiences": self.excluded_untrusted_experiences,
            "excluded_unattested_experiences": self.excluded_unattested_experiences,
            "authenticated_curriculum_experiences": self.authenticated_curriculum_experiences,
        }


class RVLOuterPlanner:
    """Convert RVL replay evidence into a safe next-round curriculum plan.

    Stale data can trigger systems interventions, but it does not increase a
    task family's curriculum weight. Only experiences inside both lag bounds
    contribute to capability-failure statistics.
    """

    def __init__(self, lag: LagConfig, health: ReplayHealthConfig | None = None) -> None:
        self.lag = lag
        self.health_gate = ReplayHealthGate(lag, health)

    def plan(
        self,
        reader: RVLTokenReplayReader,
        *,
        current_policy_version: int,
        current_verifier_version: int,
        trusted_default: bool = False,
        reward_manifest: RewardEvidenceManifest | None = None,
        require_authenticated_rewards: bool = False,
    ) -> RVLOuterPlan:
        if type(require_authenticated_rewards) is not bool:
            raise ValueError("require_authenticated_rewards must be bool")
        if require_authenticated_rewards and reward_manifest is None:
            raise ValueError("authenticated curriculum requires an operator-owned reward manifest")
        snapshot = reader.snapshot(
            current_policy_version=current_policy_version,
            current_verifier_version=current_verifier_version,
        )
        control = self.health_gate.decide(snapshot)
        experiences = reader.ready_experiences(
            current_policy_version=current_policy_version,
            current_verifier_version=current_verifier_version,
            trusted_default=trusted_default,
        )
        fresh = [
            e for e in experiences
            if e.policy_lag <= self.lag.max_policy_lag and e.verifier_lag <= self.lag.max_verifier_lag
        ]
        # A nonzero reward or reported failure is not evidence of reliable
        # task failure if its verifier provenance is untrusted. Keep these
        # rows for replay-health diagnostics, never curriculum amplification.
        trusted_fresh = [e for e in fresh if e.verification.trusted]
        eligible = (
            [e for e in trusted_fresh if reward_manifest.verified(e)]
            if reward_manifest is not None else trusted_fresh
        )
        if require_authenticated_rewards and not eligible:
            control = replace(
                control,
                freeze_training=True,
                reasons=(*control.reasons, "no_authenticated_reward_evidence"),
            )
        stats: dict[str, dict[str, float | int]] = {}
        for exp in eligible:
            family = exp.attempt.task.family
            row = stats.setdefault(family, {"count": 0, "failures": 0, "reward_sum": 0.0, "trusted": 0})
            row["count"] = int(row["count"]) + 1
            row["failures"] = int(row["failures"]) + int(not exp.verification.passed)
            row["reward_sum"] = float(row["reward_sum"]) + exp.verification.score
            row["trusted"] = int(row["trusted"]) + int(exp.verification.trusted)
        weights: dict[str, float] = {}
        normalized_stats: dict[str, dict[str, float | int]] = {}
        for family, raw in sorted(stats.items()):
            count = int(raw["count"])
            failures = int(raw["failures"])
            mean_reward = float(raw["reward_sum"]) / max(1, count)
            failure_rate = failures / max(1, count)
            trusted_fraction = int(raw["trusted"]) / max(1, count)
            # Sublinear count term prevents one high-volume family from taking over.
            weights[family] = 1.0 + failure_rate * math.log1p(count)
            normalized_stats[family] = {
                "count": count,
                "failures": failures,
                "failure_rate": failure_rate,
                "mean_reward": mean_reward,
                "trusted_fraction": trusted_fraction,
            }
        return RVLOuterPlan(
            control=control,
            family_weights=weights,
            family_stats=normalized_stats,
            fresh_experiences=len(fresh),
            excluded_stale_experiences=len(experiences) - len(fresh),
            excluded_untrusted_experiences=len(fresh) - len(trusted_fresh),
            excluded_unattested_experiences=len(trusted_fresh) - len(eligible),
            authenticated_curriculum_experiences=(
                len(eligible) if reward_manifest is not None else 0
            ),
        )
