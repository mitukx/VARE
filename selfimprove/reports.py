"""Map a dev measurement onto the existing promotion gate.

Dev has 32 tasks. ``PromotionConfig.min_eval_examples`` defaults to 64, so
this loop passes 32. ``PromotionGate`` itself is unchanged. Confirm tasks are
not added to the report.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport

DEV_COUNT = 32


@dataclass(frozen=True, slots=True)
class DevReport:
    policy_id: str
    scores: dict[str, float]
    format_compliance: float
    false_accept_rate: float
    format_reward: float
    train_reward: float
    cost: float = 1.0

    def primary(self) -> float:
        if len(self.scores) != DEV_COUNT:
            raise ValueError(f"dev report must contain {DEV_COUNT} scores")
        return sum(self.scores.values()) / DEV_COUNT

    def with_rewards(self, train_reward: float, format_reward: float) -> "DevReport":
        return replace(self, train_reward=train_reward, format_reward=format_reward)


def dev_gate() -> PromotionGate:
    return PromotionGate(
        PromotionConfig(
            min_primary_gain=0.01,
            max_slice_regression=0.03,
            max_relative_cost_increase=0.20,
            max_verifier_disagreement=0.20,
            min_eval_examples=DEV_COUNT,
            paired_confidence_gate=False,
            min_paired_examples=DEV_COUNT,
        )
    )


def to_evaluation_report(report: DevReport) -> EvaluationReport:
    primary = report.primary()
    return EvaluationReport(
        policy_id=report.policy_id,
        primary=primary,
        slices={"dev_strict": primary},
        cost=report.cost,
        verifier_disagreement=report.false_accept_rate,
        n=DEV_COUNT,
        metadata={
            "format_compliance": report.format_compliance,
            "per_task_scores": dict(report.scores),
        },
    )


def monitor_signals(before: DevReport, after: DevReport) -> tuple[str, ...]:
    signals: list[str] = []
    if after.format_reward > before.format_reward + 1e-12 and after.primary() <= before.primary() + 1e-12:
        signals.append("format_only_reward_rise")
    if after.false_accept_rate > 0.20:
        signals.append("self_verifier_disagreement")
    if after.train_reward > before.train_reward + 1e-12 and after.primary() + 1e-12 < before.primary():
        signals.append("heldout_divergence")
    return tuple(signals)
