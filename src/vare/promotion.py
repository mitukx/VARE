from __future__ import annotations

import math
import random
from numbers import Real
from statistics import fmean
from typing import Mapping

from .config import PromotionConfig
from .reward_audit import RewardAuditObservation, audit_reward_labels
from .types import EvaluationReport, PromotionDecision


class PromotionGate:
    """Fail-closed champion/challenger gate using trusted held-out evaluation.

    Evaluation backends may optionally attach ``metadata['per_task_scores']`` as
    a task-id -> bounded score mapping. When paired confidence gating is enabled,
    VARE uses a deterministic paired bootstrap lower bound rather than treating
    a noisy point estimate as sufficient promotion evidence.
    """

    def __init__(self, config: PromotionConfig) -> None:
        self.config = config
        if not 0 < config.paired_alpha < 1:
            raise ValueError("paired_alpha must be in (0,1)")
        if config.paired_bootstrap_samples <= 0:
            raise ValueError("paired_bootstrap_samples must be positive")
        if config.require_reward_audit:
            if not 0 < config.reward_audit_alpha < 1:
                raise ValueError("reward_audit_alpha must be in (0,1)")
            if not 0 <= config.reward_audit_max_false_accept_ucb <= 1:
                raise ValueError("reward_audit_max_false_accept_ucb must be in [0,1]")
            if type(config.reward_audit_min_proxy_positives) is not int or config.reward_audit_min_proxy_positives < 1:
                raise ValueError("reward_audit_min_proxy_positives must be positive")

    @staticmethod
    def _paired_scores(report: EvaluationReport) -> Mapping[str, float] | None:
        raw = report.metadata.get("per_task_scores")
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("per_task_scores must be a mapping")
        return {str(k): float(v) for k, v in raw.items()}

    @staticmethod
    def _has_valid_metrics(report: EvaluationReport) -> bool:
        if type(report.n) is not int or report.n < 0:
            return False
        if not isinstance(report.slices, dict):
            return False
        values = [report.primary, report.cost, report.verifier_disagreement]
        values.extend(report.slices.values())
        if any(
            not PromotionGate._is_finite_metric(value)
            for value in values
        ):
            return False
        if not isinstance(report.metadata, dict):
            return False
        paired = report.metadata.get("per_task_scores")
        if paired is None:
            return True
        if not isinstance(paired, dict):
            return False
        for key, value in paired.items():
            if (
                not isinstance(key, str)
                or not PromotionGate._is_finite_metric(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                return False
        return True

    @staticmethod
    def _is_finite_metric(value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        try:
            return math.isfinite(value)
        except (OverflowError, TypeError):
            return False

    def _paired_evidence(
        self, incumbent: EvaluationReport, candidate: EvaluationReport
    ) -> tuple[float | None, float | None, int, list[str]]:
        old = self._paired_scores(incumbent)
        new = self._paired_scores(candidate)
        reasons: list[str] = []
        if old is None or new is None:
            if self.config.paired_confidence_gate:
                reasons.append("missing_paired_evidence")
            return None, None, 0, reasons
        if set(old) != set(new):
            reasons.append("paired_identity_mismatch")
            return None, None, 0, reasons
        keys = sorted(old)
        diffs = [new[k] - old[k] for k in keys]
        n = len(diffs)
        if n < self.config.min_paired_examples:
            if self.config.paired_confidence_gate:
                reasons.append("insufficient_paired_examples")
            return fmean(diffs) if diffs else None, None, n, reasons
        mean_gain = fmean(diffs)
        rng = random.Random(0)
        means: list[float] = []
        for _ in range(self.config.paired_bootstrap_samples):
            means.append(fmean(diffs[rng.randrange(n)] for _ in range(n)))
        means.sort()
        index = max(0, min(len(means) - 1, int(self.config.paired_alpha * len(means))))
        lcb = means[index]
        if self.config.paired_confidence_gate and lcb < self.config.min_primary_gain:
            reasons.append("paired_uncertainty")
        return mean_gain, lcb, n, reasons

    def _reward_audit_reasons(self, candidate: EvaluationReport) -> list[str]:
        raw = candidate.metadata.get("reward_audit")
        paired = candidate.metadata.get("per_task_scores")
        if not isinstance(raw, list) or not raw:
            return ["missing_reward_audit"]
        if not isinstance(paired, dict):
            return ["reward_audit_missing_task_identity"]
        try:
            if any(not isinstance(row, dict) for row in raw):
                raise ValueError("invalid observation")
            rows = [
                RewardAuditObservation(
                    task_id=row["task_id"],
                    family=row["family"],
                    proxy_pass=row["proxy_pass"],
                    trusted_pass=row["trusted_pass"],
                )
                for row in raw
            ]
            if set(row.task_id for row in rows) != set(paired) or len(rows) != len(paired):
                return ["reward_audit_identity_mismatch"]
            result = audit_reward_labels(
                rows,
                max_false_accept_ucb=self.config.reward_audit_max_false_accept_ucb,
                min_proxy_positives=self.config.reward_audit_min_proxy_positives,
                alpha=self.config.reward_audit_alpha,
            )
        except (TypeError, ValueError, KeyError):
            return ["invalid_reward_audit"]
        return list(result.reasons)

    def decide(self, incumbent: EvaluationReport, candidate: EvaluationReport) -> PromotionDecision:
        if not self._has_valid_metrics(incumbent) or not self._has_valid_metrics(candidate):
            return PromotionDecision(
                accepted=False,
                reasons=("invalid_evaluation_metrics",),
                primary_gain=0.0,
                worst_slice_regression=0.0,
                paired_gain=None,
                paired_lcb=None,
                paired_n=0,
            )
        reasons: list[str] = []
        if self.config.require_complete_slices and set(incumbent.slices) != set(candidate.slices):
            reasons.append("evaluation_slice_identity_mismatch")
        if (self.config.require_measured_disagreement
                and candidate.metadata.get("verifier_disagreement_measured") is not True):
            reasons.append("unmeasured_verifier_disagreement")
        if self.config.require_reward_audit:
            reasons.extend(self._reward_audit_reasons(candidate))
        gain = candidate.primary - incumbent.primary
        if incumbent.n < self.config.min_eval_examples or candidate.n < self.config.min_eval_examples:
            reasons.append("insufficient_eval_examples")
        if gain < self.config.min_primary_gain:
            reasons.append("insufficient_primary_gain")
        regressions = []
        for name, old in incumbent.slices.items():
            if name in candidate.slices:
                regressions.append(old - candidate.slices[name])
        worst_regression = max(regressions, default=0.0)
        if worst_regression > self.config.max_slice_regression:
            reasons.append("slice_regression")
        if incumbent.cost > 0:
            rel_cost = (candidate.cost - incumbent.cost) / incumbent.cost
            if rel_cost > self.config.max_relative_cost_increase:
                reasons.append("cost_regression")
        if candidate.verifier_disagreement > self.config.max_verifier_disagreement:
            reasons.append("verifier_disagreement")
        paired_gain, paired_lcb, paired_n, paired_reasons = self._paired_evidence(incumbent, candidate)
        reasons.extend(paired_reasons)
        return PromotionDecision(
            accepted=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            primary_gain=gain,
            worst_slice_regression=worst_regression,
            paired_gain=paired_gain,
            paired_lcb=paired_lcb,
            paired_n=paired_n,
        )
