from __future__ import annotations

from collections import defaultdict

from .types import Experience, FailureCluster


class FailureMiner:
    """Turns rollout diagnostics into stable failure clusters for curriculum generation."""

    def label(self, exp: Experience) -> str | None:
        if exp.policy_lag > 0:
            return "policy_staleness"
        if exp.verifier_lag > 0:
            return "verifier_staleness"
        if exp.shift_score > 0.25:
            return "distribution_shift"
        if exp.verification.disagreement > 0.20:
            return "verifier_disagreement"
        tag = exp.attempt.metadata.get("failure_tag")
        if isinstance(tag, str) and tag:
            return tag
        if not exp.verification.passed:
            return "capability_failure"
        return None

    def mine(self, experiences: list[Experience]) -> tuple[FailureCluster, ...]:
        groups: dict[str, list[Experience]] = defaultdict(list)
        for exp in experiences:
            label = self.label(exp)
            exp.failure_label = label
            if label:
                groups[label].append(exp)
        clusters = []
        for label, xs in groups.items():
            families = tuple(sorted({x.attempt.task.family for x in xs}))
            clusters.append(
                FailureCluster(
                    label=label,
                    count=len(xs),
                    task_families=families,
                    mean_score=sum(x.verification.score for x in xs) / len(xs),
                )
            )
        return tuple(sorted(clusters, key=lambda c: (-c.count, c.label)))
