"""Read-only, evidence-constrained triage of post-training diagnostic signals.

Signals identify falsifiable *probes*, not causes or automatic policy updates.
No training, model execution, or promotion occurs in this module.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class TaskAudit:
    task_id: str
    format_valid: bool
    proxy_pass: bool
    trusted_pass: bool


@dataclass(frozen=True, slots=True)
class RewardGroupAudit:
    group_id: str
    rewards: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TriageSnapshot:
    run_id: str
    active_policy_version: int
    rollout_policy_version: int
    active_verifier_version: int
    rollout_verifier_version: int
    training_cohort_sha256: str
    heldout_cohort_sha256: str
    proxy_grader_sha256: str
    trusted_grader_sha256: str
    tasks: tuple[TaskAudit, ...]
    reward_groups: tuple[RewardGroupAudit, ...]
    active_tokens: int
    clipped_tokens: int


@dataclass(frozen=True, slots=True)
class TriageConfig:
    min_tasks: int = 32
    min_groups: int = 8
    false_accept_rate: float = 0.05
    invalid_format_rate: float = 0.10
    zero_reward_group_rate: float = 0.75
    clipped_token_rate: float = 0.50
    max_policy_lag: int = 0
    max_verifier_lag: int = 0


@dataclass(frozen=True, slots=True)
class TriageFinding:
    signal: str
    observed: str
    falsification_probe: str


@dataclass(frozen=True, slots=True)
class TriageReport:
    status: str
    findings: tuple[TriageFinding, ...]
    next_probes: tuple[str, ...]
    counts: dict[str, int]
    evidence_hashes: dict[str, str]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _identity(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a nonempty identifier of at most 256 characters")


def _digest(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def _integer(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")


def _rate(value: float, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")


def _validate(snapshot: TriageSnapshot, config: TriageConfig) -> None:
    _identity(snapshot.run_id, "run_id")
    for name in ("active_policy_version", "rollout_policy_version", "active_verifier_version",
                 "rollout_verifier_version", "active_tokens", "clipped_tokens"):
        _integer(getattr(snapshot, name), name)
    for name in ("min_tasks", "min_groups", "max_policy_lag", "max_verifier_lag"):
        _integer(getattr(config, name), name)
    if config.min_tasks == 0 or config.min_groups == 0:
        raise ValueError("minimum evidence sizes must be positive")
    for name in ("false_accept_rate", "invalid_format_rate", "zero_reward_group_rate", "clipped_token_rate"):
        _rate(getattr(config, name), name)
    if snapshot.rollout_policy_version > snapshot.active_policy_version:
        raise ValueError("rollout policy version is ahead of active policy")
    if snapshot.rollout_verifier_version > snapshot.active_verifier_version:
        raise ValueError("rollout verifier version is ahead of active verifier")
    if snapshot.clipped_tokens > snapshot.active_tokens:
        raise ValueError("clipped tokens exceed active tokens")
    for name in ("training_cohort_sha256", "heldout_cohort_sha256", "proxy_grader_sha256", "trusted_grader_sha256"):
        _digest(getattr(snapshot, name), name)
    if snapshot.training_cohort_sha256 == snapshot.heldout_cohort_sha256:
        raise ValueError("optimization and held-out cohort digests must differ")
    if snapshot.proxy_grader_sha256 == snapshot.trusted_grader_sha256:
        raise ValueError("proxy and trusted grader digests must differ")
    if not isinstance(snapshot.tasks, tuple) or not snapshot.tasks:
        raise ValueError("nonempty task audit tuple required")
    ids: set[str] = set()
    for task in snapshot.tasks:
        if not isinstance(task, TaskAudit):
            raise ValueError("invalid task audit")
        _identity(task.task_id, "task_id")
        if task.task_id in ids:
            raise ValueError("duplicate held-out task identity")
        ids.add(task.task_id)
        if any(type(v) is not bool for v in (task.format_valid, task.proxy_pass, task.trusted_pass)):
            raise ValueError("audit decisions must be booleans")
    if not isinstance(snapshot.reward_groups, tuple):
        raise ValueError("reward groups must be a tuple")
    groups: set[str] = set()
    for group in snapshot.reward_groups:
        if not isinstance(group, RewardGroupAudit):
            raise ValueError("invalid reward group")
        _identity(group.group_id, "group_id")
        if group.group_id in groups:
            raise ValueError("duplicate reward group identity")
        groups.add(group.group_id)
        if not isinstance(group.rewards, tuple) or len(group.rewards) < 2:
            raise ValueError("each reward group needs at least two scores")
        for reward in group.rewards:
            _rate(reward, "group reward")


def triage(snapshot: TriageSnapshot, config: TriageConfig | None = None) -> TriageReport:
    """Propose diagnostics from paired audits, abstaining when evidence is thin.

    Hash inequality is a useful consistency check, not proof of evaluator
    independence or an authenticated execution record. Findings are associations.
    """
    cfg = config or TriageConfig()
    _validate(snapshot, cfg)
    n = len(snapshot.tasks)
    false_accepts = sum(t.proxy_pass and not t.trusted_pass for t in snapshot.tasks)
    invalid_formats = sum(not t.format_valid for t in snapshot.tasks)
    zero_groups = sum(max(g.rewards) == min(g.rewards) for g in snapshot.reward_groups)
    counts = {
        "heldout_tasks": n,
        "false_accepts": false_accepts,
        "invalid_formats": invalid_formats,
        "reward_groups": len(snapshot.reward_groups),
        "zero_advantage_groups": zero_groups,
        "active_tokens": snapshot.active_tokens,
        "clipped_tokens": snapshot.clipped_tokens,
        "policy_lag": snapshot.active_policy_version - snapshot.rollout_policy_version,
        "verifier_lag": snapshot.active_verifier_version - snapshot.rollout_verifier_version,
    }
    hashes = {name: getattr(snapshot, name) for name in (
        "training_cohort_sha256", "heldout_cohort_sha256", "proxy_grader_sha256", "trusted_grader_sha256")}
    limitations = (
        "Observational signal triage only: findings are not causal attributions.",
        "Distinct hashes cannot establish grader independence, label accuracy, or genuine held-out isolation.",
        "No training, automatic refresh, checkpoint promotion, or model capability improvement was measured.",
    )
    if n < cfg.min_tasks:
        return TriageReport("insufficient_evidence", (), ("expand_independent_task_audit",),
                            counts, hashes, limitations)

    findings: list[TriageFinding] = []
    if false_accepts / n > cfg.false_accept_rate:
        findings.append(TriageFinding("proxy_false_accepts", f"{false_accepts}/{n} paired tasks",
                                      "blind_trusted_verifier_adjudication"))
    if invalid_formats / n > cfg.invalid_format_rate:
        findings.append(TriageFinding("output_contract_failures", f"{invalid_formats}/{n} outputs",
                                      "format_parser_and_generation_ablation"))
    if counts["verifier_lag"] > cfg.max_verifier_lag:
        findings.append(TriageFinding("verifier_freshness", f"lag={counts['verifier_lag']}",
                                      "version_matched_verifier_rescore"))
    if counts["policy_lag"] > cfg.max_policy_lag:
        findings.append(TriageFinding("rollout_freshness", f"lag={counts['policy_lag']}",
                                      "matched_budget_rollout_refresh_ablation"))
    if len(snapshot.reward_groups) >= cfg.min_groups:
        if zero_groups / len(snapshot.reward_groups) > cfg.zero_reward_group_rate:
            findings.append(TriageFinding("zero_group_advantage", f"{zero_groups}/{len(snapshot.reward_groups)} groups",
                                          "reward_variation_and_group_size_audit"))
    if snapshot.active_tokens > 0 and snapshot.clipped_tokens / snapshot.active_tokens > cfg.clipped_token_rate:
        findings.append(TriageFinding("high_clipped_fraction", f"{snapshot.clipped_tokens}/{snapshot.active_tokens} active tokens",
                                      "matched_optimizer_step_and_rollout_ablation"))
    status = "diagnostic_probes" if findings else "no_alert"
    probes = tuple(dict.fromkeys(f.falsification_probe for f in findings))
    return TriageReport(status, tuple(findings), probes, counts, hashes, limitations)


def load_snapshot(raw: dict[str, Any]) -> TriageSnapshot:
    """Decode an untrusted JSON object without dropping unknown keys or coercing types."""
    if not isinstance(raw, dict):
        raise ValueError("triage snapshot must be a JSON object")
    try:
        tasks = raw["tasks"]
        groups = raw["reward_groups"]
        if not isinstance(tasks, list) or not isinstance(groups, list):
            raise ValueError("tasks and reward_groups must be JSON arrays")
        if any(not isinstance(item, dict) for item in tasks + groups):
            raise ValueError("task and reward group entries must be JSON objects")
        unpacked = dict(raw)
        unpacked["tasks"] = tuple(TaskAudit(**row) for row in tasks)
        unpacked["reward_groups"] = tuple(
            RewardGroupAudit(group_id=row["group_id"], rewards=tuple(row["rewards"]))
            for row in groups
        )
        return TriageSnapshot(**unpacked)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed triage snapshot: {exc}") from exc
