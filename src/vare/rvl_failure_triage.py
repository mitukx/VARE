"""Read-only RVL replay evidence binding for externally adjudicated diagnostics.

RVL verifier rewards are proxy labels, never independent gold labels. The
adapter does not mutate training state or claim causal attribution.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .evidence import sha256_json
from .failure_triage import RewardGroupAudit, TaskAudit, TriageConfig, TriageSnapshot, triage
from .integrations.rvl import RVLTokenReplayReader


@dataclass(frozen=True, slots=True)
class ReplayAuditManifest:
    selection_sha256: str
    current_policy_version: int
    current_verifier_version: int
    rows: tuple[dict[str, Any], ...]
    reward_groups: tuple[RewardGroupAudit, ...]
    skipped_noncomparison_groups: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_sha256": self.selection_sha256,
            "current_policy_version": self.current_policy_version,
            "current_verifier_version": self.current_verifier_version,
            "rows": [dict(row) for row in self.rows],
            "eligible_reward_groups": len(self.reward_groups),
            "skipped_noncomparison_groups": self.skipped_noncomparison_groups,
            "limits": [
                "RVL rewards are proxy measurements, not independent labels.",
                "A replay storage group is not necessarily a GRPO comparison group.",
                "Content hashes do not authenticate model execution or grader independence.",
            ],
        }


def _unit(value: Any) -> None:
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("RVL proxy reward must be finite and in [0,1]")


def _sha(raw: Any, field: str) -> str:
    if (not isinstance(raw, str) or len(raw) != 64
            or any(c not in "0123456789abcdef" for c in raw)):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return raw


def _exact(raw: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"{label} contains missing or unexpected fields")
    return raw


def replay_manifest(
    reader: RVLTokenReplayReader,
    *, current_policy_version: int, current_verifier_version: int,
) -> ReplayAuditManifest:
    experiences = reader.ready_experiences(
        current_policy_version=current_policy_version,
        current_verifier_version=current_verifier_version,
    )
    if not experiences:
        raise ValueError("no ready RVL experiences available")
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[Any]] = {}
    seen: set[str] = set()
    for exp in experiences:
        group_id = exp.attempt.metadata.get("rvl_group_id")
        index = exp.attempt.metadata.get("rvl_generation_index")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("missing replay group id")
        if type(index) is not int or index < 0:
            raise ValueError("invalid generation index")
        sample_id = json.dumps([group_id, index], separators=(",", ":"), ensure_ascii=False)
        if sample_id in seen:
            raise ValueError("duplicate replay generation identity")
        seen.add(sample_id)
        _unit(exp.verification.score)
        if type(exp.verification.passed) is not bool:
            raise ValueError("invalid proxy decision")
        if not isinstance(exp.attempt.output, str):
            raise ValueError("invalid rollout response")
        rows.append({
            "sample_id": sample_id, "group_id": group_id, "generation_index": index,
            "task_id": exp.attempt.task.id,
            "prompt_sha256": hashlib.sha256(exp.attempt.task.prompt.encode()).hexdigest(),
            "response_sha256": hashlib.sha256(exp.attempt.output.encode()).hexdigest(),
            "policy_version": exp.attempt.policy_version,
            "verifier_version": exp.verification.verifier_version,
            "proxy_pass": exp.verification.passed, "proxy_reward": float(exp.verification.score),
        })
        groups.setdefault(group_id, []).append(exp)
    rows.sort(key=lambda x: str(x["sample_id"]))
    reward_groups: list[RewardGroupAudit] = []
    skipped = 0
    for group_id, members in sorted(groups.items()):
        first = members[0]
        def key(exp: Any) -> tuple[Any, ...]:
            return (exp.attempt.task.id, exp.attempt.task.prompt,
                    exp.attempt.policy_version, exp.verification.verifier_version)
        if len(members) >= 2 and all(key(x) == key(first) for x in members):
            ordered = sorted(members, key=lambda x: x.attempt.metadata["rvl_generation_index"])
            reward_groups.append(RewardGroupAudit(group_id, tuple(float(x.verification.score) for x in ordered)))
        else:
            skipped += 1
    content = {"policy_version": current_policy_version,
               "verifier_version": current_verifier_version, "rows": rows}
    return ReplayAuditManifest(sha256_json(content), current_policy_version,
                               current_verifier_version, tuple(rows),
                               tuple(reward_groups), skipped)


def adjudicate_replay(
    manifest: ReplayAuditManifest,
    audit: dict[str, Any],
    bindings: dict[str, Any],
    *, config: TriageConfig | None = None,
) -> dict[str, Any]:
    audit = _exact(audit, {"selection_sha256", "rows"}, "external audit")
    if _sha(audit["selection_sha256"], "selection_sha256") != manifest.selection_sha256:
        raise ValueError("external audit does not match replay selection")
    if not isinstance(audit["rows"], list):
        raise ValueError("audit rows must be an array")
    expected = {str(x["sample_id"]): x for x in manifest.rows}
    seen: set[str] = set()
    task_audits: list[TaskAudit] = []
    for item in audit["rows"]:
        item = _exact(item, {"sample_id", "response_sha256", "format_valid", "trusted_pass"}, "audit row")
        key = item["sample_id"]
        if not isinstance(key, str) or key not in expected:
            raise ValueError("unknown audit generation identity")
        if key in seen:
            raise ValueError("duplicate audit generation identity")
        seen.add(key)
        if _sha(item["response_sha256"], "response_sha256") != expected[key]["response_sha256"]:
            raise ValueError("audit response hash mismatch")
        if type(item["format_valid"]) is not bool or type(item["trusted_pass"]) is not bool:
            raise ValueError("audit verdicts must be booleans")
        task_audits.append(TaskAudit(key, item["format_valid"],
                                     expected[key]["proxy_pass"], item["trusted_pass"]))
    if seen != set(expected):
        raise ValueError("incomplete independent audit: all ready generations are required")
    bindings = _exact(bindings, {
        "run_id", "training_cohort_sha256", "heldout_cohort_sha256",
        "proxy_grader_sha256", "trusted_grader_sha256", "active_tokens", "clipped_tokens",
    }, "study bindings")
    for field in ("training_cohort_sha256", "heldout_cohort_sha256",
                  "proxy_grader_sha256", "trusted_grader_sha256"):
        _sha(bindings[field], field)
    snapshot = TriageSnapshot(
        run_id=bindings["run_id"],
        active_policy_version=manifest.current_policy_version,
        rollout_policy_version=min(int(x["policy_version"]) for x in manifest.rows),
        active_verifier_version=manifest.current_verifier_version,
        rollout_verifier_version=min(int(x["verifier_version"]) for x in manifest.rows),
        training_cohort_sha256=bindings["training_cohort_sha256"],
        heldout_cohort_sha256=bindings["heldout_cohort_sha256"],
        proxy_grader_sha256=bindings["proxy_grader_sha256"],
        trusted_grader_sha256=bindings["trusted_grader_sha256"],
        tasks=tuple(sorted(task_audits, key=lambda x: x.task_id)),
        reward_groups=manifest.reward_groups,
        active_tokens=bindings["active_tokens"],
        clipped_tokens=bindings["clipped_tokens"],
    )
    report = triage(snapshot, config)
    return {
        "replay_selection_sha256": manifest.selection_sha256,
        "audit_sha256": sha256_json(audit),
        "bindings_sha256": sha256_json(bindings),
        "skipped_noncomparison_groups": manifest.skipped_noncomparison_groups,
        "triage": report.to_dict(),
        "limitations": [
            "This is a replay audit, not a held-out learning experiment or causal diagnosis.",
            "Bindings are supplied, not independently attested; cohort disjointness is unverified.",
            "Current versions are operator-supplied and need external runtime attestation.",
            "Trusted grader independence and labels are external obligations.",
            "Active/clipped token counters are operator-supplied, not validated from RVL.",
        ],
    }


def load_strict_json(path: str | Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in items:
            if key in row:
                raise ValueError(f"duplicate JSON key: {key}")
            row[key] = value
        return row

    def nonfinite(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=nonfinite)
