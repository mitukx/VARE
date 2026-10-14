"""Deterministic synthetic probes for the read-only failure triage contract."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from vare.evidence import sha256_json
from vare.failure_triage import RewardGroupAudit, TaskAudit, TriageSnapshot, triage


def run() -> dict:
    base = TriageSnapshot(
        run_id="constructed-e0-v1", active_policy_version=3,
        rollout_policy_version=3, active_verifier_version=2,
        rollout_verifier_version=2, training_cohort_sha256="a" * 64,
        heldout_cohort_sha256="b" * 64, proxy_grader_sha256="c" * 64,
        trusted_grader_sha256="d" * 64,
        tasks=tuple(TaskAudit(f"task-{i}", True, i % 2 == 0, i % 2 == 0) for i in range(64)),
        reward_groups=tuple(RewardGroupAudit(f"group-{i}", (0.0, 1.0)) for i in range(8)),
        active_tokens=1000, clipped_tokens=100,
    )
    cases = {
        "clean": base,
        "proxy_false_accept": replace(base, tasks=tuple(
            replace(t, proxy_pass=True, trusted_pass=False) if i < 10 else t
            for i, t in enumerate(base.tasks))),
        "format_failure": replace(base, tasks=tuple(
            replace(t, format_valid=False) if i < 10 else t
            for i, t in enumerate(base.tasks))),
        "group_reward_tie": replace(base, reward_groups=tuple(
            RewardGroupAudit(g.group_id, (1.0, 1.0)) for g in base.reward_groups)),
        "lag_and_clipping": replace(base, rollout_policy_version=1, clipped_tokens=920),
        "short_cohort": replace(base, tasks=base.tasks[:10]),
    }
    expected = {
        "clean": ("no_alert", set()),
        "proxy_false_accept": ("diagnostic_probes", {"proxy_false_accepts"}),
        "format_failure": ("diagnostic_probes", {"output_contract_failures"}),
        "group_reward_tie": ("diagnostic_probes", {"zero_group_advantage"}),
        "lag_and_clipping": ("diagnostic_probes", {"rollout_freshness", "high_clipped_fraction"}),
        "short_cohort": ("insufficient_evidence", set()),
    }
    rows = []
    for name, snap in cases.items():
        report = triage(snap)
        signals = {finding.signal for finding in report.findings}
        if (report.status, signals) != expected[name]:
            raise AssertionError(f"{name}: unexpected status or signals")
        rows.append({
            "case": name,
            "input_sha256": sha256_json(asdict(snap)),
            "status": report.status,
            "signals": sorted(signals),
            "probes": list(report.next_probes),
        })
    return {
        "scope": "E0 deterministic constructed cases; not a causal or real-model result",
        "passed_contract_cases": len(rows),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run()
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
