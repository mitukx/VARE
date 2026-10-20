#!/usr/bin/env python3
"""Recompute tool counts and independently rerun the isolated public grader."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.container_isolation import DockerIsolationRunner, IsolationLimits  # noqa: E402
from vare.trajectory_store import audit_trajectory_store  # noqa: E402

PROTOCOL = ROOT / "protocols/agent_tool_recovery_v1.lock.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_public(workspace: Path, *, image: str) -> dict:
    limits = IsolationLimits(
        image=image,
        timeout_s=45,
        memory_mb=512,
        cpus=1.0,
        pids_limit=32,
        max_output_bytes=20_000,
    )
    result = asyncio.run(
        DockerIsolationRunner().run_candidate(
            limits,
            allowed_workspace_root=workspace.parent,
            workspace=workspace,
            command=["python", "/workspace/test_public.py"],
        )
    )
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def audit(output: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    raw = [
        json.loads(line)
        for line in (output / "raw_runs.jsonl").read_text().splitlines()
    ]
    checks: list[dict] = []
    for row in raw:
        trial = output / row["run_id"]
        operations = [
            json.loads(line)
            for line in (trial / "tool_operations.jsonl").read_text().splitlines()
        ]
        counts: dict[str, int] = {}
        for operation in operations:
            counts[operation["operation_id"]] = (
                counts.get(operation["operation_id"], 0) + 1
            )
        duplicate_count = sum(count - 1 for count in counts.values())
        public = verify_public(trial / "candidate", image=protocol["task"]["image"])
        store_report = (
            audit_trajectory_store(trial / "trajectory.sqlite3")
            if row["arm"] == "vare_durable_resume"
            else None
        )
        expected_tools = (
            2 if not row["fault"] else 4 if row["arm"] == "basic_restart" else 3
        )
        expected_dups = max(0, expected_tools - 2)
        current_source_hash = digest(trial / "candidate/grad_accum.py")
        checks.append(
            {
                "run_id": row["run_id"],
                "raw_row_matches_summary": row
                == next(x for x in summary["runs"] if x["run_id"] == row["run_id"]),
                "tool_count_recomputed": len(operations)
                == row["tool_execution_count"]
                == expected_tools,
                "duplicate_count_recomputed": duplicate_count
                == row["duplicate_operation_count"]
                == expected_dups,
                "all_tool_results_successful": all(
                    op.get("returncode", 0) == 0 and op.get("passed", True)
                    for op in operations
                ),
                "worker_exit_matches_injection": (
                    row["worker_exit_codes"][0] == (-9 if row["fault"] else 0)
                ),
                "public_grader_recheck_passed": public["passed"],
                "final_source_hash_matches": current_source_hash
                == row["workspace_source_sha256"],
                "vare_store_audit_consistent": (
                    store_report is None
                    or (
                        store_report["steps"] == row["durable_steps"] == 2
                        and store_report["effective_duplicates"] == 0
                    )
                ),
                "grader_recheck": public,
                "trajectory_audit": store_report,
            }
        )
    report = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": digest(PROTOCOL),
        "raw_summary_sha256": digest(output / "summary.json"),
        "raw_runs_sha256": digest(output / "raw_runs.jsonl"),
        "checks": checks,
        "aggregate": {},
        "status": "PASS_AUDITED_SYSTEM_SMOKE"
        if all(
            all(
                value is True
                for key, value in check.items()
                if key not in {"run_id", "grader_recheck", "trajectory_audit"}
            )
            for check in checks
        )
        else "NON_PASS",
        "claim_limit": protocol["scope"],
    }
    for fault in (False, True):
        for arm in protocol["arms"]:
            rows = [row for row in raw if row["fault"] == fault and row["arm"] == arm]
            name = f"{'fault' if fault else 'control'}:{arm}"
            report["aggregate"][name] = {
                "n": len(rows),
                "verified_tasks": sum(
                    check["public_grader_recheck_passed"]
                    for check in checks
                    if any(row["run_id"] == check["run_id"] for row in rows)
                ),
                "tool_executions": [row["tool_execution_count"] for row in rows],
                "duplicate_tool_executions": [
                    row["duplicate_operation_count"] for row in rows
                ],
                "wall_seconds_median": statistics.median(
                    row["wall_to_final_validation_seconds"] for row in rows
                ),
                "recovery_seconds_median": statistics.median(
                    row["recovery_seconds"] for row in rows
                ),
            }
    (output / "independent_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"evidence_manifest.json", "independent_audit.json"}
    }
    manifest = {
        "protocol_sha256": digest(PROTOCOL),
        "runner_source_sha256": digest(ROOT / "scripts/run_agent_tool_recovery_v1.py"),
        "auditor_source_sha256": digest(Path(__file__)),
        "files": files,
        "independent_audit_sha256": digest(output / "independent_audit.json"),
    }
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = audit(args.run.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "runs": len(report["checks"]),
                "verified": sum(
                    x["public_grader_recheck_passed"] for x in report["checks"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS_AUDITED_SYSTEM_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
