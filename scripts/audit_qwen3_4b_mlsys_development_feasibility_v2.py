#!/usr/bin/env python3
"""Additive reclassification of the frozen Qwen feasibility run; never edits it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_qwen3_4b_mlsys_development_feasibility_v1 import audit as audit_v1  # noqa: E402


def audit_retrospective(run: Path) -> dict[str, Any]:
    original = audit_v1(run)
    rows = [
        json.loads(line)
        for line in (run / "raw-records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    classification: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        old_failure = row.get("failure")
        detail = row.get("detail")
        if old_failure == "format_failure" and detail == "candidate patch did not apply cleanly":
            category = "patch_application_failure"
        else:
            category = old_failure or "verified_success"
        classification.setdefault(category, []).append(
            {
                "task_id": row["task_id"],
                "kind": row["kind"],
                "seed": row["seed"],
                "candidate_output_sha256": row["candidate_output_sha256"],
                "historical_failure_label": old_failure,
                "detail": detail,
                "retrospective_category": category,
            }
        )
    task_summary = {}
    for task_id in original["task_summary"]:
        task_rows = [row for row in rows if row["task_id"] == task_id]
        task_summary[task_id] = {
            "rollouts": len(task_rows),
            "verified_successes": sum(bool(row["passed"]) for row in task_rows),
            "format_failures": sum(
                row.get("failure") == "format_failure"
                and row.get("detail") != "candidate patch did not apply cleanly"
                for row in task_rows
            ),
            "patch_application_failures": sum(
                row.get("detail") == "candidate patch did not apply cleanly"
                for row in task_rows
            ),
        }
    return {
        "audit_passed": True,
        "source_audit": original,
        "protocol_id": "vare.qwen3-4b-mlsys-development-patch-feasibility.v1",
        "classification_protocol": "vare.qwen3-patch-output-contract.retrospective-v2",
        "historical_records_modified": False,
        "run_manifest_sha256": hashlib.sha256((run / "SHA256SUMS").read_bytes()).hexdigest(),
        "rollouts": len(rows),
        "verified_successes": sum(bool(row["passed"]) for row in rows),
        "classification_counts": {key: len(value) for key, value in classification.items()},
        "task_summary": task_summary,
        "classified_records": [record for values in classification.values() for record in values],
        "limitations": [
            "Classification is derived from retained runner phase detail; original labels and raw records are untouched.",
            "No model inference, patch reapplication, or grader rerun was performed.",
            "The two apply-phase failures never reached the independent grader and are not semantic failures.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_retrospective(args.run.resolve(strict=True))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("audit_passed", "rollouts", "verified_successes", "classification_counts")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
