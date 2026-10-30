#!/usr/bin/env python3
"""Reconcile the frozen MLSys task split with retained Stage 2/3 exposure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_PATH = Path(
    "results/mlsys-task-deep-qualification-v1/qualification.json"
)
QUALIFICATION_SHA256 = "2b2a1a7c6e64a866dd3732fcd9723999969a1ac69d86f8a4d9f35edf67322c9d"
STAGE3_LEDGER_SHA256 = "a2da858454745c86ef7e5162d47552bb659943e03a47442cd03c0e9dff77693a"
STAGE3_LEDGER_REFERENCE = "../../outputs/VARE-stage3-2026-10-31/task-exposure-ledger.json"
EXPECTED_DEVELOPMENT_TASKS = frozenset(
    {
        "transformers-45852-4d-mask",
        "mlxlm-1912-top-p-min-token",
        "mlxlm-1971-mimo-value-scale",
    }
)


class ExposureAuditError(ValueError):
    """The pinned task and exposure records cannot be reconciled safely."""


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ExposureAuditError(f"input is missing or not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_disposition(
    qualification: dict[str, Any],
    exposure_ledger: dict[str, Any],
    *,
    qualification_sha256: str,
    exposure_ledger_sha256: str,
    expected_qualification_sha256: str = QUALIFICATION_SHA256,
    expected_exposure_ledger_sha256: str = STAGE3_LEDGER_SHA256,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Mark prior model exposure without rewriting the frozen split or catalog."""
    if qualification_sha256 != expected_qualification_sha256:
        raise ExposureAuditError("deep-qualification catalog hash mismatch")
    if exposure_ledger_sha256 != expected_exposure_ledger_sha256:
        raise ExposureAuditError("Stage 2/3 exposure-ledger hash mismatch")

    tasks = qualification.get("tasks")
    ledger_tasks = exposure_ledger.get("tasks")
    attempts = exposure_ledger.get("attempts")
    if not isinstance(tasks, list) or not isinstance(ledger_tasks, list) or not isinstance(attempts, list):
        raise ExposureAuditError("task or exposure records are malformed")

    development = [task for task in tasks if task.get("split") == "development"]
    development_ids = {task.get("task_id") for task in development}
    if development_ids != EXPECTED_DEVELOPMENT_TASKS:
        raise ExposureAuditError("frozen Development task set changed")

    dispositions: list[dict[str, Any]] = []
    for task in development:
        matches = [
            item
            for item in ledger_tasks
            if item.get("task_id") == task.get("task_id")
            and item.get("source_pr") == task.get("source_pr")
            and item.get("family") == task.get("task_family")
        ]
        if len(matches) != 1:
            raise ExposureAuditError(
                f"exposure-ledger identity is missing or ambiguous: {task.get('task_id')}"
            )
        ledger_task = matches[0]
        task_attempts = [
            item for item in attempts if item.get("task_id") == task["task_id"]
        ]
        stage2_count = sum(item.get("stage") == "Stage 2" for item in task_attempts)
        stage3_count = sum(item.get("stage") == "Stage 3" for item in task_attempts)
        expected_stage2 = ledger_task.get("stage2_model_attempt_count")
        expected_stage3 = ledger_task.get("stage3_exposure_count")
        if stage2_count != expected_stage2 or stage3_count != expected_stage3:
            raise ExposureAuditError(
                f"attempt totals disagree with task ledger: {task['task_id']}"
            )
        total = stage2_count + stage3_count
        if total == 0:
            raise ExposureAuditError(
                f"no prior exposure found for expected task: {task['task_id']}"
            )
        records = [
            {
                "stage": item.get("stage"),
                "run_id": item.get("run_id"),
                "model_id": item.get("model_id"),
                "candidate_diff_bytes": item.get("candidate_diff_bytes"),
                "edit_applied": item.get("edit_applied"),
                "grader_result": item.get("grader_result"),
                "regrade_semantic": item.get("regrade_semantic"),
            }
            for item in task_attempts
        ]
        dispositions.append(
            {
                "task_id": task["task_id"],
                "source_pr": task["source_pr"],
                "task_family": task["task_family"],
                "original_split": "development",
                "original_catalog_claimed_unseen": task.get("exposure", {}).get(
                    "model_or_agent_seen_task"
                )
                is False,
                "prior_model_attempt_count": total,
                "stage2_attempt_count": stage2_count,
                "stage3_attempt_count": stage3_count,
                "attempts": records,
                "classification": "previously_model_exposed",
                "eligible_for_fresh_development_evaluation": False,
                "eligible_for_reallocation_to_training": False,
            }
        )

    return {
        "schema_version": 1,
        "disposition_id": "mlsys-task-deep-qualification-v1-exposure-disposition-v1",
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_records": [
            {
                "path": QUALIFICATION_PATH.as_posix(),
                "sha256": qualification_sha256,
                "purpose": "frozen source-level qualification catalog and Development allocation",
            },
            {
                "path": STAGE3_LEDGER_REFERENCE,
                "sha256": exposure_ledger_sha256,
                "purpose": "retained Stage 2/3 model and task exposure ledger",
            },
        ],
        "original_source_split": qualification.get("split_counts"),
        "reconciled_development": {
            "allocated_task_count": len(development),
            "confirmed_previously_exposed_count": len(dispositions),
            "available_for_fresh_development_evaluation": 0,
        },
        "dispositions": dispositions,
        "decision": {
            "status": "NO_GO_FOR_FRESH_USE_OF_THIS_DEVELOPMENT_SPLIT",
            "reason": "All three tasks in the frozen Development allocation have matching prior Stage 2 model attempts; PR #1912 also has three Stage 3 attempts.",
            "original_protocol_and_catalog_modified": False,
            "heldout_tasks_executed_or_graded_by_this_audit": False,
            "model_inference_or_training_run_by_this_audit": False,
            "claim_limit": "Retrospective exposure reconciliation only; no task was rerun and no model capability result was produced.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage3-ledger",
        type=Path,
        default=Path(STAGE3_LEDGER_REFERENCE),
        help="retained Stage 2/3 ledger; its SHA-256 is pinned in this audit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "results/mlsys-task-deep-qualification-v1-exposure-append-v1.json",
    )
    args = parser.parse_args()
    ledger_path = args.stage3_ledger
    if not ledger_path.is_absolute():
        ledger_path = ROOT / ledger_path
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to overwrite exposure disposition: {output}")

    qualification_path = ROOT / QUALIFICATION_PATH
    qualification_sha = _sha256(qualification_path)
    ledger_sha = _sha256(ledger_path)
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    exposure_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    disposition = build_disposition(
        qualification,
        exposure_ledger,
        qualification_sha256=qualification_sha,
        exposure_ledger_sha256=ledger_sha,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(disposition, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": _sha256(output),
                "reconciled_development": disposition["reconciled_development"],
                "decision": disposition["decision"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
