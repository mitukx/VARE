#!/usr/bin/env python3
"""Reconcile the manual MLSys queue with pinned qualification and exposure records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = Path("protocols/mlsys_candidate_queue_exposure_audit_v1.json")
OUTPUT_PATH = Path(
    "results/mlsys-candidate-manual-qualification-queue-exposure-v1.json"
)
QUEUE_ROW = re.compile(
    r"^\|\s*(\d+)\s*\|\s*\[([^\]]+)\]\(https://github\.com/([^/]+/[^/]+)/pull/(\d+)\)\s*\|"
    r"\s*(.*?)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|\s*(.*?)\s*\|$"
)


class QueueExposureAuditError(ValueError):
    """Pinned inputs or exposure records cannot be reconciled safely."""


def _read_regular(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise QueueExposureAuditError(f"{label} is missing or not a regular file: {path}")
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _input_path(relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else ROOT / path


def _verified_inputs(protocol: dict[str, Any]) -> dict[str, bytes]:
    raw_inputs: dict[str, bytes] = {}
    for item in protocol["pinned_inputs"]:
        raw = _read_regular(_input_path(item["path"]), item["role"])
        if _sha256(raw) != item["sha256"]:
            raise QueueExposureAuditError(f"pinned input hash mismatch: {item['path']}")
        raw_inputs[item["path"]] = raw

    queue_item = protocol["candidate_queue"]
    queue_raw = _read_regular(_input_path(queue_item["path"]), "candidate queue")
    if _sha256(queue_raw) != queue_item["sha256"]:
        raise QueueExposureAuditError("candidate queue hash mismatch")
    raw_inputs[queue_item["path"]] = queue_raw
    return raw_inputs


def _parse_queue(text: str) -> list[dict[str, Any]]:
    candidates = []
    for line in text.splitlines():
        match = QUEUE_ROW.match(line)
        if match is None:
            continue
        order, label, repository, number, title, changed, files, tests, status = (
            match.groups()
        )
        candidates.append(
            {
                "queue_order": int(order),
                "queue_label": label,
                "repository": repository,
                "pull_request": int(number),
                "identity": f"{repository}#{number}",
                "title": title,
                "changed_lines_in_frozen_survey": int(changed),
                "changed_files_in_frozen_survey": int(files),
                "test_paths_in_frozen_survey": int(tests),
                "frozen_queue_status": status,
            }
        )
    if len(candidates) != 24:
        raise QueueExposureAuditError(
            f"expected 24 queue rows; parsed {len(candidates)}"
        )
    if len({item["identity"] for item in candidates}) != len(candidates):
        raise QueueExposureAuditError("candidate queue contains duplicate PR identities")
    return candidates


def _source_pr_key(url: str) -> tuple[str, int]:
    match = re.fullmatch(r"https://github\.com/([^/]+/[^/]+)/pull/(\d+)", url)
    if match is None:
        raise QueueExposureAuditError(f"invalid upstream pull request URL: {url}")
    return match.group(1), int(match.group(2))


def _verify_evidence_index(item: dict[str, Any]) -> dict[str, int]:
    root = _input_path(item["base_path"]).resolve()
    raw = _read_regular(_input_path(item["path"]), "evidence index")
    if _sha256(raw) != item["sha256"]:
        raise QueueExposureAuditError(f"evidence index hash mismatch: {item['path']}")
    checked = missing = mismatched = listed = 0
    for line in raw.decode("utf-8").splitlines():
        if not line or line.startswith("SHA-256"):
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as exc:
            raise QueueExposureAuditError(
                f"malformed evidence-index row in {item['path']}"
            ) from exc
        listed += 1
        target = root / relative
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise QueueExposureAuditError("evidence index path escapes its root") from exc
        if target.is_symlink() or not target.is_file():
            missing += 1
            continue
        checked += 1
        if _sha256(target.read_bytes()) != expected:
            mismatched += 1
    expected_counts = {
        "listed": item["verified_entries"],
        "checked": item["verified_entries"],
        "missing": item["missing_entries"],
        "mismatched": item["mismatched_entries"],
    }
    actual = {
        "listed": listed,
        "checked": checked,
        "missing": missing,
        "mismatched": mismatched,
    }
    if actual != expected_counts:
        raise QueueExposureAuditError(
            f"evidence-index verification changed for {item['path']}: {actual}"
        )
    return actual


def build_audit(
    protocol: dict[str, Any],
    queue_text: str,
    qualification: dict[str, Any],
    stage2_summary: dict[str, Any],
    stage3_ledger: dict[str, Any],
    *,
    input_hashes: dict[str, str],
    evidence_index_results: list[dict[str, Any]],
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    candidates = _parse_queue(queue_text)
    observed_identities = [item["identity"] for item in candidates]
    if observed_identities != protocol["candidate_queue"]["expected_pull_requests"]:
        raise QueueExposureAuditError("manual queue PR identities differ from frozen protocol")

    tasks = qualification.get("tasks")
    exclusions = qualification.get("excluded_or_collapsed")
    stage2_attempts = stage2_summary.get("attempts")
    ledger_tasks = stage3_ledger.get("tasks")
    ledger_attempts = stage3_ledger.get("attempts")
    if not all(isinstance(value, list) for value in (tasks, exclusions, stage2_attempts, ledger_tasks, ledger_attempts)):
        raise QueueExposureAuditError("qualification or model-exposure inputs are malformed")

    task_by_id = {item.get("task_id"): item for item in tasks}
    if len(task_by_id) != len(tasks):
        raise QueueExposureAuditError("qualification catalog has duplicate task IDs")
    task_by_pr = {
        _source_pr_key(item["source_pr"]): item
        for item in tasks
    }
    exclusions_by_pr: dict[tuple[str, int], list[str]] = {}
    for item in exclusions:
        key = _source_pr_key(item["source_pr"])
        exclusions_by_pr.setdefault(key, []).append(str(item.get("reason", "unspecified")))

    stage3_only = [item for item in ledger_attempts if item.get("stage") == "Stage 3"]
    stage2_from_stage3_ledger = [item for item in ledger_attempts if item.get("stage") == "Stage 2"]
    summary_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in stage2_attempts:
        summary_by_task.setdefault(item.get("task_id"), []).append(item)
    ledger_stage2_by_task: dict[str, list[dict[str, Any]]] = {}
    ledger_stage3_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in stage2_from_stage3_ledger:
        ledger_stage2_by_task.setdefault(item.get("task_id"), []).append(item)
    for item in stage3_only:
        ledger_stage3_by_task.setdefault(item.get("task_id"), []).append(item)

    if len(stage2_attempts) != stage2_summary.get("total_attempts"):
        raise QueueExposureAuditError("Stage 2 attempt total does not match its summary")
    if len(stage2_from_stage3_ledger) != len(stage2_attempts):
        raise QueueExposureAuditError("combined ledger has an unexpected Stage 2 row count")
    if {item.get("run_id") for item in stage2_attempts} != {
        item.get("run_id") for item in stage2_from_stage3_ledger
    }:
        raise QueueExposureAuditError("Stage 2 summary and combined-ledger run IDs disagree")

    for ledger_task in ledger_tasks:
        task_id = ledger_task.get("task_id")
        stage2_count = len(ledger_stage2_by_task.get(task_id, []))
        stage3_count = len(ledger_stage3_by_task.get(task_id, []))
        if stage2_count != ledger_task.get("stage2_model_attempt_count"):
            raise QueueExposureAuditError(f"Stage 2 task count mismatch: {task_id}")
        if stage3_count != ledger_task.get("stage3_exposure_count"):
            raise QueueExposureAuditError(f"Stage 3 task count mismatch: {task_id}")
        qualified = task_by_id.get(task_id)
        if qualified is None or _source_pr_key(qualified["source_pr"]) != _source_pr_key(ledger_task["source_pr"]):
            raise QueueExposureAuditError(f"task identity mismatch in exposure ledger: {task_id}")

    attempts_by_pr: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    for key, task in task_by_pr.items():
        task_id = task["task_id"]
        attempts_by_pr[key] = {
            "Stage 2": summary_by_task.get(task_id, []),
            "Stage 3": ledger_stage3_by_task.get(task_id, []),
        }

    counts = {
        "previously_model_exposed": 0,
        "historical_heldout_reserved": 0,
        "historical_train_not_development": 0,
        "excluded_or_collapsed": 0,
        "manual_only_unqualified": 0,
    }
    rows = []
    for candidate in candidates:
        key = (candidate["repository"], candidate["pull_request"])
        task = task_by_pr.get(key)
        attempt_groups = attempts_by_pr.get(key, {"Stage 2": [], "Stage 3": []})
        attempts = []
        for stage in ("Stage 2", "Stage 3"):
            for item in attempt_groups[stage]:
                attempts.append(
                    {
                        "stage": stage,
                        "run_id": item.get("run_id"),
                        "model_id": item.get("model_id"),
                        "edit_applied": item.get("edit_applied"),
                        "candidate_diff_bytes": item.get("candidate_diff_bytes"),
                        "grader_outcome": item.get(
                            "regrade_semantic", item.get("attempt_outcome")
                        ),
                    }
                )

        if attempts:
            classification = "previously_model_exposed"
            eligibility_reason = "exact Stage 2/3 model attempts are recorded"
        elif task and task.get("split") == "heldout":
            classification = "historical_heldout_reserved"
            eligibility_reason = "historical Held-out assignment is preserved and was not opened"
        elif task and task.get("split") == "train":
            classification = "historical_train_not_development"
            eligibility_reason = "Train assignment cannot be silently moved into Development"
        elif key in exclusions_by_pr:
            classification = "excluded_or_collapsed"
            eligibility_reason = "frozen qualification review excludes or collapses this candidate"
        else:
            classification = "manual_only_unqualified"
            eligibility_reason = "manual queue metadata is not a task qualification or exposure clearance"
        counts[classification] += 1
        rows.append(
            {
                **candidate,
                "catalogued_task_id": task.get("task_id") if task else None,
                "catalogued_task_family": task.get("task_family") if task else None,
                "historical_split": task.get("split") if task else None,
                "qualification_status": (
                    "source_level_catalogued"
                    if task
                    else "excluded_or_collapsed"
                    if key in exclusions_by_pr
                    else "manual_only_unqualified"
                ),
                "qualification_notes": exclusions_by_pr.get(key, []),
                "stage2_model_attempt_count": len(attempt_groups["Stage 2"]),
                "stage3_model_attempt_count": len(attempt_groups["Stage 3"]),
                "model_attempts": attempts,
                "model_exposure_classification": (
                    "confirmed_prior_model_exposure"
                    if attempts
                    else "no_matching_attempt_in_audited_stage2_stage3_ledgers"
                ),
                "fresh_development_eligible_from_this_audit": False,
                "ineligibility_reason": eligibility_reason,
                "pretraining_exposure": "unknown",
            }
        )

    if counts != {
        "previously_model_exposed": 2,
        "historical_heldout_reserved": 8,
        "historical_train_not_development": 3,
        "excluded_or_collapsed": 11,
        "manual_only_unqualified": 0,
    }:
        raise QueueExposureAuditError(f"candidate classification counts changed: {counts}")
    queue_attempt_count = sum(
        item["stage2_model_attempt_count"] + item["stage3_model_attempt_count"]
        for item in rows
    )
    if queue_attempt_count != 6:
        raise QueueExposureAuditError(f"expected six queue model attempts; found {queue_attempt_count}")

    outside_queue = []
    queue_keys = {(item["repository"], item["pull_request"]) for item in candidates}
    for key, task in task_by_pr.items():
        if key in queue_keys:
            continue
        attempt_groups = attempts_by_pr[key]
        total = len(attempt_groups["Stage 2"]) + len(attempt_groups["Stage 3"])
        if total:
            outside_queue.append(
                {
                    "identity": f"{key[0]}#{key[1]}",
                    "task_id": task["task_id"],
                    "task_family": task["task_family"],
                    "stage2_model_attempt_count": len(attempt_groups["Stage 2"]),
                    "stage3_model_attempt_count": len(attempt_groups["Stage 3"]),
                }
            )

    return {
        "schema_version": 1,
        "audit_id": protocol["protocol_id"],
        "created_at_utc": created_at_utc
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "protocol_sha256": input_hashes[PROTOCOL_PATH.as_posix()],
        "source_main_commit": protocol["source_main_commit"],
        "source_input_hashes": input_hashes,
        "evidence_index_verification": evidence_index_results,
        "candidate_count": len(rows),
        "classification_counts": counts,
        "queue_model_attempt_count": queue_attempt_count,
        "candidates": rows,
        "model_exposed_tasks_outside_queue": outside_queue,
        "possible_family_overlap_notes": [
            {
                "members": ["ml-explore/mlx-lm#1904", "ml-explore/mlx-lm#1910"],
                "basis": "The frozen qualification exclusion says #1910 is the same tool-call argument parser family as #1904.",
                "effect": "Do not count them as independent families; #1904 remains Train and #1910 remains excluded/collapsed.",
            },
            {
                "members": ["ml-explore/mlx-lm#1795", "ml-explore/mlx-lm#1930"],
                "basis": "The frozen qualification exclusion says #1930 is the same MoE routing-index gradient family as #1795.",
                "effect": "Do not count them as independent families; #1795 remains Train and #1930 remains excluded/collapsed.",
            },
            {
                "members": ["ml-explore/mlx-lm#1921", "ml-explore/mlx-lm#1925"],
                "basis": "The frozen qualification exclusion says #1925 overlaps #1921 on left-padding/batched model semantics.",
                "effect": "#1921 remains Held-out and #1925 remains excluded/collapsed.",
            },
            {
                "members": ["ml-explore/mlx-lm#1787", "ml-explore/mlx-lm#1795"],
                "basis": "Their queue titles both concern gradients through MoE expert-routing indices; #1787 separately fails baseline discrimination.",
                "effect": "Do not count #1787 as an independent task; it remains excluded.",
            },
            {
                "members": ["huggingface/transformers#48319", "huggingface/transformers#49208"],
                "basis": "The survey report flags multiple Transformer checkpoint-resume candidates as correlated; the catalog assigns different narrow families.",
                "effect": "Keep both in their historical Held-out assignment and require a future manual independence decision before any split change.",
            },
            {
                "members": ["huggingface/transformers#45852", "ml-explore/mlx-lm#1938"],
                "basis": "Both queue titles mention masks/axes; the catalog assigns different repositories and narrow families, and the raw source screen is absent.",
                "effect": "Do not infer independence or reuse either for a fresh Development claim; #45852 is exposed and #1938 remains Held-out.",
            },
        ],
        "decision": {
            "status": "NO_GO_FOR_FRESH_DEVELOPMENT_FROM_CURRENT_QUEUE",
            "fresh_development_eligible_count": 0,
            "reason": "The queue contains two previously exposed Development tasks, eight tasks reserved in historical Held-out, three tasks assigned Train, and eleven candidates already excluded or collapsed. No task is currently assigned and qualified as fresh Development.",
            "original_queue_and_qualification_modified": False,
            "heldout_task_files_or_tests_opened": False,
            "model_inference_or_training_run_by_audit": False,
            "claim_limit": "Retrospective exact-PR matching to the pinned Stage 2/3 records and frozen qualification decisions. Absence from those ledgers is not a guarantee against model pretraining or unavailable legacy artifacts.",
        },
        "limitations": protocol["limitations"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / PROTOCOL_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT_PATH)
    args = parser.parse_args()
    protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if output_path.exists() or output_path.is_symlink():
        raise SystemExit(f"refusing to overwrite append-only audit output: {output_path}")

    protocol_raw = _read_regular(protocol_path, "audit protocol")
    protocol = json.loads(protocol_raw)
    raw_inputs = _verified_inputs(protocol)
    input_hashes = {protocol_path.relative_to(ROOT).as_posix(): _sha256(protocol_raw)}
    input_hashes.update({path: _sha256(raw) for path, raw in raw_inputs.items()})

    queue_text = raw_inputs[protocol["candidate_queue"]["path"]].decode("utf-8")
    by_path = {item["path"]: item for item in protocol["pinned_inputs"]}
    qualification = json.loads(raw_inputs["results/mlsys-task-deep-qualification-v1/qualification.json"])
    stage2 = json.loads(raw_inputs["../../outputs/VARE-stage2-final-2026-10-31/attempts-summary.json"])
    stage3 = json.loads(raw_inputs["../../outputs/VARE-stage3-2026-10-31/task-exposure-ledger.json"])
    evidence_indexes = [
        _verify_evidence_index(by_path[path])
        for path in (
            "../../outputs/VARE-stage2-final-2026-10-31/evidence-sha256.txt",
            "../../outputs/VARE-stage3-2026-10-31/evidence-sha256.txt",
        )
    ]
    evidence_results = [
        {"path": path, **result}
        for path, result in zip(
            (
                "../../outputs/VARE-stage2-final-2026-10-31/evidence-sha256.txt",
                "../../outputs/VARE-stage3-2026-10-31/evidence-sha256.txt",
            ),
            evidence_indexes,
        )
    ]
    audit = build_audit(
        protocol,
        queue_text,
        qualification,
        stage2,
        stage3,
        input_hashes=input_hashes,
        evidence_index_results=evidence_results,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": _sha256(output_path.read_bytes()),
                "candidate_count": audit["candidate_count"],
                "classification_counts": audit["classification_counts"],
                "queue_model_attempt_count": audit["queue_model_attempt_count"],
                "decision": audit["decision"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
