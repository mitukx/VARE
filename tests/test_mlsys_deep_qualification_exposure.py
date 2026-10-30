from __future__ import annotations

import copy

import pytest

from scripts.audit_mlsys_deep_qualification_exposure_v1 import (
    EXPECTED_DEVELOPMENT_TASKS,
    ExposureAuditError,
    build_disposition,
)


def _records():
    development = [
        {
            "task_id": task_id,
            "source_pr": f"https://example.invalid/{task_id}",
            "task_family": f"family-{task_id}",
            "split": "development",
            "exposure": {"model_or_agent_seen_task": False},
        }
        for task_id in sorted(EXPECTED_DEVELOPMENT_TASKS)
    ]
    ledger_tasks = []
    attempts = []
    for task in development:
        task_id = task["task_id"]
        stage2_count = 1 if task_id == "transformers-45852-4d-mask" else 2
        stage3_count = 3 if task_id == "mlxlm-1912-top-p-min-token" else 0
        ledger_tasks.append(
            {
                "task_id": task_id,
                "source_pr": task["source_pr"],
                "family": task["task_family"],
                "stage2_model_attempt_count": stage2_count,
                "stage3_exposure_count": stage3_count,
            }
        )
        for number in range(stage2_count):
            attempts.append(
                {
                    "task_id": task_id,
                    "stage": "Stage 2",
                    "run_id": f"stage2-{task_id}-{number}",
                    "model_id": "fixture-model",
                }
            )
        for number in range(stage3_count):
            attempts.append(
                {
                    "task_id": task_id,
                    "stage": "Stage 3",
                    "run_id": f"stage3-{task_id}-{number}",
                    "model_id": "fixture-model",
                }
            )
    return (
        {"tasks": development, "split_counts": {"development": 3}},
        {"tasks": ledger_tasks, "attempts": attempts},
    )


def _build(qualification, ledger):
    return build_disposition(
        qualification,
        ledger,
        qualification_sha256="a" * 64,
        exposure_ledger_sha256="b" * 64,
        expected_qualification_sha256="a" * 64,
        expected_exposure_ledger_sha256="b" * 64,
        created_at_utc="2026-10-31T00:00:00Z",
    )


def test_development_tasks_are_marked_exposed_without_rewriting_the_split():
    qualification, ledger = _records()
    disposition = _build(qualification, ledger)

    assert disposition["original_source_split"] == {"development": 3}
    assert disposition["reconciled_development"] == {
        "allocated_task_count": 3,
        "confirmed_previously_exposed_count": 3,
        "available_for_fresh_development_evaluation": 0,
    }
    assert all(
        item["eligible_for_fresh_development_evaluation"] is False
        for item in disposition["dispositions"]
    )
    assert disposition["decision"]["original_protocol_and_catalog_modified"] is False


def test_audit_rejects_changed_catalog_identity_or_incomplete_exposure_totals():
    qualification, ledger = _records()
    with pytest.raises(ExposureAuditError, match="catalog hash"):
        build_disposition(
            qualification,
            ledger,
            qualification_sha256="0" * 64,
            exposure_ledger_sha256="b" * 64,
            expected_qualification_sha256="a" * 64,
            expected_exposure_ledger_sha256="b" * 64,
        )

    damaged_ledger = copy.deepcopy(ledger)
    damaged_ledger["attempts"].pop()
    with pytest.raises(ExposureAuditError, match="attempt totals"):
        _build(qualification, damaged_ledger)


def test_audit_rejects_changed_source_family_even_when_task_id_matches():
    qualification, ledger = _records()
    ledger["tasks"][0]["family"] = "different-family"
    with pytest.raises(ExposureAuditError, match="missing or ambiguous"):
        _build(qualification, ledger)
