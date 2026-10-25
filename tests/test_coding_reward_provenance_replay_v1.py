from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.run_coding_reward_provenance_replay_v1 import (
    TASK_CASE_IDS,
    validate_outcome_identity,
)


def valid_receipt() -> tuple[dict, dict]:
    row = {
        "candidate": "known-fix",
        "passed": True,
        "image": "test-image@sha256:" + "a" * 64,
        "case_passed": len(TASK_CASE_IDS),
        "case_count": len(TASK_CASE_IDS),
    }
    receipt = {
        "candidate_id": row["candidate"],
        "passed": True,
        "image": row["image"],
        "grader_source_sha256": "b" * 64,
        "grader_adapter_sha256": "c" * 64,
        "evaluator_case_count": len(TASK_CASE_IDS),
        "cases": [{"case_id": case_id, "passed": True} for case_id in TASK_CASE_IDS],
    }
    return receipt, row


def validate(receipt: dict, row: dict) -> None:
    validate_outcome_identity(
        receipt,
        Path("receipt.json"),
        row,
        grader_source_sha256="b" * 64,
        grader_adapter_sha256="c" * 64,
        case_ids=TASK_CASE_IDS,
    )


def test_accepts_receipt_matching_frozen_identity() -> None:
    receipt, row = valid_receipt()
    validate(receipt, row)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda receipt: receipt.update(grader_source_sha256="d" * 64),
        lambda receipt: receipt.update(grader_adapter_sha256="d" * 64),
        lambda receipt: receipt["cases"][0].update(case_id="different-case"),
        lambda receipt: receipt["cases"].pop(),
        lambda receipt: receipt["cases"].append(copy.deepcopy(receipt["cases"][0])),
        lambda receipt: receipt["cases"][0].update(passed=1),
        lambda receipt: receipt.update(passed=False),
    ],
)
def test_rejects_receipt_with_changed_verifier_or_case_identity(mutate) -> None:
    receipt, row = valid_receipt()
    mutate(receipt)
    with pytest.raises(ValueError):
        validate(receipt, row)
