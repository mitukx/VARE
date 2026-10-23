from __future__ import annotations

import subprocess

import pytest

from vare.qwen_patch_contract_v2 import (
    PatchFormatError,
    grade_with_phase_taxonomy,
    validate_unified_diff,
)

TARGET = "src/example.py"
VALID = (
    "diff --git a/src/example.py b/src/example.py\n"
    "index 1111111..2222222 100644\n"
    "--- a/src/example.py\n"
    "+++ b/src/example.py\n"
    "@@ -1,2 +1,2 @@ def f():\n"
    " def f():\n"
    "-    return 0\n"
    "+    return 1\n"
)


def test_valid_complete_diff_with_context_and_index_is_accepted():
    assert validate_unified_diff(VALID, TARGET) == VALID


@pytest.mark.parametrize(
    "text",
    [
        "Here is the patch:\n" + VALID,
        VALID + "Commentary after the patch.\n",
        "```diff\n" + VALID + "```\n",
        VALID + "diff --git a/other.py b/other.py\n",
        VALID.replace("@@ -1,2 +1,2", "@@ -1,3 +1,2"),
        VALID.replace("+    return 1\n", "+    return 1\n\nextra\n"),
    ],
)
def test_leading_trailing_extra_block_and_malformed_hunks_are_rejected(text):
    with pytest.raises(PatchFormatError):
        validate_unified_diff(text, TARGET)


def test_in_hunk_context_is_not_mistaken_for_prose():
    diff = VALID.replace("\n def f():\n", "\n def f():\n     # ordinary source context\n")
    diff = diff.replace("@@ -1,2 +1,2", "@@ -1,3 +1,3")
    assert validate_unified_diff(diff, TARGET) == diff


def _run(raw=VALID, *, apply_error=None, grade_result=None, grade_error=None):
    def apply(_patch):
        if apply_error:
            raise apply_error
        return "candidate"

    def grade(_candidate):
        if grade_error:
            raise grade_error
        return grade_result or {"passed": True, "cases": []}

    return grade_with_phase_taxonomy(
        raw, TARGET, apply_patch=apply, grade_candidate=grade
    )


def test_invalid_format_is_rejected_before_application():
    applied = []
    result = grade_with_phase_taxonomy(
        "prose\n" + VALID,
        TARGET,
        apply_patch=lambda patch: applied.append(patch),
        grade_candidate=lambda _candidate: {"passed": True},
    )
    assert result["failure"] == "format_failure"
    assert applied == []


def test_apply_failure_is_not_misclassified_as_format():
    assert _run(apply_error=ValueError("candidate patch did not apply cleanly"))["failure"] == "patch_application_failure"


def test_grader_outcomes_and_infrastructure_are_distinguished():
    assert _run(grade_result={"passed": False, "cases": [{"failure": "wrong_output"}]})["failure"] == "grader_wrong_answer"
    assert _run(grade_result={"passed": False, "cases": [{"failure": "timeout"}]})["failure"] == "grader_timeout"
    assert _run(grade_error=RuntimeError("container unavailable"))["failure"] == "grader_infrastructure"
    assert _run(apply_error=subprocess.TimeoutExpired("git apply", 2))["failure"] == "patch_application_timeout"


def test_versioned_contract_implementation_hash_lock():
    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "protocols/qwen3_patch_output_contract_v2.json"
    lock_path = root / "protocols/qwen3_patch_output_contract_v2.lock.json"
    protocol = json.loads(protocol_path.read_text())
    lock = json.loads(lock_path.read_text())
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == lock["protocol_sha256"]
    assert protocol["implementation_sha256"] == lock["implementation_sha256"]
    assert set(lock["implementation_sha256"]) == {
        "src/vare/qwen_patch_contract_v2.py",
        "scripts/audit_qwen3_4b_mlsys_development_feasibility_v2.py",
        "tests/test_qwen_patch_contract_v2.py",
    }
    assert protocol["status"] == "retrospective-and-prospective-contract-only-no-rollouts"


def test_v2_launcher_fails_before_model_load_on_consumed_cohort(monkeypatch, tmp_path):
    import sys
    output = tmp_path / "must-not-create"
    monkeypatch.setattr(sys, "argv", ["runner", "--output", str(output)])
    from scripts.run_qwen3_4b_mlsys_development_feasibility_v2 import main

    # The frozen v2 source lock predates the integrated v9 grading code.
    # Refuse the stale implementation instead of resealing a consumed cohort.
    with pytest.raises(RuntimeError, match="v2 implementation source hash mismatch"):
        main()
    assert not output.exists()
