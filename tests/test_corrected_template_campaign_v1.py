import hashlib

import pytest

from scripts.run_corrected_template_campaign_v1 import (
    is_verified_repair,
    validate_hashed_files,
)


def test_hashed_evidence_verification_rejects_drift_and_unsafe_paths(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("original")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()

    validate_hashed_files(tmp_path, {"evidence.json": digest}, "qualification evidence")
    evidence.write_text("changed")
    with pytest.raises(ValueError, match="drift"):
        validate_hashed_files(tmp_path, {"evidence.json": digest}, "qualification evidence")
    with pytest.raises(ValueError, match="unsafe"):
        validate_hashed_files(tmp_path, {"../outside": digest}, "qualification evidence")


def test_repair_count_requires_model_identity_and_independent_grading():
    agent = {
        "stop_reason": "agent_finished",
        "public_test_result": {"passed": True},
    }
    grade = {"success": True}
    common = {
        "normal_completion": True,
        "agent": agent,
        "source_changed": True,
        "independent_grade_valid": True,
        "grade": grade,
    }

    assert not is_verified_repair(model_identity_verified=False, **common)
    assert not is_verified_repair(model_identity_verified=True, **{
        **common, "independent_grade_valid": False,
    })
    assert is_verified_repair(model_identity_verified=True, **common)
