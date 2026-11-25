"""Independent in-memory controls for the restricted source replay."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts import reproduce_trl_8520_environment_column as R


def fixture(columns, *, guard=True):
    body = f"self._signature_columns = {columns!r}"
    if guard:
        body = "if self._signature_columns is None:\n            " + body
    source = "class GRPOTrainer:\n    def _set_signature_columns_if_needed(self):\n        " + body
    return R.method_from_source(source, "GRPOTrainer", "_set_signature_columns_if_needed")


def test_environment_projection_contract_and_bad_controls():
    baseline = fixture(["prompt", "image", "images"])
    fixed = fixture(["prompt", "image", "images", "environment"])
    assert not R.exercise(R.checked_method(baseline))["environment_only"]
    assert all(R.exercise(R.checked_method(fixed)).values())
    for name in ("wrong_plural", "drop_image", "overwrite_existing"):
        assert not all(R.exercise(R.checked_method(
            R.mutant(fixed, name), allow_unconditional=name == "overwrite_existing",
        )).values())


def test_arbitrary_assignment_expression_is_refused_before_execution():
    source = """class GRPOTrainer:
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = unknown()
"""
    method = R.method_from_source(source, "GRPOTrainer", "_set_signature_columns_if_needed")
    with pytest.raises(ValueError, match="literal"):
        R.checked_method(method)



def test_retained_source_replay_evidence():
    root = Path(__file__).resolve().parents[1]
    protocol = root / "protocols/trl_8520_environment_column_source_v1.json"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        "fb72ddb537566d9c1afca9530af9e1f51b43ec5bd6aec9c0a432921cd6a60b86"
    )
    p = json.loads(protocol.read_bytes())
    path = root / "results/trl-8520-environment-column-source-v1/source-replay.zip"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "ff10fabdb1b4b1fd1224b06299fd56ade4a8e798176bbf5814f5eabca0a505f3"
    )
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["source-replay.json"]
        raw = archive.read("source-replay.json")
    assert hashlib.sha256(raw).hexdigest() == (
        "25b73032f007608f4d1edb0ffe242758de83c00ca467c6701f76fab5c3ef11d0"
    )
    result = json.loads(raw)
    assert result["source_commit"] == "f8d6c81a3088940174c9fa1bbbea32cf67bba106"
    assert result["status"] == "PASS_SOURCE_REPLAY"
    assert result["allocation"] == "UNASSIGNED_SOURCE_ONLY"
    assert result["model_inference"] is result["optimizer_updated"] is result["heldout_accessed"] is False
    assert result["upstream_training_test_executed"] is False
    assert [row["arm"] for row in result["arms"]] == p["arms"]
    assert {row["arm"]: row["pass"] for row in result["arms"]} == p["expected"]
    assert [sum(row["cases"].values()) for row in result["arms"]] == [2, 4, 2, 3, 3]
    assert all(set(row["cases"]) == set(p["cases"]) for row in result["arms"])
    assert {f["git_blob"] for f in result["source_files"]} == {
        p["source"][key] for key in ("baseline_blob", "fixed_blob", "license_blob", "test_blob")
    }
