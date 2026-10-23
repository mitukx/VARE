"""Strict semantic outcome checks for the prospective #1971 v5 calibration."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts.run_mlxlm_1971_calibration_v5 import (
    _has_semantic_decisions,
    verify_protocol_v5,
)


def test_v5_protocol_preserves_historical_lock_and_fails_closed_on_runtime_drift():
    root = Path(__file__).resolve().parents[1]
    task_root = root / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
    protocol = json.loads((task_root / "protocol-v5.json").read_text())
    lock = json.loads((task_root / "protocol-v5.lock.json").read_text())
    assert protocol["protocol_id"] == "vare.mlsys-repository-task.mlxlm-1971.v5"
    assert protocol["protocol_parent"] == "vare.mlsys-repository-task.mlxlm-1971.v4"
    assert protocol["value_scales"] == [0.25, 0.5, 0.8]
    assert (
        hashlib.sha256((task_root / "protocol-v5.json").read_bytes()).hexdigest()
        == lock["protocol_sha256"]
    )
    assert protocol["implementation_sha256"] == lock["implementation_sha256"]
    historical_revision = "9f0993ac2c71a44bcfbabd2a1399fc12e5d6d777"
    for path, expected in protocol["implementation_sha256"].items():
        if path not in {
            "src/vare/mlxlm_1971_repo_task.py",
            "tests/test_mlxlm_1971_repo_task_v4.py",
        }:
            continue
        historical = subprocess.run(
            ["git", "-C", str(root), "show", f"{historical_revision}:{path}"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == expected
    with pytest.raises(ValueError, match="v5 implementation hash mismatch"):
        verify_protocol_v5()


def test_v5_semantic_gate_accepts_only_expected_case_outcomes():
    root = Path(__file__).resolve().parents[1]
    retained = json.loads(
        (root / "results/mlxlm-1971-isolated-v4/run-1/calibration.json").read_text()
    )
    assert _has_semantic_decisions(
        retained["baseline"],
        retained["known_fix"],
        retained["constant_scale_mutant"]["result"],
        retained["scale_after_cache_mutant"]["result"],
        [0.25, 0.5, 0.8],
    )


def test_v5_semantic_gate_rejects_crash_timeout_or_wrong_case_failures():
    root = Path(__file__).resolve().parents[1]
    retained = json.loads(
        (root / "results/mlxlm-1971-isolated-v4/run-1/calibration.json").read_text()
    )
    baseline = json.loads(json.dumps(retained["baseline"]))
    baseline["cases"][0]["failure"] = "timeout"
    assert not _has_semantic_decisions(
        baseline,
        retained["known_fix"],
        retained["constant_scale_mutant"]["result"],
        retained["scale_after_cache_mutant"]["result"],
        [0.25, 0.5, 0.8],
    )
    constant = json.loads(json.dumps(retained["constant_scale_mutant"]["result"]))
    constant["cases"][1]["passed"] = False
    constant["cases"][1]["failure"] = "wrong_output"
    assert not _has_semantic_decisions(
        retained["baseline"],
        retained["known_fix"],
        constant,
        retained["scale_after_cache_mutant"]["result"],
        [0.25, 0.5, 0.8],
    )
