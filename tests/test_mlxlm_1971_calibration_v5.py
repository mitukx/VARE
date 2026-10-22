"""Strict semantic outcome checks for the prospective #1971 v5 calibration."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_mlxlm_1971_calibration_v5 import (
    _has_semantic_decisions,
    verify_protocol_v5,
)


def test_v5_protocol_binds_v4_parent_and_implementation():
    protocol = verify_protocol_v5()
    assert protocol["protocol_id"] == "vare.mlsys-repository-task.mlxlm-1971.v5"
    assert protocol["protocol_parent"] == "vare.mlsys-repository-task.mlxlm-1971.v4"
    assert protocol["value_scales"] == [0.25, 0.5, 0.8]


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
