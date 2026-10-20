"""Trusted MLSys task evaluators convert to isolated subject calls without oracle mounts."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.mlsys_task_isolated_calls import (
    IsolatedTaskAdapterMissing,
    materialize_isolated_calls,
)

ROOT = Path(__file__).resolve().parents[1]


def test_grad_accum_hidden_cases_include_short_windows_and_value_error():
    task_id = "mlsys-v1-grad-accum-tail"
    task_dir = ROOT / "benchmarks/mlsys_coding_v1/tasks/grad_accum_tail"
    spec, provenance = materialize_isolated_calls(task_id, task_dir)
    assert spec["task_kind"] == "correctness"
    assert len(spec["calls"]) == 3
    assert spec["calls"][0]["kwargs"] == {"accumulation_steps": 4}
    assert spec["calls"][1]["kwargs"] == {"accumulation_steps": 8}
    assert spec["calls"][2]["expected_exception"] == "ValueError"
    assert provenance["evaluator_sha256"]
    assert provenance["skipped_invalid_generator_indices"] == []


def test_range_sums_baseline_uses_public_task_function_name():
    task_id = "mlsys-v1-range-sums"
    task_dir = ROOT / "benchmarks/mlsys_coding_v1/tasks/range_sums"
    spec, _ = materialize_isolated_calls(task_id, task_dir)
    assert "def range_sums(" in spec["baseline_source"]
    assert "def _reference(" not in spec["baseline_source"]
    assert spec["tuple_paths"] == [[1, "*"]]


def test_unknown_task_requires_an_isolated_adapter():
    with pytest.raises(IsolatedTaskAdapterMissing, match="fixture-add"):
        materialize_isolated_calls("fixture-add", ROOT)
