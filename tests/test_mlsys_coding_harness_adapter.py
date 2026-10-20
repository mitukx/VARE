"""Adapter contract. Attack regressions for the child harness belong to PR #85."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parents[1] / "benchmarks" / "mlsys_coding_v1"


@pytest.fixture(name="adapter")
def adapter_module():
    sys.path.insert(0, str(PACK))
    import harness_adapter

    return harness_adapter


def _request(
    tmp_path: Path, adapter, *, task_type: str = "correctness", repeats: int = 1
):
    workspace = tmp_path / "subject"
    workspace.mkdir(exist_ok=True)
    (workspace / "widget.py").write_text(
        "def add(value):\n"
        "    if value < 0:\n"
        "        raise ValueError('negative')\n"
        "    return value + 1\n",
        encoding="utf-8",
    )
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps({"metadata": {"task_type": task_type}}) + "\n", encoding="utf-8"
    )

    def generator(index: int):
        return ((1 if index == 0 else -1,), {})

    def reference(value):
        if value < 0:
            raise ValueError("negative")
        return value + 1

    def checker(actual, expected, _positional, _keywords, _args, _kwargs):
        if actual != expected:
            raise AssertionError(f"{actual} != {expected}")

    return adapter.HarnessRequest(
        workspace=workspace,
        entrypoint="widget:add",
        task_type=task_type,
        trusted_task_path=task,
        input_generator=generator,
        reference=reference,
        checker=checker,
        repeats=repeats,
        timing_repeats=0,
        timeout_s=10,
    )


def test_shim_is_marked_temporary_and_checks_in_the_parent(tmp_path: Path, adapter):
    report = adapter.run_harness(_request(tmp_path, adapter, repeats=2))
    assert report.passed is True
    assert report.harness_id == "temporary_shim_candidate_call"
    assert report.replaced_by == "vare.separated_candidate_harness.grade_function"
    assert report.nonce_enforced is False
    assert report.nonce


def test_task_type_must_match_trusted_metadata(tmp_path: Path, adapter):
    request = _request(tmp_path, adapter)
    request.task_type = "performance"
    request.timing_repeats = 1
    with pytest.raises(adapter.HarnessAdapterError, match="trusted metadata"):
        adapter.run_harness(request)


def test_trusted_metadata_inside_the_workspace_is_refused(tmp_path: Path, adapter):
    request = _request(tmp_path, adapter)
    leaked = request.workspace / "task.json"
    leaked.write_text(
        request.trusted_task_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    request.trusted_task_path = leaked
    with pytest.raises(adapter.HarnessAdapterError, match="workspace"):
        adapter.run_harness(request)


def test_proposed_harness_is_preferred_and_must_enforce_the_nonce(
    tmp_path, adapter, monkeypatch
):
    seen = {}

    def proposed(payload):
        seen["entrypoint"] = payload["entrypoint"]
        seen["nonce"] = payload["nonce"]
        seen["task_type"] = payload["task_type"]
        return {
            "passed": True,
            "error": None,
            "baseline_samples_s": [],
            "candidate_samples_s": [],
            "nonce_enforced": True,
        }

    monkeypatch.setattr(adapter, "resolve_proposed_harness", lambda: proposed)
    report = adapter.run_harness(_request(tmp_path, adapter))
    assert report.harness_id == adapter.PROPOSED_SYMBOL
    assert report.nonce_enforced is True
    assert seen == {
        "entrypoint": "widget:add",
        "nonce": report.nonce,
        "task_type": "correctness",
    }

    def skipped_nonce(_payload):
        return {
            "passed": True,
            "error": None,
            "baseline_samples_s": [],
            "candidate_samples_s": [],
            "nonce_enforced": False,
        }

    monkeypatch.setattr(adapter, "resolve_proposed_harness", lambda: skipped_nonce)
    with pytest.raises(adapter.HarnessAdapterError, match="nonce"):
        adapter.run_harness(_request(tmp_path, adapter))
