from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "results/sprint_viii_20261028/external_transformers_49703/run_campaign_v5.py"
)


def _load_runner(monkeypatch):
    # These tests exercise pure protocol/journal helpers. The standalone
    # campaign runner uses psutil for resource monitoring, but importing it
    # must not make these helper tests depend on that optional runtime package.
    monkeypatch.setitem(sys.modules, "psutil", ModuleType("psutil"))
    spec = importlib.util.spec_from_file_location("sprint_viii_campaign_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_exposure_events_are_durable_and_append_only(tmp_path, monkeypatch):
    runner = _load_runner(monkeypatch)
    real_fsync = runner.os.fsync
    fsync_calls: list[int] = []

    def track_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(runner.os, "fsync", track_fsync)
    root = tmp_path / "campaign"
    root.mkdir()
    runner.append_exposure_event(root, {"arm_id": "autonomous", "heldout": False})
    runner.append_exposure_event(root, {"arm_id": "localized", "heldout": False})

    lines = (root / "task-exposure.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["arm_id"] for line in lines] == ["autonomous", "localized"]
    assert all(json.loads(line)["heldout"] is False for line in lines)
    assert len(fsync_calls) == 2


def test_campaign_protocol_shape_rejects_missing_decoding_fields(monkeypatch):
    runner = _load_runner(monkeypatch)
    protocol = {
        "protocol_id": "vare.sprint-viii.transformers-49703-development.v5",
        "frozen_before_model_inference": True,
        "code_sha256": {},
        "runtime": {},
        "model": {},
        "task": {
            "task_id": "task",
            "task_family": "family",
            "user_instruction": "repair source",
            "task_prompt_sha256": "0" * 64,
            "baseline_source_sha256": "0" * 64,
            "task_json_sha256": "0" * 64,
        },
        "arms": [
            {"arm_id": "autonomous", "tool_interface": "navigation"},
            {"arm_id": "localized", "tool_interface": "localized"},
        ],
        "inference_budget": {
            "max_turns": 1,
            "max_no_tool_retries": 1,
            "max_new_tokens_per_turn": 1,
            "max_total_new_tokens": 1,
            "max_context_tokens": 1,
            "agent_wall_seconds": 1,
            "generation_protocol": "schema_slots_v8",
            "navigation_guidance": "v2",
            "edit_format": "structured_replace",
        },
        "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
        "resource_limits": {},
        "controller_source_files_sha256_by_interface": {"navigation": {}, "localized": {}},
        "claim_limit": "development only",
    }

    with pytest.raises(ValueError, match=r"missing decoding fields: \['seed'\]"):
        runner.validate_protocol_shape(protocol)
