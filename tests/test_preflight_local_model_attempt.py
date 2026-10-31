import hashlib
import json
from unittest.mock import patch

import pytest

from scripts.preflight_local_model_attempt import (
    _check_artifact_hashes,
    available_memory_from_vm_stat,
    memory_pressure_free_percent,
    parse_swap_usage,
    run_preflight,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol_with_task_inputs(tmp_path):
    task_json = tmp_path / "task.json"
    task_prompt = tmp_path / "task-prompt.txt"
    assignment = tmp_path / "assignment.json"
    task_json.write_text('{"task_id":"example"}', encoding="utf-8")
    task_prompt.write_text("Repair the regression.", encoding="utf-8")
    assignment.write_text('{"assignment":"example"}', encoding="utf-8")
    return {
        "frozen_inputs": {
            "task": {
                "task_json_path": str(task_json),
                "task_json_sha256": _hash(task_json),
                "task_prompt_path": str(task_prompt),
                "task_prompt_sha256": _hash(task_prompt),
            }
        },
        "assignment": {
            "prospective_assignment_record": {
                "path": str(assignment),
                "sha256": _hash(assignment),
            }
        },
    }


def test_artifact_hashes_verify_both_task_files(tmp_path) -> None:
    protocol = _protocol_with_task_inputs(tmp_path)

    assert _check_artifact_hashes(tmp_path, protocol) == []


def test_artifact_hashes_reject_modified_task_json(tmp_path) -> None:
    protocol = _protocol_with_task_inputs(tmp_path)
    task_json = tmp_path / "task.json"
    task_json.write_text('{"task_id":"changed"}', encoding="utf-8")

    failures = _check_artifact_hashes(tmp_path, protocol)

    assert failures == [f"task input hash mismatch: {task_json}"]


def test_artifact_hashes_reject_modified_task_prompt(tmp_path) -> None:
    protocol = _protocol_with_task_inputs(tmp_path)
    task_prompt = tmp_path / "task-prompt.txt"
    task_prompt.write_text("Changed task prompt.", encoding="utf-8")

    failures = _check_artifact_hashes(tmp_path, protocol)

    assert failures == [f"task input hash mismatch: {task_prompt}"]


def test_artifact_hashes_reject_missing_task_json(tmp_path) -> None:
    protocol = _protocol_with_task_inputs(tmp_path)
    task_json = tmp_path / "task.json"
    task_json.unlink()

    failures = _check_artifact_hashes(tmp_path, protocol)

    assert failures == [f"task input hash mismatch: {task_json}"]


def test_artifact_hashes_reject_missing_task_prompt(tmp_path) -> None:
    protocol = _protocol_with_task_inputs(tmp_path)
    task_prompt = tmp_path / "task-prompt.txt"
    task_prompt.unlink()

    failures = _check_artifact_hashes(tmp_path, protocol)

    assert failures == [f"task input hash mismatch: {task_prompt}"]


def test_artifact_hashes_reject_incomplete_task_manifest(tmp_path) -> None:
    protocol = _protocol_with_task_inputs(tmp_path)
    del protocol["frozen_inputs"]["task"]["task_prompt_sha256"]

    failures = _check_artifact_hashes(tmp_path, protocol)

    assert failures == [
        "frozen task input manifest missing valid task_prompt_sha256"
    ]


def test_artifact_hashes_reject_absent_task_entry(tmp_path) -> None:
    protocol = _protocol_with_task_inputs(tmp_path)
    del protocol["frozen_inputs"]["task"]

    failures = _check_artifact_hashes(tmp_path, protocol)

    assert failures == ["frozen task input manifest is missing"]


def _write_preflight_fixture(tmp_path, *, include_task):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = {
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "hidden_size": 2,
        "num_attention_heads": 1,
    }
    (snapshot / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model_manifest = tmp_path / "model-manifest.json"
    model_manifest.write_text(
        json.dumps({"snapshot_path": str(snapshot), "model_weight_bytes": 0, "files": []}),
        encoding="utf-8",
    )
    assignment_path = tmp_path / "assignment.json"
    assignment_path.write_text('{"assignment":"example"}', encoding="utf-8")
    frozen_inputs = {
        "model_file_manifest": {
            "path": model_manifest.name,
            "sha256": _hash(model_manifest),
        }
    }
    if include_task:
        task_json = tmp_path / "task.json"
        task_prompt = tmp_path / "task-prompt.txt"
        task_json.write_text('{"task_id":"example"}', encoding="utf-8")
        task_prompt.write_text("Repair the regression.", encoding="utf-8")
        frozen_inputs["task"] = {
            "task_json_path": str(task_json),
            "task_json_sha256": _hash(task_json),
            "task_prompt_path": str(task_prompt),
            "task_prompt_sha256": _hash(task_prompt),
        }
    protocol = {
        "protocol_id": "preflight-test",
        "frozen_inputs": frozen_inputs,
        "assignment": {
            "workspace_path": str(workspace),
            "baseline_revision": "deadbeef",
            "prospective_assignment_record": {
                "path": str(assignment_path),
                "sha256": _hash(assignment_path),
            },
        },
        "agent_attempt": {"max_context_tokens": 1},
        "resource_policy": {
            "runtime_activation_reserve_bytes": 0,
            "gates": {
                "minimum_memory_pressure_free_percent": 0,
                "minimum_available_memory_bytes": 0,
                "maximum_swap_used_bytes": 1024,
                "minimum_disk_free_bytes": 0,
                "minimum_weight_kv_reserve_headroom_bytes": 0,
            },
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_bytes = json.dumps(protocol).encode()
    protocol_path.write_bytes(protocol_bytes)
    protocol_path.with_suffix(".sha256").write_text(
        hashlib.sha256(protocol_bytes).hexdigest() + "  protocol.json\n",
        encoding="utf-8",
    )
    return protocol_path


def _fake_resources(_root):
    return {
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "host": "test",
        "physical_memory_bytes": 1024**3,
        "available_memory_bytes": 1024**3,
        "memory_pressure_free_percent": 100,
        "total_bytes": 0,
        "used_bytes": 0,
        "free_bytes": 0,
        "disk_free_bytes": 1024**3,
    }


def test_preflight_does_not_report_missing_task_manifest_as_verified(tmp_path) -> None:
    protocol_path = _write_preflight_fixture(tmp_path, include_task=False)
    with patch("scripts.preflight_local_model_attempt.collect_host_resources", _fake_resources), patch(
        "scripts.preflight_local_model_attempt._verify_model_snapshot", return_value=[]
    ):
        result = run_preflight(protocol_path, tmp_path / "receipt.json", tmp_path)

    assert result["status"] == "NO_GO"
    assert result["frozen_inputs_verified"] is False
    assert result["model_snapshot_hashes_verified"] is True


def test_preflight_does_not_report_manifest_hash_failure_as_snapshot_verified(tmp_path) -> None:
    protocol_path = _write_preflight_fixture(tmp_path, include_task=True)
    with patch("scripts.preflight_local_model_attempt.collect_host_resources", _fake_resources), patch(
        "scripts.preflight_local_model_attempt._verify_model_snapshot",
        return_value=["model file manifest hash mismatch"],
    ):
        result = run_preflight(protocol_path, tmp_path / "receipt.json", tmp_path)

    assert result["model_snapshot_hashes_verified"] is False


def test_available_memory_uses_free_and_inactive_pages() -> None:
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages inactive: 200.\n"

    assert available_memory_from_vm_stat(text) == 300 * 16384


def test_memory_pressure_parser_requires_percentage() -> None:
    assert memory_pressure_free_percent("System-wide memory free percentage: 58%\n") == 58
    with pytest.raises(ValueError, match="free percentage"):
        memory_pressure_free_percent("memory pressure unavailable")


def test_swap_usage_parses_all_units() -> None:
    parsed = parse_swap_usage("vm.swapusage: total = 3.00G used = 512.00M free = 1024.00M")

    assert parsed == {
        "total_bytes": 3 * 1024**3,
        "used_bytes": 512 * 1024**2,
        "free_bytes": 1024 * 1024**2,
    }


def test_swap_usage_fails_closed_when_a_field_is_missing() -> None:
    with pytest.raises(ValueError, match="free"):
        parse_swap_usage("total = 3.00G used = 512.00M")
