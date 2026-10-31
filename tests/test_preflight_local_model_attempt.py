import hashlib

import pytest

from scripts.preflight_local_model_attempt import (
    _check_artifact_hashes,
    available_memory_from_vm_stat,
    memory_pressure_free_percent,
    parse_swap_usage,
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
