from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_isolated_grader", ROOT / "scripts/run_isolated_grader.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_container_command_is_offline_read_only_and_resource_limited(tmp_path: Path) -> None:
    command = module.build_command(
        docker="/usr/bin/docker",
        context="colima",
        image="python@sha256:" + "a" * 64,
        container_name="vare-grade-test",
        workspace=tmp_path / "candidate",
        task_root=tmp_path / "task",
        harness=tmp_path / "harness",
    )

    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop=ALL")] == "--cap-drop=ALL"
    assert command[command.index("--user") + 1] == "65532:65532"
    assert command[command.index("--pids-limit") + 1] == "64"
    assert command[command.index("--memory") + 1] == "536870912"
    assert all("readonly" in command[index + 1] for index, value in enumerate(command) if value == "--mount")
    assert not any("socket" in value.lower() or "credential" in value.lower() for value in command)


def test_grader_parser_keeps_structured_candidate_failure() -> None:
    record = module._parse_grade(
        'noise {"task_id":"rvl-hf-behavior-policy-parity-v1",'
        '"passed":false,"failures":["known regression"]}',
        "rvl-hf-behavior-policy-parity-v1",
    )
    assert record == {
        "task_id": "rvl-hf-behavior-policy-parity-v1",
        "passed": False,
        "failures": ["known regression"],
    }


def test_grader_parser_rejects_non_boolean_and_wrong_task_records() -> None:
    assert module._parse_grade(
        '{"task_id":"other","passed":true,"failures":[]}',
        "rvl-hf-behavior-policy-parity-v1",
    ) is None
    assert module._parse_grade(
        '{"task_id":"rvl-hf-behavior-policy-parity-v1","passed":1,"failures":[]}',
        "rvl-hf-behavior-policy-parity-v1",
    ) is None
