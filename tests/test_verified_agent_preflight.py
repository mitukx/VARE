from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import preflight_verified_agent_baseline as preflight
from scripts.preflight_verified_agent_baseline import collect_preflight, write_report


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_records_blocker_without_starting_agent_or_grader(tmp_path: Path) -> None:
    report = collect_preflight(
        root=ROOT,
        model_path=tmp_path / "missing-model",
        docker_probe=lambda: {"cli_available": True, "daemon_available": False},
    )

    assert report["status"] == "BLOCKED"
    assert "Docker daemon is unavailable" in report["blockers"]
    assert "VARE has no reviewed isolated evaluator launcher for this task" in report["blockers"]
    assert report["attempt"] == {
        "model_loaded": False,
        "agent_started": False,
        "grader_started": False,
        "candidate_code_executed": False,
    }
    assert report["measurements"] is None


def test_preflight_report_is_exclusive_and_machine_readable(tmp_path: Path) -> None:
    report = collect_preflight(
        root=ROOT,
        model_path=tmp_path / "missing-model",
        docker_probe=lambda: {"cli_available": False, "daemon_available": False},
    )
    destination = tmp_path / "evidence" / "preflight.json"

    write_report(destination, report)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved["protocol_id"] == "verified-coding-agent-baseline-v3-preflight"
    assert saved["status"] == "BLOCKED"
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_report(destination, report)


def test_preflight_checks_model_bytes_against_frozen_protocol(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("not the pinned config", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"not a model")

    report = collect_preflight(
        root=ROOT,
        model_path=model,
        docker_probe=lambda: {"cli_available": False, "daemon_available": False},
    )

    assert report["environment"]["model_snapshot_matches_protocol"] is False
    assert "the pinned local model snapshot is missing or has mismatched files" in report["blockers"]


def test_historical_task_inputs_match_the_retained_v2_lock() -> None:
    report = collect_preflight(
        root=ROOT,
        model_path=ROOT / "missing-model",
        docker_probe=lambda: {"cli_available": False, "daemon_available": False},
    )

    assert report["inputs"]["task_inputs_match_protocol"] is True
    assert "the historical task inputs do not match the v2 protocol snapshot" not in report["blockers"]


def test_docker_probe_finds_a_running_nondefault_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda _name: "/usr/bin/docker")

    def fake_run(command: list[str], **_kwargs: object) -> object:
        if command[1:3] == ["context", "ls"]:
            return type("Result", (), {"returncode": 0, "stdout": "default\ncolima\n"})()
        context = command[2]
        if context == "default":
            return type("Result", (), {"returncode": 1, "stdout": ""})()
        return type("Result", (), {"returncode": 0, "stdout": "29.5.2\n"})()

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)

    assert preflight._probe_docker() == {
        "cli_available": True,
        "daemon_available": True,
        "contexts": [
            {"context": "default", "daemon_available": False, "server_version": None},
            {"context": "colima", "daemon_available": True, "server_version": "29.5.2"},
        ],
        "working_context": "colima",
    }
