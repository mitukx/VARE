"""Source-only, no-model regression checks for the prospective v10 operator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_mlxlm_1971_v10_operator as v10


def test_v10_protocol_binds_git_commit_and_frozen_v9_without_model() -> None:
    protocol = v10.verify_v10_protocol()
    assert protocol["operator_source_commit"] == v10.OPERATOR_COMMIT
    assert protocol["reference_v9_source_commit"] == v10.V9_COMMIT
    assert protocol["allowed_actions"] == ["export", "grade"]
    assert hashlib.sha256(v10.PROTOCOL.read_bytes()).hexdigest() == v10.PROTOCOL_SHA256


def test_v10_coherent_protocol_and_lock_edit_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = json.loads(v10.PROTOCOL.read_text(encoding="utf-8"))
    protocol["allowed_actions"] = ["grade", "export"]
    candidate = tmp_path / "protocol-v10.json"
    candidate.write_text(json.dumps(protocol), encoding="utf-8")
    lock = tmp_path / "protocol-v10.lock.json"
    lock.write_text(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    monkeypatch.setattr(v10, "PROTOCOL", candidate)
    monkeypatch.setattr(v10, "LOCK", lock)
    monkeypatch.setattr(
        v10, "_git_bytes",
        lambda *args: pytest.fail("coherent protocol tampering must fail before Git"),
    )
    with pytest.raises(ValueError, match="independently pinned bytes"):
        v10.verify_v10_protocol()


def test_v10_rejects_invalid_actions_before_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        v10, "verify_v10_protocol",
        lambda: pytest.fail("bad input must fail before protocol verification"),
    )
    with pytest.raises(ValueError, match="unsupported v10 operation"):
        v10.run_v10_operation("delete", tmp_path / "candidate")
    with pytest.raises(ValueError, match="export requires source"):
        v10.run_v10_operation("export", tmp_path / "candidate")
    with pytest.raises(ValueError, match="grade cannot receive"):
        v10.run_v10_operation("grade", tmp_path / "candidate", source_repo=tmp_path)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlinked candidate"):
        v10.run_v10_operation("grade", symlink)


@pytest.mark.parametrize("action", ["grade", "export"])
def test_v10_dispatch_uses_pinned_source_and_disables_replace_refs(
    action: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    operations: list[list[str]] = []
    source = tmp_path / "upstream"
    source.mkdir()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        operations.append(command)
        assert kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
        if command[:2] == ["git", "-C"] and "worktree" in command:
            assert kwargs["check"] is True
            if "add" in command:
                checkout = Path(command[-2])
                checked = checkout / v10.SOURCE_PATH
                checked.parent.mkdir(parents=True)
                checked.write_bytes(b"trusted historical operator")
        if command[:3] == [sys.executable, "-I", "-c"]:
            request = json.loads(kwargs["input"])
            assert request["action"] == action
            if action == "export":
                assert request["source_repo"] == str(source)
            assert "run_pinned_v9_task" in command[3]
            return subprocess.CompletedProcess(command, 0, stdout='{"passed": false}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(v10, "verify_v10_protocol", lambda: {})
    monkeypatch.setattr(v10, "_git_bytes", lambda *_: b"trusted historical operator")
    monkeypatch.setattr(v10.subprocess, "run", fake_run)
    result = v10.run_v10_operation(
        action, tmp_path / "workspace",
        source_repo=source if action == "export" else None,
    )
    assert result == {"passed": False}
    assert len(operations) == 3
    assert operations[0][-1] == v10.OPERATOR_COMMIT
    assert operations[1][:3] == [sys.executable, "-I", "-c"]
    assert operations[2][3:6] == ["worktree", "remove", "--force"]


def test_v10_forged_result_is_rejected_and_worktree_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "add" in command:
            checkout = Path(command[-2])
            source = checkout / v10.SOURCE_PATH
            source.parent.mkdir(parents=True)
            source.write_bytes(b"trusted historical operator")
        if command[:3] == [sys.executable, "-I", "-c"]:
            return subprocess.CompletedProcess(
                command, 0, stdout='{"passed": true}\\n{"passed": false}'
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(v10, "verify_v10_protocol", lambda: {})
    monkeypatch.setattr(v10, "_git_bytes", lambda *_: b"trusted historical operator")
    monkeypatch.setattr(v10.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="one JSON object"):
        v10.run_v10_operation("grade", tmp_path / "workspace")
    assert len(calls) == 3
    assert "remove" in calls[-1]
