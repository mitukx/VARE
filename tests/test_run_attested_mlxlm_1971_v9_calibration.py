"""No-inference tests for immutable-source historical calibration wrapper."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_attested_mlxlm_1971_v9_calibration as calibration


def test_frozen_wrapper_disables_git_replacements_and_stages_pinned_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "upstream"
    source.mkdir()
    output = tmp_path / "new-evidence"
    commands: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append((command, kwargs))
        assert kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
        if command[:2] == ["git", "-C"] and "worktree" in command:
            assert kwargs["check"] is True
        if command[:2] == [sys.executable, "-I"]:
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    def trusted(_root: Path) -> dict[str, object]:
        return {
            "frozen_source_commit": calibration.PINNED_V9_COMMIT,
            "current_source_matches_pinned_git": True,
        }

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(calibration, "attest_current_v9", trusted)
    monkeypatch.setenv("GIT_NO_REPLACE_OBJECTS", "0")

    assert calibration.run_frozen_calibration(source, output) == 1
    assert len(commands) == 3
    add, invoke, remove = [c[0] for c in commands]
    assert add[-1] == calibration.PINNED_V9_COMMIT
    assert invoke[0:2] == [sys.executable, "-I"]
    assert invoke[invoke.index("--source-repo") + 1] == str(source)
    assert invoke[invoke.index("--output") + 1] == str(output)
    assert remove[3:6] == ["worktree", "remove", "--force"]


def test_frozen_wrapper_never_launches_on_failed_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(
        calibration, "attest_current_v9",
        lambda _root: (_ for _ in ()).throw(ValueError("tampered historical source")),
    )
    with pytest.raises(ValueError, match="tampered historical source"):
        calibration.run_frozen_calibration(source, tmp_path / "new-output")
    assert len(calls) == 2
    assert "worktree" in calls[0]
    assert "worktree" in calls[1]
    assert all(c[0] != sys.executable for c in calls)


def test_frozen_wrapper_refuses_preexisting_or_symlink_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    record = tmp_path / "retained"
    record.write_text("original evidence", encoding="utf-8")
    out = tmp_path / "new-evidence"
    out.symlink_to(record)
    monkeypatch.setattr(
        calibration.subprocess, "run",
        lambda *a, **k: pytest.fail("must not create worktree for existing output"),
    )
    with pytest.raises(FileExistsError, match="existing calibration output"):
        calibration.run_frozen_calibration(source, out)
    assert record.read_text(encoding="utf-8") == "original evidence"


def test_frozen_wrapper_rejects_changed_source_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setitem(calibration.SOURCE_COMMITS, 9, "0" * 40)
    monkeypatch.setattr(
        calibration.subprocess, "run",
        lambda *a, **k: pytest.fail("must refuse changed historical anchor"),
    )
    with pytest.raises(ValueError, match="historical v9 source anchor changed"):
        calibration.run_frozen_calibration(source, tmp_path / "new-evidence")
