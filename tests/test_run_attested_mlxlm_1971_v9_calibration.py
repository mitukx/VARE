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
    assert invoke[invoke.index("--output") + 1] == str(output / "records")
    assert output.is_dir()
    assert output.stat().st_mode & 0o777 == 0o700
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


@pytest.mark.parametrize("action", ["grade", "export"])
def test_pinned_v9_task_dispatch_isolated_and_attested(
    action: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        assert kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
        if command[:3] == [sys.executable, "-I", "-c"]:
            request = calibration.json.loads(kwargs["input"])
            assert request["action"] == action
            assert request["workspace"] == str((tmp_path / "candidate").absolute())
            assert "src" in command[3] and "export_candidate" in command[3]
            if action == "export":
                assert request["source_repo"] == str(source)
            else:
                assert "source_repo" not in request
            return subprocess.CompletedProcess(command, 0, stdout='{"passed": false}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(
        calibration, "attest_current_v9",
        lambda _path: {
            "frozen_source_commit": calibration.PINNED_V9_COMMIT,
            "current_source_matches_pinned_git": True,
        },
    )
    report = calibration.run_pinned_v9_task(
        action,
        tmp_path / "candidate",
        source_repo=source if action == "export" else None,
    )
    assert report == {"passed": False}
    assert len(calls) == 3
    assert "worktree" in calls[0][0]
    assert calls[1][0][:3] == [sys.executable, "-I", "-c"]
    assert "worktree" in calls[2][0]


def test_pinned_v9_task_refuses_unattested_or_forged_child_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == [sys.executable, "-I", "-c"]:
            return subprocess.CompletedProcess(
                command, 0, stdout='{"passed": true}\\nFORGED OUTPUT'
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(
        calibration, "attest_current_v9",
        lambda _path: {
            "frozen_source_commit": calibration.PINNED_V9_COMMIT,
            "current_source_matches_pinned_git": True,
        },
    )
    with pytest.raises(RuntimeError, match="invalid JSON"):
        calibration.run_pinned_v9_task("grade", tmp_path / "candidate")
    assert len(calls) == 3

    calls.clear()
    monkeypatch.setattr(
        calibration, "attest_current_v9",
        lambda _path: {
            "frozen_source_commit": "0" * 40,
            "current_source_matches_pinned_git": False,
        },
    )
    with pytest.raises(ValueError, match="failed immutable attestation"):
        calibration.run_pinned_v9_task("grade", tmp_path / "candidate")
    assert len(calls) == 2
    assert all(cmd[0] != sys.executable for cmd in calls)


def test_pinned_v9_task_rejects_invalid_action_before_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        calibration.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("must refuse before Git operations"),
    )
    with pytest.raises(ValueError, match="unsupported pinned operation"):
        calibration.run_pinned_v9_task("delete", tmp_path / "candidate")
    with pytest.raises(ValueError, match="export requires source repo"):
        calibration.run_pinned_v9_task("export", tmp_path / "candidate")
    with pytest.raises(ValueError, match="grade cannot receive"):
        calibration.run_pinned_v9_task(
            "grade", tmp_path / "candidate", source_repo=tmp_path
        )


@pytest.mark.parametrize("argument", ["source", "patch", "workspace"])
def test_pinned_export_rejects_symlinks_before_dispatch(
    argument: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    patch = tmp_path / "fix.patch"
    patch.write_bytes(b"trusted patch bytes")
    workspace = tmp_path / "candidate"
    alias = tmp_path / "alias"
    alias.symlink_to({"source": source, "patch": patch, "workspace": workspace}[argument])
    monkeypatch.setattr(
        calibration.subprocess, "run",
        lambda *a, **k: pytest.fail("invalid paths must fail before Git"),
    )
    with pytest.raises(ValueError, match="symlink"):
        calibration.run_pinned_v9_task(
            "export", alias if argument == "workspace" else workspace,
            source_repo=alias if argument == "source" else source,
            patch_path=alias if argument == "patch" else patch,
        )


def test_calibration_reserves_output_before_launch_and_rejects_writable_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "reserved"
    redirected = tmp_path / "redirected"

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:2] == [sys.executable, "-I"]:
            assert output.is_dir() and not output.is_symlink()
            with pytest.raises(FileExistsError):
                output.symlink_to(redirected, target_is_directory=True)
            destination = Path(command[command.index("--output") + 1])
            assert destination == output / "records"
            destination.mkdir()
            (destination / "failure.json").write_text('{"status":"fixture_failure"}')
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(calibration, "attest_current_v9", lambda _: {
        "frozen_source_commit": calibration.PINNED_V9_COMMIT,
        "current_source_matches_pinned_git": True,
    })
    assert calibration.run_frozen_calibration(source, output) == 1
    assert (output / "records/failure.json").is_file()
    assert not redirected.exists()
    with pytest.raises(FileExistsError):
        calibration.run_frozen_calibration(source, output)
    unsafe = tmp_path / "shared"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    with pytest.raises(ValueError, match="operator-owned private parent"):
        calibration.run_frozen_calibration(source, unsafe / "output")
    assert not (unsafe / "output").exists()


def test_pinned_export_uses_one_private_patch_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    patch = tmp_path / "fix.patch"
    original = b"exact original patch\n"
    patch.write_bytes(original)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[:3] == [sys.executable, "-I", "-c"]:
            request = calibration.json.loads(kwargs["input"])
            snapshot = Path(request["patch_path"])
            assert snapshot != patch
            patch.write_bytes(b"replaced original patch\n")
            assert snapshot.read_bytes() == original
            return subprocess.CompletedProcess(command, 0, stdout='{"passed": false}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    monkeypatch.setattr(calibration, "attest_current_v9", lambda _: {
        "frozen_source_commit": calibration.PINNED_V9_COMMIT,
        "current_source_matches_pinned_git": True,
    })
    calibration.run_pinned_v9_task("export", tmp_path / "candidate", source_repo=source, patch_path=patch)
