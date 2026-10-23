"""Source-only regressions for the prospective v12 source-Git boundary."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts import run_mlxlm_1971_v11_operator as v11
from scripts import run_mlxlm_1971_v12_operator as v12


def _git(repository: Path, *args: str, env=None) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *args],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def test_v12_snapshots_untrusted_git_without_running_filters_or_uploadpack_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q")
    _git(source, "config", "user.name", "VARE regression")
    _git(source, "config", "user.email", "vare-test@example.invalid")
    relative = Path("mlx_lm/models/mimo_v2_flash.py")
    source_file = source / relative
    source_file.parent.mkdir(parents=True)
    expected = b"value = 'pinned source'\n"
    source_file.write_bytes(expected)
    _git(source, "add", relative.as_posix())
    _git(source, "commit", "-qm", "pinned baseline fixture")
    commit = _git(source, "rev-parse", "HEAD")

    filter_marker = tmp_path / "filter-ran"
    uploadpack_marker = tmp_path / "uploadpack-hook-ran"
    smudge = tmp_path / "smudge.sh"
    smudge.write_text(
        '#!/bin/sh\nprintf x >> "$VARE_FILTER_MARKER"\n/bin/cat\n',
        encoding="utf-8",
    )
    smudge.chmod(0o700)
    pack_hook = tmp_path / "pack-hook.sh"
    pack_hook.write_text(
        '#!/bin/sh\nprintf x >> "$VARE_PACK_MARKER"\nexec git pack-objects "$@"\n',
        encoding="utf-8",
    )
    pack_hook.chmod(0o700)
    _git(source, "config", "filter.redteam.smudge", f"/bin/sh {smudge}")
    _git(source, "config", "uploadpack.packObjectsHook", f"/bin/sh {pack_hook}")
    (source / ".git/info/attributes").write_text(
        f"{relative.as_posix()} filter=redteam\n", encoding="utf-8"
    )
    monkeypatch.setitem(v11.GIT_ENV, "VARE_FILTER_MARKER", str(filter_marker))
    monkeypatch.setitem(v11.GIT_ENV, "VARE_PACK_MARKER", str(uploadpack_marker))

    direct_checkout = tmp_path / "direct-checkout"
    direct = subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "worktree",
            "add",
            "--detach",
            str(direct_checkout),
            commit,
        ],
        env=v11.GIT_ENV,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert direct.returncode == 0, direct.stderr
    assert filter_marker.read_text(encoding="utf-8") == "x"

    isolated = v12._snapshot_source_repository(
        source, tmp_path / "source-snapshot.git", commit
    )
    safe_checkout = tmp_path / "safe-checkout"
    subprocess.run(
        v11._git_command(
            isolated, "worktree", "add", "--detach", str(safe_checkout), commit
        ),
        env=v11.GIT_ENV,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    assert (safe_checkout / relative).read_bytes() == expected
    assert filter_marker.read_text(encoding="utf-8") == "x"
    assert not uploadpack_marker.exists()
    assert not (isolated / "info/attributes").exists()
    filters = subprocess.run(
        ["git", "-C", str(isolated), "config", "--local", "--get-regexp", r"^filter\."],
        env=v11.GIT_ENV,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert filters.returncode == 1
    assert filters.stdout == ""


def test_v12_protocol_pins_v11_dependencies_without_running_candidate_code() -> None:
    protocol = v12.verify_v12_protocol()
    assert protocol["protocol_id"] == v12.PROTOCOL_ID
    assert protocol["source_repository_checked_out_directly"] is False
    assert (
        protocol["source_snapshot_method"] == "fresh-depth-one-bare-fetch-no-checkout"
    )


def test_v12_export_passes_only_the_source_snapshot_to_v11(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "untrusted-source"
    snapshot = tmp_path / "isolated-source.git"
    source.mkdir()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        v12, "verify_v12_protocol", lambda: {"protocol_id": v12.PROTOCOL_ID}
    )
    monkeypatch.setattr(
        v11, "_trusted_v9_protocol", lambda: {"baseline_commit": v12.BASELINE_COMMIT}
    )

    def make_snapshot(source_repository: Path, destination: Path, commit: str) -> Path:
        observed["snapshot_args"] = (source_repository, destination, commit)
        return snapshot

    def run_v11(action: str, workspace: Path, **kwargs: object) -> dict[str, object]:
        observed["v11_args"] = (action, workspace, kwargs)
        return {"task_id": "mlxlm-1971-mimo-value-scale"}

    monkeypatch.setattr(v12, "_snapshot_source_repository", make_snapshot)
    monkeypatch.setattr(v11, "run_v11_operation", run_v11)
    report = v12.run_v12_operation(
        "export", tmp_path / "workspace", source_repository=source
    )
    assert observed["snapshot_args"] == (
        source,
        observed["snapshot_args"][1],
        v12.BASELINE_COMMIT,
    )
    assert observed["v11_args"] == (
        "export",
        tmp_path / "workspace",
        {"source_repo": snapshot, "patch_path": None},
    )
    assert report == {
        "task_id": "mlxlm-1971-mimo-value-scale",
        "operator_protocol_id": v12.PROTOCOL_ID,
        "operator_protocol_sha256": v12.PROTOCOL_SHA256,
        "source_snapshot_commit": v12.BASELINE_COMMIT,
    }


def test_v12_rejects_invalid_action_before_protocol_or_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        v12,
        "verify_v12_protocol",
        lambda: pytest.fail("invalid action reached protocol"),
    )
    with pytest.raises(ValueError, match="unsupported v12 operation"):
        v12.run_v12_operation("clean", tmp_path / "workspace")
