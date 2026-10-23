"""Regression controls for Git-object-anchored v9 operator preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import attest_mlxlm_1971_v9_current_git as preflight


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _frozen_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    (root / "protocols").mkdir()
    (root / "impl.py").write_text("trusted code\n", encoding="utf-8")
    source_hash = hashlib.sha256((root / "impl.py").read_bytes()).hexdigest()
    (root / "protocols/protocol-v9.json").write_text(
        json.dumps({"protocol_id": "v9-test", "implementation_sha256": {"impl.py": source_hash}}),
        encoding="utf-8",
    )
    protocol_bytes = (root / "protocols/protocol-v9.json").read_bytes()
    (root / "protocols/protocol-v9.lock.json").write_text(
        json.dumps({
            "protocol_id": "v9-test",
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "implementation_sha256": {"impl.py": source_hash},
        }),
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "freeze")
    pinned = _git(root, "rev-parse", "HEAD")
    monkeypatch.setitem(preflight.SOURCE_COMMITS, 9, pinned)
    monkeypatch.setattr(preflight, "TASK_ROOT", "protocols")
    return root, pinned


def test_v9_attestation_accepts_exact_immutable_git_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, pinned = _frozen_fixture(tmp_path, monkeypatch)
    report = preflight.attest_current_v9(root)
    assert report["frozen_source_commit"] == pinned
    assert report["implementation_files_verified"] == 1
    assert report["current_source_matches_pinned_git"] is True


def test_v9_attestation_rejects_coherent_worktree_and_commit_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, pinned = _frozen_fixture(tmp_path, monkeypatch)
    (root / "impl.py").write_text("altered code\n", encoding="utf-8")
    changed_hash = hashlib.sha256((root / "impl.py").read_bytes()).hexdigest()
    protocol_bytes = json.dumps({
        "protocol_id": "v9-test",
        "implementation_sha256": {"impl.py": changed_hash},
    }).encode()
    (root / "protocols/protocol-v9.json").write_bytes(protocol_bytes)
    (root / "protocols/protocol-v9.lock.json").write_text(
        json.dumps({
            "protocol_id": "v9-test",
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "implementation_sha256": {"impl.py": changed_hash},
        }),
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "tamper")
    replacement = _git(root, "rev-parse", "HEAD")
    _git(root, "replace", pinned, replacement)
    with pytest.raises(ValueError, match="frozen protocol differs from Git source"):
        preflight.attest_current_v9(root)


def test_v9_attestation_rejects_skip_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _frozen_fixture(tmp_path, monkeypatch)
    _git(root, "update-index", "--skip-worktree", "impl.py")
    with pytest.raises(ValueError, match="hidden v9 Git index flag"):
        preflight.attest_current_v9(root)


def test_v9_attestation_rejects_source_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = _frozen_fixture(tmp_path, monkeypatch)
    (root / "impl.py").unlink()
    external = tmp_path / "elsewhere.py"
    external.write_text("trusted code\n", encoding="utf-8")
    (root / "impl.py").symlink_to(external)
    with pytest.raises(ValueError, match="not a regular file"):
        preflight.attest_current_v9(root)


def test_attested_grade_entrypoint_checks_source_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vare import mlxlm_1971_repo_task as public

    seen: list[str] = []

    def verify(root: Path) -> dict[str, object]:
        seen.append("verify")
        return {"current_source_matches_pinned_git": True}

    def grade(workspace: Path) -> dict[str, object]:
        seen.append("grade")
        return {"passed": False}

    monkeypatch.setattr(preflight, "attest_current_v9", verify)
    monkeypatch.setattr(public, "grade_workspace", grade)
    assert preflight.grade_workspace_attested_v9(tmp_path) == {"passed": False}
    assert seen == ["verify", "grade"]

    def refuse(root: Path) -> dict[str, object]:
        raise ValueError("modified source")

    monkeypatch.setattr(preflight, "attest_current_v9", refuse)
    with pytest.raises(ValueError, match="modified source"):
        preflight.grade_workspace_attested_v9(tmp_path)
    assert seen == ["verify", "grade"]


def test_attested_export_entrypoint_checks_source_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vare import mlxlm_1971_repo_task as public

    seen: list[str] = []

    def verify(root: Path) -> dict[str, object]:
        seen.append("verify")
        return {"current_source_matches_pinned_git": True}

    def export(source: Path, target: Path, patch: Path | None) -> dict[str, object]:
        seen.append("export")
        return {"changed_files": []}

    monkeypatch.setattr(preflight, "attest_current_v9", verify)
    monkeypatch.setattr(public, "export_candidate", export)
    assert preflight.export_candidate_attested_v9(tmp_path, tmp_path / "new") == {
        "changed_files": []
    }
    assert seen == ["verify", "export"]


def test_current_checkout_matches_immutable_historical_v9_git() -> None:
    """CI gate: a coherently edited evaluator/lock cannot retain v9 identity."""
    root = Path(__file__).resolve().parents[1]
    report = preflight.attest_current_v9(root)
    assert report["current_source_matches_pinned_git"] is True
    assert report["implementation_files_verified"] == 20
