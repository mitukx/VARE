from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from vare import transformers_49752_repo_task as task


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    root = tmp_path / "source"
    package = root / "src/transformers"
    target = package / "models/minicpmv4_6/video_processing_minicpmv4_6.py"
    target.parent.mkdir(parents=True)
    target.write_text("BASELINE = True\n", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "VARE test")
    _git(root, "config", "user.email", "vare-test@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "baseline")
    monkeypatch.setattr(task, "BASELINE_COMMIT", _git(root, "rev-parse", "HEAD"))
    monkeypatch.setattr(task, "BASELINE_TARGET_SHA256", hashlib.sha256(target.read_bytes()).hexdigest())
    monkeypatch.setattr(task, "PACKAGE_PATH", "src/transformers")
    return root, target


def test_baseline_hash_rejects_clean_status_bypass(tmp_path, monkeypatch):
    root, target = _baseline(tmp_path, monkeypatch)
    _git(root, "update-index", "--assume-unchanged", str(target.relative_to(root)))
    target.write_text("TAMPERED = True\n", encoding="utf-8")
    assert not _git(root, "status", "--porcelain")
    with pytest.raises(task.Transformers49752Error, match="index flags|source SHA-256"):
        task.verify_baseline(root)


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_baseline_rejects_hidden_git_index_flags(tmp_path, monkeypatch, flag):
    root, target = _baseline(tmp_path, monkeypatch)
    _git(root, "update-index", flag, str(target.relative_to(root)))
    with pytest.raises(task.Transformers49752Error, match="hidden Git index flags"):
        task.verify_baseline(root)


def test_patch_validator_rejects_traversal_and_unlisted_files():
    traversal = (
        "diff --git a/../hidden.py b/../hidden.py\n"
        "--- a/../hidden.py\n+++ b/../hidden.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    with pytest.raises(task.Transformers49752Error):
        task._validate_diff(traversal)
    extra_file = (
        f"diff --git a/{task.TARGET_PATH} b/{task.TARGET_PATH}\n"
        f"--- a/{task.TARGET_PATH}\n+++ b/{task.TARGET_PATH}\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/extra.py b/extra.py\n--- a/extra.py\n+++ b/extra.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    with pytest.raises(task.Transformers49752Error):
        task._validate_diff(extra_file)
