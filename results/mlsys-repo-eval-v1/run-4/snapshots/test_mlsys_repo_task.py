from __future__ import annotations

from pathlib import Path

import pytest

import vare.mlsys_repo_task as repo_task
from vare.mlsys_repo_task import (
    RepositoryTaskError,
    _diff_paths,
    add_trusted_adapter,
    apply_candidate_patch,
)


def valid_diff(path: str = "mlx_lm/server.py") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "index 1234567..abcdef0 100644\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
    )


def test_patch_paths_are_confined_to_one_allowlisted_file():
    assert _diff_paths(valid_diff()) == ["mlx_lm/server.py", "mlx_lm/server.py"]
    attacks = [
        valid_diff("../../grader/hidden.py"),
        valid_diff("/tmp/escape.py"),
        valid_diff("mlx_lm/server.py") + valid_diff("mlx_lm/other.py"),
        valid_diff().replace("b/mlx_lm/server.py", "b/../../grader.py"),
        valid_diff().replace("diff --git a/mlx_lm/server.py b/mlx_lm/server.py", "diff --git a/mlx_lm/server.py"),
        valid_diff().replace("index 1234567..abcdef0 100644", "new file mode 100644"),
    ]
    for patch in attacks:
        with pytest.raises(RepositoryTaskError):
            _diff_paths(patch)


def test_trusted_adapter_cannot_be_supplied_by_candidate(tmp_path: Path):
    adapter = tmp_path / "vare_task_adapter.py"
    adapter.write_text("attacker content")
    with pytest.raises(RepositoryTaskError, match="already contains"):
        add_trusted_adapter(tmp_path)
    assert adapter.read_text() == "attacker content"


def test_trusted_adapter_uses_fixed_entrypoint_and_no_grader_cases(tmp_path: Path):
    add_trusted_adapter(tmp_path)
    source = (tmp_path / "vare_task_adapter.py").read_text()
    assert "process_message_content" in source
    assert "/subject/mlx_lm/server.py" in source
    assert "expected" not in source
    assert "grader" not in source


def test_patch_input_and_candidate_workspace_symlinks_fail_closed(tmp_path: Path, monkeypatch):
    root = tmp_path / "source"
    (root / "mlx_lm").mkdir(parents=True)
    (root / "mlx_lm/server.py").write_text("baseline")
    patch = tmp_path / "patch.diff"
    patch.write_text(valid_diff())
    patch_link = tmp_path / "patch-link.diff"
    patch_link.symlink_to(patch)
    workspace = tmp_path / "workspace"
    workspace.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(repo_task, "verify_baseline", lambda _root: None)

    with pytest.raises(RepositoryTaskError, match="regular file"):
        apply_candidate_patch(root, patch_link, tmp_path / "fresh")
    with pytest.raises(RepositoryTaskError, match="already exists"):
        apply_candidate_patch(root, patch, workspace)


def test_pinned_source_target_symlink_fails_closed(tmp_path: Path, monkeypatch):
    root = tmp_path / "source"
    package = root / "mlx_lm"
    package.mkdir(parents=True)
    (root / "real.py").write_text("safe")
    (package / "server.py").symlink_to(root / "real.py")
    monkeypatch.setattr(repo_task, "BASELINE_COMMIT", "test-commit")
    from subprocess import CompletedProcess

    calls = iter(("test-commit\n", ""))
    def fake_run(*_args, **_kwargs):
        return CompletedProcess([], 0, next(calls), "")
    monkeypatch.setattr(repo_task.subprocess, "run", fake_run)
    with pytest.raises(RepositoryTaskError, match="symlink"):
        repo_task.verify_baseline(root)

