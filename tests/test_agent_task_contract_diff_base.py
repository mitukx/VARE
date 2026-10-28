from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vare.agent_task_contract import resolve_candidate_diff_revision


def _commit_workspace(root: Path, source: str = "baseline\n") -> str:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "VARE test"], check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "vare-test@example.invalid"],
        check=True,
    )
    source_path = root / "src" / "module.py"
    source_path.parent.mkdir()
    source_path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "src/module.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "baseline"], check=True)
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_resolves_candidate_local_baseline_when_upstream_revision_is_not_present(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    commit = _commit_workspace(workspace)

    assert resolve_candidate_diff_revision(
        workspace,
        "HEAD",
        ["src/module.py"],
    ) == commit


def test_rejects_unavailable_upstream_revision_before_model_load(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    _commit_workspace(workspace)

    with pytest.raises(ValueError, match="baseline is unavailable"):
        resolve_candidate_diff_revision(
            workspace,
            "6c5f8f4186c15f4bfbd67941a1736b2b347e0155",
            ["src/module.py"],
        )


def test_rejects_baseline_that_does_not_match_candidate_starting_source(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    _commit_workspace(workspace)
    (workspace / "src" / "module.py").write_text("changed before inference\n", encoding="utf-8")

    with pytest.raises(ValueError, match="diff baseline differs"):
        resolve_candidate_diff_revision(workspace, "HEAD", ["src/module.py"])


def test_rejects_symlinked_candidate_source(tmp_path: Path) -> None:
    workspace = tmp_path / "candidate"
    commit = _commit_workspace(workspace)
    source = workspace / "src" / "module.py"
    source.unlink()
    source.symlink_to(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text("baseline\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a regular file"):
        resolve_candidate_diff_revision(workspace, commit, ["src/module.py"])
