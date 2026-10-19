"""Read-only verification of a pinned coding-agent Git workspace.

Does not import source modules, run the grader, or execute candidate code.
Git is invoked with disabled external diff filters and lazy network fetching.
This is consistency evidence, not tamper-proof external attestation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any


class WorkspaceMismatch(ValueError):
    """Workspace state is inconsistent with the agent's run record."""


def _expect(predicate: bool, message: str) -> None:
    if not predicate:
        raise WorkspaceMismatch(message)


def _git(root: Path, *args: str) -> bytes:
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0",
               GIT_EXTERNAL_DIFF="", GIT_PAGER="cat")
    completed = subprocess.run(
        ["git", "-C", str(root), "--no-optional-locks", "-c", "diff.external=",
         "-c", "core.quotePath=false", "-c", "core.fsmonitor=false", *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env=env, timeout=20,
    )
    _expect(completed.returncode == 0, f"Git verification failed: {args[0]}")
    _expect(len(completed.stdout) <= 4 * 1024 * 1024, "Git verification output oversized")
    return completed.stdout


def verify_workspace(workspace: Path, task: dict[str, Any], agent: dict[str, Any],
                     grade: dict[str, Any]) -> dict[str, Any]:
    """Bind source hashes and diff to *actual* pinned on-disk source bytes."""
    _expect(workspace.is_dir() and not workspace.is_symlink(), "missing/linked workspace")
    _expect((workspace / ".git").is_dir() and not (workspace / ".git").is_symlink(),
            "workspace must own its Git metadata directory")
    source = task.get("source", {})
    files = source.get("files")
    _expect(type(files) is list and len(files) > 0 and
            len(set(files)) == len(files), "invalid task source paths")
    for name in files:
        _expect(type(name) is str and bool(name) and
                PurePosixPath(name).as_posix() == name and
                not PurePosixPath(name).is_absolute() and
                all(piece not in ("..", ".", "") for piece in name.split("/")) and
                not name.startswith(".git/"), "unsafe task source path")
    expected_rev = source.get("base_revision")
    _expect(_git(workspace, "rev-parse", "HEAD").decode().strip() == expected_rev,
            "workspace HEAD not pinned to task base")
    # Only unstaged modifications of task source files are permitted.
    status = _git(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    allowed = set(files)
    for line in status.decode("utf-8").splitlines():
        _expect(line.startswith(" M ") and line[3:] in allowed,
                "workspace contains staged, untracked or non-allowlisted change")

    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for name in files:
        candidate = workspace / name
        _expect(candidate.is_file() and not candidate.is_symlink(),
                f"missing or linked candidate file: {name}")
        for parent in candidate.relative_to(workspace).parents:
            if parent == Path("."):
                continue
            _expect(not (workspace / parent).is_symlink(), "linked source parent directory")
        content = candidate.read_bytes()
        _expect(len(content) <= 16 * 1024 * 1024, "oversized candidate source")
        after[name] = hashlib.sha256(content).hexdigest()
        original = _git(workspace, "cat-file", "blob", f"{expected_rev}:{name}")
        before[name] = hashlib.sha256(original).hexdigest()
    _expect(agent.get("source_hashes_before") == before, "original source bytes do not match agent")
    _expect(agent.get("source_hashes_after") == after, "candidate source bytes do not match agent")
    _expect(grade.get("source_file_sha256") == after, "grader did not hash verified workspace")
    recorded_diff = agent.get("candidate_diff")
    _expect(type(recorded_diff) is str, "missing candidate diff")
    actual_diff = _git(workspace, "diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                       "--binary", expected_rev, "--", *sorted(files)).decode("utf-8").strip()
    _expect(recorded_diff == actual_diff, "Git workspace diff does not match agent record")
    _expect(grade.get("candidate_diff_sha256") == hashlib.sha256(actual_diff.encode()).hexdigest(),
            "grader did not hash verified Git diff")
    return {
        "schema_version": 1, "workspace_revision": expected_rev,
        "verified_source_sha256": after, "verified_base_sha256": before,
        "verified_diff_sha256": hashlib.sha256(actual_diff.encode()).hexdigest(),
        "workspace_verification": "read-only source/Git consistency (not external attestation)",
    }
