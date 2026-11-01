"""Run Git against a pinned candidate workspace without ambient Git overrides."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess


class PinnedWorkspaceError(ValueError):
    """The candidate workspace or its source files are not trustworthy."""


def _workspace_paths(root: Path) -> tuple[Path, Path]:
    root = Path(os.path.abspath(root))
    try:
        root_info = root.lstat()
        git_dir = root / ".git"
        git_info = git_dir.lstat()
    except OSError as exc:
        raise PinnedWorkspaceError("workspace or Git metadata is missing") from exc
    if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
        raise PinnedWorkspaceError("workspace must be a real directory")
    if not stat.S_ISDIR(git_info.st_mode) or stat.S_ISLNK(git_info.st_mode):
        raise PinnedWorkspaceError("workspace must own a real .git directory")
    return root, git_dir


def _git_environment() -> dict[str, str]:
    # Do not inherit GIT_DIR, GIT_INDEX_FILE, replacement-ref, config-injection,
    # external-diff, object-directory, or tracing variables from the caller.
    path = os.environ.get("PATH", os.defpath)
    return {
        "PATH": path,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    }


def pinned_git(
    root: Path,
    *args: str,
    timeout: int = 20,
    index_file: Path | None = None,
    max_output_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    """Run Git with an explicit repository and a minimal, noninteractive env."""
    root, git_dir = _workspace_paths(root)
    env = _git_environment()
    if index_file is not None:
        env["GIT_INDEX_FILE"] = str(Path(os.path.abspath(index_file)))
    git = shutil.which("git", path=env["PATH"])
    if git is None:
        raise PinnedWorkspaceError("Git executable is unavailable")
    command = [
        git,
        f"--git-dir={git_dir}",
        f"--work-tree={root}",
        "-c", "core.fsmonitor=false",
        "-c", "core.quotePath=false",
        "-c", "core.attributesFile=/dev/null",
        "-c", "core.excludesFile=/dev/null",
        "-c", "diff.external=",
        "-c", "core.pager=cat",
        *args,
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PinnedWorkspaceError("pinned Git command did not complete") from exc
    if completed.returncode != 0:
        raise PinnedWorkspaceError("pinned Git command failed")
    if len(completed.stdout) > max_output_bytes:
        raise PinnedWorkspaceError("pinned Git output exceeded its byte limit")
    return completed.stdout


def _safe_source_parts(name: str) -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\\" in name:
        raise PinnedWorkspaceError("invalid source path")
    path = PurePosixPath(name)
    parts = tuple(name.split("/"))
    if (path.is_absolute() or path.as_posix() != name or
            any(part in ("", ".", "..", ".git") for part in parts)):
        raise PinnedWorkspaceError("unsafe source path")
    return parts


def read_regular_source(
    root: Path, name: str, *, max_bytes: int = 16 * 1024 * 1024,
) -> bytes:
    """Read a source file without following symlinks in any path component."""
    root, _ = _workspace_paths(root)
    parts = _safe_source_parts(name)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if not directory_flag or not nofollow_flag:
        raise PinnedWorkspaceError("platform lacks no-follow directory opens")
    descriptors: list[int] = []
    try:
        current = os.open(root, os.O_RDONLY | directory_flag | nofollow_flag)
        descriptors.append(current)
        for part in parts[:-1]:
            current = os.open(
                part,
                os.O_RDONLY | directory_flag | nofollow_flag,
                dir_fd=current,
            )
            descriptors.append(current)
        file_fd = os.open(parts[-1], os.O_RDONLY | nofollow_flag, dir_fd=current)
        descriptors.append(file_fd)
        file_info = os.fstat(file_fd)
        if not stat.S_ISREG(file_info.st_mode):
            raise PinnedWorkspaceError("source path is not a regular file")
        if file_info.st_size > max_bytes:
            raise PinnedWorkspaceError("source file exceeded its byte limit")
        chunks: list[bytes] = []
        total = 0
        while block := os.read(file_fd, 1024 * 1024):
            total += len(block)
            if total > max_bytes:
                raise PinnedWorkspaceError("source file exceeded its byte limit")
            chunks.append(block)
        return b"".join(chunks)
    except OSError as exc:
        raise PinnedWorkspaceError("source path is missing, linked, or unreadable") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
