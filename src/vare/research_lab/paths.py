"""Workspace path checks for the experiment runner.

These checks reject traversal and symlinks at the moment a path is used.
They do not stop a same-user process from opening an absolute path later.
"""
from __future__ import annotations

from pathlib import Path


class PathPolicyError(ValueError):
    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class


def _root(path: Path) -> Path:
    if path.is_symlink():
        raise PathPolicyError("symlink_escape", "root path is a symlink")
    if not path.is_dir():
        raise PathPolicyError("path_traversal", "root path is not a directory")
    return path


def ensure_tree_unlinked(root: Path, leaf: Path) -> None:
    """Reject a symlink at any component from ``root`` through ``leaf``."""
    root = _root(root)
    if leaf.is_absolute():
        try:
            relative = leaf.relative_to(root)
        except ValueError as exc:
            raise PathPolicyError("path_traversal", "path is outside the root") from exc
    else:
        relative = leaf
    if relative.is_absolute() or ".." in relative.parts:
        raise PathPolicyError("path_traversal", "path escapes the root")
    current = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise PathPolicyError("path_traversal", "path escapes the root")
        current = current / part
        if current.is_symlink():
            raise PathPolicyError("symlink_escape", "path contains a symlink")
        if current.exists() and current.is_dir():
            continue
    if current.exists() or current.parent.exists():
        try:
            resolved_root = root.resolve()
            resolved = current.resolve() if current.exists() else current.parent.resolve() / current.name
        except OSError as exc:
            raise PathPolicyError("path_traversal", "path cannot be resolved") from exc
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise PathPolicyError("path_traversal", "path resolves outside the root")


def resolve_inside(root: Path, relative: str, *, expect: str = "any") -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise PathPolicyError("path_traversal", "relative path is empty or unsafe")
    if relative == ".":
        root = _root(root)
        if expect == "file":
            raise PathPolicyError("path_traversal", "expected a regular file inside the root")
        return root
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise PathPolicyError("path_traversal", "relative path escapes the root")
    root = _root(root)
    leaf = root.joinpath(*raw.parts)
    ensure_tree_unlinked(root, leaf)
    if expect == "file":
        if leaf.is_symlink():
            raise PathPolicyError("symlink_escape", "expected a regular file inside the root")
        if not leaf.exists():
            raise PathPolicyError("missing_file", "expected file is missing")
        if not leaf.is_file():
            raise PathPolicyError("path_traversal", "expected a regular file inside the root")
    elif expect == "dir":
        if leaf.is_symlink() or not leaf.is_dir():
            raise PathPolicyError("path_traversal", "expected a directory inside the root")
    return leaf


def regular_outside(path: Path, workspace: Path) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise PathPolicyError("symlink_escape", "protected path must be an absolute non-symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise PathPolicyError("path_traversal", "protected path is missing") from exc
    if resolved != path or not resolved.is_file():
        raise PathPolicyError("symlink_escape", "protected path is not a stable regular file")
    workspace_resolved = _root(workspace).resolve()
    if resolved == workspace_resolved or workspace_resolved in resolved.parents:
        raise PathPolicyError("path_traversal", "protected path must stay outside the workspace")
    return resolved
