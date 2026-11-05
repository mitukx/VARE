"""Workspace path checks for ResearchGym edits and staging.

These checks stop path escapes in the controller's edit API. They are not an
operating-system sandbox: a launched worker is still a same-user process.
"""

from __future__ import annotations

import os
from pathlib import Path


class PathViolation(ValueError):
    """An edit or stage path left the declared workspace."""


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise PathViolation("path must be a non-empty relative string")
    if "\\" in value or "\x00" in value:
        raise PathViolation("path contains a forbidden separator")
    raw = Path(value)
    if raw.is_absolute() or raw.anchor:
        raise PathViolation("absolute paths are forbidden")
    if any(part in {"", ".", ".."} for part in raw.parts):
        raise PathViolation("path traversal is forbidden")
    return raw


def resolve_inside(root: Path, relative: str) -> Path:
    """Return ``root/relative`` only if every existing component stays inside."""
    rel = safe_relative(relative)
    base = root.resolve()
    if base.is_symlink():
        raise PathViolation("workspace root is a symlink")
    current = base
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise PathViolation(f"symlink component is forbidden: {relative}")
        try:
            resolved = current.resolve()
        except OSError as exc:
            raise PathViolation(f"unresolvable path: {relative}") from exc
        if base != resolved and base not in resolved.parents:
            raise PathViolation(f"path escapes workspace: {relative}")
    final = base.joinpath(*rel.parts)
    parent = final.parent.resolve()
    if base != parent and base not in parent.parents:
        raise PathViolation(f"path escapes workspace: {relative}")
    return final


def assert_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise PathViolation(f"editable path is not a regular file: {path.name}")
    flags = path.stat().st_mode
    if not os.access(path, os.R_OK):
        raise PathViolation(f"editable path is not readable: {path.name}")
    if flags & os.X_OK and path.suffix == ".py":
        # Python sources may be marked executable; that is not itself a fault.
        return
