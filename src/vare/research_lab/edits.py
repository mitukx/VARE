"""Apply bounded text edits inside an already materialized workspace."""

from __future__ import annotations

from pathlib import Path

from .paths import PathViolation, assert_regular_file, resolve_inside


class EditError(ValueError):
    """The requested text edit does not match the current file."""


def read_files(workspace: Path, names: tuple[str, ...] | list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for name in names:
        path = resolve_inside(workspace, name)
        assert_regular_file(path)
        files[name] = path.read_text(encoding="utf-8")
    return files


def apply_edits(
    workspace: Path,
    allowed: tuple[str, ...] | list[str],
    edits: list[dict[str, object]],
) -> None:
    if not isinstance(edits, list) or not edits:
        raise EditError("edits must be a non-empty list")
    allow = set(allowed)
    for edit in edits:
        if not isinstance(edit, dict):
            raise EditError("each edit must be an object")
        path = edit.get("path")
        old = edit.get("old")
        new = edit.get("new")
        count = edit.get("count", 1)
        if not isinstance(path, str) or path not in allow:
            raise PathViolation("edit path is not an editable file")
        if not isinstance(old, str) or not isinstance(new, str) or old == "":
            raise EditError("edit old/new must be strings and old must be non-empty")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise EditError("edit count must be a positive integer")
        dest = resolve_inside(workspace, path)
        assert_regular_file(dest)
        text = dest.read_text(encoding="utf-8")
        found = text.count(old)
        if found != count:
            raise EditError(f"{path}: expected {count} occurrence(s), found {found}")
        updated = text.replace(old, new, count)
        if updated == text:
            raise EditError(f"{path}: edit did not change the file")
        dest.write_text(updated, encoding="utf-8")


def write_files(
    workspace: Path,
    allowed: tuple[str, ...] | list[str],
    files: dict[str, str],
) -> None:
    if not isinstance(files, dict) or not files:
        raise EditError("files must be a non-empty object")
    allow = set(allowed)
    for path, content in files.items():
        if path not in allow:
            raise PathViolation("write path is not an editable file")
        if not isinstance(content, str):
            raise EditError("file contents must be text")
        dest = resolve_inside(workspace, path)
        if dest.exists():
            assert_regular_file(dest)
        elif dest.parent.is_symlink():
            raise PathViolation("parent symlink is forbidden")
        dest.write_text(content, encoding="utf-8")
