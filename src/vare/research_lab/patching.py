"""Apply a unified diff inside a workspace, or leave the workspace unchanged.

The applier writes regular files only. It does not create symlinks, run a
shell, or follow a path out of the workspace.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .jsonutil import digest_bytes
from .paths import PathPolicyError, resolve_inside

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(ValueError):
    def __init__(self, failure_class: str, message: str) -> None:
        super().__init__(message)
        self.failure_class = failure_class


@dataclass(frozen=True, slots=True)
class FilePatch:
    relative: str
    old_is_null: bool
    new_is_null: bool
    hunks: tuple[tuple[int, tuple[tuple[str, str], ...]], ...]


@dataclass(frozen=True, slots=True)
class PatchApplication:
    applied: bool
    failure_class: str | None
    files: tuple[str, ...]
    patch_sha256: str


def _relative_from_diff_path(token: str) -> tuple[str, bool]:
    if token == "/dev/null":
        return "/dev/null", True
    if token.startswith('"') or "\\" in token or token.startswith("/"):
        raise PatchError("path_traversal", "patch path is not a plain relative path")
    for prefix in ("a/", "b/"):
        if token.startswith(prefix):
            token = token[len(prefix) :]
            break
    path = Path(token)
    if not token or path.is_absolute() or ".." in path.parts or ":" in token:
        raise PatchError("path_traversal", "patch path escapes the workspace")
    return token, False


def parse_unified_diff(text: str) -> tuple[FilePatch, ...]:
    if not isinstance(text, str) or not text.strip() or "\x00" in text or len(text) > 1_000_000:
        raise PatchError("invalid_patch", "patch text is empty, binary, or too large")
    if "GIT binary patch" in text or text.startswith("Binary files"):
        raise PatchError("invalid_patch", "binary patches are rejected")
    lines = text.splitlines(keepends=True)
    if not lines[0].startswith("diff --git "):
        raise PatchError("invalid_patch", "patch must be a unified diff")
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("diff --git ") and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    parsed: list[FilePatch] = []
    for block in blocks:
        header = block[0].rstrip("\n")
        parts = header.split()
        if len(parts) != 4 or parts[0] != "diff" or parts[1] != "--git":
            raise PatchError("invalid_patch", "malformed diff header")
        old_git, old_null = _relative_from_diff_path(parts[2])
        new_git, new_null = _relative_from_diff_path(parts[3])
        if any(line.startswith(("rename from ", "rename to ", "similarity index ")) for line in block):
            raise PatchError("invalid_patch", "renames are rejected")
        header_lines: list[str] = []
        for line in block[1:]:
            if line.startswith("@@ "):
                break
            header_lines.append(line)
        minus = [line for line in header_lines if line.startswith("--- ")]
        plus = [line for line in header_lines if line.startswith("+++ ")]
        if len(minus) != 1 or len(plus) != 1:
            raise PatchError("invalid_patch", "patch file header is incomplete")
        old_path, old_header_null = _relative_from_diff_path(minus[0][4:].strip())
        new_path, new_header_null = _relative_from_diff_path(plus[0][4:].strip())
        if old_null != old_header_null or new_null != new_header_null:
            raise PatchError("invalid_patch", "diff header disagrees with ---/+++ paths")
        relative = new_path if not new_header_null else old_path
        if relative in {"/dev/null"} or old_git not in {relative, "/dev/null"} or new_git not in {relative, "/dev/null"}:
            raise PatchError("invalid_patch", "diff paths disagree")
        if old_git not in {relative, "/dev/null"} or (not old_header_null and old_git != relative):
            raise PatchError("invalid_patch", "diff paths disagree")
        hunks: list[tuple[int, tuple[tuple[str, str], ...]]] = []
        index = 0
        while index < len(block) and not block[index].startswith("@@ "):
            index += 1
        while index < len(block):
            match = _HUNK.match(block[index])
            if match is None:
                raise PatchError("invalid_patch", "malformed hunk header")
            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_count = int(match.group(4) or "1")
            index += 1
            body: list[tuple[str, str]] = []
            seen_old = seen_new = 0
            while index < len(block) and not block[index].startswith("@@ "):
                line = block[index]
                index += 1
                if line.startswith("\\"):
                    if not body:
                        raise PatchError("invalid_patch", "newline marker without a line")
                    kind, content = body[-1]
                    body[-1] = (kind, content.rstrip("\n"))
                    continue
                if not line or line[0] not in " +-":
                    raise PatchError("invalid_patch", "unexpected hunk line")
                kind, content = line[0], line[1:]
                if kind == " ":
                    seen_old += 1
                    seen_new += 1
                elif kind == "-":
                    seen_old += 1
                else:
                    seen_new += 1
                body.append((kind, content if content.endswith("\n") or content == "" else content))
            if seen_old != old_count or seen_new != new_count:
                raise PatchError("invalid_patch", "hunk counts do not match")
            hunks.append((old_start, tuple(body)))
        if not hunks and not (old_header_null or new_header_null):
            raise PatchError("invalid_patch", "patch contains no hunks")
        parsed.append(FilePatch(relative, old_header_null, new_header_null, tuple(hunks)))
    if not parsed or len({item.relative for item in parsed}) != len(parsed):
        raise PatchError("invalid_patch", "patch files must be unique and nonempty")
    return tuple(parsed)


def _apply_hunks(original: bytes, hunks: tuple[tuple[int, tuple[tuple[str, str], ...]], ...]) -> bytes:
    text = original.decode("utf-8")
    old_lines = text.splitlines(keepends=True) if text else []
    new_lines: list[str] = []
    cursor = 0
    for old_start, body in hunks:
        start = 0 if old_start == 0 else old_start - 1
        if start < cursor or start > len(old_lines):
            raise PatchError("invalid_patch", "hunk does not apply")
        new_lines.extend(old_lines[cursor:start])
        cursor = start
        for kind, content in body:
            if kind in {" ", "-"}:
                if cursor >= len(old_lines) or old_lines[cursor] != content:
                    raise PatchError("invalid_patch", "hunk context does not match")
                if kind == " ":
                    new_lines.append(old_lines[cursor])
                cursor += 1
            elif kind == "+":
                new_lines.append(content)
            else:
                raise PatchError("invalid_patch", "unexpected hunk line")
    new_lines.extend(old_lines[cursor:])
    return "".join(new_lines).encode("utf-8")


def apply_unified_diff(workspace: Path, text: str, *, protected_relative: set[str] | None = None) -> PatchApplication:
    """Apply ``text`` or restore every byte this call changed."""
    digest = digest_bytes(text.encode("utf-8"))
    try:
        files = parse_unified_diff(text)
    except PatchError:
        raise
    protected_relative = protected_relative or set()
    targets: list[tuple[FilePatch, Path]] = []
    for item in files:
        if item.relative in protected_relative:
            raise PatchError("protected_path_violation", "patch names a protected path")
        try:
            if item.old_is_null:
                path = resolve_inside(workspace, item.relative)
                if path.exists():
                    raise PatchError("invalid_patch", "new file already exists")
            else:
                path = resolve_inside(workspace, item.relative, expect="file")
        except PathPolicyError as exc:
            raise PatchError(exc.failure_class, str(exc)) from exc
        targets.append((item, path))
    backups: dict[Path, bytes | None] = {}
    try:
        for item, path in targets:
            backups[path] = None if item.old_is_null else path.read_bytes()
            if item.new_is_null:
                path.unlink()
                continue
            updated = _apply_hunks(b"" if item.old_is_null else backups[path] or b"", item.hunks)
            temporary = path.with_name(path.name + ".patch-tmp")
            temporary.write_bytes(updated)
            os_replace(temporary, path)
        return PatchApplication(True, None, tuple(item.relative for item, _ in targets), digest)
    except Exception:
        for path, previous in backups.items():
            if previous is None:
                if path.exists() and not path.is_symlink():
                    path.unlink()
            else:
                path.write_bytes(previous)
        raise


def os_replace(source: Path, dest: Path) -> None:
    import os
    os.replace(source, dest)
