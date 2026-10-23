"""Hash-checked, atomic structured edits for local coding-agent workspaces."""
from __future__ import annotations

import difflib
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any


class StructuredEditError(ValueError):
    """A structured edit was invalid or stale and was not applied."""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _source_lines(source: str) -> list[str]:
    # Git counts LF-delimited lines; str.splitlines also splits source characters
    # such as bare CR and Unicode line separators, producing invalid hunk counts.
    parts = source.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _file_diff(name: str, before: str, after: str) -> str:
    lines = difflib.unified_diff(
        _source_lines(before), _source_lines(after),
        fromfile=f"a/{name}", tofile=f"b/{name}",
    )
    return "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in lines
    )


def workspace_revision(root: Path, allowed_files: set[str]) -> str:
    """Hash the current bytes of all allowlisted files in stable path order."""
    entries: list[bytes] = []
    for name in sorted(allowed_files):
        path = _safe_path(root, name, allowed_files)
        content = path.read_bytes()
        entries.extend((name.encode(), b"\0", hashlib.sha256(content).digest(), b"\n"))
    return hashlib.sha256(b"".join(entries)).hexdigest()


def _safe_path(root: Path, name: str, allowed_files: set[str]) -> Path:
    parsed = PurePosixPath(name)
    if (
        name not in allowed_files
        or not name
        or parsed.is_absolute()
        or ".." in parsed.parts
        or "." in parsed.parts
        or "\\" in name
    ):
        raise StructuredEditError("path is not an exact allowlisted file")
    if any(component.is_symlink() for component in (root.absolute(), *root.absolute().parents)):
        raise StructuredEditError("symlinked workspace roots are not allowed")
    target = root.joinpath(*parsed.parts)
    current = root
    for part in parsed.parts:
        current = current / part
        if current.is_symlink():
            raise StructuredEditError("symlink paths are not allowed")
    if not target.is_file():
        raise StructuredEditError("allowlisted file is missing")
    if target.stat().st_nlink != 1:
        raise StructuredEditError("hard-linked files are not allowed")
    try:
        target.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise StructuredEditError("path escapes the workspace") from exc
    return target


def _atomic_replace(path: Path, old: bytes, new: bytes) -> None:
    if old == new:
        raise StructuredEditError("edit would not change the file")
    mode = path.stat().st_mode & 0o777
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temp_name = stream.name
            stream.write(new)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_name, mode)
        if path.read_bytes() != old:
            raise StructuredEditError("file changed during edit; retry with fresh hashes")
        os.replace(temp_name, path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass


def apply_structured_edit(
    root: Path,
    allowed_files: set[str],
    *,
    action: str,
    path: str,
    expected_hash: str,
    expected_revision: str,
    old_text: str,
    new_text: str,
    anchor: str = "",
) -> dict[str, Any]:
    """Apply one exact replace or anchored insertion and return its generated diff."""
    target = _safe_path(root, path, allowed_files)
    before_revision = workspace_revision(root, allowed_files)
    old_bytes = target.read_bytes()
    before_hash = sha256_bytes(old_bytes)
    if before_revision != expected_revision:
        raise StructuredEditError("workspace revision is stale; inspect the current source again")
    if before_hash != expected_hash:
        raise StructuredEditError("source preimage hash is stale; reread the target file")
    try:
        old_source = old_bytes.decode("utf-8")
        new_source_text = new_text
        if action == "replace_text":
            if not old_text:
                raise StructuredEditError("old_text must not be empty")
            count = old_source.count(old_text)
            if count != 1:
                raise StructuredEditError("old_text must match exactly once")
            new_source_text = old_source.replace(old_text, new_text, 1)
        elif action == "insert_text":
            if not anchor:
                raise StructuredEditError("anchor must not be empty")
            count = old_source.count(anchor)
            if count != 1:
                raise StructuredEditError("anchor must match exactly once")
            end = old_source.index(anchor) + len(anchor)
            new_source_text = old_source[:end] + new_text + old_source[end:]
        else:
            raise StructuredEditError("unsupported structured edit action")
        new_bytes = new_source_text.encode("utf-8")
        _atomic_replace(target, old_bytes, new_bytes)
    except UnicodeDecodeError as exc:
        raise StructuredEditError("target file is not UTF-8 text") from exc
    after_bytes = target.read_bytes()
    after_hash = sha256_bytes(after_bytes)
    if after_hash == before_hash:
        raise StructuredEditError("edit would not change the file")
    diff = _file_diff(path, old_source, after_bytes.decode("utf-8"))
    return {
        "applied": True,
        "path": path,
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "workspace_revision_before": before_revision,
        "workspace_revision_after": workspace_revision(root, allowed_files),
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        "diff": diff,
    }


def unified_diff(root: Path, before: dict[str, bytes], allowed_files: set[str]) -> str:
    """Build a deterministic diff between initial source snapshots and workspace."""
    chunks: list[str] = []
    for name in sorted(allowed_files):
        original = before[name].decode("utf-8")
        current = _safe_path(root, name, allowed_files).read_bytes().decode("utf-8")
        chunks.append(_file_diff(name, original, current))
    return "".join(chunks)
