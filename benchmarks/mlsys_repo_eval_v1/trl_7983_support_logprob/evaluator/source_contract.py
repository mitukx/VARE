"""Fail-closed source preflight for a narrowly defined historical patch task."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat


MAX_SOURCE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class SourceContractResult:
    accepted: bool
    status: str
    reason: str
    source_sha256: str | None
    ast_sha256: str | None
    candidate_code_imported: bool = False
    worker_started: bool = False


def _canonical_ast_value(value: object) -> object:
    if isinstance(value, ast.AST):
        fields: list[list[object]] = []
        for name in value._fields:
            field_value = getattr(value, name)
            # Python 3.12 added this field to function and class nodes. An
            # empty value means the source has no type parameters, matching
            # Python 3.9's AST, which has no such field. Non-empty values stay
            # in the digest so the compatibility rule cannot erase semantics.
            if name == "type_params" and field_value == []:
                continue
            fields.append([name, _canonical_ast_value(field_value)])
        return [type(value).__name__, fields]
    if isinstance(value, list):
        return ["list", [_canonical_ast_value(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [_canonical_ast_value(item) for item in value]]
    if value is None:
        return ["NoneType", None]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", value.hex() if math.isfinite(value) else repr(value)]
    if type(value) is complex:
        return ["complex", value.real.hex(), value.imag.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if value is Ellipsis:
        return ["ellipsis"]
    raise TypeError(f"unsupported Python AST field value: {type(value).__name__}")


def canonical_ast_sha256(tree: ast.AST) -> str:
    """Hash AST structure with empty 3.12 type-parameter fields normalized."""
    canonical = json.dumps(
        _canonical_ast_value(tree), ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ast_sha256(source: bytes) -> str:
    """Hash Python syntax without locations, execution, or parser-version drift."""
    tree = ast.parse(source.decode("utf-8"), type_comments=True)
    return canonical_ast_sha256(tree)


def _read_candidate(root: Path, relative_path: str) -> tuple[bytes | None, str | None]:
    root = root.resolve(strict=True)
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None, "candidate path is not a safe relative path"

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(root, directory_flags)
        for part in relative.parts[:-1]:
            next_descriptor = os.open(
                part, directory_flags, dir_fd=directory_descriptor
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        assert directory_descriptor is not None
        descriptor = os.open(
            relative.parts[-1], file_flags, dir_fd=directory_descriptor
        )
    except OSError:
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        return None, "candidate path contains a symlink or could not be opened safely"
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return None, "candidate source is not a single-link regular file"
        if before.st_size > MAX_SOURCE_BYTES:
            return None, "candidate source exceeds the size limit"
        chunks: list[bytes] = []
        remaining = MAX_SOURCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        source = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(
            relative.parts[-1], dir_fd=directory_descriptor, follow_symlinks=False
        )
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ) or (
            before.st_dev,
            before.st_ino,
            before.st_mode,
        ) != (current.st_dev, current.st_ino, current.st_mode):
            return None, "candidate source changed while being read"
        if len(source) > MAX_SOURCE_BYTES:
            return None, "candidate source exceeds the size limit"
        return source, None
    finally:
        os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def check_candidate_source(
    workspace: Path, relative_path: str, expected_ast_sha256: str
) -> tuple[SourceContractResult, bytes | None]:
    """Check source before import; return the checked bytes for sealed staging."""
    source, error = _read_candidate(workspace, relative_path)
    if error is not None or source is None:
        return (
            SourceContractResult(
                accepted=False,
                status="SOURCE_CONTRACT_REJECT",
                reason=error or "candidate source unavailable",
                source_sha256=None,
                ast_sha256=None,
            ),
            None,
        )
    source_hash = hashlib.sha256(source).hexdigest()
    try:
        candidate_ast_hash = ast_sha256(source)
    except (SyntaxError, UnicodeDecodeError, RecursionError, ValueError):
        return (
            SourceContractResult(
                accepted=False,
                status="SOURCE_CONTRACT_REJECT",
                reason="candidate source is not a supported Python module",
                source_sha256=source_hash,
                ast_sha256=None,
            ),
            None,
        )
    accepted = candidate_ast_hash == expected_ast_sha256
    return (
        SourceContractResult(
            accepted=accepted,
            status="SOURCE_CONTRACT_ACCEPT" if accepted else "SOURCE_CONTRACT_REJECT",
            reason="pinned known-fix AST match"
            if accepted
            else "candidate AST differs from pinned contract",
            source_sha256=source_hash,
            ast_sha256=candidate_ast_hash,
        ),
        source if accepted else None,
    )
