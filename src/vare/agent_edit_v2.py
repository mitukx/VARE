"""Lease-bound, atomic source-range edits for constrained coding agents.

The model sees a short one-time lease; trusted controller state binds that
lease to an allowlisted UTF-8 file, full-file SHA-256, and complete line range.
This module never repairs or infers incomplete tool calls.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any, Callable

MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_REPLACEMENT_BYTES = 256 * 1024
MAX_RANGE_LINES = 120


def _path_parts(raw_path: object, allowed: set[str]) -> tuple[tuple[str, ...] | None, str | None]:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path or "\\" in raw_path:
        return None, "path_not_allowlisted"
    path = PurePosixPath(raw_path)
    if (path.is_absolute() or raw_path != path.as_posix()
            or any(part in {".", ".."} for part in path.parts)
            or raw_path not in allowed):
        return None, "path_not_allowlisted"
    return path.parts, None


def _open_parent(workspace: Path, parts: tuple[str, ...]) -> tuple[int, str]:
    if not parts:
        raise OSError("empty source path")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(workspace.resolve(strict=True), flags)
    try:
        for component in parts[:-1]:
            child_fd = os.open(component, flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
    except BaseException:
        os.close(parent_fd)
        raise
    return parent_fd, parts[-1]


def _read_current(parent_fd: int, basename: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    file_fd = os.open(basename, flags, dir_fd=parent_fd)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise OSError("source is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(file_fd, min(65536, MAX_SOURCE_BYTES + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > MAX_SOURCE_BYTES:
                raise OSError("source exceeds byte limit")
        after = os.fstat(file_fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise OSError("source changed while being read")
        return b"".join(chunks), before
    finally:
        os.close(file_fd)


def _decode_source(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf") or b"\x00" in data or b"\r" in data:
        raise ValueError("source_encoding_unsupported")
    return data.decode("utf-8", errors="strict")


def read_source_lines_v1(
    workspace: Path,
    raw_path: object,
    start_line: object,
    end_line: object,
    allowed: set[str],
) -> dict[str, Any]:
    """Read a bounded range and return the whole-file preimage hash."""
    parts, error = _path_parts(raw_path, allowed)
    if error:
        return {"status": "valid_call_rejected_by_policy", "error": error}
    if (isinstance(start_line, bool) or isinstance(end_line, bool)
            or not isinstance(start_line, int) or not isinstance(end_line, int)
            or start_line < 1 or end_line < start_line
            or end_line - start_line + 1 > MAX_RANGE_LINES):
        return {"status": "valid_call_rejected_by_policy", "error": "invalid_line_range"}
    parent_fd = -1
    try:
        parent_fd, basename = _open_parent(workspace, parts or ())
        data, _file_stat = _read_current(parent_fd, basename)
        text = _decode_source(data)
        lines = text.splitlines(keepends=True)
        if end_line > len(lines):
            return {"status": "valid_call_rejected_by_policy", "error": "line_range_out_of_bounds",
                    "line_count": len(lines)}
        return {
            "status": "read_complete",
            "path": str(PurePosixPath(*parts)),
            "start_line": start_line,
            "end_line": end_line,
            "line_count": len(lines),
            "sha256": hashlib.sha256(data).hexdigest(),
            "content": "".join(lines[start_line - 1:end_line]),
        }
    except (OSError, UnicodeError, ValueError) as exc:
        return {"status": "valid_call_rejected_by_policy", "error": str(exc)[:160]}
    finally:
        if parent_fd >= 0:
            os.close(parent_fd)


def replace_source_lines_v1(
    workspace: Path,
    raw_path: object,
    expected_sha256: object,
    start_line: object,
    end_line: object,
    replacement: object,
    allowed: set[str],
) -> dict[str, Any]:
    """Replace existing lines iff the complete source preimage still matches.

    A same-directory temporary file and atomic rename prevent partial writes.
    Every path component is opened without following symlinks.
    """
    parts, error = _path_parts(raw_path, allowed)
    if error:
        return {"applied": False, "status": "valid_call_rejected_by_policy", "error": error}
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)):
        return {"applied": False, "status": "invalid_json_or_schema", "error": "invalid_sha256"}
    if (isinstance(start_line, bool) or isinstance(end_line, bool)
            or not isinstance(start_line, int) or not isinstance(end_line, int)
            or start_line < 1 or end_line < start_line
            or end_line - start_line + 1 > MAX_RANGE_LINES):
        return {"applied": False, "status": "invalid_json_or_schema", "error": "invalid_line_range"}
    if not isinstance(replacement, str) or "\x00" in replacement or "\r" in replacement:
        return {"applied": False, "status": "invalid_json_or_schema", "error": "invalid_replacement_text"}
    try:
        replacement_bytes = replacement.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return {"applied": False, "status": "invalid_json_or_schema", "error": "invalid_utf8_replacement"}
    if len(replacement_bytes) > MAX_REPLACEMENT_BYTES:
        return {"applied": False, "status": "valid_call_rejected_by_policy", "error": "replacement_exceeds_byte_limit"}

    parent_fd = -1
    temp_name: str | None = None
    try:
        parent_fd, basename = _open_parent(workspace, parts or ())
        original, original_stat = _read_current(parent_fd, basename)
        before_hash = hashlib.sha256(original).hexdigest()
        if before_hash != expected_sha256:
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "stale_preimage", "actual_sha256": before_hash}
        text = _decode_source(original)
        lines = text.splitlines(keepends=True)
        if end_line > len(lines):
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "line_range_out_of_bounds", "line_count": len(lines)}
        selected = "".join(lines[start_line - 1:end_line])
        if replacement == selected:
            return {"applied": False, "status": "valid_call_rejected_by_policy", "error": "no_change"}
        if selected.endswith("\n") and replacement and not replacement.endswith("\n"):
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "replacement_must_preserve_final_newline"}
        updated = "".join(lines[:start_line - 1]) + replacement + "".join(lines[end_line:])
        updated_bytes = updated.encode("utf-8", errors="strict")
        if len(updated_bytes) > MAX_SOURCE_BYTES:
            return {"applied": False, "status": "valid_call_rejected_by_policy", "error": "source_exceeds_byte_limit"}

        temp_name = f".{basename}.vare-{secrets.token_hex(12)}.tmp"
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        temp_fd = os.open(temp_name, create_flags, stat.S_IMODE(original_stat.st_mode), dir_fd=parent_fd)
        try:
            view = memoryview(updated_bytes)
            while view:
                written = os.write(temp_fd, view)
                view = view[written:]
            os.fchmod(temp_fd, stat.S_IMODE(original_stat.st_mode))
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)

        # Recheck the named target immediately before the atomic replacement.
        current, current_stat = _read_current(parent_fd, basename)
        if (hashlib.sha256(current).hexdigest() != expected_sha256
                or (current_stat.st_dev, current_stat.st_ino) != (original_stat.st_dev, original_stat.st_ino)):
            return {"applied": False, "status": "valid_call_rejected_by_policy", "error": "stale_preimage"}
        os.replace(temp_name, basename, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temp_name = None
        os.fsync(parent_fd)
        return {
            "applied": True,
            "status": "edit_applied",
            "path": str(PurePosixPath(*parts)),
            "lines_replaced": end_line - start_line + 1,
            "before_sha256": before_hash,
            "after_sha256": hashlib.sha256(updated_bytes).hexdigest(),
        }
    except (OSError, UnicodeError, ValueError) as exc:
        return {"applied": False, "status": "valid_call_rejected_by_policy", "error": str(exc)[:160]}
    finally:
        if parent_fd >= 0:
            if temp_name is not None:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
            os.close(parent_fd)


class SourceEditLeaseStoreV1:
    """Bind a short one-time edit lease to an allowlisted source preimage."""

    def __init__(self, workspace: Path, allowed: set[str],
                 token_factory: Callable[[], str] | None = None) -> None:
        self.workspace = workspace
        self.allowed = set(allowed)
        self._token_factory = token_factory or (lambda: secrets.token_hex(6))
        self._leases: dict[str, dict[str, Any]] = {}

    def read_lines(self, raw_path: object, start_line: object, end_line: object) -> dict[str, Any]:
        observed = read_source_lines_v1(
            self.workspace, raw_path, start_line, end_line, self.allowed,
        )
        if observed.get("status") != "read_complete":
            return observed
        lease = self._token_factory()
        while lease in self._leases:
            lease = self._token_factory()
        record = {
            "lease": lease,
            "path": observed["path"],
            "expected_sha256": observed["sha256"],
            "start_line": observed["start_line"],
            "end_line": observed["end_line"],
            "consumed": False,
            "outcome": "lease_issued",
        }
        self._leases[lease] = record
        # Keep the long digest in the controller's evidence record, not in the
        # generated edit action. The opaque lease carries that preimage binding.
        observed.pop("sha256", None)
        observed["lease"] = lease
        return observed

    def replace_lines(self, lease: object, replacement: object) -> dict[str, Any]:
        if not isinstance(lease, str) or lease not in self._leases:
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "unknown_or_consumed_lease"}
        record = self._leases[lease]
        if record["consumed"]:
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "unknown_or_consumed_lease", "lease": lease}
        # A lease is consumed before validation or writing, so retries must
        # obtain a new source read and bind to the then-current preimage.
        record["consumed"] = True
        result = replace_source_lines_v1(
            self.workspace,
            record["path"],
            record["expected_sha256"],
            record["start_line"],
            record["end_line"],
            replacement,
            self.allowed,
        )
        record["outcome"] = result.get("status", "valid_call_rejected_by_policy")
        if result.get("after_sha256"):
            record["after_sha256"] = result["after_sha256"]
        return {**result, "lease": lease}

    def audit_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._leases.values()]
