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
    target = root.joinpath(*parsed.parts)
    current = root
    for part in parsed.parts:
        current = current / part
        if current.is_symlink():
            raise StructuredEditError("symlink paths are not allowed")
    if not target.is_file():
        raise StructuredEditError("allowlisted file is missing")
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
    diff = "".join(difflib.unified_diff(
        old_source.splitlines(keepends=True),
        after_bytes.decode("utf-8").splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}",
    ))
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
        original = before[name].decode("utf-8").splitlines(keepends=True)
        current = _safe_path(root, name, allowed_files).read_text(encoding="utf-8").splitlines(keepends=True)
        chunks.extend(difflib.unified_diff(original, current, fromfile=f"a/{name}", tofile=f"b/{name}"))
    return "".join(chunks)


class StructuredEditController:
    """Keep model-facing edits short while retaining exact host-side hashes.

    This controller deliberately handles only the bounded local repair loop:
    short reads, opaque revision/preimage handles, trusted public-test feedback,
    diff inspection, and a candidate-only finalization result. Hidden grading
    remains a separate caller responsibility and is never inferred here.
    """

    def __init__(
        self,
        root: Path,
        allowed_files: set[str],
        *,
        readable_files: set[str] | None = None,
        max_read_lines: int = 40,
        max_read_chars: int = 8_000,
        max_feedback_chars: int = 2_000,
    ) -> None:
        if not 1 <= max_read_lines <= 120 or not 256 <= max_read_chars <= 24_000:
            raise ValueError("structured read limits exceed the frozen controller bounds")
        if not 128 <= max_feedback_chars <= 2_000:
            raise ValueError("public feedback limit exceeds the frozen controller bound")
        self.root = root
        self.allowed_files = set(allowed_files)
        self.readable_files = set(readable_files if readable_files is not None else allowed_files)
        if not self.allowed_files.issubset(self.readable_files):
            raise ValueError("editable files must also be readable")
        self.max_read_lines = max_read_lines
        self.max_read_chars = max_read_chars
        self.max_feedback_chars = max_feedback_chars
        self._counter = 0
        self._handles: dict[str, tuple[str, str, str]] = {}
        self._latest_region_by_path: dict[str, tuple[str, str, str]] = {}
        self._before = {
            name: _safe_path(root, name, self.allowed_files).read_bytes()
            for name in sorted(self.allowed_files)
        }
        self._latest_diff_hash = sha256_bytes(b"")
        self._inspected_diff_hash: str | None = None
        self._latest_test_passed: bool | None = None
        self._failure_pairs: set[tuple[str, str]] = set()
        self._failed_noop_actions: set[str] = set()

    def _issue(self, path: str, revision: str, source_hash: str) -> tuple[str, str]:
        self._counter += 1
        revision_handle = f"rev-{self._counter:04d}"
        source_handle = f"src-{self._counter:04d}"
        self._handles[revision_handle] = (path, revision, source_hash)
        self._handles[source_handle] = (path, revision, source_hash)
        return revision_handle, source_handle

    def read_region(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read a short source region and return opaque edit handles."""
        source_path = _safe_path(self.root, path, self.readable_files)
        if start_line < 1 or (end_line is not None and end_line < start_line):
            raise StructuredEditError("source region line range is invalid")
        lines = source_path.read_text(encoding="utf-8").splitlines(keepends=True)
        requested_end = min(len(lines), end_line if end_line is not None else len(lines))
        requested_last = min(
            requested_end,
            start_line + self.max_read_lines - 1,
        )
        if start_line > len(lines):
            raise StructuredEditError("source region starts after end of file")
        selected: list[str] = []
        chars = 0
        for line in lines[start_line - 1:requested_last]:
            if chars + len(line) > self.max_read_chars:
                break
            selected.append(line)
            chars += len(line)
        last_line = start_line + len(selected) - 1
        content = "".join(selected)
        truncated = last_line < requested_end
        next_line = last_line + 1
        if path not in self.allowed_files:
            return {
                "path": path,
                "start_line": start_line,
                "end_line": last_line,
                "next_line": next_line,
                "truncated": truncated,
                "content": content,
                "read_only": True,
            }
        revision = workspace_revision(self.root, self.allowed_files)
        source_hash = sha256_bytes(source_path.read_bytes())
        revision_handle, source_handle = self._issue(path, revision, source_hash)
        self._latest_region_by_path[path] = (revision_handle, source_handle, content)
        return {
            "path": path,
            "start_line": start_line,
            "end_line": last_line,
            "next_line": next_line,
            "truncated": truncated,
            "content": content,
            "workspace_revision_handle": revision_handle,
            "source_preimage_handle": source_handle,
        }

    def apply_edit(
        self,
        *,
        action: str,
        path: str,
        workspace_revision_handle: str,
        source_preimage_handle: str,
        old_text: str,
        new_text: str,
        anchor: str = "",
    ) -> dict[str, Any]:
        """Resolve opaque handles and delegate to the exact-preimage atomic editor."""
        if len(old_text) > 12_000 or len(new_text) > 8_000 or len(anchor) > 12_000:
            raise StructuredEditError("structured edit text exceeds the per-action character budget")
        implicit_handles = False
        if not workspace_revision_handle and not source_preimage_handle:
            latest = self._latest_region_by_path.get(path)
            if latest is None:
                raise StructuredEditError("read the affected editable source region before editing")
            workspace_revision_handle, source_preimage_handle, region = latest
            required_text = anchor if action == "insert_text" else old_text
            if not required_text or region.count(required_text) != 1:
                raise StructuredEditError(
                    "edit anchor must match exactly once in the latest source region"
                )
            implicit_handles = True
        revision_binding = self._handles.get(workspace_revision_handle)
        source_binding = self._handles.get(source_preimage_handle)
        if revision_binding is None or source_binding is None:
            raise StructuredEditError("unknown edit handle; read the current source region")
        if revision_binding != source_binding or revision_binding[0] != path:
            raise StructuredEditError("edit handles do not refer to the same source snapshot")
        handle_path, expected_revision, expected_hash = revision_binding
        if workspace_revision(self.root, self.allowed_files) != expected_revision:
            raise StructuredEditError("workspace revision handle is stale; reread the affected source region")
        try:
            result = apply_structured_edit(
                self.root,
                self.allowed_files,
                action=action,
                path=handle_path,
                expected_hash=expected_hash,
                expected_revision=expected_revision,
                old_text=old_text,
                new_text=new_text,
                anchor=anchor,
            )
        except StructuredEditError as exc:
            if str(exc) in {"edit would not change the file", "old_text must match exactly once"}:
                signature = sha256_bytes(
                    "\0".join((action, path, old_text, new_text, anchor)).encode("utf-8")
                )
                repeated = signature in self._failed_noop_actions
                self._failed_noop_actions.add(signature)
                if repeated:
                    raise StructuredEditError(
                        "repeated no-op edit; inspect the current source and choose a different targeted change"
                    ) from exc
            raise
        current_diff = unified_diff(self.root, self._before, self.allowed_files)
        self._latest_diff_hash = sha256_bytes(current_diff.encode("utf-8"))
        self._inspected_diff_hash = None
        self._latest_test_passed = None
        self._latest_region_by_path.pop(path, None)
        result["edit_revision_source"] = (
            "implicit_latest_source_read" if implicit_handles else "explicit_opaque_handles"
        )
        return result

    def record_public_test_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Bound trusted public-test feedback and flag repeated unchanged failures."""
        passed = result.get("passed")
        status = result.get("status")
        unavailable_statuses = {
            "container_launch_error", "container_timeout", "container_terminated",
            "runner_error", "untrusted_test_input",
        }
        if status in unavailable_statuses:
            if passed is not None or result.get("gradeable") is True:
                raise StructuredEditError(
                    "unavailable public-test result cannot be marked gradeable"
                )
            error = result.get("error", "")
            if not isinstance(error, str):
                raise StructuredEditError("public-test error must be text")
            self._latest_test_passed = None
            self._inspected_diff_hash = None
            return {
                "passed": None,
                "available": False,
                "status": status,
                "feedback": error[-self.max_feedback_chars:],
                "next_action": (
                    "public tests did not produce a trusted candidate result; "
                    "do not treat this as a code failure"
                ),
            }
        if not isinstance(passed, bool):
            raise StructuredEditError("public-test result must come from the trusted test runner")
        if status is not None and (
            status not in {"passed", "assertion_failure", "test_execution_error"}
            or (passed and status != "passed")
        ):
            raise StructuredEditError("public-test status and pass result disagree")
        stderr = result.get("stderr", "")
        stdout = result.get("stdout", "")
        error = result.get("error", "")
        if not isinstance(stderr, str) or not isinstance(stdout, str):
            raise StructuredEditError("public-test output must be text")
        if not isinstance(error, str):
            raise StructuredEditError("public-test error must be text")
        feedback_text = "\n".join(part for part in (stdout, stderr, error) if part)
        bounded = feedback_text[-self.max_feedback_chars:]
        signature = sha256_bytes(bounded.encode("utf-8")) if not passed else ""
        pair = (self._latest_diff_hash, signature)
        repeated = bool(not passed and signature and pair in self._failure_pairs)
        if not passed and signature:
            self._failure_pairs.add(pair)
        self._latest_test_passed = passed
        self._inspected_diff_hash = None
        return {
            "passed": passed,
            "failure_signature": signature,
            "repeated_unchanged_failure": repeated,
            "feedback": bounded,
            "next_action": (
                "inspect the failing assertion and make a different targeted edit"
                if repeated else "use the reported public assertion to plan the next repair"
            ) if not passed else "inspect the final diff before submission",
        }

    def inspect_diff(self) -> dict[str, Any]:
        """Return the current bounded diff and mark exactly this diff as inspected."""
        diff = unified_diff(self.root, self._before, self.allowed_files)
        digest = sha256_bytes(diff.encode("utf-8"))
        self._latest_diff_hash = digest
        self._inspected_diff_hash = digest
        return {"diff": diff, "diff_sha256": digest, "changed": bool(diff)}

    def finalize_for_independent_grading(self) -> dict[str, Any]:
        """Return a candidate patch only after public tests and diff inspection."""
        current = unified_diff(self.root, self._before, self.allowed_files)
        digest = sha256_bytes(current.encode("utf-8"))
        if self._latest_test_passed is not True:
            raise StructuredEditError("candidate cannot be submitted until public tests pass")
        if self._inspected_diff_hash != digest:
            raise StructuredEditError("inspect the current diff after the last edit and test")
        if not current:
            raise StructuredEditError("candidate has no source changes to submit")
        return {
            "candidate_ready_for_independent_grading": True,
            "verified_success": False,
            "diff": current,
            "diff_sha256": digest,
        }
