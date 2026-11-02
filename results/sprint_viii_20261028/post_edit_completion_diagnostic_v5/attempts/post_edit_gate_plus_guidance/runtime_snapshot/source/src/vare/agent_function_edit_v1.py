"""Function-scoped transactional edits for repository coding agents.

The controller lists and reads candidate-visible Python functions, issues a
one-time lease bound to the full source preimage and function span, and accepts
only a replacement for the executable body. The original signature and
docstring are retained byte-for-byte. Explicit format normalizations are
versioned and recorded; semantic text is never inferred or repaired.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Callable

from .agent_edit_v2 import (
    _decode_source,
    _open_parent,
    _path_parts,
    _read_current,
    _decode_source_crlf_v2,
    _normalize_python_replacement_v1,
    replace_python_source_lines_array_transactional_v2,
    replace_python_source_lines_transactional_v1,
)


def _supported_functions(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Return multiline top-level functions and direct class methods without decorators."""
    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.decorator_list and node.body[0].lineno > node.lineno:
                found.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and not child.decorator_list and child.body[0].lineno > child.lineno):
                    found.append((f"{node.name}.{child.name}", child))
    return found


def _signature_record(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    return {
        "name": node.name,
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "args_ast": ast.dump(node.args, include_attributes=False),
        "returns_ast": ast.dump(node.returns, include_attributes=False) if node.returns else None,
        "type_comment": node.type_comment,
    }


def _is_single_line_return(
    node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str],
) -> bool:
    body_nodes = list(node.body)
    first = body_nodes[0] if body_nodes else None
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        body_nodes = body_nodes[1:]
    if len(body_nodes) != 1 or not isinstance(body_nodes[0], ast.Return):
        return False
    statement = body_nodes[0]
    if statement.lineno != statement.end_lineno:
        return False
    line = lines[statement.lineno - 1]
    return bool(line.strip()) and "\n" not in line.rstrip("\n")


class FunctionEditLeaseStoreV1:
    """Issue function-body leases and commit syntax-valid replacements atomically."""

    def __init__(self, workspace: Path, allowed: set[str],
                 token_factory: Callable[[], str] | None = None, *,
                 single_line_only: bool = False) -> None:
        self.workspace = workspace
        self.allowed = set(allowed)
        self._token_factory = token_factory or (lambda: secrets.token_hex(6))
        self._leases: dict[str, dict[str, Any]] = {}
        self.single_line_only = single_line_only

    def _read(self, raw_path: object) -> tuple[str | None, str | None, str | None]:
        parts, error = _path_parts(raw_path, self.allowed)
        if error:
            return None, None, error
        assert parts is not None
        if not str(Path(*parts)).endswith(".py"):
            return None, None, "python_source_required"
        parent_fd = -1
        try:
            parent_fd, basename = _open_parent(self.workspace, parts)
            data, _ = _read_current(parent_fd, basename)
            text = _decode_source(data)
            return text, hashlib.sha256(data).hexdigest(), None
        except (OSError, UnicodeError, ValueError) as exc:
            return None, None, str(exc)[:160]
        finally:
            if parent_fd >= 0:
                os.close(parent_fd)

    def list_functions(self, raw_path: object) -> dict[str, Any]:
        if not isinstance(raw_path, str):
            return {"status": "invalid_json_or_schema", "error": "source_path_required"}
        text, source_hash, error = self._read(raw_path)
        if error:
            return {"status": "valid_call_rejected_by_policy", "error": error}
        assert text is not None and source_hash is not None
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return {"status": "source_parse_failure", "syntax_error": {
                "message": str(exc.msg)[:160], "lineno": exc.lineno, "offset": exc.offset,
            }}
        functions = []
        source_lines = text.splitlines(keepends=True)
        for qualified_name, node in _supported_functions(tree):
            if self.single_line_only and not _is_single_line_return(node, source_lines):
                continue
            functions.append({
                "qualified_name": qualified_name,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "signature": _signature_record(node),
            })
        return {
            "status": "functions_listed",
            "path": str(Path(raw_path)),
            "source_sha256": source_hash,
            "functions": functions,
        }

    def read_function(self, raw_path: object, qualified_name: object) -> dict[str, Any]:
        if not isinstance(raw_path, str):
            return {"status": "invalid_json_or_schema", "error": "source_path_required"}
        if not isinstance(qualified_name, str) or not qualified_name:
            return {"status": "invalid_json_or_schema", "error": "function_name_required"}
        text, source_hash, error = self._read(raw_path)
        if error:
            return {"status": "valid_call_rejected_by_policy", "error": error}
        assert text is not None and source_hash is not None
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return {"status": "source_parse_failure", "syntax_error": {
                "message": str(exc.msg)[:160], "lineno": exc.lineno, "offset": exc.offset,
            }}
        matches = [node for name, node in _supported_functions(tree) if name == qualified_name]
        if len(matches) != 1:
            return {"status": "valid_call_rejected_by_policy", "error": "function_not_unique_or_not_supported"}
        node = matches[0]
        lines = text.splitlines(keepends=True)
        if self.single_line_only and not _is_single_line_return(node, lines):
            return {"status": "valid_call_rejected_by_policy",
                    "error": "single_line_return_function_required"}
        if not isinstance(node.end_lineno, int) or node.lineno < 1 or node.end_lineno > len(lines):
            return {"status": "valid_call_rejected_by_policy", "error": "function_span_unavailable"}
        start_line, end_line = node.lineno, node.end_lineno
        source_lines = lines[start_line - 1:end_line]
        selected = "".join(source_lines)
        if not selected.endswith("\n") and source_hash:
            # EOF functions may legitimately lack a final newline. Preserve that fact.
            pass
        body_start = node.body[0].lineno if node.body else end_line + 1
        if node.body:
            indent_match = re.match(r"^[ \t]*", lines[node.body[0].lineno - 1])
            body_indent_prefix = indent_match.group(0) if indent_match is not None else ""
        else:
            body_indent_prefix = ""
        first_statement = node.body[0] if node.body else None
        if (isinstance(first_statement, ast.Expr)
                and isinstance(first_statement.value, ast.Constant)
                and isinstance(first_statement.value.value, str)):
            # Preserve a leading docstring as part of the lease's immutable prefix.
            statement_end = first_statement.end_lineno
            if not isinstance(statement_end, int):
                return {"status": "valid_call_rejected_by_policy", "error": "function_span_unavailable"}
            body_start = statement_end + 1
        body_offset = max(0, body_start - start_line)
        prefix = "".join(source_lines[:body_offset])
        body = "".join(source_lines[body_offset:])
        # Bind the lease to a second safe read, rejecting a race between parsing
        # and lease creation rather than issuing a lease for mixed source states.
        reread, reread_hash, reread_error = self._read(raw_path)
        if reread_error or reread_hash != source_hash or reread != text:
            return {"status": "valid_call_rejected_by_policy", "error": "source_changed_during_read"}
        lease = self._token_factory()
        while lease in self._leases:
            lease = self._token_factory()
        self._leases[lease] = {
            "lease": lease,
            "path": str(Path(raw_path)),
            "expected_sha256": source_hash,
            "start_line": start_line,
            "end_line": end_line,
            "qualified_name": qualified_name,
            "signature": _signature_record(node),
            "single_line_only": self.single_line_only,
            "body_indent_prefix": body_indent_prefix,
            "prefix": prefix,
            "original_body": body,
            "original_function_sha256": hashlib.sha256(selected.encode("utf-8")).hexdigest(),
            "original_body_ends_with_newline": body.endswith("\n"),
            "original_function_ends_with_newline": selected.endswith("\n"),
            "line_ending": "CRLF" if "\r\n" in text else "LF",
            "consumed": False,
            "outcome": "lease_issued",
        }
        return {
            "status": "read_complete",
            "path": str(Path(raw_path)),
            "qualified_name": qualified_name,
            "start_line": start_line,
            "end_line": end_line,
            "signature": _signature_record(node),
            "body_indent_prefix": body_indent_prefix,
            "single_line_return_only": self.single_line_only,
            "preserved_prefix": prefix,
            "body": body,
            "lease": lease,
        }

    def replace_function_body_transactional(
        self, lease: object, raw_body: object, normalization: object = "none",
    ) -> dict[str, Any]:
        if not isinstance(lease, str) or lease not in self._leases:
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "unknown_or_consumed_lease"}
        record = self._leases[lease]
        if record["consumed"]:
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "unknown_or_consumed_lease", "lease": lease}
        record["consumed"] = True
        if not isinstance(raw_body, str) or "\x00" in raw_body or "\r" in raw_body:
            result = {"applied": False, "status": "invalid_json_or_schema", "error": "invalid_replacement_text"}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        raw_hash = hashlib.sha256(raw_body.encode("utf-8", errors="strict")).hexdigest()
        body, normalization_error = _normalize_python_replacement_v1(raw_body, normalization)
        if normalization_error:
            result = {
                "applied": False, "status": "valid_call_rejected_by_policy",
                "error": normalization_error,
                "normalization": normalization,
                "raw_replacement_sha256": raw_hash,
            }
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        assert body is not None
        if record["original_body_ends_with_newline"] and body and not body.endswith("\n"):
            result = {
                "applied": False, "status": "valid_call_rejected_by_policy",
                "error": "replacement_must_preserve_final_newline",
                "normalization": normalization,
                "raw_replacement_sha256": raw_hash,
                "normalized_replacement_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            }
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        candidate_function = record["prefix"] + body
        current, current_hash, read_error = self._read(record["path"])
        if read_error or current is None or current_hash != record["expected_sha256"]:
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "stale_preimage"}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        current_lines = current.splitlines(keepends=True)
        candidate_source = (
            "".join(current_lines[:record["start_line"] - 1])
            + candidate_function
            + "".join(current_lines[record["end_line"]:])
        )
        try:
            tree = ast.parse(candidate_source, filename=record["path"])
        except SyntaxError as exc:
            result = {
                "applied": False, "status": "candidate_rejected_syntax",
                "syntax_valid": False,
                "syntax_error": {"message": str(exc.msg)[:160], "lineno": exc.lineno, "offset": exc.offset},
                "normalization": normalization,
                "raw_replacement_sha256": raw_hash,
            }
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        nodes = [node for name, node in _supported_functions(tree)
                 if name == record["qualified_name"]]
        if len(nodes) != 1:
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "replacement_function_identity_mismatch", "syntax_valid": True}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        if _signature_record(nodes[0]) != record["signature"]:
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "function_signature_changed", "syntax_valid": True}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        if record.get("single_line_only") and not _is_single_line_return(
                nodes[0], candidate_source.splitlines(keepends=True)):
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "single_line_return_replacement_required", "syntax_valid": True}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        result = replace_python_source_lines_transactional_v1(
            self.workspace,
            record["path"],
            record["expected_sha256"],
            record["start_line"],
            record["end_line"],
            candidate_function,
            self.allowed,
            normalization="none",
        )
        record["outcome"] = result.get("status", "valid_call_rejected_by_policy")
        if result.get("after_sha256"):
            record["after_sha256"] = result["after_sha256"]
        for key in ("syntax_valid", "syntax_error"):
            if key in result:
                record[key] = result[key]
        record.update({
            "normalization": normalization,
            "raw_replacement_sha256": raw_hash,
            "normalized_replacement_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        })
        return {
            **result,
            "lease": lease,
            "qualified_name": record["qualified_name"],
            "normalization": normalization,
            "raw_replacement_sha256": raw_hash,
            "normalized_replacement_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }

    def audit_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._leases.values()]


class FunctionEditLeaseStoreV2(FunctionEditLeaseStoreV1):
    """Line-array function editor with source-EOL preservation and strict leases."""

    def _read(self, raw_path: object) -> tuple[str | None, str | None, str | None]:
        parts, error = _path_parts(raw_path, self.allowed)
        if error:
            return None, None, error
        assert parts is not None
        if not str(Path(*parts)).endswith(".py"):
            return None, None, "python_source_required"
        parent_fd = -1
        try:
            parent_fd, basename = _open_parent(self.workspace, parts)
            data, _ = _read_current(parent_fd, basename)
            text = _decode_source_crlf_v2(data)
            return text, hashlib.sha256(data).hexdigest(), None
        except (OSError, UnicodeError, ValueError) as exc:
            return None, None, str(exc)[:160]
        finally:
            if parent_fd >= 0:
                os.close(parent_fd)

    def replace_function_body_lines_transactional_v2(
        self, lease: object, raw_body_lines: object,
    ) -> dict[str, Any]:
        if not isinstance(lease, str) or lease not in self._leases:
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "unknown_or_consumed_lease"}
        record = self._leases[lease]
        if record["consumed"]:
            return {"applied": False, "status": "valid_call_rejected_by_policy",
                    "error": "unknown_or_consumed_lease", "lease": lease}
        record["consumed"] = True
        if (not isinstance(raw_body_lines, list) or len(raw_body_lines) > 120
                or any(not isinstance(line, str) or any(ch in line for ch in "\r\n\x00")
                       for line in raw_body_lines)):
            result = {"applied": False, "status": "invalid_json_or_schema",
                      "error": "invalid_replacement_lines"}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        try:
            raw_bytes = json.dumps(raw_body_lines, ensure_ascii=False, separators=(",", ":")) \
                .encode("utf-8", errors="strict")
        except (UnicodeEncodeError, TypeError, ValueError):
            result = {"applied": False, "status": "invalid_json_or_schema",
                      "error": "invalid_utf8_replacement"}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        if len(raw_bytes) > 256 * 1024:
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "replacement_exceeds_byte_limit"}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}

        current, current_hash, read_error = self._read(record["path"])
        if read_error or current is None or current_hash != record["expected_sha256"]:
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "stale_preimage"}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        eol = "\r\n" if record["line_ending"] == "CRLF" else "\n"
        body = eol.join(raw_body_lines)
        if raw_body_lines and record["original_body_ends_with_newline"]:
            body += eol
        candidate_function = record["prefix"] + body
        current_lines = current.splitlines(keepends=True)
        candidate_source = (
            "".join(current_lines[:record["start_line"] - 1])
            + candidate_function
            + "".join(current_lines[record["end_line"]:])
        )
        try:
            tree = ast.parse(candidate_source, filename=record["path"])
        except SyntaxError as exc:
            result = {
                "applied": False, "status": "candidate_rejected_syntax", "syntax_valid": False,
                "syntax_error": {"message": str(exc.msg)[:160], "lineno": exc.lineno, "offset": exc.offset},
                "line_ending": record["line_ending"],
                "raw_body_lines_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            }
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        nodes = [node for name, node in _supported_functions(tree)
                 if name == record["qualified_name"]]
        if len(nodes) != 1:
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "replacement_function_identity_mismatch", "syntax_valid": True}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        if _signature_record(nodes[0]) != record["signature"]:
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "function_signature_changed", "syntax_valid": True}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        if record.get("single_line_only") and not _is_single_line_return(
                nodes[0], candidate_source.splitlines(keepends=True)):
            result = {"applied": False, "status": "valid_call_rejected_by_policy",
                      "error": "single_line_return_replacement_required", "syntax_valid": True}
            record["outcome"] = result["status"]
            return {**result, "lease": lease}
        full_function_lines = candidate_function.splitlines()
        result = replace_python_source_lines_array_transactional_v2(
            self.workspace, record["path"], record["expected_sha256"],
            record["start_line"], record["end_line"], full_function_lines, self.allowed,
        )
        record["outcome"] = result.get("status", "valid_call_rejected_by_policy")
        if result.get("after_sha256"):
            record["after_sha256"] = result["after_sha256"]
        record.update({
            "raw_body_lines_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "line_ending": record["line_ending"],
        })
        for key in ("syntax_valid", "syntax_error"):
            if key in result:
                record[key] = result[key]
        return {
            **result, "lease": lease, "qualified_name": record["qualified_name"],
            "raw_body_lines_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "line_ending": record["line_ending"],
        }
