"""Bounded search over explicitly allowlisted and filename-filtered sources.

The navigator accepts only caller-supplied paths and applies VARE's local path
policy before reading them. This cannot establish that an otherwise innocuous
file contains no reference implementation or embedded answer; task
qualification must certify source visibility and grader separation. Reads use
the atomic editor's descriptor-safe primitives.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .agent_edit_v2 import (
    MAX_SOURCE_BYTES,
    _decode_source,
    _open_parent,
    _path_parts,
    _read_current,
)
from .agent_task_contract import candidate_visible_source_files

MAX_QUERY_CHARS = 256
MAX_QUERY_TERMS = 24
MAX_RESULT_COUNT = 8
MAX_OUTPUT_BYTES = 8 * 1024
MAX_CONTEXT_LINES = 1
MAX_INDEX_FILES = 128
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024
MAX_FILE_SOURCE_BYTES = MAX_SOURCE_BYTES
MAX_FILE_SOURCE_LINES = 100_000
MAX_TOTAL_SOURCE_LINES = 250_000
MAX_CONTEXT_CHARS = 512
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_LINE_BREAK = re.compile(r"\r\n|[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]")
_LINE_BREAK_CHARS = frozenset("\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029")


def _candidate_source_path(path: str) -> bool:
    """Reject conventional test/grader paths even if a caller over-allowlists."""
    return path in candidate_visible_source_files((path,))


def _terms(text: str) -> list[str]:
    return list(dict.fromkeys(token.casefold() for token in _TOKEN.findall(text)))


def _source_line_count(text: str) -> int:
    """Count Python-style splitlines without constructing a line list."""
    if not text:
        return 0
    separators = sum(1 for _match in _LINE_BREAK.finditer(text))
    return separators + int(text[-1] not in _LINE_BREAK_CHARS)


def _split_source_lines(text: str) -> list[str]:
    """Split only after byte and line-count guards have passed."""
    return text.splitlines()


def _line_match(line: str, query_terms: list[str], query_term_set: set[str]) -> list[str]:
    """Collect only matched query terms; do not materialize every line token."""
    found: set[str] = set()
    for match in _TOKEN.finditer(line):
        term = match.group(0).casefold()
        if term in query_term_set:
            found.add(term)
            if len(found) == len(query_term_set):
                break
    return [term for term in query_terms if term in found]


def _bounded_context(context: str, query_terms: list[str]) -> tuple[str, bool]:
    if len(context) <= MAX_CONTEXT_CHARS:
        return context, False
    folded = context.casefold()
    anchor = min((position for term in query_terms
                  if (position := folded.find(term)) >= 0), default=0)
    start = max(0, min(anchor - MAX_CONTEXT_CHARS // 3,
                       len(context) - MAX_CONTEXT_CHARS))
    excerpt = context[start:start + MAX_CONTEXT_CHARS]
    return ("…" if start else "") + excerpt + (
        "…" if start + MAX_CONTEXT_CHARS < len(context) else ""
    ), True


def _error(status: str, **fields: Any) -> dict[str, Any]:
    return {"status": status, **fields}


class SourceNavigatorV1:
    """Rank the supplied, locally filtered paths in a bounded snapshot.

    Search results are invalidated and rebuilt whenever any indexed source
    hash changes. The local path filter is defense in depth, not a substitute
    for independently qualified task isolation. The serialized response,
    including metadata, is capped by ``max_output_bytes``.
    """

    def __init__(
        self,
        workspace: Path,
        candidate_source_paths: set[str],
        *,
        max_results: int = MAX_RESULT_COUNT,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        context_lines: int = MAX_CONTEXT_LINES,
        max_total_source_bytes: int = MAX_TOTAL_SOURCE_BYTES,
        max_total_source_lines: int = MAX_TOTAL_SOURCE_LINES,
    ) -> None:
        if not 1 <= max_results <= MAX_RESULT_COUNT:
            raise ValueError("max_results must be between 1 and the hard result limit")
        if not 512 <= max_output_bytes <= MAX_OUTPUT_BYTES:
            raise ValueError("max_output_bytes exceeds the hard output limit")
        if not 0 <= context_lines <= MAX_CONTEXT_LINES:
            raise ValueError("context_lines exceeds the hard context limit")
        if not 1 <= max_total_source_bytes <= MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("max_total_source_bytes exceeds the hard index limit")
        if not 1 <= max_total_source_lines <= MAX_TOTAL_SOURCE_LINES:
            raise ValueError("max_total_source_lines exceeds the hard index limit")
        self.workspace = workspace
        self.allowed = {
            path for path in candidate_source_paths
            if isinstance(path, str) and _candidate_source_path(path)
        }
        if len(self.allowed) > MAX_INDEX_FILES:
            raise ValueError("candidate source file count exceeds the hard index limit")
        self.max_results = max_results
        self.max_output_bytes = max_output_bytes
        self.context_lines = context_lines
        self.max_total_source_bytes = max_total_source_bytes
        self.max_total_source_lines = max_total_source_lines
        self._snapshot: dict[str, dict[str, Any]] = {}
        self._index_sha256 = ""
        self._generation = 0
        self._last_build_error: dict[str, Any] | None = None
        if not self.allowed:
            self._last_build_error = _error("no_candidate_visible_source_files")
        else:
            self._rebuild()

    @property
    def index_sha256(self) -> str:
        return self._index_sha256

    @property
    def indexed_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshot))

    def _read_file_bytes(self, path: str) -> dict[str, Any]:
        """Read exactly once through the editor's fd-safe path primitives."""
        parts, error = _path_parts(path, self.allowed)
        if error:
            return _error("valid_call_rejected_by_policy", error=error)
        assert parts is not None
        parent_fd = -1
        try:
            parent_fd, basename = _open_parent(self.workspace, parts)
            data, _file_stat = _read_current(parent_fd, basename)
            if len(data) > MAX_FILE_SOURCE_BYTES:
                return _error("source_file_limit_exceeded", limit="per_file_bytes")
            return {
                "status": "read_complete", "path": path,
                "sha256": hashlib.sha256(data).hexdigest(), "data": data,
            }
        except (OSError, UnicodeError, ValueError) as exc:
            return _error("valid_call_rejected_by_policy", error=str(exc)[:160])
        finally:
            if parent_fd >= 0:
                os.close(parent_fd)

    def _build_snapshot(
        self,
    ) -> tuple[dict[str, dict[str, Any]], str | dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        total_lines = 0
        for path in sorted(self.allowed):
            read = self._read_file_bytes(path)
            if read.get("status") != "read_complete":
                return {}, _error(
                    "source_read_failed", path=path,
                    reason=str(read.get("error", read.get("status", "unknown")))[:120],
                )
            data = read["data"]
            total_bytes += len(data)
            if total_bytes > self.max_total_source_bytes:
                return {}, _error("source_index_limit_exceeded", limit="total_source_bytes")
            try:
                content = _decode_source(data)
            except (UnicodeError, ValueError) as exc:
                return {}, _error(
                    "source_read_failed", path=path,
                    reason=str(exc)[:120],
                )
            line_count = _source_line_count(content)
            if line_count > MAX_FILE_SOURCE_LINES:
                return {}, _error("source_index_limit_exceeded", limit="per_file_lines")
            total_lines += line_count
            if total_lines > self.max_total_source_lines:
                return {}, _error("source_index_limit_exceeded", limit="total_source_lines")
            lines = _split_source_lines(content)
            snapshot[path] = {
                "sha256": read["sha256"],
                "line_count": line_count,
                "lines": lines,
            }
        digest_input = json.dumps(
            {path: value["sha256"] for path, value in sorted(snapshot.items())},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return snapshot, hashlib.sha256(digest_input).hexdigest()

    def _rebuild(self) -> bool:
        snapshot, digest_or_error = self._build_snapshot()
        if not isinstance(digest_or_error, str):
            self._last_build_error = digest_or_error
            self._snapshot = {}
            self._index_sha256 = ""
            return False
        self._snapshot = snapshot
        self._index_sha256 = digest_or_error
        self._generation += 1
        self._last_build_error = None
        return True

    def _is_stale(self) -> bool:
        for path, indexed in self._snapshot.items():
            fresh = self._read_file_bytes(path)
            if fresh.get("status") != "read_complete":
                return True
            if fresh["sha256"] != indexed["sha256"]:
                return True
        return False

    def _bounded(self, result: dict[str, Any]) -> dict[str, Any]:
        """Drop lower-ranked results until the actual JSON payload fits."""
        results = result.get("results")
        while True:
            # Match the local agent's exact tool-result serialization.
            encoded = json.dumps(result, ensure_ascii=False).encode("utf-8")
            size = len(encoded)
            if size <= self.max_output_bytes and result.get("output_bytes") == size:
                return result
            if size <= self.max_output_bytes:
                result["output_bytes"] = size
                continue
            if not isinstance(results, list) or not results:
                return {
                    "status": "output_limit_too_small",
                    "max_output_bytes": self.max_output_bytes,
                }
            results.pop()
            result["truncated"] = True

    def search(self, query: object) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
            return self._bounded(_error("invalid_query", max_query_chars=MAX_QUERY_CHARS))
        query_text = query.strip()
        query_terms = _terms(query_text)
        if not query_terms or len(query_terms) > MAX_QUERY_TERMS:
            return self._bounded(_error("invalid_query", max_query_terms=MAX_QUERY_TERMS))

        rebuilt = False
        if self._last_build_error is not None:
            return self._bounded(dict(self._last_build_error))
        if self._is_stale():
            if not self._rebuild():
                return self._bounded(dict(self._last_build_error or _error("source_index_failed")))
            rebuilt = True

        phrase = " ".join(query_text.casefold().split())
        phrase_pattern = re.compile(
            r"\s+".join(re.escape(part) for part in phrase.split())
        ) if phrase else None
        terms_set = set(query_terms)
        matches: list[dict[str, Any]] = []
        match_count = 0
        for path, indexed in sorted(self._snapshot.items()):
            lines = indexed["lines"]
            for offset, line in enumerate(lines):
                matched = _line_match(line, query_terms, terms_set)
                phrase_match = bool(phrase_pattern and phrase_pattern.search(line.casefold()))
                if not matched and not phrase_match:
                    continue
                match_count += 1
                score = 100 * int(phrase_match) + 10 * len(matched)
                if terms_set and len(matched) == len(terms_set):
                    score += 25
                begin = max(0, offset - self.context_lines)
                end = min(len(lines), offset + self.context_lines + 1)
                context, context_truncated = _bounded_context(
                    "\n".join(lines[begin:end]), query_terms,
                )
                matches.append({
                    "path": path,
                    "line": offset + 1,
                    "score": score,
                    "matched_terms": matched,
                    "context_start_line": begin + 1,
                    "context": context,
                    "context_truncated": context_truncated,
                    "source_sha256": indexed["sha256"],
                })
                matches.sort(key=lambda item: (-item["score"], item["path"], item["line"]))
                if len(matches) > self.max_results:
                    matches.pop()
        result = {
            "status": "ok",
            "query": query_text,
            "index_sha256": self._index_sha256,
            "index_generation": self._generation,
            "index_rebuilt": rebuilt,
            "indexed_paths": list(sorted(self._snapshot)),
            "result_limit": self.max_results,
            "truncated": match_count > len(matches),
            "results": matches,
            "content_is_untrusted_source_data": True,
        }
        return self._bounded(result)
