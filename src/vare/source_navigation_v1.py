"""Bounded, hash-aware source search for candidate-visible task files.

This module indexes only the exact source paths supplied by the task
controller. It never discovers files, executes source text, or reads task
tests, graders, Git history, or known-fix artifacts. File reads and symlink
checks reuse the atomic editor's pinned source reader.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .agent_edit_v2 import read_source_lines_v1

MAX_QUERY_CHARS = 256
MAX_QUERY_TERMS = 24
MAX_RESULT_COUNT = 8
MAX_OUTPUT_BYTES = 8 * 1024
MAX_CONTEXT_LINES = 1
MAX_INDEX_FILES = 128
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_TEST_DIRS = {"test", "tests", "grader", "graders", "__tests__"}
_PROTECTED_DIRS = _TEST_DIRS | {
    ".git", "fixes", "known-fixes", "known_fixes", "history", "historical",
    "references", "reference-data",
}


def _candidate_source_path(path: str) -> bool:
    """Reject conventional test/grader paths even if a caller over-allowlists."""
    if not path:
        return False
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or path != parsed.as_posix() or any(
        part in {"", ".", ".."} for part in parsed.parts
    ):
        return False
    lowered = tuple(part.casefold() for part in parsed.parts)
    basename = lowered[-1]
    if any(part in _PROTECTED_DIRS for part in lowered[:-1]):
        return False
    if basename.startswith("test_") or basename.endswith("_test.py"):
        return False
    return basename not in {
        "test_public.py", "grader.py", "hidden_tests.py", "known_fix.py",
        "known_fix.patch", "known_fix.diff",
    }


def _terms(text: str) -> list[str]:
    return list(dict.fromkeys(token.casefold() for token in _TOKEN.findall(text)))


def _error(status: str, **fields: Any) -> dict[str, Any]:
    return {"status": status, **fields}


class SourceNavigatorV1:
    """Deterministic ranked search over a bounded source snapshot.

    Search results are invalidated and rebuilt whenever any indexed source
    hash changes. The serialized response, including metadata, is capped by
    ``max_output_bytes``.
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
    ) -> None:
        if not 1 <= max_results <= MAX_RESULT_COUNT:
            raise ValueError("max_results must be between 1 and the hard result limit")
        if not 512 <= max_output_bytes <= MAX_OUTPUT_BYTES:
            raise ValueError("max_output_bytes exceeds the hard output limit")
        if not 0 <= context_lines <= MAX_CONTEXT_LINES:
            raise ValueError("context_lines exceeds the hard context limit")
        if not 1 <= max_total_source_bytes <= MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("max_total_source_bytes exceeds the hard index limit")
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
        self._snapshot: dict[str, dict[str, Any]] = {}
        self._index_sha256 = ""
        self._generation = 0
        self._last_build_error: dict[str, Any] | None = None
        self._rebuild()

    @property
    def index_sha256(self) -> str:
        return self._index_sha256

    @property
    def indexed_paths(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshot))

    def _read_file(self, path: str) -> dict[str, Any]:
        first = read_source_lines_v1(self.workspace, path, 1, 1, self.allowed)
        if first.get("status") == "valid_call_rejected_by_policy":
            if first.get("error") == "line_range_out_of_bounds" and first.get("line_count") == 0:
                return {
                    "status": "read_complete", "path": path, "line_count": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(), "content": "",
                }
            return first
        if first.get("status") != "read_complete":
            return first

        line_count = first["line_count"]
        expected_hash = first["sha256"]
        pieces = [first["content"]]
        start = 2
        while start <= line_count:
            end = min(start + 119, line_count)
            chunk = read_source_lines_v1(self.workspace, path, start, end, self.allowed)
            if chunk.get("status") != "read_complete":
                return chunk
            if chunk["sha256"] != expected_hash or chunk["line_count"] != line_count:
                return _error("source_changed_during_index", path=path)
            pieces.append(chunk["content"])
            start = end + 1
        content = "".join(pieces)
        if hashlib.sha256(content.encode("utf-8")).hexdigest() != expected_hash:
            return _error("source_changed_during_index", path=path)
        return {
            "status": "read_complete", "path": path, "line_count": line_count,
            "sha256": expected_hash, "content": content,
        }

    def _build_snapshot(
        self,
    ) -> tuple[dict[str, dict[str, Any]], str | dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        for path in sorted(self.allowed):
            read = self._read_file(path)
            if read.get("status") != "read_complete":
                return {}, _error(
                    "source_read_failed", path=path,
                    reason=str(read.get("error", read.get("status", "unknown")))[:120],
                )
            content = read["content"]
            total_bytes += len(content.encode("utf-8"))
            if total_bytes > self.max_total_source_bytes:
                return {}, _error("source_index_limit_exceeded", limit="total_source_bytes")
            lines = content.splitlines()
            snapshot[path] = {
                "sha256": read["sha256"],
                "line_count": read["line_count"],
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
            fresh = read_source_lines_v1(self.workspace, path, 1, 1, self.allowed)
            if fresh.get("status") == "valid_call_rejected_by_policy":
                if fresh.get("error") == "line_range_out_of_bounds" and fresh.get("line_count") == 0:
                    current_hash = hashlib.sha256(b"").hexdigest()
                else:
                    return True
            elif fresh.get("status") == "read_complete":
                current_hash = fresh["sha256"]
            else:
                return True
            if current_hash != indexed["sha256"]:
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
        terms_set = set(query_terms)
        matches: list[dict[str, Any]] = []
        for path, indexed in sorted(self._snapshot.items()):
            lines = indexed["lines"]
            token_rows = [set(_terms(line)) for line in lines]
            for offset, line in enumerate(lines):
                matched = [term for term in query_terms if term in token_rows[offset]]
                phrase_match = bool(phrase and phrase in " ".join(line.casefold().split()))
                if not matched and not phrase_match:
                    continue
                score = 100 * int(phrase_match) + 10 * len(matched)
                if terms_set and len(matched) == len(terms_set):
                    score += 25
                begin = max(0, offset - self.context_lines)
                end = min(len(lines), offset + self.context_lines + 1)
                context = "\n".join(lines[begin:end])
                matches.append({
                    "path": path,
                    "line": offset + 1,
                    "score": score,
                    "matched_terms": matched,
                    "context_start_line": begin + 1,
                    "context": context,
                    "source_sha256": indexed["sha256"],
                })
        matches.sort(key=lambda item: (-item["score"], item["path"], item["line"]))
        result = {
            "status": "ok",
            "query": query_text,
            "index_sha256": self._index_sha256,
            "index_generation": self._generation,
            "index_rebuilt": rebuilt,
            "indexed_paths": list(sorted(self._snapshot)),
            "result_limit": self.max_results,
            "truncated": len(matches) > self.max_results,
            "results": matches[:self.max_results],
            "content_is_untrusted_source_data": True,
        }
        return self._bounded(result)
