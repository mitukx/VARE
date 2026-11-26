"""Bounded, source-grounded navigation helpers for coding agents.

The helpers in this module inspect source text without importing or executing it.
They are intentionally small: the caller remains responsible for task
allowlisting, safe source reads, and limiting the serialized tool response.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any


_QUERY_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_GENERIC_QUERY_WORDS = {
    "a", "an", "and", "are", "as", "at", "before", "by", "does", "for",
    "from", "in", "is", "it", "of", "on", "or", "that", "the", "then",
    "this", "to", "use", "where", "with",
}


@dataclass(frozen=True)
class _SearchFocus:
    query: str
    path: str | None
    words: frozenset[str]
    strong_identifiers: frozenset[str]


class SearchLoopGuard:
    """Detect repeated semantically similar literal searches with no matches.

    After ``empty_search_limit`` related empty searches, the last result carries
    navigation alternatives. Further related searches are intercepted until a
    source read, symbol lookup, or successful edit changes the agent's context.
    Positive or unrelated searches start a new sequence.
    """

    def __init__(self, empty_search_limit: int = 3) -> None:
        if type(empty_search_limit) is not int or empty_search_limit < 1:
            raise ValueError("empty_search_limit must be a positive integer")
        self.empty_search_limit = empty_search_limit
        self._misses: list[_SearchFocus] = []
        self._blocked_focus: _SearchFocus | None = None

    @staticmethod
    def _focus(query: str, path: str | None) -> _SearchFocus:
        identifiers = [word.lower() for word in _QUERY_WORD.findall(query)]
        words = frozenset(
            part
            for identifier in identifiers
            for part in identifier.split("_")
            if part and part not in _GENERIC_QUERY_WORDS
        )
        strong = frozenset(
            identifier for identifier in identifiers
            if "_" in identifier or len(identifier) >= 14
        )
        return _SearchFocus(query=query, path=path, words=words, strong_identifiers=strong)

    @staticmethod
    def _related(left: _SearchFocus, right: _SearchFocus) -> bool:
        if left.path is not None and right.path is not None and left.path != right.path:
            return False
        if left.strong_identifiers & right.strong_identifiers:
            return True
        if not left.words or not right.words:
            return False
        overlap = len(left.words & right.words) / min(len(left.words), len(right.words))
        return overlap >= 0.75

    def reset(self) -> None:
        """Clear the current miss sequence after source context changes."""
        self._misses.clear()
        self._blocked_focus = None

    def intercept(self, query: str, path: str | None = None) -> dict[str, Any] | None:
        """Return a redirect result instead of repeating a blocked search."""
        focus = self._focus(query, path)
        if self._blocked_focus is None or not self._related(self._blocked_focus, focus):
            return None
        return {
            "executed": False,
            "redirected": True,
            "reason": "repeated_related_searches_returned_no_matches",
            "query": query,
            "alternatives": [
                "Use list_symbols(path, query=<class-or-function concept>) to inspect Python definitions and class fields.",
                "Use read_file(path, start_line, end_line) to inspect a bounded source range.",
                "For exact text search, use a short case-sensitive literal copied from source context.",
            ],
        }

    def observe(
        self, query: str, match_count: int, path: str | None = None,
    ) -> dict[str, Any] | None:
        """Record an executed search and optionally attach a redirect hint."""
        if type(match_count) is not int or match_count < 0:
            raise ValueError("match_count must be a non-negative integer")
        focus = self._focus(query, path)
        if match_count:
            self.reset()
            return None
        if self._misses and self._related(self._misses[-1], focus):
            self._misses.append(focus)
        else:
            self._misses = [focus]
            self._blocked_focus = None
        if len(self._misses) < self.empty_search_limit:
            return None
        self._blocked_focus = focus
        return {
            "reason": "repeated_related_searches_returned_no_matches",
            "related_empty_searches": len(self._misses),
            "alternatives": [
                "Use list_symbols(path, query=<class-or-function concept>) to inspect Python definitions and class fields.",
                "Use read_file(path, start_line, end_line) to inspect a bounded source range.",
                "For exact text search, use a short case-sensitive literal copied from source context.",
            ],
        }


class SequentialReadGuide:
    """Offer a compact AST map after repeated small, contiguous source reads.

    Some agents spend their tool budget paging through a Python file from line
    one without locating a declaration. This guide is advisory: it never
    suppresses a read or changes the returned source page.
    """

    def __init__(
        self, *, page_threshold: int = 4, max_page_lines: int = 40,
        max_symbols: int = 6,
    ) -> None:
        if type(page_threshold) is not int or page_threshold < 2:
            raise ValueError("page_threshold must be an integer of at least two")
        if type(max_page_lines) is not int or max_page_lines < 1:
            raise ValueError("max_page_lines must be a positive integer")
        if type(max_symbols) is not int or not 1 <= max_symbols <= 12:
            raise ValueError("max_symbols must be between one and twelve")
        self.page_threshold = page_threshold
        self.max_page_lines = max_page_lines
        self.max_symbols = max_symbols
        self.reset()

    def reset(self) -> None:
        """Forget a page sequence after another tool or source change."""
        self._path: str | None = None
        self._source_sha256: str | None = None
        self._last_end_line: int | None = None
        self._page_count = 0
        self._hinted = False

    @staticmethod
    def _terms(text: str) -> set[str]:
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        return {
            part.lower()
            for token in _QUERY_WORD.findall(split)
            for part in token.split("_")
            if len(part) > 2 and part.lower() not in _GENERIC_QUERY_WORDS
        }

    def _outline(
        self, source: str, *, task_text: str, after_line: int,
    ) -> list[dict[str, Any]]:
        if len(source.encode("utf-8")) > 1_000_000:
            return []
        try:
            indexed = list_python_symbols(source, limit=60, max_limit=60)
        except (RecursionError, ValueError):
            return []
        symbols = indexed.get("symbols", [])
        if not symbols:
            return []
        terms = self._terms(task_text)

        def symbol_terms(symbol: dict[str, Any]) -> set[str]:
            fields = symbol.get("fields", [])
            searchable = [
                str(symbol.get("name", "")),
                *[str(item) for item in symbol.get("methods", [])],
                *[str(field.get("name", "")) for field in fields],
                *[str(field.get("annotation") or "") for field in fields],
                *[str(item) for item in symbol.get("bases", [])],
                *[str(item) for item in symbol.get("parameters", [])],
                str(symbol.get("returns") or ""),
            ]
            return self._terms(" ".join(searchable))

        ranked = [
            (len(terms & symbol_terms(symbol)), symbol)
            for symbol in symbols
        ]
        relevant = [item for item in ranked if item[0] > 0]
        if relevant:
            ordered = [
                symbol for _score, symbol in sorted(
                    relevant,
                    key=lambda item: (-item[0], item[1].get("line", 0), item[1].get("name", "")),
                )
            ]
        else:
            ordered = sorted(
                symbols,
                key=lambda item: (
                    item.get("line", 0) < after_line,
                    abs(item.get("line", 0) - after_line),
                    item.get("line", 0),
                ),
            )
        return [
            {
                "kind": symbol.get("kind"),
                "name": symbol.get("name"),
                "line": symbol.get("line"),
                "end_line": symbol.get("end_line"),
            }
            for symbol in ordered[:self.max_symbols]
        ]

    def observe(
        self, *, path: str, source: str, source_sha256: str,
        start_line: int, end_line: int, char_offset: int = 0,
        task_text: str,
    ) -> dict[str, Any] | None:
        """Return a task-oriented declaration map once a scan pattern appears."""
        if (
            not isinstance(path, str) or not path.endswith(".py")
            or not isinstance(source, str)
            or not isinstance(source_sha256, str)
            or type(start_line) is not int or type(end_line) is not int
            or start_line < 1 or end_line < start_line
            or type(char_offset) is not int or char_offset != 0
            or end_line - start_line + 1 > self.max_page_lines
        ):
            self.reset()
            return None

        contiguous = (
            self._path == path
            and self._source_sha256 == source_sha256
            and self._last_end_line is not None
            and start_line == self._last_end_line + 1
        )
        if contiguous:
            self._page_count += 1
        else:
            self._path = path
            self._source_sha256 = source_sha256
            self._page_count = 1
            self._hinted = False
        self._last_end_line = end_line
        if self._hinted or self._page_count < self.page_threshold:
            return None

        outline = self._outline(source, task_text=task_text, after_line=end_line)
        if not outline:
            self._hinted = True
            return None
        self._hinted = True
        return {
            "type": "sequential_source_scan",
            "consecutive_pages": self._page_count,
            "message": (
                "Several consecutive source pages have been read. These AST-indexed "
                "declarations may help you jump to the relevant implementation."
            ),
            "declarations": outline,
        }


def _unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return None


def list_python_symbols(
    source: str,
    *,
    query: str = "",
    offset: int = 0,
    limit: int = 30,
    max_limit: int = 60,
) -> dict[str, Any]:
    """Return a paginated AST index of Python classes and functions.

    No candidate module is imported or executed. Class entries include direct
    annotated fields and methods so a task description such as "request model"
    can locate a class even when the literal term is absent from its body.
    """
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if type(limit) is not int or limit < 1 or limit > max_limit:
        raise ValueError(f"limit must be between 1 and {max_limit}")
    if not isinstance(query, str):
        raise ValueError("query must be a string")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {
            "error": "source is not parseable Python",
            "line": exc.lineno,
            "column": exc.offset,
        }

    entries: list[dict[str, Any]] = []

    def visit_body(body: list[ast.stmt], parents: tuple[str, ...] = ()) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                class_name = ".".join((*parents, node.name))
                fields = [
                    {
                        "name": item.target.id,
                        "annotation": _unparse(item.annotation),
                        "line": item.lineno,
                    }
                    for item in node.body
                    if isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)
                ]
                methods = [
                    item.name for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                entries.append({
                    "kind": "class",
                    "name": class_name,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "bases": [base for base in (_unparse(x) for x in node.bases) if base],
                    "fields": fields,
                    "methods": methods,
                })
                visit_body(node.body, (*parents, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = ".".join((*parents, node.name))
                entries.append({
                    "kind": "method" if parents else "function",
                    "name": name,
                    "line": node.lineno,
                    "end_line": getattr(node, "end_lineno", node.lineno),
                    "async": isinstance(node, ast.AsyncFunctionDef),
                    "parameters": [arg.arg for arg in node.args.args],
                    "returns": _unparse(node.returns),
                    "decorators": [item for item in (_unparse(x) for x in node.decorator_list) if item],
                })
                visit_body(node.body, (*parents, node.name))

    visit_body(tree.body)
    terms = {part.lower() for part in _QUERY_WORD.findall(query)}
    if terms:
        def matches(entry: dict[str, Any]) -> bool:
            searchable = [entry["name"]]
            searchable.extend(entry.get("methods", []))
            searchable.extend(field["name"] for field in entry.get("fields", []))
            searchable.extend(field.get("annotation") or "" for field in entry.get("fields", []))
            searchable.extend(entry.get("bases", []))
            searchable.extend(entry.get("decorators", []))
            haystack = " ".join(searchable).lower()
            # Match all query terms so a generic term cannot swamp the result.
            return all(term in haystack for term in terms)
        entries = [entry for entry in entries if matches(entry)]

    selected = entries[offset:offset + limit]
    next_offset = offset + len(selected) if offset + len(selected) < len(entries) else None
    return {
        "symbols": selected,
        "query": query,
        "offset": offset,
        "total_symbols": len(entries),
        "has_more": next_offset is not None,
        "next_offset": next_offset,
    }
