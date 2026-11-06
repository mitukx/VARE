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
    "a", "an", "and", "are", "as", "at", "before", "by", "class", "does", "def", "for",
    "from", "in", "is", "it", "of", "on", "or", "that", "the", "then",
    "this", "to", "use", "where", "with",
}


def _normalized_words(text: str) -> list[str]:
    """Split identifiers and normalize common inflections for search-loop matching."""
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    words: list[str] = []
    for raw in _QUERY_WORD.findall(camel_split):
        parts = raw.split("_")
        for part in parts:
            word = part.lower()
            if not word:
                continue
            if word == "deferred":
                word = "defer"
            elif word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("ing") and len(word) > 5:
                word = word[:-3]
                if len(word) > 2 and word[-1] == word[-2]:
                    word = word[:-1]
            elif word.endswith("ed") and len(word) > 4:
                word = word[:-2]
                if len(word) > 2 and word[-1] == word[-2]:
                    word = word[:-1]
            elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
                word = word[:-1]
            if word and word not in _GENERIC_QUERY_WORDS:
                words.append(word)
    return words


@dataclass(frozen=True)
class _SearchFocus:
    query: str
    path: str | None
    words: frozenset[str]
    strong_identifiers: frozenset[str]


class SearchLoopGuard:
    """Detect repeated no-match searches before the agent reads source.

    When a path is supplied, misses in that task-owned file share one budget even
    if the model keeps rephrasing the query. Without a path, semantic similarity
    is used. Further related searches are intercepted until source is read, symbol
    context changes, or an edit succeeds.
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
        words = frozenset(_normalized_words(query))
        identifiers = [word.lower() for word in _QUERY_WORD.findall(query)]
        strong = frozenset(
            "_".join(_normalized_words(identifier))
            for identifier in identifiers
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
    # Tool callers often send a short natural-language question, not a single
    # identifier. Ignore generic words and split camel-case identifiers so a
    # query such as "Where is cache factory?" can locate ``CacheFactory``.
    terms = set(_normalized_words(query))
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
