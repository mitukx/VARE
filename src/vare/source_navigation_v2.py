"""AST-assisted Python qualified-name lookup over the V1 bounded index.

V1 search requires every query term to occur on one physical line. This
version keeps that behavior for ordinary searches and adds a narrow semantic
case for ``ClassName.method_name`` queries, which naturally span source lines.
It parses candidate-visible Python source only; it never imports or executes it.
"""
from __future__ import annotations

import ast
import re
from typing import Any

from .source_navigation_v1 import SourceNavigatorV1

_QUALIFIED_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")


class SourceNavigatorV2(SourceNavigatorV1):
    """Support qualified-name lookup and reject repeated searches on unchanged code."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._searched_queries: set[tuple[int, str]] = set()

    def search(self, query: object) -> dict[str, Any]:
        normalized = (
            " ".join(query.casefold().split())
            if isinstance(query, str) and query.strip() else None
        )
        if normalized is not None and self._last_build_error is None and not self._is_stale():
            key = (self._generation, normalized)
            if key in self._searched_queries:
                return self._bounded({
                    "status": "duplicate_query",
                    "query": query.strip(),
                    "index_sha256": self._index_sha256,
                    "index_generation": self._generation,
                    "retry_hint": (
                        "This exact query was already searched on the unchanged source snapshot. "
                        "Search a shorter code identifier or a distinctive term from the implementation; "
                        "do not repeat the task description."
                    ),
                })

        result = super().search(query)
        if normalized is not None and result.get("status") == "ok":
            self._searched_queries.add((self._generation, normalized))
        if (not isinstance(query, str) or not _QUALIFIED_IDENTIFIER.fullmatch(query.strip())
                or result.get("status") != "ok" or result.get("results")):
            return result

        parts = query.strip().split(".")
        class_name, method_name = parts[-2:]
        matches: list[dict[str, Any]] = []
        for path, indexed in sorted(self._snapshot.items()):
            if not path.endswith(".py"):
                continue
            lines = indexed["lines"]
            try:
                tree = ast.parse("\n".join(lines), filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef) or node.name != class_name:
                    continue
                for member in node.body:
                    if (not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                            or member.name != method_name):
                        continue
                    offset = member.lineno - 1
                    begin = max(0, offset - self.context_lines)
                    end = min(len(lines), offset + self.context_lines + 1)
                    context = "\n".join(lines[begin:end])
                    matches.append({
                        "path": path,
                        "line": member.lineno,
                        "score": 200,
                        "matched_terms": [class_name.casefold(), method_name.casefold()],
                        "context_start_line": begin + 1,
                        "context": context[:512],
                        "context_truncated": len(context) > 512,
                        "source_sha256": indexed["sha256"],
                        "match_kind": "python_class_method",
                    })

        if not matches:
            return result
        result["results"] = sorted(matches, key=lambda item: (item["path"], item["line"]))[:self.max_results]
        result["truncated"] = len(matches) > self.max_results
        result["qualified_name_resolution"] = "ast_class_method"
        result.pop("output_bytes", None)
        return self._bounded(result)
