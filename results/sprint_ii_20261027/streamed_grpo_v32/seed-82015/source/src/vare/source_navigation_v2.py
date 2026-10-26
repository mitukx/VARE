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
    """Support exact class-method searches without weakening V1 text matching."""

    def search(self, query: object) -> dict[str, Any]:
        result = super().search(query)
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
