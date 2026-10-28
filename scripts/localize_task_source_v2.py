#!/usr/bin/env python3
"""Select one bounded AST definition named explicitly in candidate-visible task text.

This source-only locator never reads tests, graders, fixes, or task history. It
fails closed when an explicit code entity cannot be matched to one definition.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.agent_task_contract import load_agent_task  # noqa: E402

WINDOW_LINES = 80
_BACKTICK = re.compile(r"`([^`]+)`")
_QUALIFIED_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _named_definitions(source: str, requested: set[str]) -> list[tuple[str, ast.AST]]:
    tree = ast.parse(source)
    found: list[tuple[str, ast.AST]] = []

    def visit(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
        for child in ast.iter_child_nodes(node):
            name = getattr(child, "name", None)
            next_parents = parents
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join((*parents, name)) if parents else name
                if name in requested or qualified in requested:
                    found.append((qualified, child))
                next_parents = (*parents, name)
            visit(child, next_parents)

    visit(tree)
    return found


def choose_definition_window(prompt: str, source: str, path: str) -> dict[str, Any]:
    requested = {
        value.strip() for value in _BACKTICK.findall(prompt)
        if _QUALIFIED_NAME.fullmatch(value.strip())
    }
    if not requested:
        raise ValueError("task prompt contains no explicit code entity")
    definitions = _named_definitions(source, requested)
    if not definitions:
        raise ValueError("explicit task code entity does not match a source definition")
    if len(definitions) > 1:
        exact_qualified = [(name, node) for name, node in definitions if name in requested]
        if len(exact_qualified) == 1:
            definitions = exact_qualified
        else:
            raise ValueError("explicit code entity matches multiple source definitions")
    name, node = definitions[0]
    lines = source.splitlines(keepends=True)
    start_at = getattr(node, "lineno", None)
    end_at = getattr(node, "end_lineno", None)
    if (not isinstance(start_at, int) or not isinstance(end_at, int)
            or start_at < 1 or end_at < start_at or end_at > len(lines)):
        raise ValueError("matched source definition has invalid line bounds")
    width = min(WINDOW_LINES, len(lines))
    start = max(1, min(start_at - (width - (end_at - start_at + 1)) // 2,
                       len(lines) - width + 1))
    end = start + width - 1
    if start > start_at or end < end_at:
        raise ValueError("bounded localization window cannot contain the matched definition")
    content = "".join(lines[start - 1:end])
    return {
        "algorithm": "task_prompt_ast_entity_window_v1",
        "path": path,
        "start_line": start,
        "end_line": end,
        "window_lines": width,
        "matched_entity": name,
        "matched_node_kind": type(node).__name__,
        "content": content,
    }


def localize(task_root: Path, workspace: Path) -> dict[str, Any]:
    contract = load_agent_task(task_root)
    source_paths = [
        path for path in contract.candidate_visible_source_files
        if path.endswith(".py") and path != "test_public.py"
    ]
    if len(source_paths) != 1:
        raise ValueError("AST localization v2 requires exactly one candidate-visible Python source file")
    path = source_paths[0]
    raw = (workspace / path).read_bytes()
    source = raw.decode("utf-8", errors="strict")
    task_text = contract.prompt_text.split("## Reproduce", 1)[0].rstrip()
    selected = choose_definition_window(task_text, source, path)
    return {
        **selected,
        "schema_version": 2,
        "task_id": contract.task_id,
        "base_revision": contract.base_revision,
        "task_prompt_sha256": _sha(task_text.encode("utf-8")),
        "source_sha256": _sha(raw),
        "content_sha256": _sha(selected["content"].encode("utf-8")),
        "selection_inputs": ["task_prompt_inline_code_entities", "candidate_source_ast", "editable_path_from_task_metadata"],
        "known_fix_or_grader_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite localization record: {args.output}")
    record = localize(args.task_root, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "task_id", "path", "start_line", "end_line", "matched_entity", "content_sha256",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
