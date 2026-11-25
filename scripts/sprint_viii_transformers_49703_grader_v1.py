#!/usr/bin/env python3
"""Non-executing semantic grader for Transformers PR 49703's registration defect."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


TARGET_CLASS = "Mask2FormerTransformerModule"
TARGET_METHOD = "__init__"
TARGET_PATH = "src/transformers/models/mask2former/modeling_mask2former.py"


def read_source(root_raw: str, relative_raw: str) -> tuple[bytes | None, str | None]:
    root = Path(root_raw).absolute()
    relative = PurePosixPath(relative_raw)
    if root.is_symlink() or not root.is_dir():
        return None, "workspace_root_not_real_directory"
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return None, "source_path_invalid"
    path = root
    for part in relative.parts:
        path = path / part
        try:
            mode = path.lstat().st_mode
        except OSError:
            return None, "source_unavailable"
        if stat.S_ISLNK(mode):
            return None, "source_symlink_rejected"
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None, "source_not_regular_file"
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(fd, 65536)
                if not block:
                    break
                total += len(block)
                if total > 2_000_000:
                    return None, "source_exceeds_byte_limit"
                chunks.append(block)
            return b"".join(chunks), None
        finally:
            os.close(fd)
    except OSError:
        return None, "source_open_failed"


def _is_input_projections_target(node: ast.AST) -> bool:
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self" and node.attr == "input_projections")


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def grade_bytes(source: bytes) -> dict[str, object]:
    digest = hashlib.sha256(source).hexdigest()
    try:
        module = ast.parse(source.decode("utf-8", "strict"), filename=TARGET_PATH)
    except (UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        return {"status": "REJECT", "reason": "source_syntax_or_encoding_invalid",
                "detail": type(exc).__name__, "source_sha256": digest}
    if sum(1 for _ in ast.walk(module)) > 100_000:
        return {"status": "REJECT", "reason": "source_ast_too_large", "source_sha256": digest}
    classes = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == TARGET_CLASS]
    if len(classes) != 1:
        return {"status": "FAIL", "reason": "target_class_missing_or_ambiguous", "source_sha256": digest}
    init_methods = [node for node in classes[0].body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == TARGET_METHOD]
    if len(init_methods) != 1:
        return {"status": "FAIL", "reason": "target_initializer_missing_or_ambiguous", "source_sha256": digest}

    assignments: list[ast.expr] = []
    append_calls = 0
    for node in ast.walk(init_methods[0]):
        if isinstance(node, ast.Assign) and any(_is_input_projections_target(target) for target in node.targets):
            assignments.append(node.value)
        elif (isinstance(node, ast.AnnAssign) and _is_input_projections_target(node.target)
              and node.value is not None):
            assignments.append(node.value)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "append" and _is_input_projections_target(node.func.value)):
            append_calls += 1
    if len(assignments) != 1:
        return {"status": "FAIL", "reason": "input_projection_container_assignment_missing_or_ambiguous",
                "source_sha256": digest, "append_call_count": append_calls}

    value = assignments[0]
    container = _dotted_name(value.func) if isinstance(value, ast.Call) else None
    accepted_container = container in {"nn.ModuleList", "torch.nn.ModuleList"}
    empty_initializer = isinstance(value, ast.Call) and (
        len(value.args) == 0 and len(value.keywords) == 0
        or len(value.args) == 1 and isinstance(value.args[0], (ast.List, ast.Tuple))
        and not value.args[0].elts and len(value.keywords) == 0
    )
    if not accepted_container or not empty_initializer:
        return {"status": "FAIL", "reason": "projection_container_is_not_an_empty_module_list",
                "source_sha256": digest, "container_expression": container,
                "append_call_count": append_calls}
    if append_calls < 2:
        return {"status": "FAIL", "reason": "projection_branches_do_not_register_both_paths",
                "source_sha256": digest, "append_call_count": append_calls}
    return {"status": "PASS", "reason": "projection_modules_registered_in_both_branches",
            "source_sha256": digest, "container_expression": container,
            "append_call_count": append_calls, "candidate_execution": "never_imported_or_executed"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--source", default=TARGET_PATH)
    args = parser.parse_args()
    source, error = read_source(args.workspace_root, args.source)
    receipt = ({"status": "REJECT", "reason": error} if error else grade_bytes(source))
    receipt["candidate_execution"] = "never_imported_or_executed"
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
