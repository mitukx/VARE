#!/usr/bin/env python3
"""Independent, non-executing grader for the Sprint VII E0 clamp fixture.

Candidate modules are read as bytes and parsed with ``ast``. They are never
imported, compiled, or executed. The tiny expression interpreter below accepts
only a fixed, audited Python AST subset and evaluates it over frozen cases.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any


def _read_candidate(root_raw: str, relative_raw: str) -> tuple[bytes | None, str | None]:
    root = Path(root_raw).absolute()
    relative = PurePosixPath(relative_raw)
    if root.is_symlink() or not root.is_dir():
        return None, "workspace_root_not_real_directory"
    ancestor = Path(root.anchor)
    for part in root.parts[1:]:
        ancestor = ancestor / part
        try:
            mode = ancestor.lstat().st_mode
        except OSError:
            return None, "workspace_ancestor_unavailable"
        if stat.S_ISLNK(mode):
            return None, "workspace_ancestor_symlink_rejected"
    if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return None, "candidate_path_invalid"
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            return None, "candidate_missing"
        if stat.S_ISLNK(mode):
            return None, "candidate_symlink_rejected"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(candidate, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                return None, "candidate_not_regular_file"
            chunks = []
            while True:
                block = os.read(fd, 65536)
                if not block:
                    break
                chunks.append(block)
                if sum(map(len, chunks)) > 256 * 1024:
                    return None, "candidate_exceeds_byte_limit"
            return b"".join(chunks), None
        finally:
            os.close(fd)
    except OSError:
        return None, "candidate_open_failed"


def _bounded_number(value: Any) -> int | float:
    if type(value) is int and value.bit_length() <= 64:
        return value
    if type(value) is float and math.isfinite(value) and abs(value) <= 1e100:
        return value
    raise ValueError("candidate_numeric_bound_exceeded")


def _eval_expr(node: ast.expr, value: int | float) -> Any:
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        return _bounded_number(node.value)
    if isinstance(node, ast.Name) and node.id == "value":
        return _bounded_number(value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_expr(node.operand, value)
        return _bounded_number(+operand if isinstance(node.op, ast.UAdd) else -operand)
    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left, value)
        right = _eval_expr(node.right, value)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
        }
        operation = operations.get(type(node.op))
        if operation is not None:
            return _bounded_number(operation())
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [_eval_expr(item, value) for item in node.args]
        if node.func.id == "max" and args:
            return _bounded_number(max(args))
        if node.func.id == "min" and args:
            return _bounded_number(min(args))
        if node.func.id == "abs" and len(args) == 1:
            return _bounded_number(abs(args[0]))
    if isinstance(node, ast.IfExp):
        test = node.test
        if (isinstance(test, ast.Compare) and len(test.ops) == len(test.comparators) == 1
                and isinstance(test.left, ast.Name) and test.left.id == "value"):
            left = value
            right = _eval_expr(test.comparators[0], value)
            comparisons = {
                ast.Lt: lambda: left < right,
                ast.LtE: lambda: left <= right,
                ast.Gt: lambda: left > right,
                ast.GtE: lambda: left >= right,
                ast.Eq: lambda: left == right,
                ast.NotEq: lambda: left != right,
            }
            compare = comparisons.get(type(test.ops[0]))
            if compare is not None:
                branch = node.body if compare() else node.orelse
                return _bounded_number(_eval_expr(branch, value))
    raise ValueError("unsupported_candidate_expression")


def _grade_source(source: bytes, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        return {"status": "REJECT", "reason": "source_hash_mismatch", "source_sha256": actual_sha256}
    try:
        text = source.decode("utf-8", errors="strict")
        module = ast.parse(text, filename="candidate.py")
    except (UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        return {"status": "REJECT", "reason": "candidate_syntax_or_encoding_invalid",
                "source_sha256": actual_sha256, "detail": type(exc).__name__}
    if sum(1 for _ in ast.walk(module)) > 256:
        return {"status": "REJECT", "reason": "candidate_ast_node_limit_exceeded",
                "source_sha256": actual_sha256}
    if (len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef)
            or module.body[0].name != "clamp"):
        return {"status": "REJECT", "reason": "candidate_module_shape_rejected",
                "source_sha256": actual_sha256}
    function = module.body[0]
    args = function.args
    if (function.decorator_list or function.returns is not None or function.type_comment is not None
            or args.posonlyargs or len(args.args) != 1 or args.args[0].arg != "value"
            or args.vararg or args.kwonlyargs or args.kw_defaults or args.kwarg or args.defaults
            or len(function.body) != 1 or not isinstance(function.body[0], ast.Return)
            or function.body[0].value is None):
        return {"status": "REJECT", "reason": "candidate_function_shape_rejected",
                "source_sha256": actual_sha256}

    cases = [(-7, 0), (-1, 0), (0, 0), (1, 1), (7, 7)]
    observed = []
    try:
        for argument, expected in cases:
            actual = _eval_expr(function.body[0].value, argument)
            observed.append({"input": argument, "expected": expected, "actual": actual,
                             "passed": type(actual) in {int, float} and actual == expected})
    except (ArithmeticError, TypeError, ValueError, RecursionError) as exc:
        return {"status": "REJECT", "reason": "candidate_expression_not_in_safe_subset",
                "source_sha256": actual_sha256, "detail": type(exc).__name__, "cases": observed}
    passed = all(case["passed"] for case in observed)
    return {"status": "PASS" if passed else "FAIL", "reason": "all_frozen_cases_passed" if passed else "frozen_case_mismatch",
            "source_sha256": actual_sha256, "cases": observed,
            "candidate_execution": "never_imported_or_executed"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    source, error = _read_candidate(args.workspace_root, args.source)
    if error:
        receipt = {"status": "REJECT", "reason": error}
    else:
        assert source is not None
        receipt = _grade_source(source, args.expected_sha256)
    receipt["candidate_execution"] = "never_imported_or_executed"
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
