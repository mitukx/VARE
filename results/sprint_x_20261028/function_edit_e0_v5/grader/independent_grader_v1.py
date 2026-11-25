#!/usr/bin/env python3
"""Side-effect-free AST interpreter for a minimum batch-size contract."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


class UnsupportedCandidate(Exception):
    pass


def eval_expr(node: ast.expr, requested: int):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and node.id == "requested":
        return requested
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in {"min", "max"} or node.keywords or not node.args:
            raise UnsupportedCandidate("unsupported_call")
        args = [eval_expr(arg, requested) for arg in node.args]
        return min(args) if node.func.id == "min" else max(args)
    raise UnsupportedCandidate("unsupported_expression")


def eval_condition(node: ast.expr, requested: int) -> bool:
    if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
        raise UnsupportedCandidate("unsupported_condition")
    left = eval_expr(node.left, requested)
    right = eval_expr(node.comparators[0], requested)
    op = node.ops[0]
    if isinstance(op, ast.Lt): return left < right
    if isinstance(op, ast.LtE): return left <= right
    if isinstance(op, ast.Gt): return left > right
    if isinstance(op, ast.GtE): return left >= right
    if isinstance(op, ast.Eq): return left == right
    if isinstance(op, ast.NotEq): return left != right
    raise UnsupportedCandidate("unsupported_operator")


_NO_RETURN = object()


def eval_statements(statements: list[ast.stmt], requested: int):
    for statement in statements:
        if isinstance(statement, ast.Return):
            if statement.value is None:
                raise UnsupportedCandidate("bare_return")
            return eval_expr(statement.value, requested)
        if isinstance(statement, ast.If):
            selected = statement.body if eval_condition(statement.test, requested) else statement.orelse
            result = eval_statements(selected, requested)
            if result is not _NO_RETURN:
                return result
            continue
        if isinstance(statement, ast.Pass):
            continue
        raise UnsupportedCandidate("unsupported_statement")
    return _NO_RETURN


def grade(candidate: Path) -> dict:
    if candidate.is_symlink() or not candidate.is_file():
        return {"status": "FAIL", "reason": "not_regular_file"}
    raw = candidate.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        module = ast.parse(raw.decode("utf-8"), filename=str(candidate))
    except Exception as exc:
        return {"status": "FAIL", "reason": "syntax_or_utf8", "error": str(exc), "source_sha256": digest}
    funcs = [node for node in module.body if isinstance(node, ast.FunctionDef)
             and node.name == "normalize_batch_size"]
    if len(funcs) != 1:
        return {"status": "FAIL", "reason": "function_count", "source_sha256": digest}
    function = funcs[0]
    body = [node for node in function.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    )]
    if (function.decorator_list or len(function.args.args) != 1
            or function.args.args[0].arg != "requested" or function.args.posonlyargs
            or function.args.kwonlyargs or function.args.vararg or function.args.kwarg):
        return {"status": "FAIL", "reason": "function_signature", "source_sha256": digest}
    inputs = [-100, -7, -1, 0, 1, 2, 100]
    observed = []
    try:
        for item in inputs:
            result = eval_statements(body, item)
            if result is _NO_RETURN or type(result) is not int:
                raise UnsupportedCandidate("missing_integer_return")
            observed.append(result)
    except Exception as exc:
        return {"status": "FAIL", "reason": "unsupported_candidate", "error": str(exc),
                "source_sha256": digest, "candidate_execution": "never_imported_or_executed"}
    expected = [max(1, item) for item in inputs]
    passed = observed == expected
    return {"status": "PASS" if passed else "FAIL", "reason": None if passed else "behavior_mismatch",
            "source_sha256": digest, "inputs": inputs, "observed": observed, "expected": expected,
            "candidate_execution": "never_imported_or_executed",
            "grader": "sprint_x_synthetic_min_batch_ast_v1"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(grade(args.candidate), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
