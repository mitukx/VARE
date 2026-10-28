#!/usr/bin/env python3
"""Independent static contract grader for minimum FPS sampling."""
import argparse
import ast
import json
from pathlib import Path


def grade(path: Path) -> dict:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"success": False, "reason": f"parse_error:{type(exc).__name__}"}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "default_sample_indices_fn"]
    if len(funcs) != 1:
        return {"success": False, "reason": "entry_point_missing_or_duplicated"}
    expected = False
    probe_value = None
    for node in ast.walk(funcs[0]):
        if not isinstance(node, ast.Assign) or not any(isinstance(t, ast.Name) and t.id == "num_frames" for t in node.targets):
            continue
        value = node.value
        if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "max" and len(value.args) == 2):
            continue
        lower, count = value.args
        minimum_is_one = isinstance(lower, ast.Constant) and lower.value == 1
        if not (isinstance(count, ast.Call) and isinstance(count.func, ast.Name) and count.func.id == "int" and len(count.args) == 1):
            continue
        expr = count.args[0]
        names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
        if {"total_num_frames", "video_fps", "fps"}.issubset(names):
            def evaluate(node):
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.Name):
                    return {"total_num_frames": 10, "video_fps": 30, "fps": 1}[node.id]
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                    return evaluate(node.left) / evaluate(node.right)
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                    return evaluate(node.left) * evaluate(node.right)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "int":
                    return int(evaluate(node.args[0]))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "max":
                    return max(evaluate(node.args[0]), evaluate(node.args[1]))
                raise ValueError("expression outside the independent grader's safe arithmetic subset")
            try:
                probe_value = evaluate(value)
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
                probe_value = None
            expected = minimum_is_one and probe_value == 1
            break
    return {
        "success": expected,
        "safe_expression_probe": {"total_num_frames": 10, "video_fps": 30, "fps": 1,
                                  "expected_minimum": 1, "observed": probe_value},
        "reason": "contract_pass" if expected else "minimum_fps_count_not_enforced",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(grade(args.source), sort_keys=True))
