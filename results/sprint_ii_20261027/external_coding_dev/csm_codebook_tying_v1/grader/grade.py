#!/usr/bin/env python3
"""Independent non-executing AST grader for the CSM configuration contract."""
import argparse
import ast
import json
from pathlib import Path


def grade(path: Path) -> dict:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"success": False, "reason": f"parse_error:{type(exc).__name__}"}
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CsmConfig"]
    if len(classes) != 1:
        return {"success": False, "reason": "CsmConfig_missing_or_duplicated"}
    methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__"]
    if len(methods) != 1:
        return {"success": False, "reason": "CsmConfig_post_init_missing_or_duplicated"}
    method = methods[0]
    pop_ok = False
    assignment_ok = False
    forced_false = False
    rejected_override = False
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and node.func.value.id == "kwargs"
                    and node.func.attr == "pop" and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "tie_word_embeddings"
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value is None):
                pop_ok = True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "self" and target.attr == "tie_word_embeddings"):
                    value = node.value
                    if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                            and value.value.id == "self" and value.attr == "tie_codebooks_embeddings"):
                        assignment_ok = True
                    if isinstance(value, ast.Constant) and value.value is False:
                        forced_false = True
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "ValueError":
                rejected_override = True
    success = pop_ok and assignment_ok and not forced_false and not rejected_override
    return {
        "success": success,
        "checks": {
            "serialized_tie_word_embeddings_removed": pop_ok,
            "derived_from_tie_codebooks_embeddings": assignment_ok,
            "not_forced_false": not forced_false,
            "does_not_reject_serialized_override": not rejected_override,
        },
        "reason": "contract_pass" if success else "configuration_contract_failed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(grade(args.source), sort_keys=True))
