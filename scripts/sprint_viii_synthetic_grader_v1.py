#!/usr/bin/env python3
"""Independent, non-executing semantic grader for the synthetic shift task."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path, PurePosixPath
import stat


def read_candidate(root_raw: str, relative_raw: str) -> tuple[bytes | None, str | None]:
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
                if total > 65536:
                    return None, "source_exceeds_byte_limit"
                chunks.append(block)
            return b"".join(chunks), None
        finally:
            os.close(fd)
    except OSError:
        return None, "source_open_failed"


def grade_bytes(source: bytes) -> dict[str, str]:
    try:
        module = ast.parse(source.decode("utf-8", "strict"), filename="subject.py")
    except (UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        return {"status": "REJECT", "reason": "source_syntax_or_encoding_invalid",
                "detail": type(exc).__name__}
    if sum(1 for _ in ast.walk(module)) > 128:
        return {"status": "REJECT", "reason": "source_ast_too_large"}
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return {"status": "FAIL", "reason": "module_shape_rejected"}
    function = module.body[0]
    if (function.name != "shift" or function.decorator_list or function.returns is not None
            or len(function.args.args) != 1 or function.args.args[0].arg != "value"
            or function.args.posonlyargs or function.args.vararg or function.args.kwonlyargs
            or function.args.kw_defaults or function.args.kwarg or function.args.defaults
            or len(function.body) != 1 or not isinstance(function.body[0], ast.Return)
            or function.body[0].value is None):
        return {"status": "FAIL", "reason": "function_shape_rejected"}
    expression = function.body[0].value
    delta = None
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        if (isinstance(expression.left, ast.Name) and expression.left.id == "value"
                and isinstance(expression.right, ast.Constant)
                and type(expression.right.value) is int):
            delta = expression.right.value
        elif (isinstance(expression.right, ast.Name) and expression.right.id == "value"
                and isinstance(expression.left, ast.Constant)
                and type(expression.left.value) is int):
            delta = expression.left.value
    if delta == 2:
        return {"status": "PASS", "reason": "exact_frozen_shift_contract"}
    if delta is not None:
        return {"status": "FAIL", "reason": "frozen_shift_contract_mismatch"}
    return {"status": "FAIL", "reason": "expression_outside_frozen_oracle_subset"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--source", default="subject.py")
    args = parser.parse_args()
    source, error = read_candidate(args.workspace_root, args.source)
    receipt = ({"status": "REJECT", "reason": error} if error else grade_bytes(source))
    receipt["candidate_execution"] = "never_imported_or_executed"
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
