#!/usr/bin/env python3
"""Non-executing independent grader for the Sprint VII synthetic shift fixture."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath


def _read_source(root_raw: str, relative_raw: str) -> tuple[bytes | None, str | None]:
    root = Path(root_raw).absolute()
    relative = PurePosixPath(relative_raw)
    if root.is_symlink() or not root.is_dir():
        return None, "workspace_root_not_real_directory"
    if relative.is_absolute() or not relative.parts or any(p in {"", ".", ".."} for p in relative.parts):
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
            chunks = []
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


def grade_bytes(source: bytes, expected_sha256: str) -> dict:
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        return {"status": "REJECT", "reason": "source_hash_mismatch", "source_sha256": actual_sha256}
    try:
        module = ast.parse(source.decode("utf-8", "strict"), filename="subject.py")
    except (UnicodeError, SyntaxError, ValueError, RecursionError) as exc:
        return {"status": "REJECT", "reason": "source_syntax_or_encoding_invalid",
                "detail": type(exc).__name__, "source_sha256": actual_sha256}
    if sum(1 for _ in ast.walk(module)) > 128:
        return {"status": "REJECT", "reason": "source_ast_too_large", "source_sha256": actual_sha256}
    if (len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef)
            or module.body[0].name != "shift"):
        return {"status": "REJECT", "reason": "module_shape_rejected", "source_sha256": actual_sha256}
    function = module.body[0]
    if (function.decorator_list or function.returns is not None
            or len(function.args.args) != 1 or function.args.args[0].arg != "value"
            or function.args.posonlyargs or function.args.vararg or function.args.kwonlyargs
            or function.args.kw_defaults or function.args.kwarg or function.args.defaults
            or len(function.body) != 1 or not isinstance(function.body[0], ast.Return)
            or function.body[0].value is None):
        return {"status": "REJECT", "reason": "function_shape_rejected", "source_sha256": actual_sha256}
    expression = function.body[0].value
    delta = None
    if isinstance(expression, ast.Name) and expression.id == "value":
        delta = 0
    elif isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        if (isinstance(expression.left, ast.Name) and expression.left.id == "value"
                and isinstance(expression.right, ast.Constant) and type(expression.right.value) is int):
            delta = expression.right.value
        elif (isinstance(expression.right, ast.Name) and expression.right.id == "value"
                and isinstance(expression.left, ast.Constant) and type(expression.left.value) is int):
            delta = expression.left.value
    if delta is None:
        return {"status": "REJECT", "reason": "expression_outside_frozen_oracle_subset",
                "source_sha256": actual_sha256}
    correct = delta == 2
    return {
        "status": "PASS" if correct else "FAIL",
        "reason": "exact_frozen_shift_contract" if correct else "frozen_shift_contract_mismatch",
        "source_sha256": actual_sha256,
        "candidate_execution": "never_imported_or_executed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--source", default="subject.py")
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    source, error = _read_source(args.workspace_root, args.source)
    receipt = ({"status": "REJECT", "reason": error} if error else
               grade_bytes(source, args.expected_sha256))
    receipt["candidate_execution"] = "never_imported_or_executed"
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
