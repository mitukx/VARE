#!/usr/bin/env python3
"""Narrow function-contract grader for three Development repairs.

Candidate modules are never imported. Only declared function bodies are
compiled after structural and closed-call admission. This is deliberately a
small Python repair language, not a general hostile-code sandbox. The trusted
caller must also enforce process timeout and memory limits.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ASSETS = ROOT / "benchmarks/vare_development_repair_v1"
TASKS = {
    "navigation-overflow": ("navigation-baseline.py.txt", "SourceNavigatorV1", "_bounded"),
    "navigation-retry": ("navigation-baseline.py.txt", "SourceNavigatorV1", "search"),
    "manifest-nonempty": ("manifest-baseline.py.txt", None, "manifest_entries"),
}
SAFE_GLOBALS = {
    "json", "re", "Any", "MAX_QUERY_CHARS", "MAX_QUERY_TERMS",
    "_terms", "_error", "_line_match", "_bounded_context",
    "len", "isinstance", "str", "dict", "list", "set", "bool", "int",
    "sorted", "enumerate", "tuple", "min", "max", "ValueError", "TypeError",
}
SAFE_ATTRIBUTES = {
    "get", "pop", "append", "items", "sort", "casefold", "split", "strip",
    "search", "encode", "dumps", "compile", "escape", "join", "add",
    "_snapshot", "_last_build_error",
    "_index_sha256", "_generation", "_is_stale", "_rebuild", "_bounded",
    "allowed", "max_results", "max_output_bytes", "context_lines",
}
SAFE_METHOD_CALLS = SAFE_ATTRIBUTES - {
    "_snapshot", "_last_build_error", "_index_sha256", "_generation",
    "allowed", "max_results", "max_output_bytes", "context_lines",
}
FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef,
    ast.With, ast.AsyncWith, ast.Await, ast.Yield, ast.YieldFrom, ast.Global,
    ast.Nonlocal, ast.Delete,
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select(tree: ast.Module, owner: str | None, method: str) -> ast.FunctionDef:
    body = tree.body
    if owner:
        classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == owner]
        if len(classes) != 1:
            raise ValueError("missing or duplicated target class")
        body = classes[0].body
    found = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == method]
    if len(found) != 1:
        raise ValueError("missing or duplicated target function")
    return found[0]


def admit(task: str, candidate: str) -> tuple[str, ast.FunctionDef]:
    asset, owner, method = TASKS[task]
    baseline = (ASSETS / asset).read_text()
    if len(candidate.encode()) > 64 * 1024:
        raise ValueError("source too large")
    original_tree, candidate_tree = ast.parse(baseline), ast.parse(candidate)
    original = select(original_tree, owner, method)
    replacement = select(candidate_tree, owner, method)
    original_body, replacement_body = original.body, replacement.body
    original.body = replacement.body = []
    if ast.dump(original_tree) != ast.dump(candidate_tree):
        raise ValueError("edits outside the declared function body")
    original.body, replacement.body = original_body, replacement_body
    local_names = {
        node.id for node in ast.walk(replacement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    } | {node.arg for node in ast.walk(replacement) if isinstance(node, ast.arg)}
    for statement in replacement.body:
        for node in ast.walk(statement):
            if isinstance(node, FORBIDDEN_NODES + (ast.FunctionDef,)):
                raise ValueError(f"unsupported syntax: {type(node).__name__}")
            if isinstance(node, ast.Name):
                if "__" in node.id or node.id not in local_names | SAFE_GLOBALS:
                    raise ValueError(f"unsupported name: {node.id}")
            if isinstance(node, ast.Attribute):
                if node.attr not in SAFE_ATTRIBUTES or not isinstance(node.ctx, ast.Load):
                    raise ValueError(f"unsupported attribute: {node.attr}")
            if isinstance(node, ast.Constant):
                if isinstance(node.value, str) and "__" in node.value:
                    raise ValueError("dunder string access is unsupported")
                if isinstance(node.value, (int, float)) and abs(node.value) > 1_000_000:
                    raise ValueError("numeric constant exceeds the repair-language bound")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in SAFE_GLOBALS - {"json", "re", "Any"}:
                        raise ValueError("indirect or unknown function call")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr not in SAFE_METHOD_CALLS:
                        raise ValueError("unknown method call")
                else:
                    raise ValueError("indirect call")
    return baseline, replacement


def load_candidate(task: str, candidate: str):
    baseline, replacement = admit(task, candidate)
    asset, owner, method = TASKS[task]
    module = types.ModuleType("vare._development_contract")
    module.__package__ = "vare"
    # Only pinned repository code executes at module scope.
    exec(compile(baseline, asset, "exec"), module.__dict__)
    transplant = ast.Module(body=[copy.deepcopy(replacement)], type_ignores=[])
    ast.fix_missing_locations(transplant)
    namespace = dict(module.__dict__)
    exec(compile(transplant, "<admitted-function>", "exec"), namespace)
    function = namespace[method]
    if owner:
        setattr(getattr(module, owner), method, function)
        return module.SourceNavigatorV1, module
    return function, module


def observe(task: str, candidate: str) -> dict[str, bool]:
    implementation, module = load_candidate(task, candidate)
    checks: dict[str, bool] = {}
    if task == "manifest-nonempty":
        invalid = [{}, {"files": {}}, {"files": []}, {"files": None},
                   {"files": 0}, {"files": "bad"}]
        for i, manifest in enumerate(invalid):
            entries, errors = implementation(manifest)
            checks[f"invalid-{i}"] = entries == [] and isinstance(errors, list) and bool(errors)
        metadata = {"bytes": 1, "sha256": "0" * 64}
        entries, errors = implementation({"files": {"a.json": metadata, "b.json": metadata}})
        checks["mapping-preserved"] = entries == [("a.json", metadata), ("b.json", metadata)] and errors == []
        rows = [{"path": "a.json", **metadata}, {"path": "b.json", **metadata}]
        entries, errors = implementation({"files": rows})
        checks["array-preserved"] = entries == [("a.json", rows[0]), ("b.json", rows[1])] and errors == []
        entries, errors = implementation({"files": [rows[0], rows[0]]})
        checks["duplicates-rejected"] = len(entries) == 1 and bool(errors)
        entries, errors = implementation({"files": [7, {}, {"path": ""}]})
        checks["malformed-array-rejected"] = entries == [] and len(errors) == 3
        return checks
    with tempfile.TemporaryDirectory(prefix="vare-contract-") as directory:
        root = Path(directory)
        if task == "navigation-overflow":
            paths = set()
            for i in range(128):
                relative = f"src/{'module_' * 16}{i:03d}.py"
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"rare_target_{i} = {i}\n")
                paths.add(relative)
            for budget in (2048, 4096, 8192):
                navigator = implementation(root, paths, max_output_bytes=budget)
                result = navigator.search("rare_target_73")
                encoded = json.dumps(result, ensure_ascii=False).encode()
                checks[f"bounded-{budget}"] = len(encoded) <= budget and result.get("output_bytes") == len(encoded)
                checks[f"retains-hit-{budget}"] = (
                    result.get("status") == "ok" and len(result.get("results", [])) == 1
                    and result["results"][0]["path"].endswith("073.py")
                    and result["results"][0]["line"] == 1
                    and result["results"][0]["source_sha256"] == sha((root / result["results"][0]["path"]).read_bytes())
                )
            navigator = implementation(root, paths)
            checks["no-fabricated-hits"] = navigator.search("absent_word").get("results") == []
            checks["invalid-query-rejected"] = navigator.search(17).get("status") == "invalid_query"
        else:
            source = root / "module.py"
            source.write_text("repair_target = 1\n")
            navigator = implementation(root, {"module.py"}, max_total_source_lines=3)
            checks["initial-success"] = navigator.search("repair_target").get("status") == "ok"
            source.write_text("invalid\n" * 4)
            failed = navigator.search("repair_target")
            checks["oversized-fails"] = failed.get("status") == "source_index_limit_exceeded"
            source.write_text("repair_target = 2\n")
            recovered = navigator.search("repair_target")
            checks["recovers-after-repair"] = recovered.get("status") == "ok" and recovered.get("index_rebuilt") is True
            checks["fresh-content"] = bool(recovered.get("results")) and recovered["results"][0]["context"] == "repair_target = 2"
            checks["stable-generation"] = navigator.search("repair_target").get("index_rebuilt") is False
            source.unlink()
            navigator = implementation(root, {"module.py"})
            checks["missing-source-fails"] = navigator.search("repair_target").get("status") == "source_read_failed"
            source.write_text("repair_target = 3\n")
            checks["initial-build-recovers"] = navigator.search("repair_target").get("status") == "ok"
            checks["empty-allowlist-fails"] = implementation(root, set()).search("repair_target").get("status") == "no_candidate_visible_source_files"
    return checks


def grade(task: str, source: Path) -> dict[str, Any]:
    data = source.read_bytes()
    try:
        checks = observe(task, data.decode())
        success = bool(checks) and all(type(value) is bool and value for value in checks.values())
        return {"task": task, "candidate_sha256": sha(data), "admitted": True,
                "checks": checks, "success": success}
    except Exception as exc:
        return {"task": task, "candidate_sha256": sha(data), "admitted": False,
                "success": False, "error": f"{type(exc).__name__}: {exc}"[:512]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(grade(args.task, args.source), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
