#!/usr/bin/env python3
"""Compare retained v7 and current v8 TRL graders on real-source mutations."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/trl_grpo_accumulation_scale"
V7_ROOT = ROOT / "results/trl-grpo-accumulation-window-normalizer-v1/cpu-calibration-v7/protocol_snapshot"


def run(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def load_grader(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def mutate_source(path: Path, kind: str) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss")
    lines = source.splitlines(keepends=True)
    if kind == "early_return":
        first = method.body[0]
        indent = " " * first.col_offset
        lines.insert(first.lineno - 1, f"{indent}return None  # mutation: bypass the loss path\n")
    elif kind == "zero_numerator":
        expected = ast.dump(ast.parse("(per_token_loss * mask).sum()").body[0].value, include_attributes=False)
        targets = []
        for node in ast.walk(method):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "loss" for t in node.targets):
                value = node.value
                if (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div)
                        and ast.dump(value.left, include_attributes=False) == expected
                        and isinstance(value.right, ast.Name) and value.right.id == "normalizer"):
                    targets.append(value)
        if len(targets) != 1:
            raise RuntimeError(f"expected one pinned loss numerator in {path}, found {len(targets)}")
        value = targets[0]
        start = (value.lineno - 1, value.col_offset)
        end = (value.end_lineno - 1, value.end_col_offset)
        if start[0] != end[0]:
            raise RuntimeError(f"multiline loss expression requires explicit mutation support: {path}")
        line = lines[start[0]]
        lines[start[0]] = line[:start[1]] + "((per_token_loss * mask).sum() * 0.0) / normalizer" + line[end[1]:]
    else:
        raise ValueError(kind)
    path.write_text("".join(lines), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def checkout(task: dict, path: Path, revision: str) -> None:
    path.mkdir(parents=True)
    run(["git", "init", "--quiet"], path)
    run(["git", "remote", "add", "origin", task["source"]["repository"]], path)
    base = task["source"]["base_revision"]
    run(["git", "fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", base], path)
    if revision != base:
        run(["git", "fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", revision], path)
    run(["git", "sparse-checkout", "init", "--no-cone"], path)
    (path / ".git/info/sparse-checkout").write_text("".join("/" + item + "\n" for item in task["source"]["files"]))
    run(["git", "checkout", "--quiet", "--detach", revision], path)
    if run(["git", "rev-parse", "HEAD"], path) != revision:
        raise RuntimeError("source checkout revision mismatch")


def grade(module, workspace: Path, task_root: Path) -> dict:
    return module.grade(workspace, task_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new output directory; defaults under results/")
    args = parser.parse_args()
    output = args.output or ROOT / "results/trl-grpo-accumulation-window-normalizer-v1/grader-mutation-v8"
    output = output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not V7_ROOT.is_dir():
        parser.error(f"retained v7 protocol snapshot is missing: {V7_ROOT}")

    task = json.loads((TASK_ROOT / "task.json").read_text(encoding="utf-8"))
    fixed = task["source"]["calibration_revision"]
    old = load_grader(V7_ROOT / "evaluator/grade.py", "vare_trl_v7_grader")
    current = load_grader(TASK_ROOT / "evaluator/grade.py", "vare_trl_v8_grader")
    output.mkdir(parents=True)
    (output / "mutations").mkdir()
    results = {}
    with tempfile.TemporaryDirectory(prefix="vare-trl-grader-mutation-") as temporary:
        source = Path(temporary) / "candidate"
        checkout(task, source, fixed)
        for label, mutation in (("control", None), ("early_return", "early_return"),
                                ("zero_numerator", "zero_numerator")):
            if mutation:
                for relative in task["source"]["files"]:
                    mutate_source(source / relative, mutation)
            old_result = grade(old, source, V7_ROOT)
            new_result = grade(current, source, TASK_ROOT)
            results[label] = {
                "v7": {"passed": old_result["passed"], "failures": old_result["failures"],
                       "candidate_diff_sha256": old_result["candidate_diff_sha256"]},
                "v8": {"passed": new_result["passed"], "failures": new_result["failures"],
                       "candidate_diff_sha256": new_result["candidate_diff_sha256"]},
            }
            write_json(output / f"{label}.v7.grade.json", old_result)
            write_json(output / f"{label}.v8.grade.json", new_result)
            if mutation:
                patch = run(["git", "diff", "--binary", task["source"]["base_revision"], "--", *task["source"]["files"]], source)
                (output / "mutations" / f"{label}.patch").write_text(patch, encoding="utf-8")
                run(["git", "checkout", "--quiet", "--", *task["source"]["files"]], source)
        expected = {
            "control": (True, True),
            "early_return": (True, False),
            "zero_numerator": (True, False),
        }
        observed_ok = all((results[name]["v7"]["passed"], results[name]["v8"]["passed"]) == pair
                          for name, pair in expected.items())

    summary = {
        "schema_version": 1,
        "study_id": "trl-grader-false-accept-mutations-v8",
        "source_repository": task["source"]["repository"],
        "source_revision": fixed,
        "base_revision": task["source"]["base_revision"],
        "protocols": {"comparison": "retained-v7", "candidate": "current-v8"},
        "expected_outcomes": {name: {"v7_passed": pair[0], "v8_passed": pair[1]}
                              for name, pair in expected.items()},
        "observed_outcomes": results,
        "acceptance_passed": observed_ok,
        "resources": {"model_weights_downloaded": False, "third_party_python_packages": 0,
                      "accelerator_hours": 0, "paid_api_calls": 0, "external_compute_usd": 0},
        "claim_limit": "This tests two source mutations against one pinned historical change; it does not establish grader soundness for arbitrary code or validate full trainer execution.",
        "runtime": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), "platform": platform.platform()},
    }
    write_json(output / "summary.json", summary)
    manifest_files = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
                      if path.is_file() and path.name != "manifest.json"}
    write_json(output / "manifest.json", {"schema_version": 1, "files": manifest_files,
                                          "v7_evaluator_sha256": sha(V7_ROOT / "evaluator/grade.py"),
                                          "v8_evaluator_sha256": sha(TASK_ROOT / "evaluator/grade.py"),
                                          "task_json_sha256": sha(TASK_ROOT / "task.json"),
                                          "protocol_lock_sha256": sha(TASK_ROOT / "protocol.lock.json")})
    print(json.dumps({"acceptance_passed": observed_ok, "output": str(output),
                      "outcomes": {name: {"v7": row["v7"]["passed"], "v8": row["v8"]["passed"]}
                                   for name, row in results.items()}}, indent=2))
    return 0 if observed_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
