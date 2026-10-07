#!/usr/bin/env python3
"""CPU-only arithmetic grader for the pinned GRPO normalizer branches."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any


BRANCHES = {
    "main_dapo_cispo_vespo": "trl/trainer/grpo_trainer.py",
    "experimental_dapo": "trl/experimental/gspo_token/grpo_trainer.py",
}
CASES = (
    {"case_id": "generation-window-larger", "mode": "train", "items": 12.0,
     "world_size": 1, "steps_per_generation": 4, "current_accumulation_steps": 2},
    {"case_id": "equal-windows", "mode": "train", "items": 12.0,
     "world_size": 1, "steps_per_generation": 2, "current_accumulation_steps": 2},
    {"case_id": "accumulation-window-larger", "mode": "train", "items": 12.0,
     "world_size": 1, "steps_per_generation": 2, "current_accumulation_steps": 4},
    {"case_id": "partial-final-window", "mode": "train", "items": 12.0,
     "world_size": 1, "steps_per_generation": 4, "current_accumulation_steps": 1},
    {"case_id": "clamped-empty-count", "mode": "train", "items": 0.0,
     "world_size": 2, "steps_per_generation": 4, "current_accumulation_steps": 2},
    {"case_id": "evaluation-is-unscaled", "mode": "eval", "items": 12.0,
     "world_size": 2, "steps_per_generation": 4, "current_accumulation_steps": 2},
)


class _Scalar:
    """Only the scalar operations used by the production normalizer statements."""

    def __init__(self, value: float):
        self.value = float(value)

    def clamp(self, *, min: float) -> "_Scalar":
        return _Scalar(max(self.value, min))

    def __truediv__(self, other: float) -> "_Scalar":
        return _Scalar(self.value / float(other))

    def __mul__(self, other: float) -> "_Scalar":
        return _Scalar(self.value * float(other))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_is_normalizer(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == "normalizer" for target in node.targets)
    if isinstance(node, ast.AugAssign):
        return isinstance(node.target, ast.Name) and node.target.id == "normalizer"
    return False


def _writes_name(node: ast.AST, name: str) -> bool:
    """Return whether a statement binds, mutates or deletes the named local."""
    targets = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        targets = [node.target]
    elif isinstance(node, ast.Delete):
        targets = node.targets

    def contains(target: ast.AST) -> bool:
        if isinstance(target, ast.Name):
            return target.id == name
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(contains(item) for item in target.elts)
        return False

    return any(contains(target) for target in targets)


def _is_mode_train(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "mode"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "train"
    )


def _loss_branch_matches(node: ast.If, branch_name: str) -> bool:
    test = node.test
    if not (isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and test.left.attr == "loss_type"
            and isinstance(test.left.value, ast.Name)
            and test.left.value.id == "self"):
        return False
    if branch_name == "experimental_dapo":
        return (
            len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "dapo"
        )
    if branch_name == "main_dapo_cispo_vespo":
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.In) or len(test.comparators) != 1:
            return False
        values = test.comparators[0]
        if not isinstance(values, (ast.List, ast.Tuple, ast.Set)):
            return False
        literals = {item.value for item in values.elts if isinstance(item, ast.Constant)}
        return {"cispo", "dapo", "vespo"}.issubset(literals)
    return False


def _extract_normalizer(path: Path, branch_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    trainer = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "GRPOTrainer"), None)
    method = next((node for node in trainer.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_compute_loss"), None) if trainer else None
    if method is None:
        return None, None

    for branch in ast.walk(method):
        if not isinstance(branch, ast.If) or not _loss_branch_matches(branch, branch_name):
            continue
        assignment = next((statement for statement in branch.body if _target_is_normalizer(statement)), None)
        if assignment is None:
            continue
        training_gate = next((statement for statement in branch.body
                              if isinstance(statement, ast.If) and _is_mode_train(statement)
                              and any(_target_is_normalizer(item) for item in statement.body)), None)
        permitted_writes = {id(assignment)}
        if training_gate is not None:
            permitted_writes.update(id(node) for node in ast.walk(training_gate)
                                    if _writes_name(node, "normalizer"))
        branch_writes = (node for statement in branch.body for node in ast.walk(statement)
                         if _writes_name(node, "normalizer"))
        if any(id(node) not in permitted_writes for node in branch_writes):
            return None, None
        return assignment, training_gate
    return None, None


def _execute_normalizer(path: Path, branch_name: str, case: dict[str, Any]) -> float | None:
    assignment, training_gate = _extract_normalizer(path, branch_name)
    if assignment is None:
        return None
    statements = [assignment]
    if training_gate is not None:
        statements.append(training_gate)
    module = ast.Module(body=statements, type_ignores=[])
    ast.fix_missing_locations(module)
    self_stub = type("TrainerStub", (), {})()
    self_stub.accelerator = type("AcceleratorStub", (), {"num_processes": case["world_size"]})()
    self_stub.current_gradient_accumulation_steps = case["current_accumulation_steps"]
    self_stub.args = type("ArgsStub", (), {"steps_per_generation": case["steps_per_generation"]})()
    namespace = {
        "inputs": {"num_items_in_batch": _Scalar(case["items"])},
        "self": self_stub,
        "mode": case["mode"],
    }
    exec(compile(module, str(path), "exec"), {}, namespace)
    normalizer = namespace.get("normalizer")
    return normalizer.value if isinstance(normalizer, _Scalar) else None


def _grade_file(path: Path, branch_name: str, tolerance: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    failures = []
    for case in CASES:
        try:
            observed = _execute_normalizer(path, branch_name, case)
        except Exception as exc:
            observed = None
            failures.append({"check": "normalizer_runtime", "case_id": case["case_id"],
                             "error": f"{type(exc).__name__}: {exc}"})
        base = max(case["items"], 1.0) / case["world_size"]
        expected = (base * case["current_accumulation_steps"] / case["steps_per_generation"]
                    if case["mode"] == "train" else base)
        if observed is None or not math.isfinite(observed) or observed <= 0:
            error = None
        else:
            error = abs(observed - expected)
        row = {
            "case_id": case["case_id"],
            "mode": case["mode"],
            "items": case["items"],
            "world_size": case["world_size"],
            "steps_per_generation": case["steps_per_generation"],
            "current_accumulation_steps": case["current_accumulation_steps"],
            "expected_normalizer": expected,
            "observed_normalizer": observed,
            "absolute_normalizer_error": error,
        }
        rows.append(row)
        if error is None or error > tolerance:
            failures.append({"check": "accumulation_window_normalizer", "case_id": case["case_id"],
                             "expected": expected, "observed": observed, "absolute_error": error})
    return {"conditions": rows, "max_absolute_normalizer_error": max(
        (row["absolute_normalizer_error"] for row in rows if row["absolute_normalizer_error"] is not None),
        default=None,
    )}, failures


def grade(workspace: Path, task_root: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    task_root = task_root.resolve()
    evaluator_file = Path(__file__).resolve()
    if evaluator_file.is_relative_to(workspace) or task_root == workspace or task_root.is_relative_to(workspace):
        raise ValueError("grader and task specification must be outside the candidate workspace")
    task = _json(task_root / "task.json")
    lock = _json(task_root / "protocol.lock.json")
    hashes = lock["locked_hashes"]
    if _sha256(evaluator_file) != hashes["evaluator_sha256"]:
        raise ValueError("evaluator hash does not match the locked protocol")
    if _sha256(task_root / "task.json") != hashes["task_json_sha256"]:
        raise ValueError("task descriptor hash does not match the locked protocol")
    if _sha256(task_root / "TASK.md") != hashes["task_brief_sha256"]:
        raise ValueError("task brief hash does not match the locked protocol")

    source_files = task["source"]["files"]
    missing = [name for name in source_files if not (workspace / name).is_file()]
    if missing:
        raise FileNotFoundError(f"candidate checkout is missing source files: {missing}")
    revision = subprocess.run(["git", "-C", str(workspace), "rev-parse", "HEAD"],
                              check=True, text=True, capture_output=True).stdout.strip()
    base_revision = task["source"]["base_revision"]
    subprocess.run(["git", "-C", str(workspace), "cat-file", "-e", f"{base_revision}^{{commit}}"],
                   check=True, capture_output=True)
    diff = subprocess.run(["git", "-C", str(workspace), "diff", "--binary", base_revision,
                           "--", *source_files], check=True, capture_output=True).stdout
    tolerance = float(lock["acceptance"]["absolute_normalizer_tolerance"])
    failures: list[dict[str, Any]] = []
    branches = {}
    for branch_name, relative in BRANCHES.items():
        result, branch_failures = _grade_file(workspace / relative, branch_name, tolerance)
        branches[branch_name] = result
        failures.extend({"source_file": relative, **failure} for failure in branch_failures)

    max_error = max(
        (row["absolute_normalizer_error"]
         for result in branches.values() for row in result["conditions"]
         if row["absolute_normalizer_error"] is not None),
        default=None,
    )
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "protocol_id": lock["protocol_id"],
        "workspace_revision": revision,
        "base_revision": base_revision,
        "candidate_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_file_sha256": {name: _sha256(workspace / name) for name in source_files},
        "evaluator_sha256": _sha256(evaluator_file),
        "task_descriptor_sha256": _sha256(task_root / "task.json"),
        "passed": not failures,
        "checks": {"branches": branches},
        "metrics": {"condition_count": sum(len(item["conditions"]) for item in branches.values()),
                    "max_absolute_normalizer_error": max_error},
        "failures": failures,
        "resource_scope": {"model_weights_loaded": False, "third_party_python_packages_used": False,
                           "accelerator_hours": 0, "paid_api_calls": 0, "external_compute_usd": 0},
        "claim_limit": task["claim_limit"],
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    try:
        result = grade(args.workspace, args.task_root)
    except Exception as exc:
        print(json.dumps({"grader_error": type(exc).__name__, "message": str(exc)}, sort_keys=True))
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] or args.allow_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
