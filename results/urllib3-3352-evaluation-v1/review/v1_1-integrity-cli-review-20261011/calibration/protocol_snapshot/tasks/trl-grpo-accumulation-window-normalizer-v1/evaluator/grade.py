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


def _local_name_aliases(method: ast.FunctionDef, root_name: str) -> set[str]:
    """Collect conservative direct local aliases of one protected Tensor name."""
    aliases = {root_name}
    changed = True
    while changed:
        changed = False
        for node in ast.walk(method):
            targets = []
            value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                targets = [node.targets[0]]
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
                value = node.value
            elif isinstance(node, ast.NamedExpr):
                targets = [node.target]
                value = node.value
            if (isinstance(value, ast.Name) and value.id in aliases
                    and len(targets) == 1 and isinstance(targets[0], ast.Name)
                    and targets[0].id not in aliases):
                aliases.add(targets[0].id)
                changed = True
    return aliases


def _mutates_name_inplace(node: ast.AST, names: set[str]) -> bool:
    """Detect mutator-method and `out=` calls targeting a protected name or alias."""
    if not isinstance(node, ast.Call):
        return False

    def rooted_in_name(value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return value.id in names
        if isinstance(value, (ast.Attribute, ast.Subscript)):
            return rooted_in_name(value.value)
        return False

    function = node.func
    if (isinstance(function, ast.Attribute) and function.attr.endswith("_")
            and rooted_in_name(function.value)):
        return True
    return any(keyword.arg == "out" and rooted_in_name(keyword.value)
               for keyword in node.keywords)


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
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            return contains(target.value)
        return False

    if any(contains(target) for target in targets):
        return True
    if _mutates_name_inplace(node, {name}):
        return True
    return False


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


def _function_returns(method: ast.FunctionDef) -> list[ast.Return]:
    """Collect returns in this method without entering nested helper functions."""
    returns: list[ast.Return] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is method:
                for statement in node.body:
                    self.visit(statement)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

        def visit_Return(self, node: ast.Return) -> None:
            returns.append(node)
            self.generic_visit(node)

    Visitor().visit(method)
    return returns


def _always_exits(statement: ast.stmt) -> bool:
    """Whether a statement unconditionally prevents following statements from running."""
    if isinstance(statement, (ast.Return, ast.Raise)):
        return True
    if isinstance(statement, ast.If):
        # Constant conditions have only one reachable arm. For a dynamic condition,
        # both arms must terminate; an absent else always leaves a fall-through path.
        if isinstance(statement.test, ast.Constant) and isinstance(statement.test.value, bool):
            arm = statement.body if statement.test.value else statement.orelse
            return bool(arm) and any(_always_exits(item) for item in arm)
        return bool(statement.orelse) and any(_always_exits(item) for item in statement.body) \
            and any(_always_exits(item) for item in statement.orelse)
    if isinstance(statement, (ast.With, ast.AsyncWith)):
        return any(_always_exits(item) for item in statement.body)
    return False


def _is_expected_downstream_loss_adjustment(statement: ast.stmt, loss_aliases: set[str]) -> bool:
    """Allow only the pinned entropy and auxiliary-loss updates after the policy branch."""
    if not isinstance(statement, ast.If):
        return False
    writes = [node for node in ast.walk(statement)
              if _writes_name(node, "loss") or _mutates_name_inplace(node, loss_aliases)]
    if len(writes) != 1 or not isinstance(writes[0], ast.Assign):
        return False
    assignment = writes[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        return False
    if assignment.targets[0].id != "loss":
        return False

    if (isinstance(statement.test, ast.Attribute)
            and isinstance(statement.test.value, ast.Name)
            and statement.test.value.id == "self"
            and statement.test.attr == "_entropy_bonus_enabled"):
        expected = ast.parse("loss = loss - apply_coef * entropy_loss").body[0].value
        return ast.dump(assignment.value, include_attributes=False) == ast.dump(expected, include_attributes=False)

    if (isinstance(statement.test, ast.Attribute)
            and isinstance(statement.test.value, ast.Name)
            and statement.test.value.id == "self"
            and statement.test.attr == "aux_loss_enabled"):
        expected = ast.parse("loss = loss + self.router_aux_loss_coef * aux_loss / normalizer").body[0].value
        return ast.dump(assignment.value, include_attributes=False) == ast.dump(expected, include_attributes=False)
    return False


def _extract_normalizer(path: Path, branch_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    trainer = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "GRPOTrainer"), None)
    method = next((node for node in trainer.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_compute_loss"), None) if trainer else None
    if method is None:
        return None, None, None

    normalizer_aliases = _local_name_aliases(method, "normalizer")
    loss_aliases = _local_name_aliases(method, "loss")
    # The pinned production branches live in a top-level if/elif decision chain.
    # Searching all descendants let an unreachable nested copy satisfy the oracle.
    matching_branches = []
    for index, statement in enumerate(method.body):
        if not isinstance(statement, ast.If):
            continue
        branch = statement
        while True:
            if _loss_branch_matches(branch, branch_name):
                matching_branches.append((branch, index))
            if len(branch.orelse) != 1 or not isinstance(branch.orelse[0], ast.If):
                break
            branch = branch.orelse[0]
    if len(matching_branches) != 1:
        return None, None, None
    branch, branch_index = matching_branches[0]
    # A return/raise before the target branch can bypass the code whose extracted
    # statements are executed below. This guard is deliberately conservative.
    if any(_always_exits(statement) for statement in method.body[:branch_index]):
        return None, None, None
    if any(not (isinstance(node.value, ast.Name) and node.value.id == "loss")
           for node in _function_returns(method)):
        return None, None, None
    if not branch.body:
        return None, None, None
    assignment = next((statement for statement in branch.body if _target_is_normalizer(statement)), None)
    if assignment is None:
        return None, None, None
    loss_assignment = next((statement for statement in branch.body
                            if isinstance(statement, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "loss"
                                    for target in statement.targets)), None)
    if (loss_assignment is None or not isinstance(loss_assignment.value, ast.BinOp)
            or not isinstance(loss_assignment.value.op, ast.Div)
            or not isinstance(loss_assignment.value.right, ast.Name)
            or loss_assignment.value.right.id != "normalizer"):
        return None, None, None
    expected_numerator = ast.parse("(per_token_loss * mask).sum()").body[0].value
    if ast.dump(loss_assignment.value.left, include_attributes=False) != ast.dump(
            expected_numerator, include_attributes=False):
        return None, None, None
    denominator = loss_assignment.value.right
    loss_index = branch.body.index(loss_assignment)
    later_loss_writes = (
        node for statement in branch.body[loss_index + 1:]
        for node in ast.walk(statement)
        if (_writes_name(node, "loss") or _mutates_name_inplace(node, loss_aliases))
    )
    if any(True for _ in later_loss_writes):
        return None, None, None
    for statement in method.body[branch_index + 1:]:
        if any(_writes_name(node, "loss") or _mutates_name_inplace(node, loss_aliases)
               for node in ast.walk(statement)):
            if not _is_expected_downstream_loss_adjustment(statement, loss_aliases):
                return None, None, None
    training_gate = next((statement for statement in branch.body
                          if isinstance(statement, ast.If) and _is_mode_train(statement)
                          and any(_target_is_normalizer(item) for item in statement.body)), None)
    permitted_writes = {id(assignment)}
    if training_gate is not None:
        gate_writes = [node for node in ast.walk(training_gate)
                       if (_writes_name(node, "normalizer")
                           or _mutates_name_inplace(node, normalizer_aliases))]
        if len(gate_writes) != 1 or not _target_is_normalizer(gate_writes[0]):
            return None, None, None
        permitted_writes.add(id(gate_writes[0]))
    branch_writes = (node for statement in branch.body for node in ast.walk(statement)
                     if (_writes_name(node, "normalizer")
                         or _mutates_name_inplace(node, normalizer_aliases)))
    if any(id(node) not in permitted_writes for node in branch_writes):
        return None, None, None
    return assignment, training_gate, denominator


def _execute_contract(path: Path, branch_name: str, case: dict[str, Any]) -> tuple[float | None, float | None]:
    assignment, training_gate, denominator = _extract_normalizer(path, branch_name)
    if assignment is None:
        return None, None
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
    if not isinstance(normalizer, _Scalar) or normalizer.value <= 0:
        return None, None
    if denominator is None:
        return normalizer.value, None
    expression = ast.Expression(body=denominator)
    ast.fix_missing_locations(expression)
    effective_denominator = eval(compile(expression, str(path), "eval"), {}, namespace)
    if not isinstance(effective_denominator, _Scalar) or effective_denominator.value <= 0:
        return normalizer.value, None
    return normalizer.value, normalizer.value / effective_denominator.value


def _execute_normalizer(path: Path, branch_name: str, case: dict[str, Any]) -> float | None:
    return _execute_contract(path, branch_name, case)[0]


def _grade_file(path: Path, branch_name: str, tolerance: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    failures = []
    for case in CASES:
        try:
            observed, effective_loss_scale = _execute_contract(path, branch_name, case)
        except Exception as exc:
            observed = None
            effective_loss_scale = None
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
            "expected_effective_loss_scale": 1.0,
            "observed_effective_loss_scale": effective_loss_scale,
        }
        rows.append(row)
        if error is None or error > tolerance:
            failures.append({"check": "accumulation_window_normalizer", "case_id": case["case_id"],
                             "expected": expected, "observed": observed, "absolute_error": error})
        if (effective_loss_scale is None or not math.isclose(effective_loss_scale, 1.0,
                                                              rel_tol=0.0, abs_tol=tolerance)):
            failures.append({"check": "effective_loss_denominator", "case_id": case["case_id"],
                             "expected_scale": 1.0, "observed_scale": effective_loss_scale})
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
