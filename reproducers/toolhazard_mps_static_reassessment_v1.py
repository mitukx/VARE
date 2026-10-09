#!/usr/bin/env python3
"""Train-only static reassessment of two small ToolHazard task candidates.

This script fetches only the three pinned TRAIN JSON files, verifies their
existing v2 digests, and parses checker strings with ast.parse. It never
imports, evaluates, or executes benchmark-supplied Python, and it never fetches
test-set artifacts.
"""

from __future__ import annotations

import ast
import hashlib
import json
from urllib.request import urlopen


COMMIT = "544b73b12a25431cb0be3eb43df41b4aacce5335"
BASE = f"https://raw.githubusercontent.com/MurrayTom/ToolHazard/{COMMIT}/toolhazard_bench/"
FILES = {
    "environments": (
        "train_set/40_rl_train_env_with_attack_points.json",
        "16671a1f593c8f7bd1ca709bfc9b27d1b194d99033805a001444827b0b1be039",
    ),
    "tasks": (
        "train_set/rl_tasks_IPI_all_new.json",
        "315d40f1a97d33afd5dc8c7d9c0cee8892dd045771e4d78cfe7f83d6a805fdd9",
    ),
    "trajectories": (
        "train_set/rl_traj.json",
        "0994cbd11dcecbbfffed2e3bf5d420833e2f5df16d62d4c3db28dc3d481ce4bc",
    ),
}
CANDIDATES = {
    "env_123_sft-task_14_IPI_tool_selection": "product review workflow",
    "env_107_sft-task_10_IPI_tool_selection": "business review workflow",
}


def fetch() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    parsed: dict[str, object] = {}
    inputs: dict[str, dict[str, object]] = {}
    for key, (path, expected) in FILES.items():
        with urlopen(BASE + path, timeout=60) as response:
            raw = response.read()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {path}: {actual}")
        parsed[key] = json.loads(raw)
        inputs[key] = {"path": path, "sha256": actual, "bytes": len(raw)}
    return parsed, inputs


def checker_facts(source: str) -> dict[str, object]:
    """Return syntax/AST observations only; do not call compiled functions."""
    tree = ast.parse(source)
    calls: list[str] = []
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    return {
        "syntax_valid": True,
        "imports_present": any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)),
        "dynamic_exec_calls": sorted(set(calls) & {"eval", "exec", "compile"}),
        "identifiers": sorted(identifiers),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }


def summarize_candidate(task: dict[str, object], env: dict[str, object], traj: dict[str, object]) -> dict[str, object]:
    base = task.get("checklist_with_func", [])
    attack = task.get("injected_checklist_with_func", [])
    checker_sources = [row["check_func"] for row in base + attack]
    facts = [checker_facts(source) for source in checker_sources]
    code = str(env.get("env_class_code", ""))
    env_tree = ast.parse(code)
    method_names = [
        line.strip().split("(", 1)[0].removeprefix("def ")
        for line in code.splitlines()
        if line.startswith("    def ")
    ]
    product_update_fields: list[str] | None = None
    for node in ast.walk(env_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update_product_info":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "allowed_fields"
                    for target in child.targets
                ) and isinstance(child.value, (ast.List, ast.Tuple)):
                    product_update_fields = [
                        elt.value for elt in child.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
            break
    actions = [
        step.get("action")
        for step in traj.get("trajectory", [])
        if isinstance(step.get("action"), dict)
    ]
    initial = task.get("init_config", {})
    encoded_initial = json.dumps(initial, ensure_ascii=False)
    return {
        "task_id": task["task_id"],
        "environment_id": task["env_id"],
        "description": CANDIDATES[task["task_id"]],
        "base_checker_count": len(base),
        "attack_checker_count": len(attack),
        "checker_ast": facts,
        "environment_method_names": method_names,
        "product_initial_fields": (
            sorted(next(iter(initial.get("products", {}).values())).keys())
            if isinstance(initial, dict) and isinstance(initial.get("products"), dict) and initial.get("products")
            else []
        ),
        "product_update_allowed_fields": product_update_fields,
        "initial_state_contains_information_injection": "<INFORMATION>" in encoded_initial,
        "released_train_trajectory": {
            "step_count": traj.get("steps"),
            "total_reward": traj.get("total_reward"),
            "action_names": [action.get("name") for action in actions],
        },
        "review_observations": (
            [
                "One base checker accepts any numeric average-rating field; it does not compare the value with an independently recomputed mean.",
                "The stated task asks to report the mean, while the environment exposes a getter and no dedicated report/result action.",
            ]
            if task["task_id"].startswith("env_123_sft-task_14")
            else [
                "The business checker accepts a name/city/coordinate-box match; it does not test the requested absence of a pre-existing same-name business in the neighborhood.",
                "The final-state checks do not bind success to calls made through the declared tools.",
                "The attack checker returns true when the injected unauthorized helpful-count change is present; it is not a safe-success predicate.",
            ]
        ),
    }


def main() -> None:
    data, input_manifest = fetch()
    task_rows = data["tasks"]
    env_rows = data["environments"]
    trajectories = data["trajectories"]
    if not isinstance(task_rows, list) or not isinstance(env_rows, dict) or not isinstance(trajectories, list):
        raise SystemExit("unexpected JSON root type")
    task_by_id = {row["task_id"]: row for row in task_rows}
    trajectory_by_id = {row["task_info"]["task_id"]: row for row in trajectories}
    candidate_results = {}
    for task_id in CANDIDATES:
        task = task_by_id[task_id]
        base_id = task_id.removesuffix("_IPI_tool_selection")
        candidate_results[task_id] = summarize_candidate(
            task, env_rows[task["env_id"]], trajectory_by_id[base_id]
        )
    print(json.dumps({
        "source_commit": COMMIT,
        "scope": "Pinned TRAIN JSON only; no test artifacts fetched; benchmark Python parsed statically and never executed.",
        "inputs": input_manifest,
        "candidate_reviews": candidate_results,
        "decision": "REJECT_TOOLHAZARD_MPS_POSTTRAINING_GATE",
        "decision_reason": "The inspected task checkers do not establish a complete independent grading contract, and VARE has no existing stateful ToolHazard execution path.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
