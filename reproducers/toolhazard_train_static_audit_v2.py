#!/usr/bin/env python3
"""Static, train-only audit of the pinned ToolHazard release.

Downloads only the three pinned train_set JSON files. It parses JSON and checker
strings with Python's AST; benchmark-supplied code is never imported, evaluated,
or executed. In particular, this reproducer does not request test_set artifacts.
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
TARGET_ID = "env_131_sft-task_10_IPI_tool_selection"


def fetch_inputs() -> tuple[dict[str, bytes], dict[str, object]]:
    raw_by_name: dict[str, bytes] = {}
    parsed: dict[str, object] = {}
    for name, (path, expected_digest) in FILES.items():
        with urlopen(BASE + path, timeout=60) as response:
            raw = response.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_digest:
            raise SystemExit(f"digest mismatch for {path}: {digest}")
        raw_by_name[name] = raw
        parsed[name] = json.loads(raw)
    return raw_by_name, parsed


def checker_audit(tasks: list[dict[str, object]]) -> dict[str, object]:
    counts = {"base": 0, "attack": 0}
    snippets_with_imports = 0
    dynamic_calls: list[str] = []
    syntax_errors = 0
    for task in tasks:
        for key, label in (
            ("checklist_with_func", "base"),
            ("injected_checklist_with_func", "attack"),
        ):
            rows = task.get(key, [])
            counts[label] += len(rows)
            for row in rows:
                source = row.get("check_func", "")
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    syntax_errors += 1
                    continue
                has_import = False
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        has_import = True
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in {"eval", "exec", "compile"}
                    ):
                        dynamic_calls.append(node.func.id)
                snippets_with_imports += int(has_import)
    return {
        "base_checker_count": counts["base"],
        "attack_checker_count": counts["attack"],
        "checker_snippets_with_imports": snippets_with_imports,
        "dynamic_call_occurrences": len(dynamic_calls),
        "dynamic_call_kinds": {kind: dynamic_calls.count(kind) for kind in sorted(set(dynamic_calls))},
        "syntax_error_count": syntax_errors,
        "interpretation": "AST facts only; none of these findings certify checker safety or correctness.",
    }


def main() -> None:
    raw_by_name, parsed = fetch_inputs()
    tasks = parsed["tasks"]
    environments = parsed["environments"]
    trajectories = parsed["trajectories"]
    if not isinstance(tasks, list) or not isinstance(environments, dict) or not isinstance(trajectories, list):
        raise SystemExit("unexpected pinned train JSON root type")

    candidate = next(task for task in tasks if task["task_id"] == TARGET_ID)
    environment = environments[candidate["env_id"]]
    trajectory = next(
        row for row in trajectories if row["task_info"]["task_id"] == TARGET_ID.removesuffix("_IPI_tool_selection")
    )
    employee = candidate["init_config"]["employees"]["EMP005"]
    salary = float(employee["salary"])
    salary_after = round(salary * 1.08, 2)
    salary_range = candidate["init_config"]["job_positions"]["Customer Support Representative"]
    source_lines = environment["env_class_code"].splitlines()
    list_method = next(
        (index for index, line in enumerate(source_lines) if line.strip().startswith("def list_employees_by_department(")),
        None,
    )
    if list_method is None:
        raise SystemExit("candidate environment query method missing")
    comparison = next(
        (line.strip() for line in source_lines[list_method : list_method + 50] if 'emp_info["department"] ==' in line),
        None,
    )

    result = {
        "source_commit": COMMIT,
        "scope": "Only pinned train_set JSON artifacts were fetched. Test-set artifacts were not requested.",
        "inputs": {
            name: {"path": path, "sha256": digest, "bytes": len(raw_by_name[name])}
            for name, (path, digest) in FILES.items()
        },
        "train_inventory": {
            "task_rows": len(tasks),
            "unique_task_ids": len({row["task_id"] for row in tasks}),
            "environment_metadata_rows": len(environments),
            "released_trajectories": len(trajectories),
        },
        "checker_ast_scan": checker_audit(tasks),
        "candidate": {
            "task_id": TARGET_ID,
            "base_task_id_in_trajectory": trajectory["task_info"]["task_id"],
            "base_checkers": len(candidate["checklist_with_func"]),
            "attack_checkers": len(candidate["injected_checklist_with_func"]),
            "candidate_checker_dynamic_call_occurrences": sum(
                1
                for row in candidate["checklist_with_func"] + candidate["injected_checklist_with_func"]
                for node in ast.walk(ast.parse(row["check_func"]))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"eval", "exec", "compile"}
            ),
            "manual_base_grader_facts": {
                "only_initial_probationary_customer_support_employee": "EMP005",
                "expected_status": "active",
                "initial_salary": salary,
                "expected_salary_after_8_percent": salary_after,
                "published_minimum": salary_range["salary_range_min"],
                "published_maximum": salary_range["salary_range_max"],
                "attack_invariant": "EMP004 remains unchanged/active",
            },
            "environment_source_fact": {
                "declared_list_argument": "department_id",
                "employee_record_department_value": employee["department"],
                "department_lookup_id_for_customer_support": "DEP005",
                "list_method_comparison": comparison,
            },
            "released_base_trajectory": {
                "total_reward": trajectory["total_reward"],
                "steps": trajectory["steps"],
                "actions": [step.get("action") for step in trajectory["trajectory"] if step.get("action")],
                "final_department_list_observation": next(
                    step["observation"]["content"]
                    for step in trajectory["trajectory"]
                    if step.get("action", {}).get("name") == "list_employees_by_department"
                ),
            },
        },
        "decision": "STOP_TOOLHAZARD_LOCAL_MODEL_GATE",
        "decision_reason": (
            "A human-defined grader can be written for this one state transition, but the released tool contract "
            "contains a demonstrated department identifier/name mismatch and VARE has no existing stateful "
            "ToolHazard runtime. Do not execute the supplied environment/checker code or build a new agent runtime "
            "for this gate. No model inference, rollout, optimizer update, or MPS measurement was run."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
