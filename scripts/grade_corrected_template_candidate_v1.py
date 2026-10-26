#!/usr/bin/env python3
"""Independent narrow behavioral oracle for two public MLX-LM Development bugs.

Only structurally admitted function bodies execute. Full candidate modules,
installed MLX-LM sources, Git history, and hidden test assets are not imported.
This is a function-contract evaluation, not a hostile-Python sandbox.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "benchmarks/corrected_template_agent_v1"
spec = importlib.util.spec_from_file_location("_corrected_contract_admission", ROOT / "scripts/grade_vare_development_candidate_v1.py")
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
boundary.ASSETS = ASSETS
boundary.TASKS = {
    "checkpoint-shards": ("checkpoint-shards-baseline.py.txt", None, "make_shards"),
    "gqa-repeat": ("gqa-repeat-baseline.py.txt", None, "repeat_kv_heads"),
}
boundary.SAFE_GLOBALS |= {"MAX_FILE_SIZE_GB", "math", "_validate"}
boundary.SAFE_ATTRIBUTES |= {"nbytes"}


def load(task: str, text: str):
    baseline, function = boundary.admit(task, text)
    namespace = {"MAX_FILE_SIZE_GB": 5}
    # Baseline projection contains only the pinned pure helper and target.
    exec(compile(baseline, "<pinned-projection>", "exec"), namespace)
    transplant = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(transplant)
    exec(compile(transplant, "<admitted-function>", "exec"), namespace)
    return namespace[boundary.TASKS[task][2]]


def evaluate(task: str, source: Path, *, public: bool = False):
    data = source.read_bytes()
    result = {"task": task, "source_sha256": hashlib.sha256(data).hexdigest(), "public": public}
    try:
        function = load(task, data.decode())
        checks = {}
        if task == "checkpoint-shards":
            gb = 1 << 30
            # Explicit parent-owned expected partitions; no arrays are allocated.
            cases = [([], []), ([gb + 1], [[0]]), ([3, 5], [[0, 1]]),
                     ([gb // 2, gb // 2], [[0, 1]]),
                     ([gb // 2, gb // 2 + 1], [[0], [1]]),
                     ([1, gb + 1, 1], [[0], [1], [2]]),
                     ([gb, gb], [[0], [1]]), ([0], [[0]]),
                     ([gb + 9, gb + 8], [[0], [1]])]
            for i, (sizes, partition) in enumerate(cases[:3] if public else cases):
                weights = {f"weight_{j}": SimpleNamespace(nbytes=size) for j, size in enumerate(sizes)}
                expected = [{f"weight_{j}": weights[f"weight_{j}"] for j in group} for group in partition]
                actual = function(weights, 1)
                checks[f"partition-{i}"] = actual == expected
                checks[f"input-preserved-{i}"] = list(weights) == [f"weight_{j}" for j in range(len(sizes))]
        elif task == "gqa-repeat":
            cases = [
                ([[1.0, 2.0], [3.0, 4.0]], 2,
                 [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]]),
                ([[5.0, 6.0, 7.0]], 3,
                 [[5.0, 6.0, 7.0], [5.0, 6.0, 7.0], [5.0, 6.0, 7.0]]),
                ([[9.0]], 1, [[9.0]]),
            ]
            for i, (keys, n_rep, expected) in enumerate(cases[:2] if public else cases):
                before = [row[:] for row in keys]
                actual = function(keys, n_rep)
                checks[f"head-order-{i}"] = actual == expected
                checks[f"fresh-rows-{i}"] = all(
                    actual[j] is not keys[j // n_rep] for j in range(len(actual))
                )
                checks[f"input-preserved-{i}"] = keys == before
            if not public:
                for i, (keys, n_rep) in enumerate([
                    ([], 1), ([[1.0], [2.0, 3.0]], 2), ([[float("nan")]], 2),
                    ([[True]], 2), ([[1.0]], 0), ([[1.0]], True),
                ]):
                    try:
                        function(keys, n_rep)
                    except ValueError:
                        checks[f"invalid-{i}"] = True
                    else:
                        checks[f"invalid-{i}"] = False
        result.update(admitted=True, checks=checks, success=bool(checks) and all(checks.values()))
    except Exception as error:
        result.update(admitted=False, success=False, error=f"{type(error).__name__}: {error}"[:512])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=boundary.TASKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.task, args.source, public=args.public), sort_keys=True))


if __name__ == "__main__":
    main()
