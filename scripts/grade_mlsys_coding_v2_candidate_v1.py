#!/usr/bin/env python3
"""Independent narrow contract grader for two unassigned MLSys v2 tasks.

Candidate modules are never imported. Only one predeclared function body is
transplanted into the pinned module after AST admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "benchmarks/corrected_template_agent_v2"
spec = importlib.util.spec_from_file_location(
    "_mlsys_v2_contract_admission", ROOT / "scripts/grade_vare_development_candidate_v1.py",
)
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)
boundary.ASSETS = ASSETS
boundary.TASKS = {
    "label-smooth": ("label-smooth-baseline.py.txt", None, "smooth_targets"),
    "ema-bias": ("ema-bias-baseline.py.txt", None, "bias_corrected_ema"),
}
boundary.SAFE_GLOBALS |= {"math", "float", "range", "type", "_validate", "_finite"}
boundary.SAFE_ATTRIBUTES |= {"isfinite"}
boundary.SAFE_METHOD_CALLS |= {"isfinite"}
HIDDEN_TESTS = {
    "label-smooth": ROOT / "benchmarks/mlsys_coding_v2/tasks/label_smooth/evaluator/test_hidden.py",
    "ema-bias": ROOT / "benchmarks/mlsys_coding_v2/tasks/ema_bias/evaluator/test_hidden.py",
}
CANDIDATE_FILENAMES = {"label-smooth": "label_smooth.py", "ema-bias": "ema_bias.py"}


def source_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def close(actual, expected) -> bool:
    return (isinstance(actual, list) and len(actual) == len(expected)
            and all(isinstance(value, (int, float)) and math.isfinite(value)
                    and abs(value - target) <= 1e-12
                    for value, target in zip(actual, expected)))


def evaluate_label_smooth(function, *, public: bool) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    cases = [(0, 1, 0.0)] if public else [
        (0, 1, 0.2), (0, 3, 0.0), (1, 3, 0.3), (4, 5, 0.9),
        (0, 2, 0.99), (2, 4, 0.2), (1, 2, 0.5),
    ]
    for index, (label, classes, epsilon) in enumerate(cases):
        before = (label, classes, epsilon)
        actual = function(label, classes, epsilon)
        share = float(epsilon) / classes
        expected = [share] * classes
        expected[label] = (1.0 - float(epsilon)) + share
        checks[f"distribution-{index}"] = close(actual, expected)
        checks[f"sum-to-one-{index}"] = (isinstance(actual, list)
                                          and abs(sum(actual) - 1.0) <= 1e-12)
        checks[f"inputs-preserved-{index}"] = before == (label, classes, epsilon)
    if not public:
        for index, args in enumerate([
            (0, 0, 0.1), (2, 2, 0.1), (0, 2, 1.0),
            (0, 2, float("nan")), (True, 2, 0.1),
        ]):
            try:
                function(*args)
            except ValueError:
                checks[f"invalid-{index}"] = True
            else:
                checks[f"invalid-{index}"] = False
    return checks


def evaluate_ema(function, *, public: bool) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    cases = [(5.0, 7.0, 0.0, 3)] if public else [
        (0.0, 10.0, 0.9, 1), (1.0, 10.0, 0.9, 2),
        (0.0, 4.0, 0.0, 1), (2.0, -3.0, 0.5, 3),
        (7.0, 7.0, 0.99, 8),
    ]
    for index, (previous, observation, beta, step) in enumerate(cases):
        raw = float(beta) * float(previous) + (1.0 - float(beta)) * float(observation)
        expected = raw / (1.0 - float(beta) ** step)
        actual = function(previous, observation, beta, step)
        checks[f"bias-correction-{index}"] = (
            isinstance(actual, (int, float)) and math.isfinite(actual)
            and abs(actual - expected) <= 1e-12
        )
    if not public:
        for index, args in enumerate([
            (0.0, 1.0, 1.0, 1), (0.0, 1.0, 0.5, 0),
            (0.0, 1.0, 0.5, True), (float("nan"), 1.0, 0.5, 1),
        ]):
            try:
                function(*args)
            except ValueError:
                checks[f"invalid-{index}"] = True
            else:
                checks[f"invalid-{index}"] = False
    return checks


def grade(task: str, source: Path, *, public: bool = False) -> dict:
    data = source.read_bytes()
    result = {"task": task, "source_sha256": source_sha(data), "public": public}
    try:
        # Execute only the declared function body in the same closed namespace
        # used by task admission; the pinned module and its helpers are trusted.
        implementation, _ = boundary.load_candidate(task, data.decode("utf-8"))
        checks = (evaluate_label_smooth(implementation, public=public)
                  if task == "label-smooth" else evaluate_ema(implementation, public=public))
        harness = None
        harness_passed = None
        if not public:
            with tempfile.TemporaryDirectory(prefix="vare-v2-hidden-grade-") as directory:
                candidate = Path(directory) / CANDIDATE_FILENAMES[task]
                candidate.write_bytes(data)
                try:
                    completed = subprocess.run(
                        [sys.executable, "-I", str(HIDDEN_TESTS[task]), directory],
                        cwd=ROOT, capture_output=True, text=True, timeout=20, check=False,
                    )
                    harness_passed = completed.returncode == 0
                    harness = {"returncode": completed.returncode,
                               "stdout": completed.stdout[-2000:],
                               "stderr": completed.stderr[-2000:],
                               "timeout": False}
                except subprocess.TimeoutExpired as error:
                    harness = {"returncode": None,
                               "stdout": str(error.stdout or "")[-2000:],
                               "stderr": str(error.stderr or "")[-2000:],
                               "timeout": True}
                    harness_passed = False
        result.update(admitted=True, checks=checks,
                      task_pack_hidden_harness=harness,
                      task_pack_hidden_harness_passed=harness_passed,
                      success=bool(checks) and all(checks.values())
                      and (harness_passed is not False))
    except Exception as error:
        result.update(admitted=False, success=False,
                      error=f"{type(error).__name__}: {error}"[:512])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=boundary.TASKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    print(json.dumps(grade(args.task, args.source, public=args.public), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
