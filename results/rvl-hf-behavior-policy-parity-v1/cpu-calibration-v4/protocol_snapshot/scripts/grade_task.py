#!/usr/bin/env python3
"""Run the locked evaluator outside a candidate-controlled source checkout."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/rvl_behavior_policy_parity"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT,
                        help="task directory containing task.json (defaults to the HF parity task)")
    parser.add_argument("--result", type=Path,
                        help="optional JSON result path outside the candidate workspace")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    task_root = args.task_root.expanduser().resolve()
    if not workspace.is_dir():
        parser.error(f"candidate workspace does not exist: {workspace}")
    if not (task_root / "task.json").is_file():
        parser.error(f"task descriptor not found under: {task_root}")
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    evaluator = (ROOT / task["grader"]["path"]).resolve()
    if not evaluator.is_relative_to(ROOT):
        parser.error("grader path must remain inside the VARE repository")
    if evaluator.is_relative_to(workspace) or task_root == workspace or task_root.is_relative_to(workspace):
        parser.error("grader and task specification must stay outside the candidate workspace")

    with tempfile.TemporaryDirectory(prefix="vare-grade-") as temporary:
        result_path = args.result.expanduser().resolve() if args.result else Path(temporary) / "grade.json"
        if result_path.is_relative_to(workspace) or result_path.is_relative_to(task_root):
            parser.error("grade result must be outside the candidate workspace and locked task files")
        command = [sys.executable, str(evaluator), "--workspace", str(workspace),
                   "--task-root", str(task_root), "--json-out", str(result_path)]
        try:
            completed = subprocess.run(command, check=False, text=True,
                                       capture_output=True, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"grader exceeded {args.timeout_seconds} seconds", file=sys.stderr)
            return 1
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
