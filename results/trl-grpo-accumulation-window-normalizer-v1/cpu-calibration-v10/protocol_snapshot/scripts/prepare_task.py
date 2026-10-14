#!/usr/bin/env python3
"""Create an isolated candidate checkout at the task's pinned base revision."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/rvl_behavior_policy_parity"


def run_git(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                            capture_output=True)
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="new directory for the candidate checkout")
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT,
                        help="task directory containing task.json (defaults to the HF parity task)")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    task_root = args.task_root.expanduser().resolve()
    if workspace.exists():
        parser.error(f"workspace must not already exist: {workspace}")
    if workspace == ROOT or workspace.is_relative_to(ROOT) or ROOT.is_relative_to(workspace):
        parser.error("candidate workspace must be separate from the VARE repository")
    if task_root == workspace or task_root.is_relative_to(workspace):
        parser.error("task specification must be outside the candidate workspace")

    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    workspace.parent.mkdir(parents=True, exist_ok=True)
    workspace.mkdir()
    run_git(["init", "--quiet"], cwd=workspace)
    run_git(["remote", "add", "origin", task["source"]["repository"]], cwd=workspace)
    run_git(["fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin",
             task["source"]["base_revision"]], cwd=workspace)
    run_git(["sparse-checkout", "init", "--no-cone"], cwd=workspace)
    sparse_patterns = "\n".join("/" + item for item in task["source"]["files"]) + "\n"
    (workspace / ".git/info/sparse-checkout").write_text(sparse_patterns, encoding="utf-8")
    run_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=workspace)
    revision = run_git(["rev-parse", "HEAD"], cwd=workspace)
    if revision != task["source"]["base_revision"]:
        parser.error(f"source checkout mismatch: expected {task['source']['base_revision']}, got {revision}")
    missing = [item for item in task["source"]["files"] if not (workspace / item).is_file()]
    if missing:
        parser.error(f"sparse checkout omitted required source files: {missing}")

    print(f"Prepared {task['task_id']} at {revision}")
    print(f"Candidate workspace: {workspace}")
    print(f"Task brief: {task_root / 'TASK.md'}")
    print(f"Grade after the change: python3 {ROOT / 'scripts/grade_task.py'} "
          f"--workspace {workspace} --task-root {task_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
