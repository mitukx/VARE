#!/usr/bin/env python3
"""Run locked base and known-fix grader calibration without retaining a gold patch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "urllib3-preserve-ambiguous-request-target-v1"
PYTHON = "/opt/homebrew/bin/python3.11"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(output: Path, *, fix: bool, task_python: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    preparation_dir = output / "preparation"
    prep = [sys.executable, str(ROOT / "scripts/prepare_urllib3_3352_task.py"), "--output", str(preparation_dir)]
    if fix:
        prep.append("--calibration-fix")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    preparation = subprocess.run(prep, check=True, capture_output=True, text=True, env=env)
    record_path = preparation_dir / "source-record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    command = [
        sys.executable,
        str(ROOT / "scripts/grade_urllib3_3352_task.py"),
        "--workspace", str(preparation_dir / "workspace"),
        "--source-record", str(record_path),
        "--task-id", TASK_ID,
        "--task-python", task_python,
    ]
    grading = subprocess.run(command, check=False, capture_output=True, text=True, env=env)
    (output / "prepare.stdout.json").write_text(preparation.stdout, encoding="utf-8")
    (output / "grade.stdout.json").write_text(grading.stdout, encoding="utf-8")
    (output / "grade.stderr.txt").write_text(grading.stderr, encoding="utf-8")
    try:
        payload = json.loads(grading.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        payload = {"unparsed_stdout": grading.stdout}
    result = {
        "name": "known-fix" if fix else "pinned-base",
        "process_exit_code": grading.returncode,
        "passed": payload.get("passed", False),
        "task_grade": payload.get("task_grade", {}),
        "source_record_sha256": digest(record_path),
        "prepare_stdout_sha256": digest(output / "prepare.stdout.json"),
        "grade_stdout_sha256": digest(output / "grade.stdout.json"),
        "grade_stderr_sha256": digest(output / "grade.stderr.txt"),
        "workspace_baseline_commit": record["workspace_baseline_commit"],
        "workspace_baseline_tree": record["workspace_baseline_tree"],
        "calibration_reference_revision": record["calibration_reference_revision"],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new run root outside the VARE checkout")
    parser.add_argument("--task-python", default=PYTHON)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing calibration output: {output}")
    output.mkdir(parents=True)
    baseline = run_one(output / "pinned-base", fix=False, task_python=args.task_python)
    fixed = run_one(output / "known-fix", fix=True, task_python=args.task_python)
    expected_baseline = (
        baseline["passed"] is False
        and baseline["task_grade"].get("fail_to_pass", {}).get("double_slash_colon") is False
        and baseline["task_grade"].get("fail_to_pass", {}).get("double_slash_colon_query") is False
        and all(baseline["task_grade"].get("pass_to_pass", {}).values())
    )
    expected_fix = fixed["passed"] is True and all(fixed["task_grade"].get("fail_to_pass", {}).values())
    result = {
        "protocol_id": "urllib3-3352-local-http-feasibility-v1",
        "expected_baseline_failure_reproduced": expected_baseline,
        "known_fix_passed_calibration": expected_fix,
        "pinned_base": baseline,
        "known_fix": fixed,
    }
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest(path)}  {path.relative_to(output).as_posix()}\n"
                for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"),
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if expected_baseline and expected_fix else 1


if __name__ == "__main__":
    raise SystemExit(main())
