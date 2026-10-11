#!/usr/bin/env python3
"""Run the task-specific isolated grader through VARE's ExecutableEvaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from vare.environments.runner import ExecutableEvaluator
from vare.environments.spec import EngineeringTaskSpec


TASK_ID = "urllib3-preserve-ambiguous-request-target-v1"
ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/urllib3_3352"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-python", default="/opt/homebrew/bin/python3.11")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    source_record = args.source_record.expanduser().resolve()
    if args.task_id != TASK_ID:
        print(json.dumps({"grader_error": "task identity mismatch"}, sort_keys=True))
        return 2
    if not source_record.is_file():
        print(json.dumps({"grader_error": "source record missing"}, sort_keys=True))
        return 2

    spec_raw = json.loads((TASK_ROOT / "vare_spec.json").read_text(encoding="utf-8"))
    if spec_raw["id"] != TASK_ID:
        print(json.dumps({"grader_error": "VARE task spec identity mismatch"}, sort_keys=True))
        return 2
    spec_raw["source"]["local_path"] = str(workspace)
    command = spec_raw["tests"][0]
    argv = list(command["argv"])
    argv[0] = sys.executable
    command["argv"] = [argv[0]] + [
        item.replace("{source_record}", str(source_record))
        .replace("{task_id}", TASK_ID)
        .replace("{task_python}", args.task_python)
        for item in argv[1:]
    ]
    spec = EngineeringTaskSpec.from_dict(spec_raw, source_base=TASK_ROOT)
    result = ExecutableEvaluator(spec, evaluator_root=TASK_ROOT).evaluate(
        workspace,
        provenance={
            "task_id": TASK_ID,
            "source_record_sha256": hashlib.sha256(source_record.read_bytes()).hexdigest(),
            "grader_script_sha256": hashlib.sha256((TASK_ROOT / "evaluator/grade.py").read_bytes()).hexdigest(),
        },
    )
    test_stdout = result.tests[0].stdout if result.tests else ""
    try:
        task_result = json.loads(test_stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        task_result = {"grader_output_unparsed": test_stdout}
    output = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "passed": result.passed,
        "vare_environment_result": result.to_dict(),
        "task_grade": task_result,
    }
    print(json.dumps(output, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
