#!/usr/bin/env python3
"""Calibrate that locked task inputs reject isolated, deliberate mutations."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = ROOT / "benchmarks/historical"
EXPECTED_CONTROL_ERROR = "candidate checkout is missing source files"
EXPECTED_LOCK_ERROR = "does not match the locked protocol"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def task_directories() -> list[Path]:
    return sorted(path.parent for path in TASKS_ROOT.glob("*/task.json")
                  if (path.parent / "protocol.lock.json").is_file())


def checked_inputs(task_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    lock = json.loads((task_root / "protocol.lock.json").read_text(encoding="utf-8"))
    evaluator = (ROOT / task["grader"]["path"]).resolve()
    if not evaluator.is_relative_to(ROOT) or not evaluator.is_file():
        raise ValueError(f"grader must be an existing file inside the repository: {evaluator}")
    expected = lock["locked_hashes"]
    actual = {
        "task_json_sha256": sha256_file(task_root / "task.json"),
        "task_brief_sha256": sha256_file(task_root / "TASK.md"),
        "evaluator_sha256": sha256_file(evaluator),
    }
    if task["task_id"] != lock["task_id"]:
        raise ValueError(f"task id mismatch in {task_root}")
    for name, value in actual.items():
        if expected[name] != value:
            raise ValueError(f"protocol lock mismatch for {name} in {task_root}")
    return task, lock, evaluator


def copy_task_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for name in ("task.json", "TASK.md", "protocol.lock.json"):
        shutil.copy2(source / name, destination / name)


def invoke_grader(evaluator: Path, workspace: Path, task_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(evaluator), "--workspace", str(workspace),
         "--task-root", str(task_root)],
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"grader returned non-JSON output (exit {completed.returncode}): "
            f"{completed.stdout[:300]} {completed.stderr[:300]}"
        ) from exc
    return {"exit_code": completed.returncode, "payload": payload}


def run_task_calibration(task_root: Path, scratch: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task, lock, evaluator = checked_inputs(task_root)
    task_id = task["task_id"]
    workspace = scratch / "empty-candidate-workspace"
    workspace.mkdir()

    control_root = scratch / "control-task"
    copy_task_files(task_root, control_root)
    control = invoke_grader(evaluator, workspace, control_root)
    control_error = control["payload"].get("message", "")
    if (control["exit_code"] != 1
            or control["payload"].get("grader_error") != "FileNotFoundError"
            or EXPECTED_CONTROL_ERROR not in control_error):
        raise RuntimeError(f"untampered control did not reach candidate validation for {task_id}: {control}")
    control_record = {
        "task_id": task_id,
        "expected": "locked inputs pass; empty candidate fails at source-file validation",
        "observed": "locked inputs pass; missing candidate source files rejected",
        "grader_error": control["payload"]["grader_error"],
        "passed": True,
    }

    cases: list[dict[str, Any]] = []
    for mutation in ("task_descriptor", "task_brief", "evaluator"):
        case_root = scratch / mutation
        copied_task = case_root / "task"
        copy_task_files(task_root, copied_task)
        case_evaluator = evaluator

        if mutation == "task_descriptor":
            descriptor_path = copied_task / "task.json"
            descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
            descriptor["title"] = str(descriptor["title"]) + " [tampered]"
            write_json(descriptor_path, descriptor)
        elif mutation == "task_brief":
            brief = copied_task / "TASK.md"
            brief.write_bytes(brief.read_bytes() + b"\nTampering calibration mutation.\n")
        else:
            case_evaluator = case_root / "evaluator" / "grade.py"
            case_evaluator.parent.mkdir(parents=True)
            shutil.copy2(evaluator, case_evaluator)
            case_evaluator.write_bytes(case_evaluator.read_bytes() + b"\n")

        observed = invoke_grader(case_evaluator, workspace, copied_task)
        message = observed["payload"].get("message", "")
        rejected = (observed["exit_code"] == 1
                    and observed["payload"].get("grader_error") == "ValueError"
                    and EXPECTED_LOCK_ERROR in message)
        cases.append({
            "task_id": task_id,
            "mutation": mutation,
            "grader_error": observed["payload"].get("grader_error"),
            "error_message": message,
            "expected": "ValueError with locked-protocol hash mismatch",
            "passed": rejected,
        })
        if not rejected:
            raise RuntimeError(f"{mutation} mutation was not rejected for {task_id}: {observed}")

    record = {
        "task_id": task_id,
        "protocol_id": lock["protocol_id"],
        "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
        "task_json_sha256": sha256_file(task_root / "task.json"),
        "task_brief_sha256": sha256_file(task_root / "TASK.md"),
        "evaluator_sha256": sha256_file(evaluator),
    }
    return {"control": control_record, "cases": cases}, record


def snapshot(output: Path, task_roots: list[Path], input_records: list[dict[str, Any]]) -> None:
    files: dict[str, Path] = {
        "scripts/calibrate_integrity.py": Path(__file__).resolve(),
    }
    for task_root, record in zip(task_roots, input_records):
        prefix = Path("tasks") / record["task_id"]
        task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
        evaluator = (ROOT / task["grader"]["path"]).resolve()
        for name, source in (
            ("task.json", task_root / "task.json"),
            ("TASK.md", task_root / "TASK.md"),
            ("protocol.lock.json", task_root / "protocol.lock.json"),
            ("evaluator/grade.py", evaluator),
        ):
            files[str(prefix / name)] = source
    sums: dict[str, str] = {}
    for relative, source in files.items():
        target = output / "protocol_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        sums[relative] = sha256_file(target)
    write_json(output / "protocol_snapshot/SHA256SUMS.json", sums)


def write_manifest(output: Path, input_records: list[dict[str, Any]]) -> None:
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(output / "manifest.json", {"schema_version": 1, "inputs": input_records,
                                           "files": files})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/protocol-integrity/cpu-calibration-v1",
                        help="new evidence directory")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists; choose a new directory: {output}")
    if output == ROOT or ROOT.is_relative_to(output):
        parser.error("output cannot contain the repository")

    task_roots = task_directories()
    if not task_roots:
        parser.error(f"no locked tasks found under {TASKS_ROOT}")
    input_records: list[dict[str, Any]] = []
    for task_root in task_roots:
        task, lock, evaluator = checked_inputs(task_root)
        input_records.append({
            "task_id": task["task_id"],
            "protocol_id": lock["protocol_id"],
            "task_json_sha256": sha256_file(task_root / "task.json"),
            "task_brief_sha256": sha256_file(task_root / "TASK.md"),
            "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
            "evaluator_sha256": sha256_file(evaluator),
        })

    output.mkdir(parents=True)
    snapshot(output, task_roots, input_records)
    controls: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="vare-integrity-calibration-") as scratch_name:
            scratch = Path(scratch_name)
            for index, task_root in enumerate(task_roots):
                task_scratch = scratch / str(index)
                task_scratch.mkdir()
                outcome, _ = run_task_calibration(task_root, task_scratch)
                controls.append(outcome["control"])
                cases.extend(outcome["cases"])
        write_json(output / "controls.json", controls)
        write_json(output / "tamper_cases.json", cases)
        summary = {
            "schema_version": 1,
            "status": "calibrated",
            "calibration_id": "locked-task-input-tamper-calibration-v1",
            "control_count": len(controls),
            "tamper_case_count": len(cases),
            "tamper_rejections": sum(bool(case["passed"]) for case in cases),
            "observed": {"all_controls_reached_candidate_validation": all(row["passed"] for row in controls),
                         "all_mutations_rejected": all(row["passed"] for row in cases)},
            "resources": {"model_weights_downloaded": False, "gpu_hours": 0,
                          "paid_api_calls": 0, "external_compute_usd": 0,
                          "third_party_python_packages": 0, "public_source_fetches": 0},
            "trust_boundary": "The unchanged, version-controlled protocol lock is the trust anchor; this run detects isolated changes to the task descriptor, brief, or grader. It does not authenticate coordinated edits to the lock itself and is not a hostile-code sandbox.",
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "created_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        if not summary["observed"]["all_controls_reached_candidate_validation"] or not summary["observed"]["all_mutations_rejected"]:
            raise RuntimeError("integrity calibration did not produce the preregistered outcome")
        write_json(output / "summary.json", summary)
        write_manifest(output, input_records)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"Evidence and SHA-256 manifest: {output}")
        return 0
    except Exception as exc:
        write_json(output / "failure.json", {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "created_at_utc": datetime.now(timezone.utc).isoformat()},
        })
        write_manifest(output, input_records)
        print(f"Integrity calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
