#!/usr/bin/env python3
"""Calibrate the historical-task grader on pinned failing and fixed revisions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/rvl_behavior_policy_parity"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_git(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                            capture_output=True)
    return result.stdout.strip()


def verify_lock(task: dict, lock: dict, task_root: Path, evaluator: Path) -> None:
    expected = lock["locked_hashes"]
    actual = {
        "task_json_sha256": sha256_file(task_root / "task.json"),
        "task_brief_sha256": sha256_file(task_root / "TASK.md"),
        "evaluator_sha256": sha256_file(evaluator),
    }
    for name, value in actual.items():
        if expected[name] != value:
            raise SystemExit(f"protocol lock mismatch for {name}: expected {expected[name]}, got {value}")
    if task["task_id"] != lock["task_id"]:
        raise SystemExit("task id does not match protocol lock")


def checkout_revision(
    repo_url: str,
    revision: str,
    base_revision: str,
    path: Path,
    source_files: list[str],
) -> str:
    path.mkdir()
    run_git(["init", "--quiet"], cwd=path)
    run_git(["remote", "add", "origin", repo_url], cwd=path)
    run_git(["fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", base_revision], cwd=path)
    if revision != base_revision:
        run_git(["fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", revision], cwd=path)
    run_git(["sparse-checkout", "init", "--no-cone"], cwd=path)
    sparse_patterns = "\n".join("/" + item for item in source_files) + "\n"
    (path / ".git/info/sparse-checkout").write_text(sparse_patterns, encoding="utf-8")
    run_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=path)
    actual = run_git(["rev-parse", "HEAD"], cwd=path)
    if actual != revision:
        raise RuntimeError(f"revision mismatch: expected {revision}, got {actual}")
    missing = [item for item in source_files if not (path / item).is_file()]
    if missing:
        raise RuntimeError(f"sparse checkout omitted required source files: {missing}")
    return actual


def grade_revision(
    label: str,
    revision: str,
    task: dict,
    temporary: Path,
    task_root: Path,
    evaluator: Path,
) -> dict:
    source = temporary / label
    actual_revision = checkout_revision(task["source"]["repository"], revision,
                                        task["source"]["base_revision"], source,
                                        task["source"]["files"])
    result_path = temporary / f"{label}.grade.json"
    command = [sys.executable, str(evaluator), "--workspace", str(source),
               "--task-root", str(task_root), "--json-out", str(result_path), "--allow-fail"]
    completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=45)
    if completed.returncode != 0:
        raise RuntimeError(
            f"grader could not evaluate {label} (exit {completed.returncode}): "
            f"{completed.stdout}\n{completed.stderr}"
        )
    if not result_path.is_file():
        raise RuntimeError(f"grader produced no result for {label}: {completed.stdout} {completed.stderr}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "grader_error" in result:
        raise RuntimeError(f"grader error on {label}: {result}")
    if result["workspace_revision"] != actual_revision:
        raise RuntimeError(f"graded revision mismatch for {label}")
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def snapshot_protocol(output: Path, task_root: Path, evaluator: Path) -> None:
    destination = output / "protocol_snapshot"
    files = {
        "task.json": task_root / "task.json",
        "TASK.md": task_root / "TASK.md",
        "protocol.lock.json": task_root / "protocol.lock.json",
        str(Path("evaluator") / evaluator.name): evaluator,
        "scripts/calibrate_task.py": Path(__file__).resolve(),
        "scripts/grade_task.py": ROOT / "scripts/grade_task.py",
        "scripts/prepare_task.py": ROOT / "scripts/prepare_task.py",
    }
    for relative, source in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    sums = {relative: sha256_file(destination / relative) for relative in files}
    write_json(destination / "SHA256SUMS.json", sums)


def write_manifest(output: Path, task_root: Path, evaluator: Path) -> None:
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "task_json_sha256": sha256_file(task_root / "task.json"),
        "task_brief_sha256": sha256_file(task_root / "TASK.md"),
        "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
        "evaluator_sha256": sha256_file(evaluator),
        "calibration_script_sha256": sha256_file(Path(__file__).resolve()),
        "files": files,
    }
    write_json(output / "manifest.json", manifest)


def result_metrics(result: dict) -> dict:
    if isinstance(result.get("metrics"), dict):
        return result["metrics"]
    checks = result.get("checks", {})
    rollout_rows = checks.get("rollout_cases", [])
    trainer_rows = checks.get("trainer", {}).get("conditions", [])
    return {
        "rollout_conditions": len(rollout_rows),
        "trainer_conditions": len(trainer_rows),
        "max_rollout_abs_logprob_error": max(
            (row["max_abs_logprob_error"] for row in rollout_rows
             if isinstance(row.get("max_abs_logprob_error"), (int, float))),
            default=None,
        ),
        "max_trainer_abs_logprob_error": max(
            (row["max_abs_logprob_error"] for row in trainer_rows
             if isinstance(row.get("max_abs_logprob_error"), (int, float))),
            default=None,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT,
                        help="task directory containing task.json")
    parser.add_argument("--output", type=Path,
                        help="new evidence directory (defaults under results/<task-id>)")
    args = parser.parse_args()
    task_root = args.task_root.expanduser().resolve()
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    lock = json.loads((task_root / "protocol.lock.json").read_text(encoding="utf-8"))
    evaluator = (ROOT / task["grader"]["path"]).resolve()
    if not evaluator.is_relative_to(ROOT):
        parser.error("grader path must remain inside the VARE repository")
    protocol_version = task["grader"].get("protocol_version", 1)
    default_output = ROOT / "results" / task["task_id"] / f"cpu-calibration-v{protocol_version}"
    output = args.output.expanduser().resolve() if args.output else default_output
    if output.exists():
        parser.error(f"output already exists; choose a new path to preserve prior evidence: {output}")
    if output == ROOT or ROOT.is_relative_to(output):
        parser.error("output cannot contain the VARE repository")

    verify_lock(task, lock, task_root, evaluator)
    output.mkdir(parents=True)
    snapshot_protocol(output, task_root, evaluator)
    with tempfile.TemporaryDirectory(prefix="vare-calibrate-") as scratch:
        temporary = Path(scratch)
        try:
            baseline = grade_revision("baseline", task["source"]["base_revision"], task,
                                      temporary, task_root, evaluator)
            write_json(output / "baseline.grade.json", baseline)
            calibration = grade_revision("calibration", task["source"]["calibration_revision"], task,
                                         temporary, task_root, evaluator)
            write_json(output / "calibration.grade.json", calibration)
        except Exception as exc:
            failure = {
                "schema_version": 1,
                "task_id": task["task_id"],
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "created_at_utc": datetime.now(timezone.utc).isoformat()},
            }
            write_json(output / "failure.json", failure)
            write_manifest(output, task_root, evaluator)
            print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
            return 1

    expected = lock["expected_calibration"]
    observed = {"baseline_passed": baseline["passed"], "calibration_passed": calibration["passed"]}
    if observed != expected:
        failure = {
            "schema_version": 1,
            "task_id": task["task_id"],
            "status": "failed",
            "error_type": "CalibrationOutcomeMismatch",
            "message": f"observed {observed} but preregistered outcome is {expected}",
                "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
        }
        write_json(output / "failure.json", failure)
        write_manifest(output, task_root, evaluator)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    summary = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "protocol_id": lock["protocol_id"],
        "status": "calibrated",
        "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
        "pre_fix_revision": task["source"]["base_revision"],
        "fixed_revision": task["source"]["calibration_revision"],
        "observed": observed,
        "baseline_metrics": result_metrics(baseline),
        "fixed_metrics": result_metrics(calibration),
        "resources": {
            "model_weights_downloaded": False,
            "gpu_hours": 0,
            "paid_api_calls": 0,
            "external_compute_usd": 0,
            "third_party_python_packages": 0,
        },
        "claim_limit": "This run calibrates one deterministic CPU grader on a real historical source change; it measures no agent, model learning, or capability gain.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()},
    }
    write_json(output / "summary.json", summary)
    write_manifest(output, task_root, evaluator)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Evidence and SHA-256 manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
