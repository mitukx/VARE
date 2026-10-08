#!/usr/bin/env python3
"""Re-run the explicitly frozen current VARE regression controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/regression_calibration_v1.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def run_git(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                            capture_output=True)
    return result.stdout.strip()


def protocol_tasks() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    tasks = protocol.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("regression protocol must declare a non-empty task allowlist")
    ids = [entry.get("task_id") for entry in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("regression protocol has duplicate task ids")
    return protocol, tasks


def checked_task(entry: dict[str, Any], expected: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    task_root = (ROOT / entry["task_root"]).resolve()
    if not task_root.is_relative_to(ROOT):
        raise ValueError(f"task root escapes repository: {task_root}")
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    lock = json.loads((task_root / "protocol.lock.json").read_text(encoding="utf-8"))
    evaluator = (ROOT / task["grader"]["path"]).resolve()
    if not evaluator.is_relative_to(ROOT) or not evaluator.is_file():
        raise ValueError(f"grader must be an existing repository file: {evaluator}")
    if task.get("task_id") != entry["task_id"] or lock.get("task_id") != entry["task_id"]:
        raise ValueError(f"task id mismatch for {entry['task_id']}")
    lock_hash = sha256_file(task_root / "protocol.lock.json")
    if lock_hash != entry["protocol_lock_sha256"]:
        raise ValueError(f"protocol lock changed for {entry['task_id']}: {lock_hash}")
    locked_hashes = lock["locked_hashes"]
    observed = {
        "task_json_sha256": sha256_file(task_root / "task.json"),
        "task_brief_sha256": sha256_file(task_root / "TASK.md"),
        "evaluator_sha256": sha256_file(evaluator),
    }
    for name, value in observed.items():
        if locked_hashes.get(name) != value:
            raise ValueError(f"protocol input hash mismatch for {entry['task_id']} {name}")
    if lock.get("expected_calibration") != expected:
        raise ValueError(f"expected outcomes differ from calibration protocol for {entry['task_id']}")
    source = task["source"]
    if not source.get("files") or any(not name.startswith("src/vare/") for name in source["files"]):
        raise ValueError(f"only VARE source tasks are supported: {entry['task_id']}")
    if not REVISION_RE.fullmatch(source.get("base_revision", "")):
        raise ValueError(f"base revision must be a full commit SHA for {entry['task_id']}")
    if not REVISION_RE.fullmatch(entry.get("fixed_revision", "")):
        raise ValueError(f"fixed revision must be a full commit SHA for {entry['task_id']}")
    return task_root, task, lock, evaluator


def checkout_revision(task: dict[str, Any], revision: str, path: Path) -> str:
    path.mkdir(parents=True)
    repository = task["source"]["repository"]
    run_git(["init", "--quiet"], cwd=path)
    run_git(["remote", "add", "origin", repository], cwd=path)
    run_git(["fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", revision], cwd=path)
    run_git(["sparse-checkout", "init", "--no-cone"], cwd=path)
    # The grader imports VARE modules beyond the candidate edit target. Keep the
    # complete package from the same immutable revision in the candidate tree.
    (path / ".git/info/sparse-checkout").write_text("/src/vare/**\n", encoding="utf-8")
    run_git(["checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=path)
    actual = run_git(["rev-parse", "HEAD"], cwd=path)
    if actual != revision:
        raise RuntimeError(f"revision mismatch: expected {revision}, got {actual}")
    for name in task["source"]["files"]:
        if not (path / name).is_file():
            raise RuntimeError(f"sparse checkout omitted task source: {name}")
    return actual


def source_hashes(workspace: Path, files: list[str]) -> dict[str, str]:
    return {name: sha256_file(workspace / name) for name in files}


def run_grader(evaluator: Path, workspace: Path, task_root: Path,
               result_path: Path) -> tuple[dict[str, Any], int]:
    help_result = subprocess.run([sys.executable, str(evaluator), "--help"],
                                 check=False, text=True, capture_output=True, timeout=15)
    help_text = help_result.stdout + help_result.stderr
    command = [sys.executable, str(evaluator), "--workspace", str(workspace)]
    if "--task-root" in help_text:
        command.extend(["--task-root", str(task_root)])
    command.extend(["--json-out", str(result_path)])
    completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=90)
    if not result_path.is_file():
        raise RuntimeError(
            f"grader emitted no JSON (exit {completed.returncode}): "
            f"{completed.stdout[-1000:]} {completed.stderr[-1000:]}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result.get("passed"), bool):
        raise RuntimeError("grader JSON does not contain a boolean passed field")
    if completed.returncode not in (0, 1, 2):
        raise RuntimeError(f"grader exited unexpectedly ({completed.returncode}): {completed.stderr[-1000:]}")
    return result, completed.returncode


def snapshot_protocol(output: Path, task_root: Path, evaluator: Path) -> None:
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    paths = {
        "task.json": task_root / "task.json",
        "TASK.md": task_root / "TASK.md",
        "protocol.lock.json": task_root / "protocol.lock.json",
        "evaluator/grade.py": evaluator,
        "calibration_protocol.json": PROTOCOL,
    }
    for relative, source in paths.items():
        target = output / "protocol_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(output / "protocol_snapshot/SHA256SUMS.json", {
        relative: sha256_file(output / "protocol_snapshot" / relative)
        for relative in paths
    })


def write_manifest(output: Path, entry: dict[str, Any]) -> None:
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "protocol_id": "vare-current-regression-calibration-v1",
        "task_id": entry["task_id"],
        "files": files,
    })


def calibrate(entry: dict[str, Any], protocol: dict[str, Any], output: Path) -> dict[str, Any]:
    task_root, task, lock, evaluator = checked_task(entry, protocol["expected_outcomes"])
    output.mkdir(parents=True)
    snapshot_protocol(output, task_root, evaluator)
    files = task["source"]["files"]
    baseline_revision = task["source"]["base_revision"]
    fixed_revision = entry["fixed_revision"]
    with tempfile.TemporaryDirectory(prefix="vare-regression-calibration-") as temporary:
        temporary_root = Path(temporary)
        baseline_workspace = temporary_root / "baseline"
        fixed_workspace = temporary_root / "fixed"
        checkout_revision(task, baseline_revision, baseline_workspace)
        checkout_revision(task, fixed_revision, fixed_workspace)
        baseline_hashes = source_hashes(baseline_workspace, files)
        fixed_hashes = source_hashes(fixed_workspace, files)
        baseline, baseline_exit = run_grader(evaluator, baseline_workspace, task_root,
                                             temporary_root / "baseline.grade.json")
        fixed, fixed_exit = run_grader(evaluator, fixed_workspace, task_root,
                                       temporary_root / "fixed.grade.json")
        write_json(output / "baseline.grade.json", baseline)
        write_json(output / "fixed.grade.json", fixed)
        for label, workspace in (("baseline-source", baseline_workspace), ("fixed-source", fixed_workspace)):
            for name in files:
                target = output / label / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((workspace / name).read_bytes())
        patch_parts: list[str] = []
        for name in files:
            before = (baseline_workspace / name).read_text(encoding="utf-8").splitlines(keepends=True)
            after = (fixed_workspace / name).read_text(encoding="utf-8").splitlines(keepends=True)
            import difflib
            patch_parts.extend(difflib.unified_diff(before, after, fromfile=f"a/{name}",
                                                    tofile=f"b/{name}"))
        (output / "candidate-fix.patch").write_text("".join(patch_parts), encoding="utf-8")

    expected = protocol["expected_outcomes"]
    observed = {"baseline_passed": baseline["passed"], "fixed_passed": fixed["passed"]}
    if observed != expected:
        write_json(output / "failure.json", {
            "schema_version": 1,
            "task_id": entry["task_id"],
            "status": "failed",
            "expected": expected,
            "observed": observed,
            "baseline_exit_code": baseline_exit,
            "fixed_exit_code": fixed_exit,
        })
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "task_id": entry["task_id"],
        "status": "calibrated" if observed == expected else "failed",
        "baseline_revision": baseline_revision,
        "fixed_revision": fixed_revision,
        "baseline_source_hashes": baseline_hashes,
        "fixed_source_hashes": fixed_hashes,
        "baseline_passed": baseline["passed"],
        "fixed_passed": fixed["passed"],
        "baseline_exit_code": baseline_exit,
        "fixed_exit_code": fixed_exit,
        "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
        "calibration_protocol_sha256": sha256_file(PROTOCOL),
        "calibrator_sha256": sha256_file(Path(__file__).resolve()),
        "claim_limit": lock.get("interpretation_limit", "One frozen deterministic CPU regression; no general error-rate or model-capability claim."),
        "resources": {"gpu_hours": 0, "paid_api_calls": 0, "external_compute_usd": 0},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "created_at_utc": datetime.now(timezone.utc).isoformat()},
    }
    write_json(output / "summary.json", summary)
    write_manifest(output, entry)
    if observed != expected:
        raise RuntimeError(f"calibration outcomes differ for {entry['task_id']}: {observed} vs {expected}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="new directory outside the repository for reconstructed calibration evidence")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        parser.error(f"output root already exists: {output_root}")
    if output_root == ROOT or ROOT.is_relative_to(output_root):
        parser.error("output root cannot contain the repository")
    protocol, entries = protocol_tasks()
    summaries = []
    try:
        for entry in entries:
            summaries.append(calibrate(entry, protocol,
                                       output_root / entry["task_id"]))
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(output_root / "failure.json", {
            "schema_version": 1,
            "protocol_id": protocol.get("protocol_id"),
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        print(f"calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    write_json(output_root / "summary.json", {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "calibrated",
        "task_count": len(summaries),
        "calibration_protocol_sha256": sha256_file(PROTOCOL),
        "calibrator_sha256": sha256_file(Path(__file__).resolve()),
        "evidence_auditor_sha256": sha256_file(ROOT / "scripts/audit_regression_evidence.py"),
        "tasks": summaries,
    })
    write_json(output_root / "manifest.json", {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "files": {
            str(path.relative_to(output_root)): sha256_file(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file() and path != output_root / "manifest.json"
        },
    })
    print(json.dumps({"status": "calibrated", "task_count": len(summaries),
                      "tasks": [{"task_id": row["task_id"], "status": row["status"]}
                                for row in summaries]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
