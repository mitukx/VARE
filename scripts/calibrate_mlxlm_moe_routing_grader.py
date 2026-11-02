#!/usr/bin/env python3
"""Calibrate the pinned MoE grader on baseline, known fix, and wrong mutant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


EDITABLE = (
    "mlx_lm/models/granitemoe.py",
    "mlx_lm/models/granitemoehybrid.py",
    "mlx_lm/models/lfm2_moe.py",
)
BASELINE = "77c33b14373ac70d7abd6f82af15962852adadbb"
KNOWN_FIX = "acb0240394dfc17ffd6db77a3ccecbdbde522ef9"
SELECTED_WRONG_FAILURE = "test_lfm2_moe_backward"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git(root: Path, *args: str) -> str:
    git = "/usr/bin/git" if Path("/usr/bin/git").exists() else shutil.which("git")
    if not git:
        raise RuntimeError("git is unavailable")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    result = subprocess.run(
        [git, "-c", "core.fsmonitor=false", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        timeout=30,
    )
    return result.stdout.strip()


def _verify_snapshot(root: Path, revision: str, hashes: dict[str, str]) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"trusted source snapshot is missing or symlinked: {root}")
    if _git(root, "rev-parse", "HEAD") != revision:
        raise ValueError(f"trusted source snapshot revision mismatch: {root}")
    if _git(root, "rev-parse", "--is-shallow-repository") != "true":
        raise ValueError(f"trusted source snapshot is not shallow: {root}")
    if _git(root, "remote"):
        raise ValueError(f"trusted source snapshot still has a Git remote: {root}")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"trusted source snapshot is not clean: {root}")
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            raise ValueError(f"trusted source snapshot file hash mismatch: {relative}")


def prepare_arm(source_root: Path, known_fix: Path, arm_root: Path, arm: str) -> Path:
    if arm_root.exists():
        raise FileExistsError(f"refusing to overwrite calibration arm: {arm_root}")
    arm_root.mkdir(parents=True)
    workspace = arm_root / "workspace"
    shutil.copytree(source_root, workspace, symlinks=True)
    files: tuple[str, ...]
    if arm == "known-fix":
        files = EDITABLE
    elif arm == "wrong-mutant":
        files = EDITABLE[:2]
    elif arm == "baseline":
        files = ()
    else:
        raise ValueError(f"unsupported calibration arm: {arm}")
    for relative in files:
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(known_fix / relative, destination)
    return workspace


def expected_counts(arm: str) -> dict[str, int]:
    return {
        "baseline": {"passed": 0, "failed": 3, "errors": 0},
        "known-fix": {"passed": 3, "failed": 0, "errors": 0},
        "wrong-mutant": {"passed": 2, "failed": 1, "errors": 0},
    }[arm]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--grader", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--pythonpath", type=Path, required=True)
    args = parser.parse_args()

    task_root = args.task_root.resolve(strict=True)
    prepared_root = args.prepared_root.resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite calibration evidence: {output_root}")
    if task_root in prepared_root.parents or prepared_root in task_root.parents:
        raise SystemExit("task definition and prepared source must be separate trees")
    output_root.mkdir(parents=True)
    manifest_path = prepared_root / "prepared-source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = json.loads(
        (task_root / "protocol.lock.json").read_text(encoding="utf-8")
    )
    if (
        sha256_file(Path(__file__).resolve())
        != protocol["calibration"]["runner_sha256"]
    ):
        raise SystemExit(
            "calibration runner source hash differs from the task protocol"
        )
    if manifest["protocol_lock_sha256"] != sha256_file(
        task_root / "protocol.lock.json"
    ):
        raise SystemExit("prepared source manifest uses a different task protocol")
    if manifest["known_fix_revision"] != KNOWN_FIX:
        raise SystemExit("prepared known-fix revision differs from the task protocol")
    if manifest["candidate_workspace_has_test_file"]:
        raise SystemExit("grader test file is present in the candidate source manifest")
    if Path(manifest["trusted_grader_test"]).parent.joinpath(".git").exists():
        raise SystemExit("grader test directory contains Git metadata")
    source_root = Path(manifest["candidate_workspace"])
    known_fix = Path(manifest["trusted_known_fix_source"])
    trusted_baseline = Path(manifest["trusted_baseline_source"])
    grader_test = Path(manifest["trusted_grader_test"])
    baseline_hashes = manifest["baseline_file_sha256"]
    known_fix_hashes = manifest["known_fix_file_sha256"]
    if baseline_hashes != protocol["source"]["baseline_file_sha256"]:
        raise SystemExit("prepared baseline hashes differ from the task protocol")
    if known_fix_hashes != protocol["source"]["known_fix_file_sha256"]:
        raise SystemExit("prepared known-fix hashes differ from the task protocol")
    if manifest["license_sha256"] != protocol["source"]["license_file_sha256"]:
        raise SystemExit("prepared baseline license differs from the task protocol")
    _verify_snapshot(source_root, BASELINE, baseline_hashes)
    _verify_snapshot(trusted_baseline, BASELINE, baseline_hashes)
    _verify_snapshot(known_fix, KNOWN_FIX, known_fix_hashes)
    if sha256_file(trusted_baseline / "LICENSE") != manifest["license_sha256"]:
        raise SystemExit("trusted baseline license differs from the prepared manifest")
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    expected_arm_order = ("baseline", "known-fix", "wrong-mutant")
    overall_ok = True

    for arm in expected_arm_order:
        arm_root = output_root / "arms" / arm
        workspace = prepare_arm(source_root, known_fix, arm_root, arm)
        grade_output = arm_root / "grade-output"
        command = [
            sys.executable,
            str(args.grader.resolve(strict=True)),
            "--task-root",
            str(task_root),
            "--source-manifest",
            str(manifest_path),
            "--workspace",
            str(workspace),
            "--trusted-baseline",
            str(trusted_baseline),
            "--grader-test",
            str(grader_test),
            "--python",
            str(args.python.resolve(strict=True)),
            "--pythonpath",
            str(args.pythonpath.resolve(strict=True)),
            "--output",
            str(grade_output),
            "--arm",
            arm,
        ]
        process = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=780,
            check=False,
        )
        grade_path = grade_output / "grade.json"
        grade_log = grade_output / "pytest.log"
        attempt: dict[str, Any] = {
            "arm": arm,
            "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
            "launcher_stdout": process.stdout[-4096:],
            "launcher_returncode": process.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        if grade_path.is_file():
            grade = json.loads(grade_path.read_text(encoding="utf-8"))
            actual = {
                key: int(grade["pytest_counts"].get(key, 0))
                for key in ("passed", "failed", "errors")
            }
            arm_ok = actual == expected_counts(arm) and not any(
                (
                    grade["timed_out"],
                    grade["rss_limit_exceeded"],
                    grade["output_limit_exceeded"],
                    grade["monitor_error"],
                )
            )
            arm_ok = arm_ok and grade["passed"] is (arm == "known-fix")
            arm_ok = arm_ok and process.returncode == (0 if arm == "known-fix" else 1)
            if arm == "wrong-mutant":
                log_text = (
                    grade_log.read_text(encoding="utf-8", errors="replace")
                    if grade_log.is_file()
                    else ""
                )
                arm_ok = arm_ok and bool(
                    re.search(r"FAILED .*" + SELECTED_WRONG_FAILURE, log_text)
                )
            attempt.update(
                {
                    "grade_json_sha256": sha256_file(grade_path),
                    "pytest_log_sha256": sha256_file(grade_log)
                    if grade_log.is_file()
                    else None,
                    "pytest_counts": actual,
                    "grader_status": grade["grader_status"],
                    "arm_calibration_passed": arm_ok,
                }
            )
        else:
            arm_ok = False
            attempt.update(
                {
                    "arm_calibration_passed": False,
                    "failure": "grader produced no structured result",
                }
            )
        records.append(attempt)
        if not arm_ok:
            overall_ok = False
            # A broken baseline or known-fix arm makes later results uninterpretable.
            break

    record = {
        "schema_version": 1,
        "task_id": "mlxlm-moe-routing-backward-train-01",
        "status": "CALIBRATION_PASS" if overall_ok and len(records) == 3 else "NO_GO",
        "required_arm_order": list(expected_arm_order),
        "completed_arms": [item["arm"] for item in records],
        "arms": records,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "prepared_source_manifest_sha256": sha256_file(manifest_path),
        "protocol_lock_sha256": sha256_file(task_root / "protocol.lock.json"),
        "grader_source_sha256": sha256_file(args.grader.resolve(strict=True)),
        "test_source_sha256": sha256_file(grader_test),
    }
    write_json(output_root / "calibration.json", record)
    print(
        json.dumps(
            {
                "task_id": record["task_id"],
                "status": record["status"],
                "completed_arms": record["completed_arms"],
                "elapsed_seconds": record["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if record["status"] == "CALIBRATION_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
