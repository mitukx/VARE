#!/usr/bin/env python3
"""Calibrate candidate-import attribution and runtime aborts under Colima."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_local_agent import (  # noqa: E402
    classify_public_test_exit,
    run_public_tests,
)

PROTOCOL = ROOT / "experiments/coding_agent_phase_a_v6/public-test-execution-v9.json"
LOCK = ROOT / "experiments/coding_agent_phase_a_v6/public-test-execution-v9.lock.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen() -> tuple[dict, dict]:
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    lock = json.loads(LOCK.read_bytes())
    if hashlib.sha256(protocol_bytes).hexdigest() != lock.get("protocol_sha256"):
        raise ValueError("v9 protocol hash mismatch")
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("v9 protocol identity mismatch")
    for relative, expected in lock["implementation_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"v9 implementation hash mismatch: {relative}")
    return protocol, lock


def image_id(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        capture_output=True, text=True, check=True, timeout=20,
        env={**os.environ, "DOCKER_CONTEXT": "colima"},
    )
    return result.stdout.strip()


def run(output: Path) -> dict:
    protocol, lock = load_frozen()
    if os.environ.get("DOCKER_CONTEXT") != "colima":
        raise RuntimeError("DOCKER_CONTEXT=colima is required")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite calibration output: {output}")
    image = protocol["image"]
    actual_image_id = image_id(image)
    if actual_image_id != protocol["image_id"]:
        raise RuntimeError("local image differs from the v9 pinned image identity")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vare-v9-calibration-") as root_text:
        root = Path(root_text)
        (root / "src").mkdir()
        (root / "test_public.py").write_text(
            protocol["public_test"], encoding="utf-8",
        )
        (root / "src/value.py").write_text(
            protocol["source_variants"]["broken_baseline"], encoding="utf-8",
        )
        expected_hashes = {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file()
        }
        expected_files = set(expected_hashes)
        baseline = run_public_tests(
            root, image, expected_hashes["test_public.py"],
            expected_workspace_files=expected_files,
            expected_file_hashes=expected_hashes,
            allowed_modified_files={"src/value.py"},
        )
        (root / "src/value.py").write_text(
            protocol["source_variants"]["known_fix"], encoding="utf-8",
        )
        known_fix = run_public_tests(
            root, image, expected_hashes["test_public.py"],
            expected_workspace_files=expected_files,
            expected_file_hashes=expected_hashes,
            allowed_modified_files={"src/value.py"},
        )

    classifier_cases = {
        "candidate_virtual_final_frame": classify_public_test_exit(
            1, 'Traceback (most recent call last):\n  File "/workspace/src/value.py", line 1\n'
            '  File "<string>", line 1\nModuleNotFoundError: candidate_dep',
        ),
        "candidate_frame_after_evaluator_import_frame": classify_public_test_exit(
            1, 'Traceback (most recent call last):\n  File "/workspace/test_public.py", line 1\n'
            '  File "/workspace/src/value.py", line 2\n  File "<string>", line 1\n'
            'ModuleNotFoundError: candidate_dep',
        ),
        "evaluator_only_virtual_final_frame": classify_public_test_exit(
            1, 'Traceback (most recent call last):\n  File "/workspace/test_public.py", line 1\n'
            '  File "<string>", line 1\nModuleNotFoundError: grader_dep',
        ),
    }
    if classifier_cases != {
        "candidate_virtual_final_frame": "test_execution_error",
        "candidate_frame_after_evaluator_import_frame": "test_execution_error",
        "evaluator_only_virtual_final_frame": "evaluator_unavailable",
    }:
        raise RuntimeError("v9 missing-dependency attribution differs from protocol")

    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    runner_test = subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         "tests/test_run_local_agent_structured_v2.py",
         "-k", "test_real_runner_loop_stops_after_evaluator_failure"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90,
    )
    if runner_test.returncode != 0:
        raise RuntimeError("actual runner-loop abort regression failed: " + runner_test.stdout[-2000:] + runner_test.stderr[-2000:])
    if not (
        baseline.get("classification") == "assertion_failure"
        and baseline.get("passed") is False
        and baseline.get("gradeable") is True
        and known_fix.get("classification") == "passed"
        and known_fix.get("passed") is True
        and known_fix.get("gradeable") is True
    ):
        raise RuntimeError("v9 baseline/fix calibration did not satisfy the frozen contract")

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True, timeout=10,
    ).stdout
    record = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": lock["protocol_sha256"],
        "git_commit": commit,
        "git_worktree_clean_before_run": not bool(dirty),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": sys.version,
        "docker_context": "colima",
        "image": image,
        "image_id": actual_image_id,
        "implementation_sha256": lock["implementation_sha256"],
        "results": {"broken_baseline": baseline, "known_fix": known_fix},
        "missing_dependency_classification": classifier_cases,
        "runtime_abort_regression": {
            "command": [sys.executable, "-m", "pytest", "-q",
                        "tests/test_run_local_agent_structured_v2.py",
                        "-k", "test_real_runner_loop_stops_after_evaluator_failure"],
            "returncode": runner_test.returncode,
            "stdout_tail": runner_test.stdout[-2000:],
            "stderr_tail": runner_test.stderr[-2000:],
            "passed": runner_test.returncode == 0,
        },
        "wall_seconds": round(time.monotonic() - started, 3),
        "model_loaded": False,
        "optimizer_updates": 0,
        "limitations": [
            "Synthetic fixture calibration, not a repository task or model rollout.",
            "Runtime-abort claim is verified by the runner-loop regression test, not inferred from a classifier predicate.",
            "HF Development candidates in PR #107 were previously model-exposed and are not fresh evidence.",
        ],
        "calibration_passed": True,
    }
    output.mkdir(parents=True)
    target = output / "calibration.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(target)}  calibration.json\n", encoding="utf-8",
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = run(args.output)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"calibration_passed": record["calibration_passed"],
                      "image_id": record["image_id"],
                      "wall_seconds": record["wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
