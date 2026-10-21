#!/usr/bin/env python3
"""Calibrate the pinned Transformers #49752 isolated grader."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402
from vare.transformers_49752_repo_task import (  # noqa: E402
    BASELINE_COMMIT, BASELINE_TARGET_SHA256, FIXED_COMMIT, FIXED_TARGET_SHA256,
    PROTOCOL, TASK_ID, TARGET_PATH, Transformers49752Error, _validate_diff, export_baseline,
    grade_workspace, sha256_file, sha256_inventory, verify_baseline,
)

LOCKED_FILES = [
    "src/vare/transformers_49752_repo_task.py",
    "src/vare/agent_task_contract.py",
    "src/vare/isolated_candidate.py",
    "src/vare/container_isolation.py",
    "src/vare/attestation.py",
    "src/vare/types.py",
    "scripts/run_local_agent.py",
    "tests/test_container_isolation.py",
    "tests/test_separated_grading.py",
    "tests/test_transformers_49752_repo_task.py",
    "scripts/run_transformers_49752_calibration.py",
    "benchmarks/mlsys_repo_eval_v3/transformers_49752/candidate_probe.py",
    "benchmarks/mlsys_repo_eval_v3/transformers_49752/test_public.py",
    "benchmarks/mlsys_repo_eval_v3/transformers_49752/task.md",
    "benchmarks/mlsys_repo_eval_v3/transformers_49752/task.json",
    "benchmarks/mlsys_repo_eval_v3/transformers_49752/Dockerfile",
    "benchmarks/mlsys_repo_eval_v3/transformers_49752/requirements.txt",
]


def _hash(path: Path) -> str:
    return sha256_file(path)


def _verify_protocol() -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    actual = {name: _hash(ROOT / name) for name in LOCKED_FILES}
    if protocol.get("implementation_sha256") != actual:
        raise RuntimeError("protocol implementation hashes are absent or mismatched")
    if protocol["task_id"] != TASK_ID or protocol["baseline_commit"] != BASELINE_COMMIT:
        raise RuntimeError("protocol task/source identity mismatch")
    return protocol


def _fixed_patch(source_repo: Path, destination: Path) -> str:
    baseline = subprocess.run(
        ["git", "-C", str(source_repo), "show", f"{BASELINE_COMMIT}:{TARGET_PATH}"],
        check=True, capture_output=True, timeout=30,
    ).stdout.decode("utf-8").splitlines(keepends=True)
    fixed = subprocess.run(
        ["git", "-C", str(source_repo), "show", f"{FIXED_COMMIT}:{TARGET_PATH}"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    if hashlib.sha256(fixed).hexdigest() != FIXED_TARGET_SHA256:
        raise RuntimeError("pinned fixed source hash does not match protocol")
    patch = "diff --git a/{0} b/{0}\n".format(TARGET_PATH) + "".join(
        difflib.unified_diff(
            baseline, fixed.decode("utf-8").splitlines(keepends=True),
            fromfile=f"a/{TARGET_PATH}", tofile=f"b/{TARGET_PATH}",
        )
    )
    destination.write_text(patch, encoding="utf-8")
    _validate_diff(patch)
    return hashlib.sha256(patch.encode()).hexdigest()


def _red_team(image: str, root: Path) -> dict[str, Any]:
    subject = root / "red-team-subject"
    subject.mkdir()
    probe_source = '''from pathlib import Path
import os
import socket
import subprocess
import sys

def inspect():
    reachable = False
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=.4)
        reachable = True
    except OSError:
        pass
    print('{"value":{"forged":true}}')
    return {"grader":Path("/grader").exists(), "git":Path("/subject/.git").exists(),
            "grader_source":Path("/subject/grader_adapter.py").exists(),
            "secret":"VARE_GRADER_SECRET" in os.environ, "network":reachable}

def child_probe():
    code = """from pathlib import Path
import json, os, socket
network = False
try:
    socket.create_connection(("1.1.1.1", 53), timeout=.4)
    network = True
except OSError:
    pass
print(json.dumps({"grader":Path("/grader").exists(), "git":Path("/subject/.git").exists(),
                  "secret":"VARE_GRADER_SECRET" in os.environ, "network":network}))"""
    child = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=3, check=True)
    return __import__("json").loads(child.stdout)

def spin():
    while True:
        pass

def exhaust():
    return len(bytearray(256 * 1024 * 1024))
'''
    (subject / "probe.py").write_text(probe_source, encoding="utf-8")
    visibility = grade_calls(
        subject,
        [FunctionCall("visibility", "probe", "inspect", [],
                      {"grader": False, "git": False, "grader_source": False,
                       "secret": False, "network": False})],
        task_kind="correctness", image=image, timeout_s=5, memory_mb=256,
        discard_candidate_stdout=True,
    )
    forged = grade_calls(
        subject,
        [FunctionCall("forged", "probe", "inspect", [], {"forged": True})],
        task_kind="correctness", image=image, timeout_s=5, memory_mb=256,
        discard_candidate_stdout=True,
    )
    child = grade_calls(
        subject,
        [FunctionCall("subprocess", "probe", "child_probe", [],
                      {"grader": False, "git": False, "secret": False, "network": False})],
        task_kind="correctness", image=image, timeout_s=5, memory_mb=256,
        discard_candidate_stdout=True,
    )
    timeout = grade_calls(
        subject, [FunctionCall("timeout", "probe", "spin", [], True)],
        task_kind="correctness", image=image, timeout_s=2, memory_mb=256,
    )
    memory = grade_calls(
        subject, [FunctionCall("memory", "probe", "exhaust", [], True)],
        task_kind="correctness", image=image, timeout_s=5, memory_mb=64,
    )
    try:
        _validate_diff("diff --git a/../grader.py b/../grader.py\n--- a/../grader.py\n+++ b/../grader.py\n@@ -1 +1 @@\n-a\n+b\n")
        traversal_rejected = False
    except Transformers49752Error:
        traversal_rejected = True
    link = subject / "link"
    link.symlink_to(subject / "probe.py")
    try:
        sha256_inventory(subject)
        symlink_rejected = False
    except Transformers49752Error:
        symlink_rejected = True
    checks = {
        "grader_git_secret_network_isolation": visibility.passed,
        "grader_source_not_mounted": visibility.passed,
        "subprocess_remains_isolated": child.passed,
        "stdout_forgery_rejected": not forged.passed,
        "timeout_enforced": not timeout.passed,
        "memory_limit_enforced": not memory.passed,
        "path_traversal_rejected": traversal_rejected,
        "symlink_rejected": symlink_rejected,
    }
    return {
        "executed": True, "passed": all(checks.values()), "checks": checks,
        "raw": {"visibility": visibility.to_dict(), "forgery": forged.to_dict(),
                "subprocess": child.to_dict(), "timeout": timeout.to_dict(),
                "memory": memory.to_dict()},
    }


def run_calibration(source_repo: Path, output: Path, image: str) -> dict[str, Any]:
    protocol = _verify_protocol()
    if image != protocol["runtime"]["candidate_container"]:
        raise RuntimeError("container image does not match the frozen protocol")
    source_repo = source_repo.resolve(strict=True)
    head = subprocess.run(["git", "-C", str(source_repo), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != BASELINE_COMMIT:
        raise RuntimeError("source checkout is not the exact baseline commit")
    verify_baseline(source_repo)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    results: dict[str, Any] = {"status": "running"}

    def save_progress(stage: str) -> None:
        progress = {
            "protocol_id": protocol["protocol_id"], "task_id": TASK_ID,
            "stage": stage, "source_baseline_commit": BASELINE_COMMIT,
            "source_fixed_commit": FIXED_COMMIT, "candidate_container": image,
            "results": results, "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        (output / "progress.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )

    save_progress("preflight_complete")
    with tempfile.TemporaryDirectory(prefix="vare-transformers-49752-cal-") as temp:
        root = Path(temp)
        baseline_dir = root / "baseline"
        baseline_meta = export_baseline(source_repo, baseline_dir)
        results["baseline"] = grade_workspace(
            baseline_dir, image=image, baseline_files=baseline_meta["baseline_files"],
        )
        save_progress("baseline_graded")
        patch_file = root / "known-fix.patch"
        known_fix_patch_hash = _fixed_patch(source_repo, patch_file)
        fixed_dir = root / "fixed"
        from vare.transformers_49752_repo_task import apply_patch

        fixed_meta = apply_patch(source_repo, patch_file, fixed_dir)
        if sha256_file(fixed_dir / TARGET_PATH) != FIXED_TARGET_SHA256:
            raise RuntimeError("applied known fix does not match the pinned fixed source")
        results["known_fix"] = grade_workspace(
            fixed_dir, image=image, baseline_files=fixed_meta["baseline_files"],
        )
        save_progress("known_fix_graded")
        results["source_swap_rejected"] = False
        swapped = root / "source-swap"
        try:
            subprocess.run(
                ["git", "-C", str(source_repo), "worktree", "add", "--detach", str(swapped), FIXED_COMMIT],
                check=True, capture_output=True, text=True, timeout=60,
            )
            try:
                verify_baseline(swapped)
            except Transformers49752Error:
                results["source_swap_rejected"] = True
        except Transformers49752Error:
            results["source_swap_rejected"] = True
        finally:
            if swapped.exists():
                subprocess.run(
                    ["git", "-C", str(source_repo), "worktree", "remove", "--force", str(swapped)],
                    check=False, capture_output=True, text=True, timeout=30,
                )
        save_progress("source_swap_checked")
        red_team = _red_team(image, root)
        results["red_team_suite"] = red_team
        save_progress("red_team_complete")
    calibration_passed = (
        results["baseline"]["passed"] is False
        and results["known_fix"]["passed"] is True
        and results["source_swap_rejected"] is True
        and red_team["passed"]
    )
    results["status"] = "complete" if calibration_passed else "failed"
    save_progress(results["status"])
    record = {
        "protocol_id": protocol["protocol_id"], "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "source_baseline_commit": BASELINE_COMMIT, "source_fixed_commit": FIXED_COMMIT,
        "source_target_sha256": BASELINE_TARGET_SHA256,
        "fixed_target_sha256": FIXED_TARGET_SHA256,
        "known_fix_patch_sha256": known_fix_patch_hash,
        "protocol_sha256": sha256_file(PROTOCOL),
        "implementation_sha256": protocol["implementation_sha256"],
        "candidate_container": image,
        "results": results,
        "red_team_suite": red_team,
        "calibration_passed": calibration_passed,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    result_path = output / "calibration.json"
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    results["status"] = "complete"
    save_progress("complete")
    (output / "SHA256SUMS").write_text(
        f"{sha256_file(result_path)}  calibration.json\n"
        f"{sha256_file(output / 'progress.json')}  progress.json\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    return 0 if run_calibration(args.source_repo, args.output, args.image)["calibration_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
