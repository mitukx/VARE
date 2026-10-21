#!/usr/bin/env python3
"""Reproduce the pinned MLX-LM #1904 baseline/fix grader calibration."""

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
from typing import Any

from vare.mlsys_repo_task import (
    BASELINE_COMMIT,
    BASELINE_SERVER_SHA256,
    FIXED_COMMIT,
    export_baseline,
    grade_workspace,
    prepare_workspace,
    sha256_file,
    verify_baseline,
    verify_protocol_implementation,
    verify_red_team_evidence,
    verify_protocol_image,
    verify_no_index_flags,
    CALIBRATION_INPUT_FILES,
    calibration_accepts,
    RepositoryTaskError,
    verify_task_prompt,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--docker-context", default=None)
    parser.add_argument(
        "--image",
        default="python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258",
    )
    args = parser.parse_args()
    if args.docker_context:
        os.environ["DOCKER_CONTEXT"] = args.docker_context
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    patch = Path(__file__).resolve().parents[1] / (
        "benchmarks/mlsys_repo_eval_v1/mlxlm_1904_tool_arguments/known_fix.patch"
    )
    protocol = patch.parent / "protocol-v8.json"
    task = patch.parent / "task.md"
    started = time.monotonic()
    implementation_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    implementation_worktree_clean = not subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if not implementation_worktree_clean:
        raise SystemExit(
            "calibration requires a clean tracked worktree containing the frozen protocol and evaluator bytes"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol_data = json.loads(protocol.read_text(encoding="utf-8"))
    if protocol_data.get("protocol_id") != "mlsys-repository-task-mlxlm-1904-v8":
        raise SystemExit("calibration requires the v8 protocol")
    try:
        verify_no_index_flags(ROOT, CALIBRATION_INPUT_FILES)
    except RepositoryTaskError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        verify_protocol_image(protocol_data, args.image)
    except RepositoryTaskError as exc:
        raise SystemExit(str(exc)) from exc
    if protocol_data["source"]["baseline_server_sha256"] != BASELINE_SERVER_SHA256:
        raise SystemExit("protocol baseline source hash disagrees with the grader")
    try:
        verify_task_prompt(protocol_data, task)
    except RepositoryTaskError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        verified_implementation = verify_protocol_implementation(protocol_data, ROOT)
    except RepositoryTaskError as exc:
        raise SystemExit(str(exc)) from exc
    red_team_script = ROOT / "scripts/red_team_mlxlm_1904.py"
    red_team_output = output.with_name(output.stem + ".red-team.json")
    if red_team_output.exists() or red_team_output.is_symlink():
        raise SystemExit(f"refusing to overwrite red-team evidence {red_team_output}")
    red_team_command = [
        sys.executable,
        str(red_team_script),
        "--source-root",
        str(args.source_root.resolve()),
        "--output",
        str(red_team_output),
        "--image",
        args.image,
    ]
    if args.docker_context:
        red_team_command.extend(["--docker-context", args.docker_context])
    red_team_exit = 127
    red_team_error = None
    try:
        red_team = subprocess.run(
            red_team_command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        red_team_exit = red_team.returncode
        if red_team_exit != 0:
            red_team_error = (red_team.stderr or red_team.stdout)[-4000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        red_team_error = f"{type(exc).__name__}: {exc}"
    try:
        red_team_evidence = verify_red_team_evidence(red_team_output, image=args.image)
        red_team_verified = red_team_exit == 0
    except RepositoryTaskError as exc:
        red_team_evidence = None
        red_team_verified = False
        red_team_error = str(exc)
    arms: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1904-calibration-") as folder:
        root = Path(folder)
        swapped_checkout = root / "fixed-source-swap"
        subprocess.run(
            [
                "git",
                "-C",
                str(args.source_root),
                "worktree",
                "add",
                "--detach",
                str(swapped_checkout),
                FIXED_COMMIT,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        try:
            verify_baseline(swapped_checkout)
            fixed_swap_rejected = False
        except RepositoryTaskError:
            fixed_swap_rejected = True
        finally:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(args.source_root),
                    "worktree",
                    "remove",
                    "--force",
                    str(swapped_checkout),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
        baseline_workspace = root / "baseline"
        fixed_workspace = root / "fixed"
        arms["baseline"] = {
            "source_revision": BASELINE_COMMIT,
            "workspace": export_baseline(args.source_root, baseline_workspace),
            "grade": grade_workspace(baseline_workspace, image=args.image),
        }
        arms["fixed"] = {
            "source_revision": FIXED_COMMIT,
            "workspace": prepare_workspace(args.source_root, patch, fixed_workspace),
            "grade": grade_workspace(fixed_workspace, image=args.image),
        }
        expected_fixed_source = subprocess.run(
            [
                "git",
                "-C",
                str(args.source_root),
                "show",
                f"{FIXED_COMMIT}:mlx_lm/server.py",
            ],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        expected_fixed_sha256 = hashlib.sha256(expected_fixed_source).hexdigest()
        arms["fixed"]["matches_pinned_fixed_source"] = (
            arms["fixed"]["workspace"]["candidate_source_sha256"]
            == expected_fixed_sha256
        )
    elapsed = time.monotonic() - started
    docker_version = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}/{{.Server.Arch}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    result = {
        "schema_version": 1,
        "experiment_id": "mlsys-repository-task-mlxlm-1904-calibration-v1",
        "implementation_commit": implementation_commit,
        "implementation_worktree_clean_at_start": implementation_worktree_clean,
        "protocol_sha256": sha256_file(protocol),
        "verified_implementation_sha256": verified_implementation,
        "task_sha256": sha256_file(task),
        "patch_sha256": sha256_file(patch),
        "red_team_command_exit_code": red_team_exit,
        "red_team_evidence": red_team_evidence,
        "red_team_verified": red_team_verified,
        "red_team_error": red_team_error,
        "host": {"system": platform.platform(), "machine": platform.machine()},
        "python": sys.version,
        "docker_context": os.environ.get("DOCKER_CONTEXT", "default"),
        "docker_server": docker_version,
        "container_image": args.image,
        "resource_limits": {
            "memory_mb": 512,
            "cpus": 1.0,
            "pids_limit": 64,
            "timeout_s": 5,
        },
        "configured_memory_limit_bytes": 512 * 1024 * 1024,
        "peak_memory_bytes": None,
        "elapsed_wall_s": elapsed,
        "arms": arms,
        "integrity_probes": {"fixed_source_swap_rejected": fixed_swap_rejected},
        "decision": {
            "baseline_failed": arms["baseline"]["grade"]["passed"] is False,
            "known_fix_passed": arms["fixed"]["grade"]["passed"] is True,
            "known_fix_matches_pinned_revision": arms["fixed"][
                "matches_pinned_fixed_source"
            ],
            "fixed_source_swap_rejected": fixed_swap_rejected,
            "calibration_passed": calibration_accepts(
                baseline_failed=arms["baseline"]["grade"]["passed"] is False,
                known_fix_passed=arms["fixed"]["grade"]["passed"] is True,
                fixed_source_matches=arms["fixed"]["matches_pinned_fixed_source"]
                is True,
                source_swap_rejected=fixed_swap_rejected is True,
                red_team_verified=red_team_verified is True,
                implementation_worktree_clean=implementation_worktree_clean is True,
            ),
            "evidence_level": "E1 narrow historical function-contract calibration",
        },
        "limitations": [
            "The original upstream test starts the HTTP/model-serving fixture; this isolated adapter grades only the pure message-normalization function contract.",
            "This calibration uses a known fix, not an agent-generated patch or model rollout.",
            "Docker memory is capped at the recorded limit; peak container memory was not measured.",
        ],
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"output": str(output), "decision": result["decision"]}, sort_keys=True
        )
    )
    return 0 if result["decision"]["calibration_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
