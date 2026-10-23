#!/usr/bin/env python3
"""Run the frozen matched greedy/sampled MLX-LM #1910 feasibility study."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observe() -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import monitor_local_agent_resources_v1 as monitor
    return monitor.observe()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve(strict=True)
    python_exe = args.python.absolute()
    if not python_exe.is_file():
        parser.error("pinned Python executable is missing")
    protocol_path = args.protocol.resolve(strict=True)
    lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite experiment evidence: {output}")
    if lock.get("status") != "FROZEN_NOT_RUN":
        parser.error("protocol is not frozen and not run")
    sidecar = protocol_path.with_name("protocol.sha256").read_text().split()[0]
    if sha256(protocol_path) != sidecar:
        parser.error("protocol SHA-256 sidecar mismatch")
    for relative, expected in lock["code_sha256"].items():
        if sha256(repo / relative) != expected:
            parser.error(f"locked script hash mismatch: {relative}")
    for relative, expected in lock["task"]["task_bundle_sha256"].items():
        if sha256(repo / relative) != expected:
            parser.error(f"locked task bundle hash mismatch: {relative}")
    model_manifest = protocol_path.parent / "model-manifest.json"
    # Agent protocol v1-v3 share a manifest stored beside the first lock.
    if not model_manifest.exists():
        model_manifest = protocol_path.parent.parent / "mlsys-repo-task-mlx-lm-1910-agent-v1" / "model-manifest.json"
    if sha256(model_manifest) != lock["model"]["manifest_sha256"]:
        parser.error("model manifest hash mismatch")
    manifest = json.loads(model_manifest.read_text(encoding="utf-8"))
    model_dir = args.model_path.resolve(strict=True)
    for name, identity in manifest["files"].items():
        file_path = model_dir / name
        if not file_path.is_file() or file_path.stat().st_size != identity["size_bytes"] or sha256(file_path) != identity["sha256"]:
            parser.error(f"model checkpoint file mismatch: {name}")
    dep_lock = repo / lock["runtime"]["dependency_lock_path"]
    if sha256(dep_lock) != lock["runtime"]["dependency_lock_sha256"]:
        parser.error("MLX dependency lock hash mismatch")
    package_probe = subprocess.run(
        [str(python_exe), "-c",
         "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ('mlx-lm','mlx','transformers')}))"],
        capture_output=True, text=True, check=False,
    )
    if package_probe.returncode or json.loads(package_probe.stdout) != {"mlx-lm": "0.32.0", "mlx": "0.32.4", "transformers": "5.20.2"}:
        parser.error("runtime package version mismatch")
    if lock["grader"]["grader_sha256"] != sha256(repo / "scripts/grade_mlx_lm_1910_candidate_v1.py"):
        parser.error("grader identity mismatch")
    calibration = protocol_path.parent / "grader-calibration" / "calibration.json"
    if not calibration.exists():
        calibration = protocol_path.parent.parent / "mlsys-repo-task-mlx-lm-1910-agent-v3" / "grader-calibration" / "calibration.json"
    calibrated = json.loads(calibration.read_text(encoding="utf-8"))
    if (calibrated.get("status") != "CALIBRATION_PASS_DEVELOPMENT_ONLY"
            or sha256(calibration) != lock["grader"]["calibration_result_sha256"]):
        parser.error("candidate grader has not passed baseline/fix/mutant calibration")
    if args.model_path.resolve(strict=True).name != lock["model"]["revision"]:
        parser.error("model snapshot revision mismatch")

    output.mkdir(parents=True)
    started = time.monotonic()
    preflight_resources = observe()
    if (preflight_resources["swap_used_mib"] >= lock["resources"]["swap_used_abort_mib"]
            or preflight_resources["memory_free_percent"] < lock["resources"]["memory_free_abort_percent"]):
        blocked = {"schema_version": 1, "protocol_id": lock["protocol_id"], "status": "BLOCKED",
                   "reason": "resource_abort_threshold_exceeded_before_model_load",
                   "resource_before_model_load": preflight_resources, "attempts": [],
                   "model_loaded": False, "elapsed_seconds": round(time.monotonic() - started, 3)}
        (output / "summary.json").write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"status": "BLOCKED", "model_loaded": False}))
        return 3
    task_root = repo / "benchmarks/mlsys_repo_eval_v3/mlxlm_1910_agent_task"
    agent_runner = repo / "scripts/run_local_agent.py"
    monitor_runner = repo / "scripts/monitor_local_agent_resources_v1.py"
    grader_runner = repo / "scripts/grade_mlx_lm_1910_candidate_v1.py"
    arms = []
    abort_all = False
    for arm in lock["arms"]:
        if abort_all:
            arms.append({"arm": arm["arm"], "status": "NOT_RUN", "reason": "earlier resource abort"})
            continue
        attempt = output / arm["arm"]
        attempt.mkdir()
        workspace = attempt / "workspace"
        shutil.copytree(task_root / "workspace", workspace, symlinks=False)
        arm_started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        start_resources = observe()
        command = [str(python_exe), str(agent_runner),
                   "--workspace", str(workspace), "--model-path", str(args.model_path.resolve(strict=True)),
                   "--model-repository", lock["model"]["repository"], "--backend", "mlx",
                   "--public-test-image", lock["runtime"]["public_test_image"],
                   "--task-root", str(task_root), "--output", str(attempt / "agent.json"),
                   "--instruction", lock["agent"]["instruction"], "--seed", str(arm["seed"]),
                   "--max-turns", str(lock["agent"]["turn_budget"]),
                   "--max-new-tokens-per-turn", str(lock["agent"]["max_new_tokens_per_turn"]),
                   "--max-total-new-tokens", str(lock["agent"]["max_total_new_tokens"]),
                   "--max-context-tokens", str(lock["agent"]["max_context_tokens"]),
                   "--wall-seconds", str(lock["agent"]["wall_seconds_per_attempt"]),
                   "--temperature", str(arm["temperature"]), "--top-p", str(arm["top_p"]),
                   "--top-k", str(arm["top_k"]), "--edit-format", lock["agent"]["edit_format"]]
        command.append("--do-sample" if arm["do_sample"] else "--no-do-sample")
        monitor_cmd = [str(python_exe), str(monitor_runner),
                       "--output", str(attempt / "process"), "--max-seconds", "600",
                       "--swap-abort-mib", "2900", "--memory-free-abort-percent", "15", "--", *command]
        process = subprocess.run(monitor_cmd, cwd=repo, capture_output=True, text=True, check=False)
        (attempt / "launcher.stdout.txt").write_text(process.stdout)
        (attempt / "launcher.stderr.txt").write_text(process.stderr)
        monitor_result = json.loads((attempt / "process/resource-monitor.json").read_text())
        end_resources = observe()
        hashes_after = {"qwen3_coder.py": sha256(workspace / "qwen3_coder.py"),
                        "test_public.py": sha256(workspace / "test_public.py")}
        task_locked = lock["task"]["task_bundle_sha256"]
        public_test_expected = task_locked[
            "benchmarks/mlsys_repo_eval_v3/mlxlm_1910_agent_task/workspace/test_public.py"
        ]
        if hashes_after["test_public.py"] != public_test_expected:
            abort_all = True
            grade_error = "public test hash changed in candidate workspace"
        else:
            grade_output = attempt / "independent-grade"
            grade_cmd = [str(python_exe), str(grader_runner),
                         "--git-source", str(args.git_source.resolve(strict=True)),
                         "--candidate-source", str(workspace / "qwen3_coder.py"),
                         "--output", str(grade_output)]
            graded = subprocess.run(grade_cmd, cwd=repo, capture_output=True, text=True, check=False)
            (attempt / "grader-launcher.stdout.txt").write_text(graded.stdout)
            (attempt / "grader-launcher.stderr.txt").write_text(graded.stderr)
            grade_error = None if (grade_output / "grade.json").is_file() else "candidate grader did not emit grade.json"
            if grade_error:
                abort_all = True
        agent_record = None
        if (attempt / "agent.json").is_file():
            agent_record = json.loads((attempt / "agent.json").read_text())
        grade_record = None
        if (attempt / "independent-grade/grade.json").is_file():
            grade_record = json.loads((attempt / "independent-grade/grade.json").read_text())
        entry = {"arm": arm["arm"], "status": "COMPLETED" if monitor_result["status"] == "COMPLETED" and grade_record else monitor_result["status"],
                 "seed": arm["seed"], "started_at_utc": arm_started_utc,
                 "monitor": monitor_result, "grader_error": grade_error,
                 "resource_before": start_resources, "resource_after": end_resources,
                 "candidate_source_sha256": hashes_after["qwen3_coder.py"],
                 "agent_metrics": None if agent_record is None else {k: agent_record.get(k) for k in
                     ["stop_reason", "elapsed_seconds", "new_tokens", "tool_turns", "edit_attempted", "edit_applied",
                      "format_parse_failure_count", "public_test_result", "max_rss_bytes", "runtime", "candidate_diff_sha256"]},
                 "independent_grade_status": None if grade_record is None else grade_record["status"]}
        (attempt / "attempt-summary.json").write_text(json.dumps(entry, indent=2, sort_keys=True) + "\n")
        arms.append(entry)
        if monitor_result["status"] == "ABORTED_RESOURCE":
            abort_all = True
    accepted = sum(1 for arm in arms if arm.get("independent_grade_status") == "PASS")
    completed = sum(1 for arm in arms if arm.get("status") == "COMPLETED")
    status = "COMPLETED" if len(arms) == 2 and completed == 2 else "PARTIAL" if completed or accepted else "FAILED"
    summary = {"schema_version": 1, "protocol_id": lock["protocol_id"], "status": status,
               "attempts": arms, "unique_development_tasks": 1,
               "resource_before_model_load": preflight_resources,
               "accepted_attempts": accepted, "attempt_denominator": len([x for x in arms if x.get("status") != "NOT_RUN"]),
               "unique_task_successes": 1 if accepted else 0, "unique_task_denominator": 1,
               "elapsed_seconds": round(time.monotonic() - started, 3),
               "claim_limit": "Two attempts on one Development task; not an estimate of held-out task success or learning improvement."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "accepted_attempts": accepted,
                      "attempt_denominator": summary["attempt_denominator"],
                      "unique_task_successes": summary["unique_task_successes"],
                      "unique_task_denominator": 1}))
    return 0 if status == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
