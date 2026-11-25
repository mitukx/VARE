#!/usr/bin/env python3
"""Run a frozen, nonblind diagnostic of model progress and post-edit control."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_local_agent.py"
GRADER = ROOT / "scripts" / "sprint_viii_synthetic_grader_v1.py"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise OSError("short write while retaining experiment evidence")
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def resource_snapshot(path: Path, pid: int | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    rss = 0
    if pid is not None:
        try:
            root = psutil.Process(pid)
            rss = sum(
                child.memory_info().rss
                for child in [root, *root.children(recursive=True)]
                if child.is_running()
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss = 0
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "available_memory_bytes": memory.available,
        "memory_total_bytes": memory.total,
        "swap_used_bytes": swap.used,
        "swap_free_bytes": max(0, swap.total - swap.used),
        "swap_total_bytes": swap.total,
        "disk_free_bytes": shutil.disk_usage(path).free,
        "process_tree_rss_bytes": rss,
    }


def stop_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        process.terminate()
        _gone, alive = psutil.wait_procs(children, timeout=5)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def grade_candidate(workspace: Path, attempt_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable, "-I", str(GRADER), "--workspace-root", str(workspace),
        "--source", "subject.py",
    ]
    result = subprocess.run(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=20, check=False,
    )
    write_new(attempt_dir / "grader.stdout.log", result.stdout.encode())
    write_new(attempt_dir / "grader.stderr.log", result.stderr.encode())
    try:
        receipt = json.loads(result.stdout) if result.returncode == 0 else None
    except json.JSONDecodeError:
        receipt = None
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_sha256": sha256_bytes(result.stdout.encode()),
        "stderr_sha256": sha256_bytes(result.stderr.encode()),
        "grader_sha256": sha256_file(GRADER),
        "receipt": receipt,
    }


def run_attempt(
    *, arm: dict[str, Any], protocol: dict[str, Any], protocol_sha: str,
    attempt_dir: Path, model_dir: Path, task_root: Path,
) -> dict[str, Any]:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    workspace = attempt_dir / "workspace"
    shutil.copytree(Path(protocol["task"]["workspace_template_path"]), workspace)
    baseline = workspace / "subject.py"
    expected_baseline = protocol["task"]["baseline_source_sha256"]
    if sha256_file(baseline) != expected_baseline:
        raise RuntimeError("synthetic candidate workspace differs from frozen baseline")
    tracked = subprocess.run(
        ["git", "-C", str(workspace), "ls-tree", "-r", "--name-only", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
    ).stdout.splitlines()
    if tracked != ["subject.py"]:
        raise RuntimeError(f"candidate workspace has unexpected tracked paths: {tracked}")

    limits = protocol["resource_limits"]
    budget = protocol["inference_budget"]
    before = resource_snapshot(attempt_dir)
    failures = [
        name for name, key in (
            ("available_memory_below_minimum", "available_memory_bytes"),
            ("free_swap_below_minimum", "swap_free_bytes"),
            ("disk_free_below_minimum", "disk_free_bytes"),
        ) if before[key] < limits[
            {"available_memory_bytes": "minimum_available_memory_bytes",
             "swap_free_bytes": "minimum_free_swap_bytes",
             "disk_free_bytes": "minimum_disk_free_bytes"}[key]
        ]
    ]
    preflight = {
        "protocol_sha256": protocol_sha,
        "arm": arm["arm_id"],
        "resource_snapshot": before,
        "failures": failures,
        "model_inference_started": False,
    }
    write_new(attempt_dir / "preflight.json", (json.dumps(preflight, indent=2, sort_keys=True) + "\n").encode())
    if failures:
        result = {
            "protocol_sha256": protocol_sha,
            "arm": arm,
            "status": "NOT_STARTED_RESOURCE_PREFLIGHT",
            "resource_before": before,
            "resource_failures": failures,
        }
        write_new(attempt_dir / "attempt-result.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        return result

    instruction = protocol["task"]["instruction"]
    command = [
        sys.executable, str(RUNNER),
        "--workspace", str(workspace),
        "--model-path", str(model_dir),
        "--model-repository", protocol["model"]["repository"],
        "--backend", "mlx", "--mlx-device", "metal",
        "--task-root", str(task_root),
        "--diff-base", "HEAD",
        "--output", str(attempt_dir / "agent.json"),
        "--event-journal", str(attempt_dir / "events.jsonl"),
        "--instruction", instruction,
        "--seed", str(budget["seed"]),
        "--max-turns", str(budget["max_turns"]),
        "--max-no-tool-retries", str(budget["max_no_tool_retries"]),
        "--max-new-tokens-per-turn", str(budget["max_new_tokens_per_turn"]),
        "--max-total-new-tokens", str(budget["max_total_new_tokens"]),
        "--max-context-tokens", str(budget["max_context_tokens"]),
        "--wall-seconds", str(budget["agent_wall_seconds"]),
        "--temperature", str(budget["temperature"]),
        "--top-p", str(budget["top_p"]),
        "--top-k", str(budget["top_k"]),
        "--no-do-sample", "--edit-format", "structured_replace",
        "--tool-interface", arm["tool_interface"],
        "--generation-protocol", budget["generation_protocol"],
        "--require-source-read-before-edit",
        "--grader-entrypoint", str(GRADER),
    ]
    if arm["post_edit_navigation_gate"]:
        command.append("--post-edit-navigation-gate")
    if arm["post_edit_completion_guidance"]:
        command.append("--post-edit-completion-guidance")
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "0",
    })
    safe_env = {key: env[key] for key in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "PYTHONHASHSEED",
    )}
    write_new(attempt_dir / "command.json", (json.dumps({
        "argv": command, "cwd": str(ROOT), "environment": safe_env,
        "instruction": instruction,
    }, indent=2, sort_keys=True) + "\n").encode())

    sample_path = attempt_dir / "resource-samples.jsonl"
    started = time.monotonic()
    peak_rss = 0
    abort_reason = None
    with (attempt_dir / "stdout.log").open("xb") as stdout, \
            (attempt_dir / "stderr.log").open("xb") as stderr, sample_path.open("xb") as samples:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            observed = resource_snapshot(attempt_dir, process.pid)
            peak_rss = max(peak_rss, observed["process_tree_rss_bytes"])
            samples.write((json.dumps(observed, sort_keys=True) + "\n").encode())
            samples.flush()
            os.fsync(samples.fileno())
            if observed["available_memory_bytes"] < limits["minimum_available_memory_bytes"]:
                abort_reason = "available_memory_below_abort_threshold"
            elif observed["swap_free_bytes"] < limits["minimum_free_swap_bytes"]:
                abort_reason = "free_swap_below_abort_threshold"
            elif observed["swap_used_bytes"] - before["swap_used_bytes"] > limits["max_swap_growth_bytes"]:
                abort_reason = "swap_growth_above_frozen_ceiling"
            elif observed["disk_free_bytes"] < limits["minimum_disk_free_bytes"]:
                abort_reason = "disk_free_below_abort_threshold"
            elif observed["process_tree_rss_bytes"] > limits["max_model_process_rss_bytes"]:
                abort_reason = "model_process_rss_above_abort_threshold"
            elif time.monotonic() - started > limits["attempt_wall_seconds"]:
                abort_reason = "attempt_wall_budget_exhausted"
            if abort_reason:
                stop_process_tree(process)
                break
            time.sleep(limits["sample_interval_seconds"])
        exit_code = process.wait()
    after = resource_snapshot(attempt_dir)

    agent_record = None
    agent_path = attempt_dir / "agent.json"
    if agent_path.is_file():
        try:
            agent_record = json.loads(agent_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            agent_record = None
    source = workspace / "subject.py"
    candidate_sha = sha256_file(source) if source.is_file() and not source.is_symlink() else None
    diff_proc = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--", "subject.py"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if diff_proc.returncode != 0:
        raise RuntimeError(f"candidate diff failed: {diff_proc.stderr.decode('utf-8', 'replace')[:500]}")
    diff = diff_proc.stdout
    write_new(attempt_dir / "candidate.patch", diff)
    grader = grade_candidate(workspace, attempt_dir)
    runtime_audit = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_agent_runtime_snapshot_v1.py"),
         str(attempt_dir / "runtime_snapshot")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    try:
        runtime_audit_record = json.loads(runtime_audit.stdout) if runtime_audit.returncode == 0 else None
    except json.JSONDecodeError:
        runtime_audit_record = None
    result = {
        "protocol_sha256": protocol_sha,
        "arm": arm,
        "status": "RESOURCE_ABORTED" if abort_reason else ("COMPLETED" if exit_code == 0 else "RUNNER_FAILED"),
        "exit_code": exit_code,
        "abort_reason": abort_reason,
        "wall_seconds": round(time.monotonic() - started, 3),
        "peak_process_tree_rss_bytes": peak_rss,
        "resource_before": before,
        "resource_after": after,
        "agent_record_sha256": sha256_file(agent_path) if agent_path.is_file() else None,
        "agent_stop_reason": agent_record.get("stop_reason") if isinstance(agent_record, dict) else None,
        "model_explicitly_finished": agent_record.get("model_explicitly_finished") if isinstance(agent_record, dict) else None,
        "edit_attempted": agent_record.get("edit_attempted") if isinstance(agent_record, dict) else None,
        "edit_applied": agent_record.get("edit_applied") if isinstance(agent_record, dict) else None,
        "controller_interventions": agent_record.get("controller_interventions") if isinstance(agent_record, dict) else None,
        "candidate_source_sha256": candidate_sha,
        "candidate_changed": candidate_sha != expected_baseline,
        "candidate_diff_sha256": sha256_bytes(diff),
        "candidate_diff_bytes": len(diff),
        "independent_grade": grader,
        "runtime_snapshot_audit": runtime_audit_record,
        "runtime_snapshot_audit_exit_code": runtime_audit.returncode,
        "raw_artifact_sha256": {
            path.name: sha256_file(path)
            for path in sorted(attempt_dir.iterdir())
            if path.is_file() and path.name != "attempt-result.json"
        },
    }
    result_bytes = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    write_new(attempt_dir / "attempt-result.json", result_bytes)
    write_new(attempt_dir / "sha256-manifest.json", (json.dumps({
        "attempt_result_sha256": sha256_bytes(result_bytes),
        "protocol_sha256": protocol_sha,
        "raw_artifact_sha256": result["raw_artifact_sha256"],
    }, indent=2, sort_keys=True) + "\n").encode())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = sha256_bytes(protocol_bytes)
    sidecar = protocol_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="ascii").split()[0] != protocol_sha:
        raise SystemExit("frozen protocol SHA-256 does not match sidecar")
    protocol = json.loads(protocol_bytes)
    if protocol.get("protocol_id") != "vare.sprint-viii.controller-diagnostic.v5":
        raise SystemExit("unsupported diagnostic protocol")
    if protocol.get("frozen_before_model_inference") is not True:
        raise SystemExit("protocol is not frozen")
    arms = protocol.get("arms")
    if not isinstance(arms, list) or not arms:
        raise SystemExit("protocol has no diagnostic arms")
    for arm in arms:
        if (not isinstance(arm, dict)
                or not isinstance(arm.get("arm_id"), str)
                or not isinstance(arm.get("tool_interface"), str)
                or not isinstance(arm.get("post_edit_navigation_gate"), bool)
                or not isinstance(arm.get("post_edit_completion_guidance"), bool)):
            raise SystemExit("protocol arm is missing a required interface or controller setting")
    if sys.executable != protocol["runtime"]["python_executable"]:
        raise SystemExit("Python executable differs from frozen protocol")
    for relative, expected in protocol["code_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise SystemExit(f"runtime source differs from frozen protocol: {relative}")
    model_dir = Path(protocol["model"]["snapshot_path"])
    actual_model_files = {
        path.name: sha256_file(path) for path in sorted(model_dir.iterdir()) if path.is_file()
    }
    if actual_model_files != protocol["model"]["files_sha256"]:
        raise SystemExit("cached model snapshot differs from frozen file manifest")
    task_root = Path(protocol["task"]["task_root_path"])
    if sha256_file(task_root / "task.json") != protocol["task"]["task_json_sha256"]:
        raise SystemExit("synthetic task contract differs from frozen task SHA")

    output_root = args.output.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise SystemExit(f"refusing to overwrite diagnostic evidence: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    initial = resource_snapshot(output_root)
    limits = protocol["resource_limits"]
    failures = [
        name for name, key, minimum in (
            ("available_memory_below_minimum", "available_memory_bytes", limits["minimum_available_memory_bytes"]),
            ("free_swap_below_minimum", "swap_free_bytes", limits["minimum_free_swap_bytes"]),
            ("disk_free_below_minimum", "disk_free_bytes", limits["minimum_disk_free_bytes"]),
        ) if initial[key] < minimum
    ]
    preflight = {
        "protocol_sha256": protocol_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_snapshot_verified": True,
        "task_contract_verified": True,
        "resource_snapshot": initial,
        "failures": failures,
        "model_inference_started": False,
    }
    write_new(output_root / "preflight.json", (json.dumps(preflight, indent=2, sort_keys=True) + "\n").encode())
    if failures:
        summary = {"status": "NOT_STARTED_RESOURCE_PREFLIGHT", "protocol_sha256": protocol_sha,
                   "preflight": preflight, "arms": []}
        write_new(output_root / "campaign-summary.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps({"status": summary["status"], "failures": failures}, sort_keys=True))
        return 2

    results = []
    arms = protocol["arms"]
    for index, arm in enumerate(arms):
        result = run_attempt(
            arm=arm,
            protocol=protocol,
            protocol_sha=protocol_sha,
            attempt_dir=output_root / "attempts" / arm["arm_id"],
            model_dir=model_dir,
            task_root=task_root,
        )
        results.append(result)
        # The full result and raw files are durable before any next model load.
        if result["status"] in {"RESOURCE_ABORTED", "NOT_STARTED_RESOURCE_PREFLIGHT"}:
            for remaining in arms[index + 1:]:
                skipped = {
                    "protocol_sha256": protocol_sha,
                    "arm": remaining,
                    "status": "NOT_STARTED_AFTER_RESOURCE_STOP",
                }
                remaining_dir = output_root / "attempts" / remaining["arm_id"]
                remaining_dir.mkdir(parents=True, exist_ok=False)
                write_new(remaining_dir / "attempt-result.json", (json.dumps(skipped, indent=2, sort_keys=True) + "\n").encode())
                results.append(skipped)
            break
    summary = {
        "protocol_sha256": protocol_sha,
        "status": "COMPLETED" if len(results) == len(arms) and all(
            not row["status"].startswith("NOT_STARTED") for row in results
        ) else "PARTIAL",
        "claim_limit": protocol["claim_limit"],
        "task_family_denominator": 1,
        "arms": arms,
        "results": results,
        "explicit_finish_by_arm": {
            row["arm"]["arm_id"]: row.get("model_explicitly_finished")
            for row in results if isinstance(row.get("arm"), dict)
        },
        "independent_grade_by_arm": {
            row["arm"]["arm_id"]: (
                row.get("independent_grade", {}).get("receipt", {}).get("status")
                if isinstance(row.get("independent_grade"), dict) else None
            ) for row in results if isinstance(row.get("arm"), dict)
        },
        "resource_after_campaign": resource_snapshot(output_root),
        "evidence_directory": str(output_root),
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    write_new(output_root / "campaign-summary.json", summary_bytes)
    write_new(output_root / "sha256-manifest.json", (json.dumps({
        "protocol_sha256": protocol_sha,
        "campaign_summary_sha256": sha256_bytes(summary_bytes),
        "attempt_results_sha256": {
            row["arm"]["arm_id"]: sha256_file(output_root / "attempts" / row["arm"]["arm_id"] / "attempt-result.json")
            for row in results if isinstance(row.get("arm"), dict)
            and (output_root / "attempts" / row["arm"]["arm_id"] / "attempt-result.json").is_file()
        },
    }, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"status": summary["status"], "explicit_finish_by_arm": summary["explicit_finish_by_arm"],
                      "independent_grade_by_arm": summary["independent_grade_by_arm"],
                      "evidence_directory": str(output_root)}, sort_keys=True))
    return 0 if summary["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
