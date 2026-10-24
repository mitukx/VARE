#!/usr/bin/env python3
"""Execute the frozen four-run interface pilot with resource and grader gates."""
from __future__ import annotations

import argparse
import asyncio
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
sys.path.insert(0, str(ROOT / "src"))
from vare.container_isolation import DockerIsolationRunner, IsolationLimits  # noqa: E402

PROTOCOL = ROOT / "protocols/real_agent_interface_v1.json"
TASK_ROOT = ROOT / "benchmarks/mlsys_real_agent_interface_v1/tasks"
PREPARE = ROOT / "scripts/prepare_real_agent_interface_v1.py"
AGENT_RUNNER = ROOT / "scripts/run_local_agent.py"
MODEL_NAME = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path: Path) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "available_memory_bytes": memory.available,
        "swap_used_bytes": swap.used,
        "swap_free_bytes": max(0, swap.total - swap.used),
        "disk_free_bytes": shutil.disk_usage(path).free,
    }


def verify_pins(protocol: dict, source_repo: Path, model_dir: Path,
                e0_dir: Path) -> None:
    if protocol.get("frozen_before_model_inference") is not True:
        raise RuntimeError("protocol is not frozen")
    for relative, expected in protocol["implementation_sha256"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"implementation hash mismatch: {relative}")
    for task_id, expected in protocol["task_lock_sha256"].items():
        task_root = TASK_ROOT / task_id
        actual = sha256(task_root / "protocol.lock.json")
        if actual != expected:
            raise RuntimeError(f"task lock hash mismatch: {task_id}")
        task_lock = json.loads((task_root / "protocol.lock.json").read_text(encoding="utf-8"))
        if task_lock.get("split") != "development" or task_lock.get("task_id") != task_id:
            raise RuntimeError(f"task is not the frozen Development candidate: {task_id}")
        for relative, input_hash in task_lock["candidate_inputs"].items():
            candidate_input = task_root / relative
            if not candidate_input.is_file() or sha256(candidate_input) != input_hash:
                raise RuntimeError(f"candidate input hash mismatch: {task_id}/{relative}")
        hidden_path = task_root / task_lock["trusted_grader"]["file"]
        if (not hidden_path.is_file()
                or sha256(hidden_path) != task_lock["trusted_grader"]["sha256"]):
            raise RuntimeError(f"trusted grader hash mismatch: {task_id}")
    if not model_dir.is_dir() or model_dir.name != protocol["model"]["revision"]:
        raise RuntimeError("model snapshot path must be the frozen commit revision")
    expected_files = protocol["model"]["files_sha256"]
    actual_files = {item.name: sha256(item) for item in model_dir.iterdir() if item.is_file()}
    if actual_files != expected_files:
        raise RuntimeError("model snapshot file set or SHA-256 differs from the frozen protocol")
    if not source_repo.is_dir():
        raise RuntimeError("source repository checkout does not exist")
    planned = protocol["preregistration"]["attempt_order"]
    if planned != protocol["attempt_order"]:
        raise RuntimeError("execution order differs from the preregistered order")
    if len(planned) != 4 or len({(row["task_id"], row["edit_format"]) for row in planned}) != 4:
        raise RuntimeError("the frozen paired task/interface design is incomplete")
    expected_e0 = (ROOT / protocol["runtime"]["e0_result"]).resolve()
    if e0_dir.resolve() != expected_e0:
        raise RuntimeError("E0 evidence path differs from the frozen protocol")
    e0_observations = json.loads((e0_dir / "observations.json").read_text(encoding="utf-8"))
    if e0_observations.get("accepted") is not True or e0_observations.get("image") != protocol["runtime"]["candidate_image"]:
        raise RuntimeError("matching E0 isolation evidence is missing or rejected")
    e0_manifest = json.loads((e0_dir / "manifest.json").read_text(encoding="utf-8"))
    if e0_manifest.get("protocol_sha256") != sha256(PROTOCOL):
        raise RuntimeError("E0 evidence was not produced against this frozen protocol")
    if e0_manifest.get("observations_sha256") != sha256(e0_dir / "observations.json"):
        raise RuntimeError("E0 observations differ from their recorded manifest")


def preflight(protocol: dict, path: Path) -> dict[str, Any]:
    observed = snapshot(path)
    limits = protocol["resource_limits"]
    failures = [
        key for key, minimum in (
            ("available_memory_bytes", limits["minimum_available_memory_bytes"]),
            ("swap_free_bytes", limits["minimum_free_swap_bytes"]),
            ("disk_free_bytes", limits["minimum_disk_free_bytes"]),
        ) if observed[key] < minimum
    ]
    if failures:
        raise RuntimeError("resource preflight failed: " + ", ".join(failures))
    return observed


def process_tree_rss(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        descendants = [process, *process.children(recursive=True)]
        return sum(item.memory_info().rss for item in descendants if item.is_running())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def run_agent(command: list[str], env: dict[str, str], *, attempt_dir: Path,
              protocol: dict, baseline: dict[str, Any]) -> dict[str, Any]:
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    samples_path = attempt_dir / "resources.jsonl"
    swap_growth_limit = protocol["resource_limits"]["max_swap_growth_bytes"]
    memory_limit = protocol["resource_limits"]["max_model_process_rss_bytes"]
    stdout_file = stdout_path.open("w", encoding="utf-8")
    stderr_file = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command, cwd=ROOT, env=env,
        stdout=stdout_file,
        stderr=stderr_file,
    )
    started = time.monotonic()
    peak_rss = 0
    abort_reason = None
    with samples_path.open("w", encoding="utf-8") as stream:
        while proc.poll() is None:
            observed = snapshot(attempt_dir)
            observed["process_tree_rss_bytes"] = process_tree_rss(proc.pid)
            peak_rss = max(peak_rss, observed["process_tree_rss_bytes"])
            stream.write(json.dumps(observed, sort_keys=True) + "\n")
            stream.flush()
            if observed["available_memory_bytes"] < protocol["resource_limits"]["minimum_available_memory_bytes"]:
                abort_reason = "available_memory_below_abort_threshold"
            elif observed["swap_free_bytes"] < protocol["resource_limits"]["minimum_free_swap_bytes"]:
                abort_reason = "free_swap_below_abort_threshold"
            elif observed["swap_used_bytes"] - baseline["swap_used_bytes"] > swap_growth_limit:
                abort_reason = "swap_growth_above_abort_threshold"
            elif observed["disk_free_bytes"] < protocol["resource_limits"]["minimum_disk_free_bytes"]:
                abort_reason = "disk_free_below_abort_threshold"
            elif observed["process_tree_rss_bytes"] > memory_limit:
                abort_reason = "model_process_rss_above_abort_threshold"
            elif time.monotonic() - started > protocol["inference_budget"]["wall_seconds"]:
                abort_reason = "attempt_wall_budget_exhausted"
            if abort_reason:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                break
            time.sleep(protocol["resource_limits"]["sample_interval_seconds"])
    stdout_file.close()
    stderr_file.close()
    finished = snapshot(attempt_dir)
    return {
        "exit_code": proc.returncode,
        "aborted": abort_reason is not None,
        "abort_reason": abort_reason,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_process_tree_rss_bytes": peak_rss,
        "resource_before": baseline,
        "resource_after": finished,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "resource_samples_sha256": sha256(samples_path),
    }


async def grade_hidden(workspace: Path, hidden_test: Path, image: str,
                       output_path: Path) -> dict[str, Any]:
    grade_root = output_path.parent / "trusted-grader"
    grader_workspace = grade_root / "workspace"
    if grade_root.exists():
        raise RuntimeError("refusing to overwrite trusted grader workspace")
    grader_workspace.mkdir(parents=True)
    shutil.copytree(workspace / "mlx_lm", grader_workspace / "mlx_lm")
    shutil.copy2(hidden_test, grader_workspace / "test_hidden.py")
    limits = IsolationLimits(
        image=image, timeout_s=180, memory_mb=2048, cpus=1.0,
        pids_limit=32, max_output_bytes=20_000,
    )
    result = await DockerIsolationRunner().run_candidate(
        limits, allowed_workspace_root=grade_root, workspace=grader_workspace,
        command=["python", "/workspace/test_hidden.py"],
    )
    record = {
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "hidden_test_sha256": sha256(hidden_test),
        "candidate_mount": "separate post-inference container; hidden test is not copied into model workspace",
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--e0-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source_repo = args.source_repo.resolve()
    model_dir = args.model_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to overwrite evidence directory: {output}")
    verify_pins(protocol, source_repo, model_dir, args.e0_result.resolve())
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        parser.error("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    output.mkdir(parents=True)
    global_baseline = preflight(protocol, output)
    protocol_lock = {
        "protocol_sha256": sha256(PROTOCOL),
        "image": protocol["runtime"]["candidate_image"],
        "model_files_sha256": protocol["model"]["files_sha256"],
        "implementation_sha256": protocol["implementation_sha256"],
        "task_lock_sha256": protocol["task_lock_sha256"],
    }
    (output / "protocol.lock.json").write_text(
        json.dumps(protocol_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    attempts = []
    interpreter = Path(sys.executable)
    model_python = Path(protocol["runtime"]["inference_python"])
    if not model_python.is_file():
        raise RuntimeError(f"pinned inference Python is unavailable: {model_python}")

    for index, item in enumerate(protocol["attempt_order"], 1):
        attempt_id = f"{index:02d}-{item['task_id']}-{item['edit_format']}"
        attempt_dir = output / attempt_id
        attempt_dir.mkdir()
        before = preflight(protocol, attempt_dir)
        workspace = attempt_dir / "candidate-workspace"
        prep = subprocess.run([
            str(interpreter), str(PREPARE),
            "--source-repo", str(source_repo), "--task-id", item["task_id"],
            "--workspace", str(workspace),
        ], cwd=ROOT, check=True, text=True, capture_output=True)
        (attempt_dir / "materialization.stdout.json").write_text(prep.stdout, encoding="utf-8")
        (attempt_dir / "materialization.stderr.log").write_text(prep.stderr, encoding="utf-8")
        task_root = TASK_ROOT / item["task_id"]
        command = [
            str(model_python), str(AGENT_RUNNER),
            "--workspace", str(workspace), "--task-root", str(task_root),
            "--model-path", str(model_dir), "--model-repository", MODEL_NAME,
            "--backend", "mlx", "--public-test-image", protocol["runtime"]["candidate_image"],
            "--public-test-timeout-s", str(protocol["inference_budget"]["public_test_timeout_seconds"]),
            "--public-test-memory-mb", str(protocol["inference_budget"]["public_test_memory_mb"]),
            "--max-public-test-runs", str(protocol["inference_budget"]["max_public_test_runs"]),
            "--output", str(attempt_dir / "agent.json"),
            "--instruction", protocol["instruction"],
            "--seed", str(protocol["seed"]),
            "--max-turns", str(protocol["inference_budget"]["max_turns"]),
            "--max-no-tool-retries", str(protocol["inference_budget"]["max_no_tool_retries"]),
            "--max-new-tokens-per-turn", str(protocol["inference_budget"]["max_new_tokens_per_turn"]),
            "--max-total-new-tokens", str(protocol["inference_budget"]["max_total_new_tokens"]),
            "--max-context-tokens", str(protocol["inference_budget"]["max_context_tokens"]),
            "--wall-seconds", str(protocol["inference_budget"]["wall_seconds"]),
            "--temperature", str(protocol["inference_budget"]["temperature"]),
            "--top-p", str(protocol["inference_budget"]["top_p"]),
            "--top-k", str(protocol["inference_budget"]["top_k"]),
            "--no-do-sample", "--edit-format", item["edit_format"],
            "--require-split", "train",
        ]
        env = os.environ.copy()
        env.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "DOCKER_CONTEXT": protocol["runtime"]["docker_context"],
            "PYTHONPATH": str(ROOT / "src"),
            "TOKENIZERS_PARALLELISM": "false",
        })
        execution = run_agent(
            command, env, attempt_dir=attempt_dir, protocol=protocol, baseline=before,
        )
        (attempt_dir / "execution.json").write_text(
            json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        if execution["aborted"]:
            attempts.append({"attempt_id": attempt_id, "execution": execution, "status": "aborted"})
            break
        agent_path = attempt_dir / "agent.json"
        if not agent_path.is_file():
            attempts.append({"attempt_id": attempt_id, "execution": execution,
                             "status": "runner_failed_without_agent_record"})
            break
        agent = json.loads(agent_path.read_text(encoding="utf-8"))
        hidden = asyncio.run(grade_hidden(
            workspace, task_root / "evaluator/test_hidden.py",
            protocol["runtime"]["candidate_image"], attempt_dir / "trusted-grade.json",
        ))
        target = protocol["tasks"][item["task_id"]]["target_file"]
        before_hash = agent["source_hashes_before"][target]
        after_hash = agent["source_hashes_after"][target]
        changed_paths = sorted(
            name for name in agent["source_hashes_before"]
            if agent["source_hashes_before"][name] != agent["source_hashes_after"].get(name)
        )
        only_target_changed = changed_paths == [target]
        test_unchanged = "test_public.py" not in changed_paths
        public_pass = (agent.get("public_test_result") or {}).get("passed") is True
        source_changed = before_hash != after_hash
        budget = protocol["inference_budget"]
        model_files_valid = agent.get("model_files") == protocol["model"]["files_sha256"]
        budget_valid = (
            agent.get("new_tokens", 0) <= budget["max_total_new_tokens"]
            and agent.get("tool_turns", 0) <= budget["max_turns"]
            and agent.get("public_test_runs", 0) <= budget["max_public_test_runs"]
            and agent.get("elapsed_seconds", 0) <= budget["wall_seconds"]
            and execution["exit_code"] == 0
            and not execution["aborted"]
            and execution["elapsed_seconds"] <= budget["wall_seconds"]
        )
        mlx_peak = agent.get("runtime", {}).get("mlx_peak_memory_bytes")
        if isinstance(mlx_peak, int) and mlx_peak > protocol["resource_limits"]["max_model_process_rss_bytes"]:
            budget_valid = False
        success = (
            source_changed and agent.get("edit_applied") is True
            and only_target_changed and test_unchanged
            and public_pass and hidden["passed"] and budget_valid and model_files_valid
        )
        row = {
            "attempt_id": attempt_id,
            "task_id": item["task_id"],
            "edit_format": item["edit_format"],
            "source_changed": source_changed,
            "only_target_changed": only_target_changed,
            "changed_paths": changed_paths,
            "valid_edit_applied": agent.get("edit_applied") is True,
            "public_test_passed": public_pass,
            "independent_hidden_passed": hidden["passed"],
            "model_files_valid": model_files_valid,
            "budget_valid": budget_valid,
            "verified_success": success,
            "stop_reason": agent.get("stop_reason"),
            "new_tokens": agent.get("new_tokens"),
            "tool_turns": agent.get("tool_turns"),
            "elapsed_seconds": agent.get("elapsed_seconds"),
            "public_test_runs": agent.get("public_test_runs"),
            "peak_process_tree_rss_bytes": execution["peak_process_tree_rss_bytes"],
            "mlx_peak_memory_bytes": agent.get("runtime", {}).get("mlx_peak_memory_bytes"),
            "agent_record_sha256": sha256(agent_path),
            "hidden_record_sha256": sha256(attempt_dir / "trusted-grade.json"),
            "status": "complete",
        }
        (attempt_dir / "outcome.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        attempts.append(row)

    by_arm = {}
    for arm in {item["edit_format"] for item in protocol["attempt_order"]}:
        rows = [row for row in attempts if row.get("edit_format") == arm]
        by_arm[arm] = {
            "completed_attempts": len(rows),
            "verified_successes": sum(bool(row.get("verified_success")) for row in rows),
            "development_task_denominator": len(protocol["tasks"]),
        }
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "attempts": attempts,
        "by_arm": by_arm,
        "all_planned_attempts_completed": len(attempts) == len(protocol["attempt_order"])
            and all(row.get("status") == "complete" for row in attempts),
        "claim_limit": protocol["claim_limit"],
        "resource_global_start": global_baseline,
        "analysis_note": "Two task clusters; exact paired outcomes are descriptive feasibility evidence and do not establish statistical capability improvement.",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "by_arm": by_arm,
                      "all_planned_attempts_completed": summary["all_planned_attempts_completed"]}))
    return 0 if summary["all_planned_attempts_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
