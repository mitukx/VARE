#!/usr/bin/env python3
"""Run the frozen two-arm, synthetic model/tool compatibility pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
GRADER_PATH = ROOT / "scripts" / "sprint_vii_model_compat_grader.py"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_new(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            count = os.write(fd, view)
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)


def file_manifest(model_dir: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(model_dir.iterdir()) if path.is_file()}


def snapshot(disk_path: Path, pid: int | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process_rss = 0
    if pid is not None:
        try:
            root = psutil.Process(pid)
            process_rss = sum(
                child.memory_info().rss for child in [root, *root.children(recursive=True)]
                if child.is_running()
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            process_rss = 0
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "available_memory_bytes": memory.available,
        "memory_total_bytes": memory.total,
        "swap_used_bytes": swap.used,
        "swap_free_bytes": max(0, swap.total - swap.used),
        "swap_total_bytes": swap.total,
        "disk_free_bytes": shutil.disk_usage(disk_path).free,
        "process_tree_rss_bytes": process_rss,
    }


def git(root: Path, *args: str, env: dict[str, str] | None = None,
        check: bool = True) -> subprocess.CompletedProcess[bytes]:
    git_env = os.environ.copy()
    git_env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
    })
    if env:
        git_env.update(env)
    git_env.pop("GIT_DIR", None)
    git_env.pop("GIT_WORK_TREE", None)
    proc = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=git_env, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {proc.stderr.decode('utf-8', 'replace')[:500]}")
    return proc


def canonical_workspace_diff(root: Path, relative_path: str) -> subprocess.CompletedProcess[bytes]:
    """Produce a diff without user-configured external diff or textconv hooks."""
    return git(
        root, "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--",
        relative_path, check=False,
    )


def independent_grade(workspace: Path, expected_sha: str) -> dict[str, Any]:
    command = [
        sys.executable, "-I", str(GRADER_PATH), "--workspace-root", str(workspace),
        "--source", "subject.py", "--expected-sha256", expected_sha,
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=20, cwd=ROOT, check=False)
    try:
        receipt = json.loads(proc.stdout) if proc.returncode == 0 else None
    except json.JSONDecodeError:
        receipt = None
    return {
        "command": command,
        "returncode": proc.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "receipt": receipt,
        "grader_sha256": sha256_file(GRADER_PATH),
    }


def run_attempt(
    arm: dict[str, Any], protocol: dict[str, Any], protocol_sha: str,
    task_root: Path, model_dir: Path, output_root: Path,
) -> dict[str, Any]:
    limits = protocol["resource_limits"]
    budget = protocol["inference_budget"]
    attempt_dir = output_root / "attempts" / arm["tool_interface"]
    attempt_dir.mkdir(parents=True, exist_ok=False)
    workspace = attempt_dir / "workspace"
    shutil.copytree(protocol["workspace_template_path"], workspace)
    revision = git(workspace, "rev-parse", "HEAD").stdout.decode().strip()
    base_source = git(workspace, "show", "HEAD:subject.py").stdout
    tracked = git(workspace, "ls-tree", "-r", "--name-only", "HEAD").stdout.decode().splitlines()
    if (revision != protocol["workspace_base_commit"] or tracked != ["subject.py"]
            or sha256_bytes(base_source) != protocol["baseline_source_sha256"]
            or git(workspace, "status", "--porcelain=v1").stdout):
        raise RuntimeError("synthetic task workspace base commit differs from the frozen protocol")

    sample_path = attempt_dir / "resource-samples.jsonl"
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    agent_path = attempt_dir / "agent.json"
    journal_path = attempt_dir / "events.jsonl"
    command = [
        sys.executable, str(ROOT / "scripts/run_local_agent.py"),
        "--workspace", str(workspace),
        "--model-path", str(model_dir),
        "--model-repository", protocol["model"]["repository"],
        "--backend", "mlx", "--mlx-device", "metal",
        "--task-root", str(task_root),
        "--diff-base", "HEAD",
        "--output", str(agent_path),
        "--event-journal", str(journal_path),
        "--instruction", protocol["instruction"],
        "--seed", str(budget["seed"]),
        "--max-turns", str(budget["max_turns"]),
        "--max-no-tool-retries", str(budget["max_no_tool_retries"]),
        "--max-new-tokens-per-turn", str(budget["max_new_tokens_per_turn"]),
        "--max-total-new-tokens", str(budget["max_total_new_tokens"]),
        "--max-context-tokens", str(budget["max_context_tokens"]),
        "--wall-seconds", str(budget["agent_wall_seconds"]),
        "--temperature", "0.0", "--top-p", "1.0", "--top-k", "0",
        "--no-do-sample", "--edit-format", "structured_replace",
        "--tool-interface", arm["tool_interface"],
        "--generation-protocol", protocol["generation_protocol"],
        "--require-source-read-before-edit",
    ]
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "0",
    })
    before = snapshot(output_root)
    resource_reason = None
    for key, minimum in (
        ("available_memory_bytes", limits["minimum_available_memory_bytes"]),
        ("swap_free_bytes", limits["minimum_free_swap_bytes"]),
        ("disk_free_bytes", limits["minimum_disk_free_bytes"]),
    ):
        if before[key] < minimum:
            resource_reason = f"preflight_{key}_below_minimum"
            break
    if resource_reason:
        outcome = {
            "protocol_sha256": protocol_sha,
            "arm": arm,
            "status": "NOT_STARTED_RESOURCE_PREFLIGHT",
            "resource_abort_reason": resource_reason,
            "resource_before": before,
            "command": command,
        }
        write_new(attempt_dir / "attempt-result.json",
                  (json.dumps(outcome, indent=2, sort_keys=True) + "\n").encode())
        return outcome

    started = time.monotonic()
    peak_rss = 0
    abort_reason = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr, \
            sample_path.open("x", encoding="utf-8") as samples:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        while process.poll() is None:
            observed = snapshot(output_root, process.pid)
            peak_rss = max(peak_rss, observed["process_tree_rss_bytes"])
            samples.write(json.dumps(observed, sort_keys=True) + "\n")
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
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                break
            time.sleep(limits["sample_interval_seconds"])
        exit_code = process.wait()
    after = snapshot(output_root)
    agent_record = None
    if agent_path.is_file():
        try:
            agent_record = json.loads(agent_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            agent_record = None
    candidate = workspace / "subject.py"
    candidate_sha = sha256_file(candidate) if candidate.is_file() and not candidate.is_symlink() else None
    grader_record = independent_grade(workspace, candidate_sha) if candidate_sha else None
    diff = canonical_workspace_diff(workspace, "subject.py")
    changed_paths = git(workspace, "diff", "--name-only", "HEAD", "--").stdout.decode().splitlines()
    baseline_source = git(workspace, "show", "HEAD:subject.py").stdout
    final_source = candidate.read_bytes() if candidate_sha else b""
    resource_aborted = abort_reason is not None
    verified_repair = bool(
        not resource_aborted and exit_code == 0 and agent_record is not None
        and agent_record.get("edit_applied") is True
        and final_source != baseline_source
        and changed_paths == ["subject.py"]
        and grader_record is not None
        and isinstance(grader_record.get("receipt"), dict)
        and grader_record["receipt"].get("status") == "PASS"
        and grader_record["receipt"].get("candidate_execution") == "never_imported_or_executed"
    )
    outcome = {
        "protocol_sha256": protocol_sha,
        "arm": arm,
        "status": "RESOURCE_ABORT" if resource_aborted else (
            "COMPLETED" if exit_code == 0 and agent_record is not None else "RUNNER_FAILURE"
        ),
        "exit_code": exit_code,
        "resource_abort_reason": abort_reason,
        "command": command,
        "environment": {
            "HF_HUB_OFFLINE": env["HF_HUB_OFFLINE"],
            "TRANSFORMERS_OFFLINE": env["TRANSFORMERS_OFFLINE"],
            "TOKENIZERS_PARALLELISM": env["TOKENIZERS_PARALLELISM"],
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "resource_before": before,
        "resource_after": after,
        "peak_process_tree_rss_bytes": peak_rss,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "raw_files_sha256": {
            path.name: sha256_file(path) for path in
            [stdout_path, stderr_path, sample_path, journal_path, agent_path]
            if path.is_file()
        },
        "agent_record": agent_record,
        "independent_grade": grader_record,
        "verified_synthetic_repair": verified_repair,
        "source_identity": {
            "workspace_base_commit": revision,
            "baseline_source_sha256": sha256_bytes(baseline_source),
            "candidate_source_sha256": candidate_sha,
            "candidate_source_utf8": final_source.decode("utf-8", "replace"),
            "candidate_diff_sha256": sha256_bytes(diff.stdout),
            "candidate_diff_utf8": diff.stdout.decode("utf-8", "replace"),
            "changed_paths": changed_paths,
            "source_changed": final_source != baseline_source,
            "workspace_only_subject_change": changed_paths in ([], ["subject.py"]),
        },
        "measurement_counts": {
            "tool_turns": agent_record.get("tool_turns") if agent_record else None,
            "generated_tokens": agent_record.get("new_tokens") if agent_record else None,
            "edit_attempted": agent_record.get("edit_attempted") if agent_record else None,
            "edit_applied": agent_record.get("edit_applied") if agent_record else None,
            "independent_status": (
                grader_record.get("receipt", {}).get("status") if grader_record else None
            ),
            "public_tests": "NOT_RUN (Docker unavailable; no public-test tool exposed)",
        },
    }
    result_bytes = (json.dumps(outcome, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    write_new(attempt_dir / "attempt-result.json", result_bytes)
    write_new(attempt_dir / "sha256-manifest.json", (json.dumps({
        "attempt_result_sha256": sha256_bytes(result_bytes),
        "raw_files_sha256": outcome["raw_files_sha256"],
    }, indent=2, sort_keys=True) + "\n").encode())
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    protocol_sha = sha256_bytes(protocol_bytes)
    output_root = args.output.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise SystemExit(f"refusing to overwrite model compatibility evidence: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    if protocol.get("frozen_before_model_inference") is not True:
        raise SystemExit("model compatibility protocol was not frozen")
    for relative, expected in protocol["code_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise SystemExit(f"controller/grader code hash mismatch: {relative}")
    model_dir = Path(protocol["model"]["snapshot_path"])
    task_root = Path(protocol["task_root_path"])
    if file_manifest(model_dir) != protocol["model"]["files_sha256"]:
        raise SystemExit("model snapshot file set or SHA-256 differs from protocol")
    if sha256_file(task_root / "task.json") != protocol["task_json_sha256"]:
        raise SystemExit("synthetic task JSON differs from protocol")
    if sha256_file(Path(protocol["e0_result_path"])) != protocol["e0_result_sha256"]:
        raise SystemExit("E0 result changed after the model protocol was frozen")
    e0 = json.loads(Path(protocol["e0_result_path"]).read_text(encoding="utf-8"))
    if e0.get("status") != "PASS":
        raise SystemExit("the frozen E0 gate is not PASS")
    if sys.executable != protocol["python_executable"]:
        raise SystemExit("runtime interpreter differs from the frozen MLX environment")
    import mlx_lm
    import transformers
    versions = {
        "mlx": importlib_version("mlx"),
        "mlx_lm": getattr(mlx_lm, "__version__", importlib_version("mlx-lm")),
        "transformers": transformers.__version__,
    }
    if versions != protocol["runtime_packages"]:
        raise SystemExit(f"runtime package versions differ: {versions}")
    resource_before = snapshot(output_root)
    failures = [key for key, minimum in (
        ("available_memory_bytes", protocol["resource_limits"]["minimum_available_memory_bytes"]),
        ("swap_free_bytes", protocol["resource_limits"]["minimum_free_swap_bytes"]),
        ("disk_free_bytes", protocol["resource_limits"]["minimum_disk_free_bytes"]),
    ) if resource_before[key] < minimum]
    preflight = {
        "protocol_sha256": protocol_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_snapshot_files_verified": True,
        "task_json_verified": True,
        "e0_pass_verified": True,
        "runtime_packages": versions,
        "resource_snapshot": resource_before,
        "failures": failures,
        "model_inference_started": False,
    }
    write_new(output_root / "preflight.json",
              (json.dumps(preflight, indent=2, sort_keys=True) + "\n").encode())
    if failures:
        summary = {"status": "NOT_STARTED_RESOURCE_PREFLIGHT", "protocol_sha256": protocol_sha,
                   "preflight": preflight, "attempts": []}
        write_new(output_root / "campaign-summary.json",
                  (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
        print(json.dumps({"status": summary["status"], "failures": failures}, sort_keys=True))
        return 2

    attempts = []
    for index, arm in enumerate(protocol["arms"], start=1):
        attempts.append(run_attempt(arm, protocol, protocol_sha, task_root, model_dir, output_root))
        # The saved attempt result is durable before the next model is loaded.
        if attempts[-1]["status"] in {"RESOURCE_ABORT", "NOT_STARTED_RESOURCE_PREFLIGHT"}:
            for remaining in protocol["arms"][index:]:
                skipped = {"protocol_sha256": protocol_sha, "arm": remaining,
                           "status": "NOT_STARTED_AFTER_RESOURCE_STOP"}
                attempts.append(skipped)
                write_new(output_root / "attempts" / remaining["tool_interface"] / "attempt-result.json",
                          (json.dumps(skipped, indent=2, sort_keys=True) + "\n").encode())
            break
    grades = [
        attempt.get("independent_grade", {}).get("receipt", {}).get("status")
        if isinstance(attempt.get("independent_grade"), dict) else None
        for attempt in attempts
    ]
    summary = {
        "protocol_sha256": protocol_sha,
        "status": "COMPLETED" if (
            len(attempts) == len(protocol["arms"])
            and all(not attempt["status"].startswith("NOT_STARTED") for attempt in attempts)
        ) else "PARTIAL",
        "capability_claim": False,
        "primary_outcome": "one synthetic task family; no inferential comparison",
        "arms": protocol["arms"],
        "attempts": attempts,
        "independent_status_by_arm": dict(zip(
            [attempt["arm"]["tool_interface"] for attempt in attempts], grades,
        )),
        "task_family_denominator": 1,
        "verified_synthetic_repairs": sum(
            attempt.get("verified_synthetic_repair") is True for attempt in attempts
        ),
        "resource_after_campaign": snapshot(output_root),
        "evidence_directory": str(output_root),
    }
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    write_new(output_root / "campaign-summary.json", summary_bytes)
    write_new(output_root / "sha256-manifest.json", (json.dumps({
        "protocol_sha256": protocol_sha,
        "campaign_summary_sha256": sha256_bytes(summary_bytes),
        "attempt_results_sha256": {
            attempt["arm"]["tool_interface"]: sha256_file(
                output_root / "attempts" / attempt["arm"]["tool_interface"] / "attempt-result.json"
            ) for attempt in attempts
            if (output_root / "attempts" / attempt["arm"]["tool_interface"] / "attempt-result.json").is_file()
        },
    }, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"status": summary["status"], "independent_status_by_arm": summary["independent_status_by_arm"],
                      "evidence_directory": str(output_root)}, sort_keys=True))
    return 0


def importlib_version(name: str) -> str:
    from importlib.metadata import version
    return version(name)


if __name__ == "__main__":
    raise SystemExit(main())
