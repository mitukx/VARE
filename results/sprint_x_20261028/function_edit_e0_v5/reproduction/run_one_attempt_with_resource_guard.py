#!/usr/bin/env python3
"""Execute the frozen v4 attempt under unchanged host/process guards."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

REPO = Path("/Users/user/Documents/Codex/2026-10-26/vare-flagship-research-sprint-real-coding/work/VARE")
BASE = Path("/tmp/vare-sprint-x-function-e0-005")
EVIDENCE = Path(__file__).resolve().parents[1]
RUN = EVIDENCE / "attempt-001"
MODEL = Path("/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit/snapshots/019cc73c45c770444708a6dd8690c66243cc5c80")
PYTHON = Path("/tmp/vare-sprint-vii-mlx/bin/python")
LIMITS = json.loads((EVIDENCE / "protocol.json").read_text(encoding="utf-8"))["resources"]


def terminate_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def main() -> int:
    if RUN.exists():
        raise SystemExit("refusing to overwrite attempt evidence")
    if not (EVIDENCE / "execution-freeze.json").is_file():
        raise SystemExit("protocol is not frozen")
    RUN.mkdir(parents=True)
    swap0 = psutil.swap_memory().used
    memory0 = psutil.virtual_memory().available
    disk0 = psutil.disk_usage(str(EVIDENCE)).free
    free_swap0 = psutil.swap_memory().free
    if (memory0 < LIMITS["minimum_available_memory_bytes"]
            or free_swap0 < LIMITS["minimum_free_swap_bytes"]
            or disk0 < LIMITS["minimum_disk_free_bytes"]):
        (RUN / "resource-preflight-no-go.json").write_text(json.dumps({
            "decision": "NO_GO", "available_memory_bytes": memory0, "free_swap_bytes": free_swap0,
            "disk_free_bytes": disk0, "limits": LIMITS,
        }, indent=2) + "\n", encoding="utf-8")
        return 75
    command = [
        str(PYTHON), str(REPO / "scripts/run_local_agent.py"),
        "--workspace", str(BASE / "candidate-workspace"), "--model-path", str(MODEL),
        "--model-repository", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "--backend", "mlx", "--mlx-device", "metal", "--task-root", str(BASE / "task-root"),
        "--diff-base", "HEAD", "--output", str(RUN / "agent-result.json"),
        "--event-journal", str(RUN / "events.jsonl"),
        "--instruction", "Find clamp_ten in src/quant.py, read its exact definition, apply the smallest correct bounded-clamp edit, then finish. Do not only describe the change.",
        "--seed", "20261036", "--max-turns", "8", "--max-no-tool-retries", "0",
        "--max-new-tokens-per-turn", "512", "--max-total-new-tokens", "1536",
        "--max-context-tokens", "8192", "--wall-seconds", "600", "--temperature", "0.0",
        "--top-p", "1.0", "--top-k", "0", "--no-do-sample", "--edit-format", "structured_replace",
        "--generation-protocol", "schema_slots_v9", "--tool-interface", "lease_bound_function_edit_v2",
        "--require-source-read-before-edit", "--post-edit-navigation-gate",
        "--no-progress-guard", "--post-edit-completion-guidance",
    ]
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(REPO / "src"), "TOKENIZERS_PARALLELISM": "false"})
    (RUN / "command.json").write_text(json.dumps({
        "argv": command, "cwd": str(REPO), "limits": LIMITS,
        "initial_available_memory_bytes": memory0, "initial_swap_used_bytes": swap0,
        "initial_free_swap_bytes": free_swap0, "initial_disk_free_bytes": disk0,
        "environment": {key: env[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONPATH", "TOKENIZERS_PARALLELISM")},
        "protocol_sha256": (EVIDENCE / "protocol.sha256").read_text(encoding="utf-8").split()[0],
    }, indent=2) + "\n", encoding="utf-8")
    (RUN / "stdout.log").touch()
    (RUN / "stderr.log").touch()
    with (RUN / "stdout.log").open("wb") as stdout, (RUN / "stderr.log").open("wb") as stderr:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        start = time.monotonic()
        max_rss = 0
        min_available = memory0
        max_swap_growth = 0
        abort = None
        with (RUN / "resource-samples.jsonl").open("a", encoding="utf-8") as samples:
            while proc.poll() is None:
                elapsed = time.monotonic() - start
                try:
                    root = psutil.Process(proc.pid)
                    tree = [root, *root.children(recursive=True)]
                    rss = sum(p.memory_info().rss for p in tree if p.is_running())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    rss = 0
                vm, sw = psutil.virtual_memory(), psutil.swap_memory()
                growth = max(0, sw.used - swap0)
                disk = psutil.disk_usage(str(EVIDENCE)).free
                max_rss = max(max_rss, rss)
                min_available = min(min_available, vm.available)
                max_swap_growth = max(max_swap_growth, growth)
                record = {"elapsed_seconds": round(elapsed, 3), "process_tree_rss_bytes": rss,
                          "process_tree_rss_peak_bytes": max_rss, "host_available_memory_bytes": vm.available,
                          "host_swap_used_bytes": sw.used, "host_swap_growth_bytes": growth,
                          "host_free_swap_bytes": sw.free, "disk_free_bytes": disk}
                samples.write(json.dumps(record, sort_keys=True) + "\n")
                samples.flush()
                os.fsync(samples.fileno())
                if elapsed > LIMITS["max_outer_wall_seconds"]: abort = "outer_wall_limit"
                elif rss > LIMITS["max_process_tree_rss_bytes"]: abort = "process_tree_rss_limit"
                elif growth > LIMITS["max_swap_growth_bytes"]: abort = "swap_growth_limit"
                elif vm.available < LIMITS["minimum_available_memory_bytes"]: abort = "host_available_memory_floor"
                elif sw.free < LIMITS["minimum_free_swap_bytes"]: abort = "host_free_swap_floor"
                elif disk < LIMITS["minimum_disk_free_bytes"]: abort = "disk_free_floor"
                if abort:
                    terminate_group(proc)
                    break
                time.sleep(LIMITS["sample_interval_seconds"])
    result = proc.poll()
    summary = {
        "status": "RESOURCE_ABORT" if abort else ("COMPLETED" if result == 0 else "RUNTIME_FAILURE"),
        "abort_reason": abort, "returncode": result, "outer_elapsed_seconds": round(time.monotonic() - start, 3),
        "peak_process_tree_rss_bytes": max_rss, "minimum_host_available_memory_bytes": min_available,
        "maximum_swap_growth_bytes": max_swap_growth, "initial_swap_used_bytes": swap0,
        "final_swap_used_bytes": psutil.swap_memory().used, "final_swap_free_bytes": psutil.swap_memory().free,
        "initial_available_memory_bytes": memory0, "initial_free_swap_bytes": free_swap0,
        "initial_disk_free_bytes": disk0, "final_disk_free_bytes": psutil.disk_usage(str(EVIDENCE)).free,
        "limits": LIMITS,
    }
    (RUN / "resource-summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["status"] == "COMPLETED" else 75 if abort else 1


if __name__ == "__main__":
    raise SystemExit(main())
