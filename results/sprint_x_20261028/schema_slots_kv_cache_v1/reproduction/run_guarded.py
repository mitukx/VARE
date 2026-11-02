#!/usr/bin/env python3
"""Run the frozen MLX prefill benchmark under the existing VARE host caps."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil


EVIDENCE = Path(__file__).resolve().parents[1]
REPO = Path("/tmp/vare-sprint-x-kv-cache-v1")
MODEL = Path(
    "/Users/user/.cache/huggingface/hub/models--mlx-community--Qwen2.5-Coder-7B-Instruct-4bit"
    "/snapshots/019cc73c45c770444708a6dd8690c66243cc5c80"
)
PYTHON = Path("/tmp/vare-sprint-vii-mlx/bin/python")
PROTOCOL = EVIDENCE / "protocol-v2.json"
FREEZE = EVIDENCE / "execution-freeze-v2.json"
RUN = EVIDENCE / "run-002"
MODEL_OUTPUT = RUN / "model-output"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def terminate_group(proc: subprocess.Popen[bytes]) -> None:
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
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol_sha = digest(PROTOCOL)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if (protocol.get("status") != "FROZEN_BEFORE_MODEL_INFERENCE"
            or protocol.get("protocol_id") != "vare_sprint_x_schema_slots_kv_cache_v2"
            or freeze.get("protocol_sha256") != protocol_sha
            or freeze.get("model_inference_started") is not False):
        raise SystemExit("refusing model execution without an intact frozen protocol")
    if RUN.exists():
        raise SystemExit(f"refusing to overwrite benchmark run: {RUN}")
    if not REPO.is_dir() or not MODEL.is_dir() or not PYTHON.is_file():
        raise SystemExit("pinned repository, model snapshot, or Python environment is missing")

    implementation = protocol["implementation"]
    for relative_path, expected_sha in implementation["script_hashes"].items():
        if digest(EVIDENCE / relative_path) != expected_sha:
            raise SystemExit(f"frozen script identity mismatch: {relative_path}")
    if digest(EVIDENCE / "input.json") != protocol["task_and_exposure"]["input_sha256"]:
        raise SystemExit("frozen synthetic input identity mismatch")
    if digest(EVIDENCE / "tokenizer-preflight.json") != implementation["tokenizer_preflight_sha256"]:
        raise SystemExit("frozen tokenizer preflight identity mismatch")
    if digest(EVIDENCE / "model-snapshot-manifest.json") != protocol["model"]["snapshot_manifest_sha256"]:
        raise SystemExit("frozen model manifest identity mismatch")
    if digest(MODEL / "model.safetensors") != protocol["model"]["weights_sha256"]:
        raise SystemExit("cached model weights do not match the frozen SHA-256")
    repo_revision = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
    ).strip()
    if repo_revision != protocol["implementation"]["repository_commit"]:
        raise SystemExit("pinned repository revision mismatch")
    dirty = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise SystemExit("pinned repository has tracked modifications")
    for arm, paths in implementation["controller_source_hashes"].items():
        for relative_path, expected_sha in paths.items():
            if digest(REPO / relative_path) != expected_sha:
                raise SystemExit(f"frozen controller source identity mismatch: {arm}:{relative_path}")

    limits = protocol["resources"]
    RUN.mkdir(parents=True)
    swap0 = psutil.swap_memory().used
    memory0 = psutil.virtual_memory().available
    disk0 = psutil.disk_usage(str(EVIDENCE)).free
    free_swap0 = psutil.swap_memory().free
    if (memory0 < limits["minimum_available_memory_bytes"]
            or free_swap0 < limits["minimum_free_swap_bytes"]
            or disk0 < limits["minimum_disk_free_bytes"]):
        write_json(RUN / "resource-preflight-no-go.json", {
            "decision": "NO_GO",
            "available_memory_bytes": memory0,
            "free_swap_bytes": free_swap0,
            "disk_free_bytes": disk0,
            "limits": limits,
        })
        return 75

    command = [
        str(PYTHON),
        str(EVIDENCE / "reproduction" / "run_microbenchmark.py"),
        "--repo", str(REPO), "--model", str(MODEL),
        "--input", str(EVIDENCE / "input.json"), "--output", str(MODEL_OUTPUT),
    ]
    env = os.environ.copy()
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src"))),
    })
    write_json(RUN / "command.json", {
        "argv": command,
        "cwd": str(REPO),
        "limits": limits,
        "protocol_sha256": protocol_sha,
        "initial_available_memory_bytes": memory0,
        "initial_swap_used_bytes": swap0,
        "initial_free_swap_bytes": free_swap0,
        "initial_disk_free_bytes": disk0,
        "offline_environment": {key: env[key] for key in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM",
        )},
    })
    (RUN / "stdout.log").touch()
    (RUN / "stderr.log").touch()
    with (RUN / "stdout.log").open("wb") as stdout, (RUN / "stderr.log").open("wb") as stderr:
        proc = subprocess.Popen(
            command, cwd=REPO, env=env, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        start = time.monotonic()
        max_rss = 0
        min_available = memory0
        max_swap_growth = 0
        abort = None
        with (RUN / "resource-samples.jsonl").open("x", encoding="utf-8") as samples:
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
                current_attempt_limit = None
                for start_record in sorted(MODEL_OUTPUT.glob("attempt-*/attempt-start.json")):
                    attempt_dir = start_record.parent
                    if (attempt_dir / "attempt-summary.json").exists():
                        continue
                    try:
                        started_unix = json.loads(start_record.read_text())["started_unix_seconds"]
                        current_attempt_limit = max(0.0, time.time() - started_unix)
                    except (OSError, KeyError, json.JSONDecodeError, TypeError):
                        current_attempt_limit = 0.0
                    break
                record = {
                    "elapsed_seconds": round(elapsed, 3),
                    "process_tree_rss_bytes": rss,
                    "process_tree_rss_peak_bytes": max_rss,
                    "host_available_memory_bytes": vm.available,
                    "host_swap_used_bytes": sw.used,
                    "host_swap_growth_bytes": growth,
                    "host_free_swap_bytes": sw.free,
                    "disk_free_bytes": disk,
                    "active_attempt_wall_seconds": current_attempt_limit,
                }
                samples.write(json.dumps(record, sort_keys=True) + "\n")
                samples.flush()
                os.fsync(samples.fileno())
                if elapsed > limits["max_outer_wall_seconds"]:
                    abort = "outer_wall_limit"
                elif current_attempt_limit is not None and current_attempt_limit > limits["max_agent_wall_seconds"]:
                    abort = "attempt_wall_limit"
                elif rss > limits["max_process_tree_rss_bytes"]:
                    abort = "process_tree_rss_limit"
                elif growth > limits["max_swap_growth_bytes"]:
                    abort = "swap_growth_limit"
                elif vm.available < limits["minimum_available_memory_bytes"]:
                    abort = "host_available_memory_floor"
                elif sw.free < limits["minimum_free_swap_bytes"]:
                    abort = "host_free_swap_floor"
                elif disk < limits["minimum_disk_free_bytes"]:
                    abort = "disk_free_floor"
                if abort:
                    terminate_group(proc)
                    break
                time.sleep(limits["sample_interval_seconds"])
    returncode = proc.poll()
    summary = {
        "status": "RESOURCE_ABORT" if abort else ("COMPLETED" if returncode == 0 else "RUNTIME_FAILURE"),
        "abort_reason": abort,
        "returncode": returncode,
        "outer_elapsed_seconds": round(time.monotonic() - start, 3),
        "peak_process_tree_rss_bytes": max_rss,
        "minimum_host_available_memory_bytes": min_available,
        "maximum_swap_growth_bytes": max_swap_growth,
        "initial_swap_used_bytes": swap0,
        "final_swap_used_bytes": psutil.swap_memory().used,
        "final_free_swap_bytes": psutil.swap_memory().free,
        "initial_available_memory_bytes": memory0,
        "initial_free_swap_bytes": free_swap0,
        "initial_disk_free_bytes": disk0,
        "final_disk_free_bytes": psutil.disk_usage(str(EVIDENCE)).free,
        "limits": limits,
    }
    write_json(RUN / "resource-summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if summary["status"] == "COMPLETED" else 75 if abort else 1


if __name__ == "__main__":
    raise SystemExit(main())
