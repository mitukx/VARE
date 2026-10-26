#!/usr/bin/env python3
"""Execute the frozen six-cell local campaign with one controller SIGKILL.

This experiment reuses run_local_agent.py for every model/tool interaction.
Recovery is limited to the boundary after durable rollout completion and
before grading, and makes no claim about recovery inside model generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "benchmarks/vare_development_repair_v1"
PROTOCOL = ASSETS / "campaign-protocol.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def durable(path: Path, value: dict) -> None:
    data = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_protocol(protocol: dict) -> None:
    for name, digest in protocol["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"protocol source mismatch: {name}")


def kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def monitored(command: list[str], output: Path, *, seconds: int, memory_bytes: int,
              swap_growth_bytes: int, env: dict | None = None) -> dict:
    start = time.monotonic()
    swap_start = psutil.swap_memory().used
    samples = []
    abort = None
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env,
                                   cwd=ROOT, start_new_session=True)
        while process.poll() is None:
            elapsed = time.monotonic() - start
            try:
                parent = psutil.Process(process.pid)
                rss = sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)] if item.is_running())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            swap_growth = psutil.swap_memory().used - swap_start
            samples.append({"elapsed_s": elapsed, "rss_bytes": rss,
                            "swap_used_bytes": psutil.swap_memory().used,
                            "available_memory_bytes": psutil.virtual_memory().available})
            if elapsed > seconds:
                abort = "wall_limit"
            elif rss > memory_bytes:
                abort = "rss_limit"
            elif swap_growth > swap_growth_bytes:
                abort = "host_swap_growth_limit"
            if abort:
                kill_group(process)
                break
            time.sleep(0.25)
        result = {"command": command, "returncode": process.wait(),
                  "elapsed_s": time.monotonic() - start, "abort": abort,
                  "samples": samples, "peak_sampled_rss_bytes": max((s["rss_bytes"] for s in samples), default=0),
                  "swap_used_start_bytes": swap_start, "swap_used_end_bytes": psutil.swap_memory().used}
    durable(output / "process.json", result)
    return result


def check_chain(path: Path) -> tuple[int, str]:
    previous = "0" * 64
    invocations = 0
    for sequence, line in enumerate(path.read_text().splitlines(), 1):
        record = json.loads(line)
        event = record["event"]
        canonical = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        if (event["sequence"] != sequence or record["previous_event_sha256"] != previous
                or record["event_sha256"] != digest):
            raise ValueError("invalid agent event chain")
        invocations += event["event_type"] == "invocation_started"
        previous = digest
    return invocations, previous


def run_case(output: Path, case: dict, protocol: dict, *, fault: bool) -> None:
    validate_protocol(protocol)
    directory = output / case["id"]
    directory.mkdir(exist_ok=True)
    if (directory / "result.json").exists():
        raise ValueError("completed cell cannot be rerun")
    source = directory / "workspace" / case["source_path"]
    pending_path = directory / "pending-grade.json"
    recovered = pending_path.exists()
    if recovered:
        pending = json.loads(pending_path.read_text())
        if pending["protocol_sha256"] != sha(PROTOCOL) or pending["case"] != case:
            raise ValueError("recovery protocol/cell mismatch")
        for name, digest in pending["files_sha256"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"recovery artifact mismatch: {name}")
    else:
        if (directory / "inference").exists():
            raise ValueError("incomplete inference is consumed; manual audit required")
        for name, digest in case["model_files_sha256"].items():
            if sha(Path(case["model_path"]) / name) != digest:
                raise ValueError(f"model snapshot mismatch: {name}")
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        if memory.available < protocol["resources"]["minimum_available_memory_bytes"]:
            durable(directory / "preflight-rejection.json", {"reason": "available_memory", "available_bytes": memory.available})
            return
        if swap.free < protocol["resources"]["minimum_free_swap_bytes"]:
            durable(directory / "preflight-rejection.json", {"reason": "free_swap", "free_bytes": swap.free})
            return
        source.parent.mkdir(parents=True)
        source.write_bytes((ASSETS / case["baseline_asset"]).read_bytes())
        workspace = directory / "workspace"
        for args in (["init", "-q"], ["add", "."],
                     ["-c", "user.name=VARE experiment", "-c", "user.email=vare@localhost", "commit", "-qm", "Pinned candidate source"]):
            subprocess.run(["git", "-C", str(workspace), *args], check=True)
        base = subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "HEAD"], text=True).strip()
        task_root = directory / "task"
        task_root.mkdir()
        task = {"id": case["task"], "prompt": case["prompt"],
                "source": {"base_revision": base},
                "metadata": {"split": "development", "editable_files": [case["source_path"]],
                             "upstream_repository": "mitukx/VARE", "upstream_revision": case["upstream_revision"]}}
        durable(task_root / "task.json", task)
        inference = directory / "inference"
        inference.mkdir()
        command = [protocol["model_python"], str(ROOT / "scripts/development_campaign_model_entry_v1.py"),
                   "--workspace", str(workspace), "--model-path", case["model_path"],
                   "--model-repository", case["model_repository"], "--backend", "mlx",
                   "--task-root", str(task_root), "--output", str(directory / "agent-record.json"),
                   "--event-journal", str(directory / "events.jsonl"), "--instruction", case["instruction"],
                   "--seed", str(case["seed"]), "--max-turns", "12", "--max-no-tool-retries", "1",
                   "--max-new-tokens-per-turn", "1024", "--max-total-new-tokens", "4096",
                   "--max-context-tokens", "12288", "--wall-seconds", "600", "--no-do-sample",
                   "--edit-format", "structured_replace", "--tool-interface", "lease_bound_navigation_v1",
                   "--require-split", "development", "--temperature", "0", "--top-p", "0", "--top-k", "0"]
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        process = monitored(command, inference, seconds=660,
                            memory_bytes=protocol["resources"]["rss_limit_bytes"],
                            swap_growth_bytes=protocol["resources"]["swap_growth_limit_bytes"], env=env)
        if process["returncode"] != 0 or not (directory / "agent-record.json").exists():
            durable(directory / "result.json", {"case_id": case["id"], "status": "INFERENCE_FAILED",
                                               "process_returncode": process["returncode"], "abort": process["abort"]})
            return
        invocations, final_hash = check_chain(directory / "events.jsonl")
        if invocations != 1:
            raise ValueError("expected exactly one model invocation")
        retained = ["agent-record.json", "events.jsonl", source.relative_to(directory).as_posix()]
        for name in retained:
            with (directory / name).open("rb") as stream:
                os.fsync(stream.fileno())
        durable(pending_path, {"case": case, "protocol_sha256": sha(PROTOCOL),
                               "event_chain_final_sha256": final_hash,
                               "files_sha256": {name: sha(directory / name) for name in retained}})
        if fault:
            durable(directory / "fault-injected.json", {"signal": "SIGKILL", "pid": os.getpid(),
                                                       "boundary": "durable_rollout_before_grading"})
            os.kill(os.getpid(), signal.SIGKILL)
    invocations, final_hash = check_chain(directory / "events.jsonl")
    grading = directory / "grading"
    grading.mkdir()
    process = monitored([sys.executable, "-I", str(ROOT / "scripts/grade_vare_development_candidate_v1.py"),
                         "--task", case["task"], "--source", str(source)], grading,
                        seconds=30, memory_bytes=512 * 1024**2, swap_growth_bytes=128 * 1024**2)
    grade = None
    if process["returncode"] == 0:
        try:
            grade = json.loads((grading / "stdout.log").read_text())
        except (json.JSONDecodeError, OSError):
            pass
    valid = isinstance(grade, dict) and grade.get("candidate_sha256") == sha(source) and grade.get("task") == case["task"]
    durable(directory / "result.json", {
        "case_id": case["id"], "task": case["task"], "model": case["model_repository"],
        "status": "GRADED" if valid else "GRADER_FAILED", "grade": grade if valid else None,
        "candidate_sha256": sha(source), "recovered_after_sigkill": recovered,
        "model_invocations": invocations, "event_chain_final_sha256": final_hash,
        "source_changed": sha(source) != sha(ASSETS / case["baseline_asset"]),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--fault", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    validate_protocol(protocol)
    output = args.output.resolve()
    output.mkdir(exist_ok=True, parents=True)
    if args.case:
        case = next(item for item in protocol["cases"] if item["id"] == args.case)
        run_case(output, case, protocol, fault=args.fault)
        return 0
    start = time.monotonic()
    rows = []
    for case in protocol["cases"]:
        print(f"START {case['id']}", flush=True)
        command = [sys.executable, str(Path(__file__)), "--output", str(output), "--case", case["id"]]
        fault = case["id"] == protocol["fault_case"]
        process = subprocess.run(command + (["--fault"] if fault else []), check=False)
        if fault and process.returncode == -signal.SIGKILL:
            process = subprocess.run(command, check=False)
        result_path = output / case["id"] / "result.json"
        row = json.loads(result_path.read_text()) if result_path.exists() else {"case_id": case["id"], "status": "NOT_COMPLETED", "returncode": process.returncode}
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if row["status"] in {"NOT_COMPLETED", "INFERENCE_FAILED"}:
            # A shared resource failure stops remaining model admission.
            break
    durable(output / "summary.json", {"protocol_sha256": sha(PROTOCOL), "rows": rows,
                                      "elapsed_s": time.monotonic() - start})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
