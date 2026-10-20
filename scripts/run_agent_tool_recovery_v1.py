#!/usr/bin/env python3
"""Compare restart-from-scratch with VARE step-boundary recovery after tool success."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.container_isolation import DockerIsolationRunner, IsolationLimits  # noqa: E402
from vare.trajectory_store import (  # noqa: E402
    AgentStep,
    PolicyCheckpoint,
    TrajectoryStore,
    audit_trajectory_store,
)
from vare.types import Task  # noqa: E402

PROTOCOL = ROOT / "protocols/agent_tool_recovery_v1.lock.json"
TASK_ROOT = ROOT / "benchmarks/mlsys_coding_v1/tasks/grad_accum_tail"
POLICY = PolicyCheckpoint("recovery-smoke", 0, "a" * 64)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def swap_usage() -> dict[str, int]:
    raw = subprocess.run(
        ["sysctl", "vm.swapusage"], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"used = ([0-9.]+)M.*free = ([0-9.]+)M", raw)
    if match is None:
        raise RuntimeError("cannot parse system swap usage")
    return {
        "used_bytes": int(float(match.group(1)) * 1024**2),
        "free_bytes": int(float(match.group(2)) * 1024**2),
    }


def tool_record(path: Path, operation_id: str, result: dict[str, Any]) -> None:
    encoded = (
        json.dumps({"operation_id": operation_id, **result}, sort_keys=True) + "\n"
    )
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, encoded.encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def execute_tool(
    workspace: Path, operation_log: Path, step: int, run_id: str
) -> dict[str, Any]:
    if step == 0:
        source = (TASK_ROOT / "subject/grad_accum.py").read_text(encoding="utf-8")
        old = "scale = float(accumulation_steps)"
        new = "scale = float(min(len(microbatches), accumulation_steps))"
        if old in source:
            source = source.replace(old, new, 1)
        target = workspace / "grad_accum.py"
        target.write_text(source, encoding="utf-8")
        operation_id = f"{run_id}:write-file"
        result = {"tool": "write_file", "source_sha256": file_hash(target)}
    elif step == 1:
        runner = DockerIsolationRunner()
        limits = IsolationLimits(
            image=json.loads(PROTOCOL.read_text())["task"]["image"],
            timeout_s=45,
            memory_mb=512,
            cpus=1.0,
            pids_limit=32,
            max_output_bytes=20_000,
        )
        checked = asyncio.run(
            runner.run_candidate(
                limits,
                allowed_workspace_root=workspace.parent,
                workspace=workspace,
                command=["python", "/workspace/test_public.py"],
            )
        )
        operation_id = f"{run_id}:public-test"
        result = {
            "tool": "run_public_tests",
            "returncode": checked.returncode,
            "passed": checked.returncode == 0,
            "stdout_sha256": hashlib.sha256(checked.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(checked.stderr.encode()).hexdigest(),
        }
        if not result["passed"]:
            raise RuntimeError(f"public test failed: {checked.stderr[-1000:]}")
    else:
        raise ValueError("unknown step")
    tool_record(operation_log, operation_id, result)
    return {"operation_id": operation_id, **result}


def _basic_worker(
    workspace_s: str,
    operation_log_s: str,
    fault_marker_s: str,
    inject_fault: bool,
    run_id: str,
) -> None:
    workspace, operation_log, fault_marker = map(
        Path, (workspace_s, operation_log_s, fault_marker_s)
    )
    for step in range(2):
        execute_tool(workspace, operation_log, step, run_id)
        if inject_fault and step == 1 and not fault_marker.exists():
            fault_marker.write_text("sigkill-after-tool-before-progress\n")
            os.kill(os.getpid(), signal.SIGKILL)


def _vare_worker(
    db_s: str,
    workspace_s: str,
    operation_log_s: str,
    fault_marker_s: str,
    inject_fault: bool,
    run_id: str,
    worker_id: str,
) -> None:
    store = TrajectoryStore(db_s)
    lease = store.claim("execute", worker_id, lease_seconds=3)
    if lease is None:
        raise RuntimeError("no VARE execute lease")
    state = store.resume_state(run_id)
    while state["next_step"] < 2:
        index = state["next_step"]
        result = execute_tool(Path(workspace_s), Path(operation_log_s), index, run_id)
        marker = Path(fault_marker_s)
        if inject_fault and index == 1 and not marker.exists():
            marker.write_text("sigkill-after-tool-before-step-commit\n")
            os.kill(os.getpid(), signal.SIGKILL)
        step = AgentStep(
            index=index,
            event_id=f"{run_id}-step-{index}",
            policy=POLICY,
            action={
                "kind": "tool",
                "operation_id": result["operation_id"],
                "semantics": "externally_idempotent",
            },
            observation={
                "tool": result["tool"],
                "result_sha256": hashlib.sha256(
                    json.dumps(result, sort_keys=True).encode()
                ).hexdigest(),
            },
            state={"next_step": index + 1},
        )
        store.append_step(lease, step)
        state = {**state, "next_step": index + 1, "state": step.state}
    store.finish(lease)


def start_worker(target, args: tuple) -> tuple[int | None, float]:
    context = multiprocessing.get_context("spawn")
    proc = context.Process(target=target, args=args)
    started = time.perf_counter()
    proc.start()
    proc.join(120)
    if proc.is_alive():
        proc.kill()
        proc.join(5)
        raise TimeoutError("tool worker exceeded 120 seconds")
    return proc.exitcode, time.perf_counter() - started


def final_public_validation(workspace: Path) -> bool:
    runner = DockerIsolationRunner()
    limits = IsolationLimits(
        image=json.loads(PROTOCOL.read_text())["task"]["image"],
        timeout_s=45,
        memory_mb=512,
        cpus=1.0,
        pids_limit=32,
        max_output_bytes=20_000,
    )
    result = asyncio.run(
        runner.run_candidate(
            limits,
            allowed_workspace_root=workspace.parent,
            workspace=workspace,
            command=["python", "/workspace/test_public.py"],
        )
    )
    return result.returncode == 0


def run_trial(arm: str, fault: bool, repeat: int, output: Path) -> dict[str, Any]:
    run_id = f"{arm}-{('fault' if fault else 'control')}-{repeat}"
    trial = output / run_id
    trial.mkdir(parents=True)
    workspace = trial / "candidate"
    workspace.mkdir()
    shutil.copy2(TASK_ROOT / "subject/grad_accum.py", workspace / "grad_accum.py")
    shutil.copy2(TASK_ROOT / "subject/test_public.py", workspace / "test_public.py")
    operation_log = trial / "tool_operations.jsonl"
    marker = trial / "fault_injected"
    db_path = trial / "trajectory.sqlite3"
    if arm == "vare_durable_resume":
        store = TrajectoryStore.create(db_path, run_id=run_id, max_inflight=1)
        store.register_policy(POLICY)
        store.register_verifier("public-test", 0, "b" * 64)
        store.submit(
            run_id,
            Task("grad-accum", "repair the development fixture"),
            expected_steps=2,
            group_id=run_id,
            member=0,
            group_size=1,
        )
    start = time.perf_counter()
    first_target = _basic_worker if arm == "basic_restart" else _vare_worker
    first_args = (
        (str(workspace), str(operation_log), str(marker), fault, run_id)
        if arm == "basic_restart"
        else (
            str(db_path),
            str(workspace),
            str(operation_log),
            str(marker),
            fault,
            run_id,
            run_id + "-worker-1",
        )
    )
    first_exit, first_wall = start_worker(first_target, first_args)
    recovery_started = time.perf_counter()
    retries = 0
    if first_exit == -signal.SIGKILL:
        if not fault:
            raise RuntimeError("worker was killed without a frozen fault condition")
        if arm == "vare_durable_resume":
            time.sleep(3.05)  # Wait for the persisted worker lease to expire.
        second_args = (
            (str(workspace), str(operation_log), str(marker), False, run_id)
            if arm == "basic_restart"
            else (
                str(db_path),
                str(workspace),
                str(operation_log),
                str(marker),
                False,
                run_id,
                run_id + "-worker-2",
            )
        )
        second_exit, second_wall = start_worker(first_target, second_args)
        retries = 1
        if second_exit != 0:
            raise RuntimeError(f"recovery worker exit code {second_exit}")
    elif first_exit != 0:
        raise RuntimeError(f"initial worker exit code {first_exit}")
    recovery_s = time.perf_counter() - recovery_started if retries else 0.0
    verified = final_public_validation(workspace)
    total_s = time.perf_counter() - start
    operations = [json.loads(line) for line in operation_log.read_text().splitlines()]
    counts: dict[str, int] = {}
    for row in operations:
        counts[row["operation_id"]] = counts.get(row["operation_id"], 0) + 1
    vare_audit = (
        audit_trajectory_store(db_path) if arm == "vare_durable_resume" else None
    )
    committed_steps = vare_audit["steps"] if vare_audit else 0
    return {
        "run_id": run_id,
        "arm": arm,
        "fault": fault,
        "repeat": repeat,
        "worker_exit_codes": [first_exit, *([0] if retries else [])],
        "worker_wall_seconds": round(first_wall + (second_wall if retries else 0), 6),
        "recovery_seconds": round(recovery_s, 6),
        "wall_to_final_validation_seconds": round(total_s, 6),
        "tool_execution_count": len(operations),
        "duplicate_operation_count": sum(count - 1 for count in counts.values()),
        "operation_counts": counts,
        "public_test_pass_after_recovery": verified,
        "durable_steps": committed_steps,
        "trajectory_audit": vare_audit,
        "workspace_source_sha256": file_hash(workspace / "grad_accum.py"),
        "tool_operations_sha256": file_hash(operation_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    source = TASK_ROOT / "subject/grad_accum.py"
    public_test = TASK_ROOT / "subject/test_public.py"
    if file_hash(source) != protocol["task"]["source_sha256"]:
        raise RuntimeError("task source hash mismatch")
    if file_hash(public_test) != protocol["task"]["public_test_sha256"]:
        raise RuntimeError("public test hash mismatch")
    initial_swap = swap_usage()
    disk_free_start = shutil.disk_usage(output).free
    if initial_swap["free_bytes"] < 512 * 1024**2:
        raise RuntimeError("free swap below frozen preflight minimum")
    if disk_free_start < 15_000_000_000:
        raise RuntimeError("free disk below frozen preflight minimum")
    image = protocol["task"]["image"]
    inspect = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if inspect.returncode != 0:
        raise RuntimeError("pinned ARM64 container image is unavailable")
    probe = (output / "isolation-canary").resolve()
    probe.mkdir()
    # Check network denial and absence of the hidden grader mount before model/task work.
    runner = DockerIsolationRunner()
    limits = IsolationLimits(
        image=image,
        timeout_s=10,
        memory_mb=128,
        cpus=0.5,
        pids_limit=16,
        max_output_bytes=2000,
    )
    result = asyncio.run(
        runner.run_candidate(
            limits,
            allowed_workspace_root=probe.parent,
            workspace=probe,
            command=[
                "python",
                "-c",
                "import os,socket\nassert not os.path.exists('/grader')\ns=socket.socket();\ntry:\n s.connect(('1.1.1.1',80)); raise SystemExit(3)\nexcept OSError: pass\nprint('ISOLATION_OK')",
            ],
        )
    )
    shutil.rmtree(probe)
    if result.returncode != 0 or "ISOLATION_OK" not in result.stdout:
        raise RuntimeError("candidate isolation canary failed")
    rows = []
    for fault in (False, True):
        for arm in protocol["arms"]:
            for repeat in range(1, protocol["repeats"] + 1):
                rows.append(run_trial(arm, fault, repeat, output))
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": file_hash(PROTOCOL),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "resources": {
            "swap_start": initial_swap,
            "swap_end": swap_usage(),
            "disk_free_start_bytes": disk_free_start,
            "disk_free_end_bytes": shutil.disk_usage(output).free,
        },
        "isolation_canary": {
            "passed": True,
            "network_disabled": True,
            "grader_path_unmounted": True,
            "container_image": image,
        },
        "task_hashes": {
            "source": file_hash(source),
            "public_test": file_hash(public_test),
        },
        "runs": rows,
        "status": "PASS_SYSTEM_SMOKE"
        if all(r["public_test_pass_after_recovery"] for r in rows)
        else "NON_PASS",
        "claim_limit": protocol["scope"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output / "raw_runs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    manifest = {
        path.name: file_hash(path)
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    (output / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "protocol_sha256": file_hash(PROTOCOL),
                "files": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "runs": len(rows),
                "verified": sum(r["public_test_pass_after_recovery"] for r in rows),
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "PASS_SYSTEM_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
