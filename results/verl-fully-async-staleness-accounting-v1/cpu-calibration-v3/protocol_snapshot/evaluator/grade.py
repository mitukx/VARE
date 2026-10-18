#!/usr/bin/env python3
"""Grade the pinned fully-async rollout staleness defect with CPU fixtures."""
from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


TASK_ID = "verl-fully-async-staleness-accounting-v1"
SOURCE_PATH = "verl/experimental/fully_async_policy/fully_async_rollouter.py"
TASK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[2]
MAX_SOURCE_BYTES = 2_000_000
MAX_RESULT_BYTES = 65_536


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_task_lock(task_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task_path = task_root / "task.json"
    brief_path = task_root / "TASK.md"
    grader_path = Path(__file__).resolve()
    lock = read_json(task_root / "protocol.lock.json")
    task = read_json(task_path)
    expected = lock["locked_hashes"]
    actual = {
        "task_json_sha256": sha256(task_path),
        "task_brief_sha256": sha256(brief_path),
        "evaluator_sha256": sha256(grader_path),
    }
    if task.get("task_id") != TASK_ID or lock.get("task_id") != TASK_ID:
        raise ValueError("task id differs from the protocol lock")
    if task.get("grader", {}).get("path") != str(grader_path.relative_to(REPOSITORY_ROOT)):
        raise ValueError("grader path differs from the locked task descriptor")
    for name, value in actual.items():
        if expected.get(name) != value:
            raise ValueError(f"{name} does not match the locked protocol")
    source = task.get("source", {})
    if source.get("files") != [SOURCE_PATH] or source.get("candidate_paths") != [SOURCE_PATH]:
        raise ValueError("candidate source scope differs from the task")
    if lock.get("source", {}).get("base_revision") != source.get("base_revision"):
        raise ValueError("base revision differs from the protocol lock")
    if lock.get("source", {}).get("calibration_revision") != source.get("calibration_revision"):
        raise ValueError("calibration revision differs from the protocol lock")
    return task, lock


def git_output(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(workspace), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip()


def validate_workspace(workspace: Path, task: dict[str, Any]) -> str:
    if not workspace.is_dir():
        raise ValueError("candidate workspace does not exist")
    if workspace == TASK_ROOT or workspace.is_relative_to(TASK_ROOT):
        raise ValueError("candidate workspace overlaps trusted task files")
    source_path = workspace / SOURCE_PATH
    if source_path.is_symlink() or not source_path.resolve().is_relative_to(workspace.resolve()):
        raise ValueError("candidate source path escapes its workspace")
    if not source_path.is_file():
        raise FileNotFoundError(f"candidate checkout is missing source files: {SOURCE_PATH}")
    if source_path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("candidate source exceeds the source-size limit")
    revision = git_output(workspace, "rev-parse", "HEAD")
    allowed = {
        task["source"]["base_revision"],
        task["source"]["calibration_revision"],
    }
    if revision not in allowed:
        raise ValueError("candidate workspace is not based on a pinned task revision")
    changed = git_output(workspace, "status", "--porcelain", "--untracked-files=all")
    for line in changed.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path != SOURCE_PATH:
            raise ValueError("workspace changes a path outside the task scope")
    return revision


def _self_attribute(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _find_methods(tree: ast.Module) -> tuple[ast.AsyncFunctionDef, ast.AsyncFunctionDef]:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullyAsyncRollouter"]
    if len(classes) != 1:
        raise ValueError("FullyAsyncRollouter class is missing or ambiguous")
    methods = {
        node.name: node
        for node in classes[0].body
        if isinstance(node, ast.AsyncFunctionDef)
    }
    if "reset_staleness" not in methods or "_processor_worker" not in methods:
        raise ValueError("required async rollout methods are missing")
    return methods["reset_staleness"], methods["_processor_worker"]


class _Task:
    def __init__(self, done: bool):
        self._done = done

    def done(self) -> bool:
        return self._done


class _ResumeEvent:
    def set(self) -> None:
        return None


class _QueueClient:
    def __init__(self, total_produced: int, queue_size: int):
        self.total_produced = total_produced
        self.queue_size = queue_size

    async def get_queue_size(self) -> int:
        return self.queue_size

    async def get_statistics(self) -> dict[str, int]:
        return {"total_produced": self.total_produced, "queue_size": self.queue_size}


class _Owner:
    def __init__(self, case: dict[str, int]):
        self.lock = asyncio.Lock()
        self.paused = True
        self._resume_event = _ResumeEvent()
        self.staleness_samples = -1
        self.active_tasks = {
            *(_Task(done=True) for _ in range(case["done_tasks"])),
            *(_Task(done=False) for _ in range(case["inflight_tasks"])),
        }
        self.message_queue_client = _QueueClient(case["total_produced"], case["queue_size"])
        self._total_dispatched_samples = case["total_dispatched"]
        self.step_start_time = time.time() - 2
        self.idle_start_time = 0
        self._completed_steps = 0
        self._step_generated_samples = 0
        self._step_samples_history: deque[tuple[int, int]] = deque(maxlen=10)
        self._STEP_HISTORY_SIZE = 10

    def _compute_rollout_resource_utilization(self) -> float:
        return 0.0


def evaluate_reset(method: ast.AsyncFunctionDef) -> list[dict[str, Any]]:
    function = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace: dict[str, Any] = {
        "asyncio": asyncio,
        "time": time,
        "__builtins__": {"len": len, "max": max, "list": list, "print": print},
    }
    exec(compile(function, "<candidate-reset-staleness>", "exec"), namespace)
    reset = namespace["reset_staleness"]
    cases = [
        {
            "case_id": "completed-workers-with-unconsumed-tail",
            "done_tasks": 45,
            "inflight_tasks": 0,
            "total_dispatched": 45,
            "total_produced": 45,
            "queue_size": 7,
            "expected": 7,
        },
        {
            "case_id": "completed-workers-after-full-consumption",
            "done_tasks": 38,
            "inflight_tasks": 0,
            "total_dispatched": 38,
            "total_produced": 38,
            "queue_size": 0,
            "expected": 0,
        },
        {
            "case_id": "completed-workers-plus-live-and-queued-work",
            "done_tasks": 3,
            "inflight_tasks": 4,
            "total_dispatched": 10,
            "total_produced": 6,
            "queue_size": 6,
            "expected": 10,
        },
    ]
    results = []
    for case in cases:
        owner = _Owner(case)
        metrics = asyncio.run(asyncio.wait_for(reset(owner), timeout=2.0))
        observed = owner.staleness_samples
        results.append(
            {
                "case_id": case["case_id"],
                "expected": case["expected"],
                "observed": observed,
                "passed": observed == case["expected"],
                "returned_metrics": isinstance(metrics, dict),
            }
        )
    return results


def evaluate_dispatch_order(method: ast.AsyncFunctionDef) -> dict[str, Any]:
    capacity_waits = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.While)
        and isinstance(node.test, ast.Compare)
        and "max_concurrent_samples" in ast.unparse(node.test)
    ]
    increments = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and _self_attribute(node.target, "staleness_samples")
    ]
    creates = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "safe_create_task"
    ]
    reservations = []
    for lock_block in [node for node in ast.walk(method) if isinstance(node, ast.AsyncWith)]:
        is_self_lock = any(
            isinstance(item.context_expr, ast.Attribute)
            and item.context_expr.attr == "lock"
            and isinstance(item.context_expr.value, ast.Name)
            and item.context_expr.value.id == "self"
            for item in lock_block.items
        )
        if not is_self_lock:
            continue
        body_nodes = list(ast.walk(ast.Module(body=lock_block.body, type_ignores=[])))
        has_increment = any(
            node in increments for node in body_nodes
        )
        has_dispatch = any(node in creates for node in body_nodes)
        for capacity_wait in capacity_waits:
            reservations.append(
                lock_block.lineno > capacity_wait.lineno and has_increment and has_dispatch
            )
    return {
        "passed": bool(reservations) and any(reservations),
        "capacity_wait_line": min((node.lineno for node in capacity_waits), default=None),
        "staleness_increment_count": len(increments),
        "locked_dispatch_reservation_blocks": sum(reservations),
    }


def grade(workspace: Path, task_root: Path) -> dict[str, Any]:
    task, _lock = verify_task_lock(task_root)
    revision = validate_workspace(workspace, task)
    source_path = workspace / SOURCE_PATH
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=SOURCE_PATH)
    reset, processor = _find_methods(tree)
    runtime_cases = evaluate_reset(reset)
    dispatch_order = evaluate_dispatch_order(processor)
    passed = all(case["passed"] and case["returned_metrics"] for case in runtime_cases) and dispatch_order[
        "passed"
    ]
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "workspace_revision": revision,
        "passed": passed,
        "checks": {"reset_staleness": runtime_cases, "dispatch_reservation_order": dispatch_order},
        "metrics": {
            "reset_cases": len(runtime_cases),
            "reset_cases_passed": sum(case["passed"] for case in runtime_cases),
            "dispatch_reservation_order_passed": dispatch_order["passed"],
        },
        "runtime": {"python": platform.python_version(), "cpu_only": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, default=TASK_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    task_root = args.task_root.expanduser().resolve()
    result_path = args.json_out.expanduser().resolve() if args.json_out is not None else None
    if result_path is not None and (result_path.is_relative_to(workspace) or result_path.is_relative_to(task_root)):
        parser.error("result file must be outside the candidate workspace and locked task files")
    try:
        result = grade(workspace, task_root)
    except Exception as exc:
        result = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "grader_error": type(exc).__name__,
            "message": str(exc),
        }
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if result_path is not None:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 1
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        print("grader result exceeds the output limit", file=sys.stderr)
        return 2
    if result_path is not None:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] or args.allow_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
