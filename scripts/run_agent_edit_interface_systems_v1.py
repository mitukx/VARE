#!/usr/bin/env python3
"""Systems-only known-fix replay on two real MLX-LM Development tasks.

This runner compares serial and two-worker execution of the same deterministic
known-fix controller. It does not run a language model and is not capability
or learning evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.agent_edit_v2 import SourceEditLeaseStoreV1  # noqa: E402
from scripts.run_local_agent import read_file_range, replace_text  # noqa: E402
from scripts.run_local_agent import DurableEventJournal  # noqa: E402
from scripts.calibrate_agent_edit_interface_v1 import tool_output  # noqa: E402


TASKS: dict[str, dict[str, Any]] = {
    "mlxlm-2004-jamba-moe-routing": {
        "base": "4cfc6478473b0668ede3eb0cc93f2a9181a15052",
        "fix": "2cd1088e83aabf132e33647b97a25cfad23af759",
        "target": "mlx_lm/models/jamba.py",
        "grader": "benchmarks/mlsys_agent_development_v1/tasks/mlxlm-2004-jamba-moe-routing/evaluator/test_hidden.py",
    },
    "mlxlm-1996-nested-parameter-tags": {
        "base": "72be8c955d9a0d3099bd01865af288e0e947c3fa",
        "fix": "733925103546131e0e79bce0bcb2ff813939656c",
        "target": "mlx_lm/tool_parsers/qwen3_coder.py",
        "grader": "benchmarks/mlsys_agent_development_v1/tasks/mlxlm-1996-nested-parameter-tags/evaluator/test_hidden.py",
    },
}
INTERFACES = ("lease_bound_lines_v1", "legacy_structured_replace")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_bytes(path: Path, payload: bytes) -> None:
    """Create a durable, non-overwriting evidence file."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("evidence write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def write_new_json(path: Path, value: Any) -> None:
    write_new_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


class ResourceMonitor:
    """Record host and process-tree telemetry and fail closed at frozen limits."""

    def __init__(self, output: Path, watched_path: Path, limits: dict[str, int | float]):
        import psutil

        self.psutil = psutil
        self.process = psutil.Process(os.getpid())
        self.watched_path = watched_path
        self.limits = limits
        self.interval = float(limits["sample_interval_seconds"])
        self.path = output / "resources.jsonl"
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._summary: dict[str, Any] | None = None
        self.abort_reason: str | None = None
        self.baseline_swap = psutil.swap_memory().used
        self.started_monotonic = time.monotonic()
        self.samples: list[dict[str, Any]] = []
        self._cpu_by_pid: dict[int, float] = {}
        self._record("preflight")
        preflight = self.samples[-1]
        if preflight["available_memory_bytes"] < limits["minimum_available_memory_bytes"]:
            os.close(self._fd)
            self._fd = -1
            raise RuntimeError("preflight available memory is below the frozen floor")
        if preflight["free_disk_bytes"] < limits["minimum_disk_free_bytes"]:
            os.close(self._fd)
            self._fd = -1
            raise RuntimeError("preflight free disk is below the frozen floor")
        if preflight["free_swap_bytes"] < limits["minimum_free_swap_bytes"]:
            os.close(self._fd)
            self._fd = -1
            raise RuntimeError("preflight free swap is below the frozen floor")
        if (preflight["process_tree_rss_bytes"] > limits["max_process_tree_rss_bytes"]):
            os.close(self._fd)
            self._fd = -1
            raise RuntimeError("preflight process-tree RSS exceeds the frozen ceiling")

    def _process_tree_rss(self) -> int:
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except self.psutil.Error:
            pass
        total = 0
        seen: set[int] = set()
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                total += process.memory_info().rss
                cpu = process.cpu_times()
                self._cpu_by_pid[process.pid] = max(
                    self._cpu_by_pid.get(process.pid, 0.0), cpu.user + cpu.system,
                )
            except self.psutil.Error:
                continue
        return total

    def _record(self, phase: str) -> None:
        memory = self.psutil.virtual_memory()
        swap = self.psutil.swap_memory()
        swap_used = swap.used
        disk_free = self.psutil.disk_usage(self.watched_path).free
        row = {
            "timestamp_unix_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "phase": phase,
            "available_memory_bytes": memory.available,
            "total_memory_bytes": memory.total,
            "swap_used_bytes": swap_used,
            "free_swap_bytes": swap.free,
            "swap_growth_from_preflight_bytes": max(0, swap_used - self.baseline_swap),
            "free_disk_bytes": disk_free,
            "process_tree_rss_bytes": self._process_tree_rss(),
            "observed_process_tree_cpu_seconds": round(sum(self._cpu_by_pid.values()), 6),
        }
        self.samples.append(row)
        encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        view = memoryview(encoded)
        while view:
            written = os.write(self._fd, view)
            if written <= 0:
                raise OSError("resource sampler write made no progress")
            view = view[written:]
        os.fsync(self._fd)
        if row["available_memory_bytes"] < self.limits["minimum_available_memory_bytes"]:
            self.abort_reason = "available_memory_below_frozen_floor"
        elif row["swap_growth_from_preflight_bytes"] > self.limits["max_swap_growth_bytes"]:
            self.abort_reason = "swap_growth_above_frozen_ceiling"
        elif row["free_disk_bytes"] < self.limits["minimum_disk_free_bytes"]:
            self.abort_reason = "free_disk_below_frozen_floor"
        elif row["free_swap_bytes"] < self.limits["minimum_free_swap_bytes"]:
            self.abort_reason = "free_swap_below_frozen_floor"
        elif row["process_tree_rss_bytes"] > self.limits["max_process_tree_rss_bytes"]:
            self.abort_reason = "process_tree_rss_above_frozen_ceiling"
        elif time.monotonic() - self.started_monotonic > self.limits["maximum_wall_seconds"]:
            self.abort_reason = "maximum_experiment_wall_time_exceeded"
        if self.abort_reason:
            self._stop.set()

    def start(self) -> None:
        if self.abort_reason:
            raise RuntimeError(f"resource preflight failed: {self.abort_reason}")
        self._thread = threading.Thread(target=self._sample_loop, name="vare-resource-sampler", daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._record("running")
            except Exception as exc:  # fail closed if telemetry itself fails
                self.abort_reason = f"resource_sampling_error:{type(exc).__name__}"
                self._stop.set()

    def check(self) -> None:
        if self.abort_reason:
            raise RuntimeError(f"resource abort: {self.abort_reason}")

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self._summary or {}
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=max(1.0, self.interval * 4))
            self._thread = None
        try:
            if self._fd >= 0:
                self._record("final")
        finally:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            self._closed = True
        self._summary = {
            "sample_count": len(self.samples),
            "minimum_available_memory_bytes": min(
                item["available_memory_bytes"] for item in self.samples
            ),
            "maximum_swap_growth_bytes": max(
                item["swap_growth_from_preflight_bytes"] for item in self.samples
            ),
            "minimum_free_disk_bytes": min(item["free_disk_bytes"] for item in self.samples),
            "minimum_free_swap_bytes": min(item["free_swap_bytes"] for item in self.samples),
            "peak_process_tree_rss_bytes": max(
                item["process_tree_rss_bytes"] for item in self.samples
            ),
            "observed_process_tree_cpu_seconds": max(
                item["observed_process_tree_cpu_seconds"] for item in self.samples
            ),
            "abort_reason": self.abort_reason,
            "resource_sample_sha256": sha256_path(self.path),
        }
        return self._summary


class TokenizerCounter:
    """Use the calibrated tokenizer environment without loading model weights."""

    def __init__(self, python: Path, tokenizer_dir: Path, stderr_path: Path) -> None:
        self._lock = threading.Lock()
        self._stderr = stderr_path.open("xb")
        environment = os.environ.copy()
        environment.update({
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
        })
        counter = '''
import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    text = json.loads(line)
    print(len(tokenizer.encode(text, add_special_tokens=False)), flush=True)
'''
        self.process = subprocess.Popen(
            [str(python), "-c", counter, str(tokenizer_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, env=environment, bufsize=1,
        )
        ready = self.process.stdout.readline() if self.process.stdout else ""
        if ready != '{"ready": true}\n':
            self.close()
            raise RuntimeError(f"pinned tokenizer worker failed to initialize: {ready!r}")

    def count(self, serialized_call: str) -> int:
        with self._lock:
            if self.process.poll() is not None or self.process.stdin is None or self.process.stdout is None:
                raise RuntimeError("pinned tokenizer worker is not alive")
            self.process.stdin.write(json.dumps(serialized_call, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            response = self.process.stdout.readline()
            if not response:
                raise RuntimeError("pinned tokenizer worker returned no token count")
            try:
                count = int(response)
            except ValueError as exc:
                raise RuntimeError(f"invalid pinned tokenizer response: {response!r}") from exc
            if count < 0:
                raise RuntimeError("pinned tokenizer returned a negative token count")
            return count

    def close(self) -> int | None:
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            if process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process is not None and process.stdout:
            process.stdout.close()
        if getattr(self, "_stderr", None) is not None and not self._stderr.closed:
            self._stderr.close()
        return process.returncode if process is not None else None


def git_blob(repo: Path, revision: str, target: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{revision}:{target}"],
        check=True, capture_output=True,
    ).stdout


def changed_opcodes(base: str, fixed: str) -> list[tuple[str, int, int, int, int]]:
    a = base.splitlines(keepends=True)
    b = fixed.splitlines(keepends=True)
    return [
        (tag, i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            a=a, b=b, autojunk=False,
        ).get_opcodes()
        if tag != "equal"
    ]


def _contextual_replacement(
    lines: list[str], fixed_lines: list[str], *, start: int, end: int,
    fixed_start: int, fixed_end: int, require_unique: bool,
) -> tuple[int, int, str, str]:
    """Build a nonempty contextual replacement around one diff opcode."""
    left = max(0, start - 1)
    right = min(len(lines), max(end, start + 1) + 1)
    while True:
        old = "".join(lines[left:right])
        new = "".join(lines[left:start] + fixed_lines[fixed_start:fixed_end] + lines[end:right])
        if old and (not require_unique or "".join(lines).count(old) == 1):
            return left, right, old, new
        if left > 0:
            left -= 1
        elif right < len(lines):
            right += 1
        else:
            raise ValueError("could not construct a unique nonempty replacement context")


def _apply_known_fix(
    workspace: Path, task_id: str, interface: str, fixed_text: str,
    tokenizer: Any, journal: DurableEventJournal | None = None,
) -> dict[str, Any]:
    task = TASKS[task_id]
    target = task["target"]
    allowed = {target}
    path = workspace / target
    base_text = path.read_text(encoding="utf-8")
    fixed_lines = fixed_text.splitlines(keepends=True)
    operations = changed_opcodes(base_text, fixed_text)
    if not operations:
        raise ValueError("pinned known fix has no target-file change")

    lease_counter = 0

    def lease_token() -> str:
        nonlocal lease_counter
        lease_counter += 1
        return f"{lease_counter:012x}"

    leases = SourceEditLeaseStoreV1(workspace, allowed, token_factory=lease_token)
    actions: list[dict[str, Any]] = []

    def count_tokens(serialized: str) -> int | None:
        if tokenizer is None:
            return None
        if hasattr(tokenizer, "count"):
            return tokenizer.count(serialized)
        return len(tokenizer.encode(serialized, add_special_tokens=False))

    offset = 0
    for ordinal, (_tag, i1, i2, j1, j2) in enumerate(operations, 1):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        current_start = i1 + offset
        current_end = i2 + offset
        left, right, old_text, new_text = _contextual_replacement(
            lines, fixed_lines, start=current_start, end=current_end,
            fixed_start=j1, fixed_end=j2,
            require_unique=interface == "legacy_structured_replace",
        )
        start_line, end_line = left + 1, right
        if interface == "lease_bound_lines_v1":
            read_action = tool_output("read_lines", {
                "path": target, "start": start_line, "end": end_line,
            })
            observed = leases.read_lines(target, start_line, end_line)
            if observed.get("status") != "read_complete" or observed.get("content") != old_text:
                raise RuntimeError(f"lease read did not match selected source range: {observed}")
            actions.append({
                "ordinal": ordinal, "name": "read_lines", "arguments": {
                    "path": target, "start": start_line, "end": end_line,
                }, "serialized_tool_call": read_action,
                "output_tokens": count_tokens(read_action),
                "tool_response": observed,
                "observed_content_sha256": sha256_bytes(observed["content"].encode("utf-8")),
            })
            replace_action = tool_output("replace_lines", {
                "lease": observed["lease"], "text": new_text,
            })
            applied = leases.replace_lines(observed["lease"], new_text)
            if applied.get("applied") is not True:
                raise RuntimeError(f"hash-bound operation rejected known fix: {applied}")
            actions.append({
                "ordinal": ordinal, "name": "replace_lines", "arguments": {
                    "lease": observed["lease"], "text": new_text,
                }, "serialized_tool_call": replace_action,
                "output_tokens": count_tokens(replace_action),
                "tool_response": applied,
            })
        elif interface == "legacy_structured_replace":
            read_action = tool_output("read_file_range", {
                "path": target, "start_line": start_line, "end_line": end_line,
            })
            read_result = read_file_range(workspace, target, start_line, end_line, allowed)
            normalized_context = "\n".join(old_text.splitlines())
            if (read_result.get("resolved_path") != target
                    or read_result.get("content") != normalized_context):
                raise RuntimeError(f"legacy bounded read did not return expected source: {read_result}")
            actions.append({
                "ordinal": ordinal, "name": "read_file_range", "arguments": {
                    "path": target, "start_line": start_line, "end_line": end_line,
                }, "serialized_tool_call": read_action,
                "output_tokens": count_tokens(read_action),
                "tool_response": read_result,
                "observed_content_sha256": sha256_bytes(old_text.encode("utf-8")),
            })
            replace_action = tool_output("replace_text", {
                "path": target, "old_text": old_text, "new_text": new_text,
            })
            applied = replace_text(workspace, target, old_text, new_text, allowed)
            if applied.get("applied") is not True:
                raise RuntimeError(f"exact structured replacement rejected known fix: {applied}")
            actions.append({
                "ordinal": ordinal, "name": "replace_text", "arguments": {
                    "path": target, "old_text": old_text, "new_text": new_text,
                }, "serialized_tool_call": replace_action,
                "output_tokens": count_tokens(replace_action),
                "tool_response": applied,
            })
        else:
            raise ValueError(f"unknown interface: {interface}")

        if journal is not None:
            actions_this_hunk = actions[-2:]
            journal.append(
                "deterministic_tool_calls", task_id=task_id,
                interface=interface, ordinal=ordinal, calls=actions_this_hunk,
            )
            journal.append(
                "deterministic_tool_action_committed", task_id=task_id,
                interface=interface, ordinal=ordinal,
                before_sha256=sha256_bytes(old_text.encode("utf-8")),
                after_sha256=sha256_path(path),
            )
        offset += (j2 - j1) - (i2 - i1)

    actual_hash = sha256_path(path)
    expected_hash = sha256_bytes(fixed_text.encode("utf-8"))
    if actual_hash != expected_hash:
        raise RuntimeError(f"controller output differs from exact upstream fix: {actual_hash}")
    return {
        "task_id": task_id,
        "interface": interface,
        "base_sha256": sha256_bytes(base_text.encode("utf-8")),
        "fixed_sha256": actual_hash,
        "expected_fixed_sha256": expected_hash,
        "operation_count": len(operations),
        "tool_action_count": len(actions),
        "serialized_tool_call_tokens": sum(item["output_tokens"] or 0 for item in actions),
        "actions": actions,
        "lease_audit": leases.audit_records() if interface == "lease_bound_lines_v1" else [],
    }


def _grade(
    workspace: Path, task_id: str, venv_python: Path, monitor: ResourceMonitor,
) -> dict[str, Any]:
    grader = ROOT / TASKS[task_id]["grader"]
    command = [
        "sandbox-exec", "-p", "(version 1) (deny network*) (allow default)",
        str(venv_python), "-m", "unittest", "discover", "-s", str(grader.parent),
        "-p", grader.name, "-v",
    ]
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(workspace), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command, cwd=workspace, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if monitor.abort_reason or time.monotonic() - started > 180:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        stdout, stderr = process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        stdout, stderr = process.communicate()
                    if monitor.abort_reason:
                        return {
                            "command": command, "network_denied": True,
                            "grader_sha256": sha256_path(grader), "returncode": process.returncode,
                            "passed": False, "aborted": True,
                            "abort_reason": monitor.abort_reason,
                            "stdout": stdout, "stderr": stderr,
                            "stdout_sha256": sha256_bytes(stdout.encode()),
                            "stderr_sha256": sha256_bytes(stderr.encode()),
                            "elapsed_seconds": round(time.monotonic() - started, 4),
                        }
                    return {
                        "command": command, "network_denied": True,
                        "grader_sha256": sha256_path(grader), "returncode": process.returncode,
                        "passed": False, "timed_out": True,
                        "stdout": stdout, "stderr": stderr,
                        "stdout_sha256": sha256_bytes(stdout.encode()),
                        "stderr_sha256": sha256_bytes(stderr.encode()),
                        "elapsed_seconds": round(time.monotonic() - started, 4),
                    }
        monitor.check()
        return {
            "command": command, "network_denied": True,
            "grader_sha256": sha256_path(grader), "returncode": process.returncode,
            "passed": process.returncode == 0,
            "stdout": stdout, "stderr": stderr,
            "stdout_sha256": sha256_bytes(stdout.encode()),
            "stderr_sha256": sha256_bytes(stderr.encode()),
            "elapsed_seconds": round(time.monotonic() - started, 4),
        }
    except Exception:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        raise


def execute_cell(
    *, source_repo: Path, workspace: Path, output: Path, task_id: str,
    interface: str, tokenizer: Any, venv_python: Path,
    queue_submitted_at: float, monitor: ResourceMonitor,
) -> dict[str, Any]:
    task = TASKS[task_id]
    started = time.monotonic()
    monitor.check()
    base_bytes = git_blob(source_repo, task["base"], task["target"])
    fix_bytes = git_blob(source_repo, task["fix"], task["target"])
    if sha256_bytes((workspace / task["target"]).read_bytes()) != sha256_bytes(base_bytes):
        raise RuntimeError("candidate workspace target does not match immutable base blob")
    fixed_text = fix_bytes.decode("utf-8", errors="strict")
    journal_path = output / "events.jsonl"
    journal = DurableEventJournal(journal_path)
    journal.append(
        "deterministic_run_context", task_id=task_id, interface=interface,
        base_revision=task["base"], fix_revision=task["fix"],
        target=task["target"], base_sha256=sha256_bytes(base_bytes),
        fix_sha256=sha256_bytes(fix_bytes),
    )
    try:
        controller = _apply_known_fix(
            workspace, task_id, interface, fixed_text, tokenizer, journal=journal,
        )
        monitor.check()
        journal.append(
            "known_fix_controller_complete", task_id=task_id, interface=interface,
            target_sha256=sha256_path(workspace / task["target"]),
        )
        journal.append(
            "known_fix_controller_complete", task_id=task_id, interface=interface,
            target_sha256=sha256_path(workspace / task["target"]),
        )
    finally:
        journal.close()
    grade = _grade(workspace, task_id, venv_python, monitor)
    write_new_bytes(output / "trusted-grade.stdout.log", grade["stdout"].encode())
    write_new_bytes(output / "trusted-grade.stderr.log", grade["stderr"].encode())
    final = {
        "schema_version": 1, "task_id": task_id, "interface": interface,
        "workspace_revision": subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "workspace_path": str(workspace),
        "controller": controller, "independent_grade": {
            key: value for key, value in grade.items() if key not in {"stdout", "stderr"}
        },
        "diff_sha256": sha256_bytes(subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", "HEAD", "--", task["target"]],
            check=True, capture_output=True,
        ).stdout),
        "queue_inclusive_elapsed_seconds": round(time.monotonic() - queue_submitted_at, 4),
        "service_elapsed_seconds": round(time.monotonic() - started, 4),
        "journal_sha256": sha256_path(journal_path),
        "journal_event_count": len(journal_path.read_text(encoding="utf-8").splitlines()),
    }
    result_path = output / "cell-result.json"
    write_new_json(result_path, final)
    return final


def apply_and_crash_after_commit(
    source_repo_text: str, workspace_text: str, task_id: str,
    journal_text: str,
) -> None:
    """Crash-injection worker: fsync one real edit and kill before grading."""
    source_repo, workspace, journal_path = map(Path, (source_repo_text, workspace_text, journal_text))
    task = TASKS[task_id]
    base = git_blob(source_repo, task["base"], task["target"])
    if sha256_path(workspace / task["target"]) != sha256_bytes(base):
        raise RuntimeError("crash-injection workspace does not match pinned base target")
    fixed = git_blob(source_repo, task["fix"], task["target"]).decode("utf-8")
    journal = DurableEventJournal(journal_path)
    journal.append("crash_test_context", task_id=task_id, interface="lease_bound_lines_v1")
    _apply_known_fix(workspace, task_id, "lease_bound_lines_v1", fixed, tokenizer=None, journal=journal)
    journal.append("worker_killed_after_edit_before_independent_grade")
    journal.close()
    os.kill(os.getpid(), signal.SIGKILL)


def verify_event_chain(path: Path) -> tuple[int, str]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise RuntimeError("event journal is empty")
    previous = "0" * 64
    for sequence, record in enumerate(records, 1):
        if record["event"].get("sequence") != sequence:
            raise RuntimeError("event journal sequence is not contiguous")
        if record["previous_event_sha256"] != previous:
            raise RuntimeError("event journal predecessor hash mismatch")
        canonical = json.dumps(
            record["event"], ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        event_hash = sha256_bytes(canonical.encode("utf-8"))
        if event_hash != record["event_sha256"]:
            raise RuntimeError("event journal event hash mismatch")
        previous = event_hash
    return len(records), previous


def git_text(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def validate_protocol(
    protocol: dict[str, Any], protocol_path: Path, source_repo: Path,
    tokenizer_dir: Path, venv_python: Path,
) -> dict[str, Any]:
    """Check immutable code, task, grader, tokenizer and runtime identities."""
    sidecar = Path(str(protocol_path) + ".sha256")
    expected_protocol_sha = sidecar.read_text(encoding="utf-8").split()[0]
    if sha256_path(protocol_path) != expected_protocol_sha:
        raise RuntimeError("frozen systems protocol SHA-256 does not match sidecar")
    if protocol.get("protocol_id") != "agent-edit-interface-systems-only-v1":
        raise RuntimeError("unexpected systems-only protocol identity")
    if protocol.get("status") != "FROZEN_BEFORE_SYSTEMS_REPLAY":
        raise RuntimeError("systems-only protocol is not frozen")
    if protocol.get("interfaces") != list(INTERFACES):
        raise RuntimeError("frozen interface list differs from runner")
    if protocol.get("execution_modes") != [
        {"name": "synchronous", "worker_count": 1},
        {"name": "asynchronous", "worker_count": 2},
    ]:
        raise RuntimeError("frozen execution modes differ from runner")
    if protocol.get("cells_per_mode") != 4 or protocol.get("recovery_injections") != 1:
        raise RuntimeError("frozen workload size differs from runner")
    if (protocol.get("grader_timeout_seconds") != 180
            or protocol["resource_limits"].get("maximum_wall_seconds") != 1200):
        raise RuntimeError("frozen time budgets differ from runner")

    expected_runner_commit = protocol["runner_commit"]
    actual_commit = git_text(ROOT, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", expected_runner_commit, "HEAD"],
        check=False, capture_output=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("frozen runner commit is not an ancestor of the experiment checkout")
    for relative, expected in {
        **protocol["implementation_files_sha256"],
        **protocol["supporting_files_sha256"],
    }.items():
        path = (ROOT / relative).resolve(strict=True)
        if ROOT not in path.parents or sha256_path(path) != expected:
            raise RuntimeError(f"frozen file hash mismatch: {relative}")

    source_repo = source_repo.resolve(strict=True)
    origin = git_text(source_repo, "remote", "get-url", "origin")
    if origin != protocol["source_repository"]:
        raise RuntimeError("upstream source repository origin differs from frozen protocol")
    pinned_tasks = protocol["task_cases"]
    if set(pinned_tasks) != set(TASKS):
        raise RuntimeError("protocol task set differs from runner task set")
    for task_id, task in TASKS.items():
        pinned = pinned_tasks[task_id]
        for key in ("base", "fix", "target", "grader"):
            if pinned[key] != task[key]:
                raise RuntimeError(f"protocol {key} identity mismatch for {task_id}")
        base_blob = git_blob(source_repo, task["base"], task["target"])
        fix_blob = git_blob(source_repo, task["fix"], task["target"])
        if sha256_bytes(base_blob) != pinned["base_target_sha256"]:
            raise RuntimeError(f"base target hash mismatch for {task_id}")
        if sha256_bytes(fix_blob) != pinned["fix_target_sha256"]:
            raise RuntimeError(f"fix target hash mismatch for {task_id}")
        grader_path = ROOT / task["grader"]
        if sha256_path(grader_path) != pinned["grader_sha256"]:
            raise RuntimeError(f"trusted grader hash mismatch for {task_id}")
        for control_name, control in pinned["prior_qualification_controls"].items():
            control_path = (ROOT / control["path"]).resolve(strict=True)
            if ROOT not in control_path.parents or sha256_path(control_path) != control["sha256"]:
                raise RuntimeError(f"qualification control hash mismatch for {task_id}/{control_name}")
            record = json.loads(control_path.read_text(encoding="utf-8"))
            if (record.get("task_id") != task_id
                    or record.get("state") != control["state"]
                    or record.get("expected") != control["expected"]
                    or record.get("expectation_met") is not True
                    or record.get("grader_sha256") != pinned["grader_sha256"]):
                raise RuntimeError(f"qualification control record mismatch for {task_id}/{control_name}")

    tokenizer_dir = tokenizer_dir.resolve(strict=True)
    for relative, expected in protocol["tokenizer_files_sha256"].items():
        if sha256_path(tokenizer_dir / relative) != expected:
            raise RuntimeError(f"tokenizer identity mismatch: {relative}")
    if tokenizer_dir.name != protocol["tokenizer_snapshot"]:
        raise RuntimeError("tokenizer snapshot directory does not match frozen identity")

    import psutil
    venv_python = Path(os.path.abspath(venv_python))
    if not venv_python.is_file():
        raise RuntimeError("grader/tokenizer Python executable does not exist")
    tokenizer_runtime = json.loads(subprocess.run(
        [str(venv_python), "-c",
         "import importlib.metadata as m, json, platform; "
         "print(json.dumps({'python':platform.python_version(),"
         "'transformers':m.version('transformers'),"
         "'tokenizers':m.version('tokenizers'),"
         "'mlx_lm':m.version('mlx-lm')}))"],
        check=True, capture_output=True, text=True,
    ).stdout)
    actual_runtime = {
        "sampler_python": platform.python_version(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "sampler_psutil": psutil.__version__,
        "total_memory_bytes": psutil.virtual_memory().total,
        "cpu_brand": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "tokenizer_runtime": tokenizer_runtime,
    }
    if actual_runtime != protocol["runtime"]:
        raise RuntimeError(f"runtime differs from frozen protocol: {actual_runtime}")
    return {
        "protocol_sha256": expected_protocol_sha,
        "runner_checkout_commit": actual_commit,
        "runner_source_commit": expected_runner_commit,
        "source_repository_origin": origin,
        "runtime": actual_runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--venv-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_repo = args.source_repo.resolve(strict=True)
    tokenizer_dir = args.tokenizer_dir.resolve(strict=True)
    venv_python = Path(os.path.abspath(args.venv_python))
    if not venv_python.is_file():
        parser.error("--venv-python must name an existing executable")
    identities = validate_protocol(
        protocol, protocol_path, source_repo, tokenizer_dir, venv_python,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    worktree_root = args.worktree_root.resolve()
    worktree_root.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(output, output, protocol["resource_limits"])
    try:
        monitor.start()
        tokenizer = TokenizerCounter(
            venv_python, tokenizer_dir, output / "tokenizer.stderr.log",
        )
        attempts: list[dict[str, Any]] = []
        order = [(task_id, interface) for task_id in TASKS for interface in INTERFACES]
        mode_summaries: dict[str, dict[str, Any]] = {}
        for mode in protocol["execution_modes"]:
            monitor.check()
            mode_output = output / mode["name"]
            mode_output.mkdir()
            cells = []
            for index, (task_id, interface) in enumerate(order, 1):
                monitor.check()
                task = TASKS[task_id]
                workspace = worktree_root / mode["name"] / f"{index}-{task_id}-{interface}"
                workspace.parent.mkdir(parents=True, exist_ok=True)
                if workspace.exists():
                    raise FileExistsError(f"refusing to reuse worktree: {workspace}")
                subprocess.run(
                    ["git", "-C", str(source_repo), "worktree", "add", "--detach",
                     str(workspace), task["base"]],
                    check=True, capture_output=True, text=True,
                )
                version = workspace / "mlx_lm" / "_version.py"
                write_new_bytes(version, b'__version__ = "0.32.2"\n')
                cell_output = mode_output / f"{index}-{task_id}-{interface}"
                cell_output.mkdir()
                cells.append((task_id, interface, workspace, cell_output))

            mode_started = time.monotonic()
            mode_attempts: list[dict[str, Any]] = []
            queue_submitted_at = time.monotonic()
            if mode["worker_count"] == 1:
                for task_id, interface, workspace, cell_output in cells:
                    result = execute_cell(
                        source_repo=source_repo, workspace=workspace, output=cell_output,
                        task_id=task_id, interface=interface, tokenizer=tokenizer,
                        venv_python=venv_python, queue_submitted_at=queue_submitted_at,
                        monitor=monitor,
                    )
                    attempts.append(result)
                    mode_attempts.append(result)
            else:
                with ThreadPoolExecutor(
                    max_workers=mode["worker_count"], thread_name_prefix="vare-system-replay",
                ) as pool:
                    futures = [pool.submit(
                        execute_cell, source_repo=source_repo, workspace=workspace,
                        output=cell_output, task_id=task_id, interface=interface,
                        tokenizer=tokenizer, venv_python=venv_python,
                        queue_submitted_at=queue_submitted_at, monitor=monitor,
                    ) for task_id, interface, workspace, cell_output in cells]
                    for future in as_completed(futures):
                        result = future.result()
                        attempts.append(result)
                        mode_attempts.append(result)
            elapsed = round(time.monotonic() - mode_started, 4)
            mode_record = {
                "mode": mode["name"], "worker_count": mode["worker_count"],
                "task_interface_cell_count": len(cells),
                "verified_known_fix_count": sum(
                    bool(item["independent_grade"]["passed"]) for item in mode_attempts
                ),
                "execution_and_grading_seconds_after_worktree_materialization": elapsed,
                "queue_inclusive_cell_latencies_seconds": sorted(
                    item["queue_inclusive_elapsed_seconds"] for item in mode_attempts
                ),
                "service_cell_latencies_seconds": sorted(
                    item["service_elapsed_seconds"] for item in mode_attempts
                ),
                "serialized_tool_call_tokens_by_interface": {
                    interface: [
                        item["controller"]["serialized_tool_call_tokens"]
                        for item in sorted(mode_attempts, key=lambda row: (row["task_id"], row["interface"]))
                        if item["interface"] == interface
                    ] for interface in INTERFACES
                },
            }
            write_new_json(mode_output / "mode-summary.json", mode_record)
            mode_summaries[mode["name"]] = mode_record

        recovery_output = output / "recovery-after-commit-before-grade"
        recovery_output.mkdir()
        recovery_workspace = worktree_root / "recovery" / "mlxlm-2004-jamba-moe-routing"
        recovery_workspace.parent.mkdir(parents=True, exist_ok=True)
        if recovery_workspace.exists():
            raise FileExistsError(f"refusing to reuse recovery worktree: {recovery_workspace}")
        subprocess.run(
            ["git", "-C", str(source_repo), "worktree", "add", "--detach",
             str(recovery_workspace), TASKS["mlxlm-2004-jamba-moe-routing"]["base"]],
            check=True, capture_output=True, text=True,
        )
        write_new_bytes(
            recovery_workspace / "mlx_lm" / "_version.py", b'__version__ = "0.32.2"\n',
        )
        recovery_journal = recovery_output / "events.jsonl"
        crash_started = time.monotonic()
        monitor.check()
        crash_worker = multiprocessing.get_context("spawn").Process(
            target=apply_and_crash_after_commit,
            args=(str(source_repo), str(recovery_workspace),
                  "mlxlm-2004-jamba-moe-routing", str(recovery_journal)),
        )
        crash_worker.start()
        crash_worker.join(timeout=60)
        if crash_worker.is_alive():
            crash_worker.kill()
            crash_worker.join(timeout=5)
        monitor.check()
        task = TASKS["mlxlm-2004-jamba-moe-routing"]
        expected_fix = git_blob(source_repo, task["fix"], task["target"])
        recovered_target = recovery_workspace / task["target"]
        before_recovery_sha = sha256_path(recovered_target)
        event_count, previous = verify_event_chain(recovery_journal)
        if crash_worker.exitcode != -signal.SIGKILL:
            raise RuntimeError(f"crash injection did not SIGKILL the worker: {crash_worker.exitcode}")
        if before_recovery_sha != sha256_bytes(expected_fix):
            raise RuntimeError("recovered workspace does not contain the exact pinned fix")
        grade = _grade(recovery_workspace, "mlxlm-2004-jamba-moe-routing", venv_python, monitor)
        after_recovery_sha = sha256_path(recovered_target)
        if after_recovery_sha != before_recovery_sha:
            raise RuntimeError("recovery changed source; duplicate effective edit is possible")
        write_new_bytes(recovery_output / "trusted-grade.stdout.log", grade["stdout"].encode())
        write_new_bytes(recovery_output / "trusted-grade.stderr.log", grade["stderr"].encode())
        recovery_record = {
            "schema_version": 1, "worker_exitcode": crash_worker.exitcode,
            "injected_boundary": "after atomic edit and fsynced action-journal records, before independent grade",
            "journal_event_count": event_count, "journal_sha256": sha256_path(recovery_journal),
            "journal_final_event_sha256": previous,
            "target_sha256_before_recovery": before_recovery_sha,
            "expected_upstream_fix_sha256": sha256_bytes(expected_fix),
            "target_sha256_after_recovery": after_recovery_sha,
            "duplicate_edit_during_recovery": False,
            "independent_grade": {
                key: value for key, value in grade.items() if key not in {"stdout", "stderr"}
            },
            "recovery_wall_seconds": round(time.monotonic() - crash_started, 4),
        }
        write_new_json(recovery_output / "recovery.json", recovery_record)
        tokenizer_returncode = tokenizer.close()
        if tokenizer_returncode != 0:
            raise RuntimeError(f"pinned tokenizer worker exited with {tokenizer_returncode}")
        tokenizer_stderr_sha256 = sha256_path(output / "tokenizer.stderr.log")
        resource_summary = monitor.close()
        attempts.sort(key=lambda item: (
            "synchronous" if "synchronous" in item["workspace_path"] else "asynchronous",
            item["task_id"], item["interface"],
        ))
        final = {
            "schema_version": 1,
            "protocol_sha256": identities["protocol_sha256"],
            "identities": identities,
            "claim_boundary": "systems-only deterministic known-fix control; no model capability or learning evidence",
            "attempts": attempts,
            "mode_summaries": mode_summaries,
            "crash_recovery": recovery_record,
            "resource_summary": resource_summary,
            "tokenizer_stderr_sha256": tokenizer_stderr_sha256,
            "tokenizer_worker_returncode": tokenizer_returncode,
        }
        final_path = output / "all-cell-results.json"
        write_new_json(final_path, final)
        print(json.dumps({
            "attempt_count": len(attempts),
            "all_grades_passed": all(item["independent_grade"]["passed"] for item in attempts),
            "crash_grade_passed": recovery_record["independent_grade"]["passed"],
            "resource_abort_reason": resource_summary["abort_reason"],
            "results_sha256": sha256_path(final_path),
        }, indent=2))
        return 0 if (
            len(attempts) == 8
            and all(item["independent_grade"]["passed"] for item in attempts)
            and recovery_record["independent_grade"]["passed"]
            and not resource_summary["abort_reason"]
        ) else 1
    except BaseException as exc:
        try:
            if "tokenizer" in locals() and tokenizer is not None:
                tokenizer.close()
            if getattr(monitor, "_fd", -1) >= 0:
                summary = monitor.close()
            else:
                summary = {"abort_reason": monitor.abort_reason}
            write_new_json(output / "experiment-failure.json", {
                "protocol_sha256": identities["protocol_sha256"],
                "exception_type": type(exc).__name__, "error": str(exc),
                "resource_summary": summary,
            })
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
