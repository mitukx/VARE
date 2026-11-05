"""Bounded worker execution for untrusted candidate functions.

The parent owns expected values. The child imports only the staged subject
and returns JSON values. This process group is not a hostile-code sandbox:
there is no separate user, network namespace, or filesystem jail. Commands
are not taken from the candidate. A timeout is reported as ``timeout`` and
is not rewritten into a correctness failure.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vare.environments.spec import ResourceLimits

CLASS_OK = "ok"
CLASS_TIMEOUT = "timeout"
CLASS_OUTPUT_LIMIT = "output_limit"
CLASS_NONZERO = "nonzero_exit"
CLASS_MALFORMED = "malformed_output"
CLASS_MEMORY = "memory_budget"

# Trusted child program. It contains no expected answers and no task seeds.
_WORKER = r"""
import importlib
import json
import os
import sys
import time

_perf = time.perf_counter
try:
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    subject = os.path.realpath(request["subject"])
    saved = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = open(1, "w", closefd=False)
    kept = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = os.path.realpath(entry)
        except OSError:
            continue
        if resolved == subject:
            continue
        kept.append(entry)
    sys.path = [subject] + kept
    os.chdir(subject)

    def _one(spec):
        module_name = spec["module"]
        function_name = spec["function"]
        if not str(module_name).isidentifier() or not str(function_name).isidentifier():
            raise ValueError("illegal callable name")
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
        args = spec.get("args") or []
        kwargs = spec.get("kwargs") or {}
        op = spec.get("op") or "call"
        if op == "call":
            return {"id": spec.get("id"), "ok": True, "value": fn(*args, **kwargs)}
        if op != "bench":
            raise ValueError("unsupported op")
        for _ in range(int(spec.get("warmup") or 0)):
            fn(*args, **kwargs)
        samples = []
        repeats = int(spec["repeats"])
        if repeats < 1 or repeats > 20:
            raise ValueError("repeats out of range")
        for _ in range(repeats):
            started = _perf()
            fn(*args, **kwargs)
            samples.append(_perf() - started)
        return {"id": spec.get("id"), "ok": True, "elapsed_s": samples}

    results = []
    for spec in request["calls"]:
        try:
            results.append(_one(spec))
        except Exception as exc:
            results.append({
                "id": spec.get("id") if isinstance(spec, dict) else None,
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    payload = {"ok": True, "results": results}
except Exception as exc:
    payload = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
try:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
except (TypeError, ValueError):
    encoded = json.dumps({
        "ok": False,
        "error_type": "ValueError",
        "error": "worker result is not finite JSON",
    }).encode("utf-8") + b"\n"
os.write(saved, encoded)
"""


@dataclass(frozen=True, slots=True)
class Execution:
    classification: str
    returncode: int
    wall_s: float
    stdout: str
    stderr: str
    timed_out: bool
    output_limited: bool
    peak_rss_kb: int | None
    payload: dict[str, Any] | None
    memory_limit_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "returncode": self.returncode,
            "wall_s": self.wall_s,
            "timed_out": self.timed_out,
            "output_limited": self.output_limited,
            "peak_rss_kb": self.peak_rss_kb,
            "memory_limit_applied": self.memory_limit_applied,
            "payload": self.payload,
        }


def _preexec(limits: ResourceLimits):
    def apply() -> None:
        try:
            import resource
        except ImportError:
            return
        if limits.cpu_time_s is not None:
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (int(limits.cpu_time_s), int(limits.cpu_time_s) + 1),
            )
        # RLIMIT_AS is a poor fit for CPython's virtual map and is not applied.
        # Peak resident set is sampled from the parent instead.

    return apply


def _peak_rss_kb(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    try:
        text = status.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def _child_env() -> dict[str, str]:
    """Minimal environment. Repository PYTHONPATH is not forwarded."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    return env


def run_bounded(
    argv: list[str],
    *,
    cwd: Path,
    stdin: bytes,
    limits: ResourceLimits,
    env: dict[str, str] | None = None,
) -> Execution:
    if not argv or argv[0] != sys.executable:
        raise ValueError("bounded execution must start with this Python interpreter")
    cwd = cwd.resolve()
    start = time.perf_counter()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env if env is not None else _child_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        start_new_session=os.name == "posix",
        # Drain threads start after Popen. The limit callback runs only in the child.
        preexec_fn=_preexec(limits) if os.name == "posix" else None,  # noqa: PLW1509
    )
    assert proc.stdin is not None
    try:
        proc.stdin.write(stdin)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    stdout_buf, stderr_buf = bytearray(), bytearray()
    overflow = threading.Event()

    def drain(stream, retained: bytearray) -> None:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                return
            remaining = limits.max_output_bytes - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
            if len(chunk) > remaining:
                overflow.set()

    readers = [
        threading.Thread(target=drain, args=(proc.stdout, stdout_buf), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, stderr_buf), daemon=True),
    ]
    for reader in readers:
        reader.start()
    peak: int | None = None

    def terminate() -> None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        except OSError:
            if proc.poll() is None:
                proc.kill()

    deadline = start + limits.wall_time_s
    timed_out = output_limited = False
    try:
        while True:
            sample = _peak_rss_kb(proc.pid)
            if sample is not None:
                peak = sample if peak is None else max(peak, sample)
            if overflow.is_set():
                output_limited = True
                terminate()
                break
            if proc.poll() is not None and not any(reader.is_alive() for reader in readers):
                break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                timed_out = True
                terminate()
                break
            overflow.wait(min(0.01, remaining))
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            terminate()
            proc.wait()
        for reader in readers:
            reader.join(timeout=2)
        if any(reader.is_alive() for reader in readers):
            terminate()
            for reader in readers:
                reader.join(timeout=0.1)
        returncode = int(proc.returncode if proc.returncode is not None else 124)
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()
    wall = time.perf_counter() - start
    stdout = bytes(stdout_buf).decode("utf-8", errors="replace")
    stderr = bytes(stderr_buf).decode("utf-8", errors="replace")
    if timed_out:
        classification = CLASS_TIMEOUT
    elif output_limited:
        classification = CLASS_OUTPUT_LIMIT
    elif returncode != 0:
        classification = CLASS_NONZERO
    else:
        classification = CLASS_OK
    if limits.memory_mb is not None and peak is not None and peak > limits.memory_mb * 1024:
        classification = CLASS_MEMORY
    return Execution(
        classification=classification,
        returncode=124 if timed_out else returncode,
        wall_s=wall,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        output_limited=output_limited,
        peak_rss_kb=peak,
        payload=None,
        memory_limit_applied=False,
    )


def parse_worker_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("worker produced no JSON")
    lines = text.splitlines()
    if len(lines) != 1:
        raise ValueError("worker JSON must be a single line")
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("worker JSON is malformed") from exc
    if not isinstance(payload, dict) or "ok" not in payload:
        raise ValueError("worker JSON is missing ok")
    return payload


def run_worker(
    subject: Path,
    calls: list[dict[str, Any]],
    limits: ResourceLimits,
) -> Execution:
    subject = subject.resolve()
    request = {"subject": str(subject), "calls": calls}
    try:
        stdin = json.dumps(request, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("worker request is not finite JSON") from exc
    if len(stdin) > 8_000_000:
        raise ValueError("worker request exceeds 8MB")
    execution = run_bounded(
        [sys.executable, "-c", _WORKER],
        cwd=subject,
        stdin=stdin,
        limits=limits,
    )
    if execution.classification != CLASS_OK:
        return execution
    try:
        payload = parse_worker_stdout(execution.stdout)
    except ValueError:
        return Execution(
            classification=CLASS_MALFORMED,
            returncode=execution.returncode,
            wall_s=execution.wall_s,
            stdout=execution.stdout,
            stderr=execution.stderr,
            timed_out=False,
            output_limited=False,
            peak_rss_kb=execution.peak_rss_kb,
            payload=None,
            memory_limit_applied=False,
        )
    return Execution(
        classification=CLASS_OK,
        returncode=execution.returncode,
        wall_s=execution.wall_s,
        stdout=execution.stdout,
        stderr=execution.stderr,
        timed_out=False,
        output_limited=False,
        peak_rss_kb=execution.peak_rss_kb,
        payload=payload,
        memory_limit_applied=False,
    )


def limits_from_task(compute: dict[str, Any]) -> ResourceLimits:
    return ResourceLimits(
        wall_time_s=float(compute["wall_time_s"]),
        cpu_time_s=int(compute["cpu_time_s"]),
        memory_mb=int(compute["memory_mb"]),
        max_output_bytes=int(compute["max_output_bytes"]),
    )
