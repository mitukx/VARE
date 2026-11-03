"""Run the checked source bytes in a hash-pinned macOS Seatbelt worker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from .grade import (
    EXPECTED_KNOWN_FIX_AST_SHA256,
    WorkerRunnerError,
    grade_candidate,
)


@dataclass(frozen=True)
class WorkerConfig:
    python: Path
    python_sha256: str
    python_prefix: Path
    runtime_root: Path
    runtime_read_paths: tuple[Path, ...]
    worker: Path
    worker_sha256: str
    protected_paths: tuple[Path, ...]
    scratch_root: Path
    sandbox_exec: Path = Path("/usr/bin/sandbox-exec")
    max_wall_seconds: float = 600.0
    max_process_tree_rss_bytes: int = 2_684_354_560
    max_output_bytes: int = 1_048_576


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exact(descriptor: int, data: bytes) -> None:
    written = 0
    while written < len(data):
        count = os.write(descriptor, data[written:])
        if count <= 0:
            raise OSError("short write while staging a sandbox input")
        written += count


def _quote_seatbelt_path(path: Path) -> str:
    value = str(path)
    if any(char in value for char in ("\n", "\r", "\x00")):
        raise ValueError("sandbox path contains an unsupported control character")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _read_rule(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if resolved.is_dir():
        return f"(allow file-read* (subpath {_quote_seatbelt_path(resolved)}))"
    return f"(allow file-read* (literal {_quote_seatbelt_path(resolved)}))"


def _profile_text(config: WorkerConfig, run_dir: Path, tmp_dir: Path) -> str:
    allowed = [
        Path("/System"),
        Path("/usr/bin"),
        Path("/usr/lib"),
        Path("/bin"),
        Path("/sbin"),
        Path("/usr/share"),
        config.python_prefix,
        config.runtime_root,
        *config.runtime_read_paths,
        run_dir,
    ]
    lines = [
        "(version 1)",
        "(deny default)",
        "(deny network*)",
        "(allow sysctl-read)",
        f"(allow process-exec (literal {_quote_seatbelt_path(config.python.resolve(strict=True))}))",
        '(allow file-read* (literal "/dev/null"))',
        '(allow file-write* (literal "/dev/null"))',
        '(allow file-read* (literal "/dev/urandom"))',
        '(allow file-read-data (literal "/"))',
    ]
    for path in allowed:
        if path.exists():
            lines.append(_read_rule(path))
    lines.append(f"(allow file-write* (subpath {_quote_seatbelt_path(tmp_dir)}))")
    for path in config.protected_paths:
        if path.exists():
            resolved = path.resolve(strict=True)
            lines.append(
                f"(deny file-read* (subpath {_quote_seatbelt_path(resolved)}))"
            )
            lines.append(
                f"(deny file-write* (subpath {_quote_seatbelt_path(resolved)}))"
            )
    return "\n".join(lines) + "\n"


def _process_tree_rss_bytes(pid: int) -> int:
    output = subprocess.check_output(
        ["/bin/ps", "-axo", "pid=,ppid=,rss="],
        text=True,
        timeout=5,
    )
    parents: dict[int, int] = {}
    rss_kib: dict[int, int] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        process_id, parent_id, resident_kib = map(int, fields)
        parents[process_id] = parent_id
        rss_kib[process_id] = resident_kib

    descendants = {pid}
    changed = True
    while changed:
        changed = False
        for process_id, parent_id in parents.items():
            if parent_id in descendants and process_id not in descendants:
                descendants.add(process_id)
                changed = True
    return sum(rss_kib.get(process_id, 0) for process_id in descendants) * 1024


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _validate_config(config: WorkerConfig) -> None:
    if not config.sandbox_exec.is_file() or not os.access(config.sandbox_exec, os.X_OK):
        raise WorkerRunnerError(
            "macOS sandbox-exec is unavailable", worker_started=False
        )
    if not config.python.is_file() or not os.access(config.python, os.X_OK):
        raise WorkerRunnerError(
            "pinned Python executable is unavailable", worker_started=False
        )
    if (
        len(config.python_sha256) != 64
        or any(char not in "0123456789abcdef" for char in config.python_sha256)
        or _sha256(config.python) != config.python_sha256
    ):
        raise WorkerRunnerError(
            "pinned Python executable hash mismatch", worker_started=False
        )
    if not config.runtime_root.is_dir():
        raise WorkerRunnerError(
            "pinned runtime root is unavailable", worker_started=False
        )
    if not config.worker.is_file() or _sha256(config.worker) != config.worker_sha256:
        raise WorkerRunnerError("candidate worker hash mismatch", worker_started=False)
    if not config.python_prefix.is_dir():
        raise WorkerRunnerError(
            "pinned Python prefix is unavailable", worker_started=False
        )
    if not all(path.exists() for path in config.runtime_read_paths):
        raise WorkerRunnerError(
            "runtime read allowlist is incomplete", worker_started=False
        )
    if config.max_wall_seconds <= 0 or config.max_process_tree_rss_bytes <= 0:
        raise WorkerRunnerError(
            "worker resource limits are invalid", worker_started=False
        )
    if config.max_output_bytes <= 0:
        raise WorkerRunnerError("worker output limit is invalid", worker_started=False)


def run_worker(
    checked_source: bytes,
    cases: list[dict[str, object]],
    config: WorkerConfig,
) -> bytes:
    """Run the preflight-approved bytes; preserve stdout in the run directory."""
    _validate_config(config)
    config.scratch_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix="trl7983-grade-", dir=config.scratch_root)
    ).resolve(strict=True)
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(mode=0o700)
    candidate_path = run_dir / "candidate.py"
    worker_path = run_dir / "worker.py"
    worker_bytes = config.worker.read_bytes()
    if hashlib.sha256(worker_bytes).hexdigest() != config.worker_sha256:
        raise WorkerRunnerError(
            "candidate worker changed before staging", worker_started=False
        )
    worker_descriptor = os.open(
        worker_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        _write_exact(worker_descriptor, worker_bytes)
        os.fsync(worker_descriptor)
    finally:
        os.close(worker_descriptor)
    worker_path.chmod(0o400)
    descriptor = os.open(
        candidate_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        _write_exact(descriptor, checked_source)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    candidate_path.chmod(0o400)
    profile_path = run_dir / "profile.sb"
    profile_text = _profile_text(config, run_dir, tmp_dir)
    profile_path.write_text(profile_text, encoding="utf-8")
    profile_path.chmod(0o400)
    profile_sha256 = hashlib.sha256(profile_text.encode("utf-8")).hexdigest()

    payload = json.dumps(
        {"cases": cases}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    command = [
        str(config.sandbox_exec),
        "-f",
        str(profile_path),
        str(config.python),
        "-S",
        str(worker_path),
        str(config.runtime_root),
        str(candidate_path),
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(tmp_dir),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TZ": "UTC",
    }
    try:
        process = subprocess.Popen(
            command,
            cwd=run_dir,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        raise WorkerRunnerError(
            f"worker process failed to start: {type(exc).__name__}",
            worker_started=False,
        ) from exc

    worker_stdin = process.stdin
    worker_stdout = process.stdout
    if worker_stdin is None or worker_stdout is None:
        _terminate_process_group(process)
        raise WorkerRunnerError("worker pipes were unavailable", worker_started=True)

    output_chunks: list[bytes] = []
    output_size = 0
    output_lock = threading.Lock()
    output_limit_exceeded = threading.Event()

    def capture_output() -> None:
        nonlocal output_size
        while True:
            block = worker_stdout.read(65536)
            if not block:
                return
            with output_lock:
                output_size += len(block)
                if output_size > config.max_output_bytes:
                    output_limit_exceeded.set()
                elif not output_limit_exceeded.is_set():
                    output_chunks.append(block)

    reader = threading.Thread(target=capture_output, daemon=True)
    reader.start()
    started_at = time.monotonic()
    try:
        worker_stdin.write(payload)
        worker_stdin.close()
    except (BrokenPipeError, OSError) as exc:
        _terminate_process_group(process)
        reader.join(timeout=5)
        raise WorkerRunnerError(
            f"worker input failed: {type(exc).__name__}", worker_started=True
        ) from exc

    peak_rss = 0
    while process.poll() is None:
        if output_limit_exceeded.is_set():
            _terminate_process_group(process)
            reader.join(timeout=5)
            raise WorkerRunnerError("worker output limit exceeded", worker_started=True)
        if time.monotonic() - started_at > config.max_wall_seconds:
            _terminate_process_group(process)
            reader.join(timeout=5)
            raise WorkerRunnerError(
                "worker wall-time limit exceeded", worker_started=True
            )
        try:
            peak_rss = max(peak_rss, _process_tree_rss_bytes(process.pid))
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            _terminate_process_group(process)
            reader.join(timeout=5)
            raise WorkerRunnerError(
                f"process-tree monitor failed: {type(exc).__name__}",
                worker_started=True,
            ) from exc
        if peak_rss > config.max_process_tree_rss_bytes:
            _terminate_process_group(process)
            reader.join(timeout=5)
            raise WorkerRunnerError("worker RSS limit exceeded", worker_started=True)
        time.sleep(0.2)

    return_code = process.wait()
    reader.join(timeout=5)
    if reader.is_alive():
        raise WorkerRunnerError(
            "worker stdout reader did not stop", worker_started=True
        )
    if output_limit_exceeded.is_set() or output_size > config.max_output_bytes:
        raise WorkerRunnerError("worker output limit exceeded", worker_started=True)
    if return_code != 0:
        raise WorkerRunnerError(
            f"worker exited with status {return_code}", worker_started=True
        )
    output = b"".join(output_chunks)
    (run_dir / "worker.stdout").write_bytes(output)
    (run_dir / "run-metadata.json").write_text(
        json.dumps(
            {
                "worker_sha256": config.worker_sha256,
                "python_sha256": config.python_sha256,
                "sandbox_profile_sha256": profile_sha256,
                "rendered_profile_sha256": profile_sha256,
                "returncode": return_code,
                "wall_seconds": round(time.monotonic() - started_at, 6),
                "sampled_peak_process_tree_rss_bytes": peak_rss,
                "output_bytes": len(output),
                "candidate_source_sha256": hashlib.sha256(checked_source).hexdigest(),
                "candidate_source_mode": oct(candidate_path.stat().st_mode & 0o777),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def grade_workspace(
    workspace: Path,
    relative_source_path: str,
    seed: int,
    config: WorkerConfig,
) -> dict[str, object]:
    """Public grader entry: source preflight always precedes worker launch."""
    return grade_candidate(
        workspace,
        relative_source_path,
        EXPECTED_KNOWN_FIX_AST_SHA256,
        seed,
        worker_runner=lambda checked_source, cases: run_worker(
            checked_source, cases, config
        ),
    )
