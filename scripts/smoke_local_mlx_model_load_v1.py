#!/usr/bin/env python3
"""Run one offline, resource-bounded MLX model-load smoke without generation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / (
    "results/mlsys-repo-task-mlx-lm-1910-agent-v8-smollm2-360m-load-smoke"
)
PROTOCOL_PATH = BUNDLE / "protocol.json"
PROTOCOL_SHA_PATH = BUNDLE / "protocol.sha256"
DEFAULT_RUN = BUNDLE / "run-1"


class SmokeError(RuntimeError):
    """The frozen smoke contract cannot be verified."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SmokeError(f"cannot read frozen JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise SmokeError(f"frozen JSON must contain an object: {path.name}")
    return value


def _load_protocol() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    expected = PROTOCOL_SHA_PATH.read_text(encoding="ascii").strip()
    if sha256_file(PROTOCOL_PATH) != expected:
        raise SmokeError("frozen protocol SHA-256 mismatch")
    if protocol.get("protocol_id") != "vare.local-mlx-model-load-smoke.v1":
        raise SmokeError("unexpected model-load smoke protocol")
    if protocol.get("runner_sha256") != sha256_file(Path(__file__)):
        raise SmokeError("runner differs from frozen protocol")
    return protocol


def _command_output(argv: list[str], *, timeout: float = 3.0) -> str:
    result = subprocess.run(
        argv, check=True, capture_output=True, text=True, timeout=timeout
    )
    return result.stdout


def _memory_free_percent() -> int:
    output = _command_output(["memory_pressure"])
    match = re.search(r"System-wide memory free percentage:\s*(\d+)%", output)
    if match is None:
        raise SmokeError("cannot parse macOS memory-pressure report")
    return int(match.group(1))


def _swap_used_mib() -> float:
    output = _command_output(["sysctl", "vm.swapusage"])
    match = re.search(r"used\s*=\s*([0-9.]+)M", output)
    if match is None:
        raise SmokeError("cannot parse macOS swap usage")
    return float(match.group(1))


def _rss_mib(pid: int) -> float:
    output = _command_output(["ps", "-o", "rss=", "-p", str(pid)])
    value = output.strip()
    if not value:
        return 0.0
    return int(value) / 1024


def _verify_environment(protocol: dict[str, Any]) -> Path:
    runtime = protocol["runtime"]
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SmokeError("protocol requires macOS on Apple Silicon")
    if list(sys.version_info[:3]) != runtime["python_version"]:
        raise SmokeError("Python version differs from frozen runtime")
    for package, expected in runtime["packages"].items():
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SmokeError(f"required package is missing: {package}") from exc
        if observed != expected:
            raise SmokeError(f"package version differs from frozen runtime: {package}")

    model = protocol["model"]
    cache_root = Path.home() / ".cache/huggingface/hub"
    snapshot = (
        cache_root
        / model["cache_directory"]
        / "snapshots"
        / model["revision"]
    )
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise SmokeError("pinned local model snapshot is unavailable")
    expected_files = model["files"]
    actual_names: set[str] = set()
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            if not path.resolve(strict=True).is_file():
                raise SmokeError("model snapshot contains a non-file symlink")
            resolved = path.resolve(strict=True)
        elif path.is_file():
            resolved = path
        else:
            continue
        relative = path.relative_to(snapshot).as_posix()
        actual_names.add(relative)
        expected_file = expected_files.get(relative)
        if not isinstance(expected_file, dict):
            raise SmokeError(f"unlocked model file: {relative}")
        if (
            resolved.stat().st_size != expected_file.get("bytes")
            or sha256_file(resolved) != expected_file.get("sha256")
        ):
            raise SmokeError(f"pinned model file hash mismatch: {relative}")
    if actual_names != set(expected_files):
        raise SmokeError("local model file inventory differs from the frozen protocol")
    return snapshot


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _child(model_path: Path, result_path: Path) -> int:
    # This child receives only the local model path and has network access
    # disabled. It never loads task prompts, candidate workspaces, or graders.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from mlx_lm import load

    model, tokenizer = load(str(model_path))
    _write_json(
        result_path,
        {
            "loaded": True,
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "generated_tokens": 0,
        },
    )
    return 0


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-model", type=Path)
    parser.add_argument("--child-result", type=Path)
    args = parser.parse_args()
    if args.child_model is not None:
        if args.child_result is None:
            raise SystemExit("--child-result is required with --child-model")
        return _child(args.child_model, args.child_result)

    protocol = _load_protocol()
    model_path = _verify_environment(protocol)
    limits = protocol["resource_budget"]
    start_swap = _swap_used_mib()
    start_free = _memory_free_percent()
    if (
        start_swap > limits["max_start_swap_mib"]
        or start_free < limits["min_start_free_memory_percent"]
    ):
        print("NO_GO: resource preflight outside frozen limits", file=sys.stderr)
        return 3

    DEFAULT_RUN.mkdir(parents=True, exist_ok=False)
    (DEFAULT_RUN / "protocol.json").write_bytes(PROTOCOL_PATH.read_bytes())
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    stdout_path = DEFAULT_RUN / "stdout.log"
    stderr_path = DEFAULT_RUN / "stderr.log"
    child_result = DEFAULT_RUN / "load.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-model",
        str(model_path),
        "--child-result",
        str(child_result),
    ]
    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
    }
    samples: list[dict[str, Any]] = []
    status = "FAILED"
    reason: str | None = None
    peak_rss_mib = 0.0
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
        try:
            while process.poll() is None:
                now = time.monotonic()
                swap = _swap_used_mib()
                free = _memory_free_percent()
                rss = _rss_mib(process.pid)
                peak_rss_mib = max(peak_rss_mib, rss)
                samples.append(
                    {
                        "elapsed_seconds": round(now - started, 3),
                        "swap_used_mib": swap,
                        "free_memory_percent": free,
                        "child_rss_mib": round(rss, 2),
                    }
                )
                if (
                    swap - start_swap > limits["max_swap_growth_mib"]
                    or swap >= limits["max_absolute_swap_mib"]
                    or free < limits["min_runtime_free_memory_percent"]
                    or rss > limits["max_child_rss_mib"]
                ):
                    status = "ABORTED_RESOURCE"
                    reason = "resource threshold exceeded"
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                if now - started > limits["max_wall_seconds"]:
                    status = "ABORTED_TIMEOUT"
                    reason = "wall-clock limit exceeded"
                    process.kill()
                    break
                time.sleep(limits["poll_interval_seconds"])
            return_code = process.wait(timeout=5)
            elapsed = time.monotonic() - started
            if status not in {"ABORTED_RESOURCE", "ABORTED_TIMEOUT"}:
                status = "COMPLETED" if return_code == 0 and child_result.is_file() else "FAILED"
                if status == "FAILED":
                    reason = f"model loader exited with status {return_code}"
        except Exception as exc:
            process.kill()
            process.wait(timeout=5)
            elapsed = time.monotonic() - started
            return_code = process.returncode
            reason = f"monitor failure: {type(exc).__name__}: {exc}"
            status = "FAILED"

    end_swap = _swap_used_mib()
    end_free = _memory_free_percent()
    evidence = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "status": status,
        "reason": reason,
        "started_at_utc": started_at,
        "wall_seconds": round(elapsed, 3),
        "return_code": return_code,
        "generated_tokens": 0,
        "optimizer_steps": 0,
        "resource": {
            "start_swap_mib": start_swap,
            "end_swap_mib": end_swap,
            "peak_sampled_swap_mib": max(
                [start_swap, *(sample["swap_used_mib"] for sample in samples)]
            ),
            "swap_growth_mib": round(end_swap - start_swap, 2),
            "start_free_memory_percent": start_free,
            "minimum_sampled_free_memory_percent": min(
                [start_free, *(sample["free_memory_percent"] for sample in samples)]
            ),
            "end_free_memory_percent": end_free,
            "peak_sampled_child_rss_mib": round(peak_rss_mib, 2),
            "samples": samples,
        },
        "load_result": _read_json(child_result) if child_result.is_file() else None,
        "command": command,
    }
    _write_json(DEFAULT_RUN / "result.json", evidence)
    files = {}
    for path in sorted(DEFAULT_RUN.iterdir()):
        if path.is_file() and path.name != "evidence.sha256.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    _write_json(
        DEFAULT_RUN / "evidence.sha256.json",
        {"protocol_sha256": sha256_file(PROTOCOL_PATH), "files": files},
    )
    print(json.dumps({"status": status, "wall_seconds": round(elapsed, 3)}, sort_keys=True))
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(run())
