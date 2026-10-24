#!/usr/bin/env python3
"""Run the Development study with explicit resource limits and failure retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_mlx_lm_1912_qwen25_agent_v1 as harness  # noqa: E402

BUNDLE = ROOT / "results/mlsys-repo-task-mlx-lm-1912-qwen25-3b-agent-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminate_processes_from_traceback(tb: Any) -> list[dict[str, Any]]:
    """Stop Popen groups owned by a failing monitored call before unwinding."""
    stopped = []
    seen: set[int] = set()
    while tb is not None:
        process = tb.tb_frame.f_locals.get("process")
        if isinstance(process, subprocess.Popen) and process.pid not in seen:
            seen.add(process.pid)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            stopped.append({"pid": process.pid, "returncode": process.poll()})
        tb = tb.tb_next
    return stopped


class _PreservedDirectory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, *_exc: object) -> Literal[False]:
        return False


class _TemporaryDirectoryProxy:
    """Keep candidate workspaces under run evidence even after exceptions."""

    def __init__(self, original: Any, output: Path) -> None:
        self.original = original
        self.output = output

    def TemporaryDirectory(self, *, prefix: str, dir: str | None = None, **kwargs: Any) -> Any:
        for arm in ("greedy", "sampled"):
            if prefix.startswith(f"vare-1912-{arm}-"):
                path = self.output / arm / "candidate-work-root"
                path.mkdir(parents=True, exist_ok=False)
                return _PreservedDirectory(path)
        return self.original(prefix=prefix, dir=dir, **kwargs)


def _record_exception(output: Path, protocol_id: str, exc: BaseException,
                      stopped: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "supervisor-exception.json"
    detail = {
        "schema_version": 1,
        "status": "FAILED",
        "exception_type": type(exc).__name__,
        "exception_text": str(exc)[:2000],
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
        "stopped_process_groups": stopped,
        "generated_tokens_in_durable_logs": 0,
    }
    for raw_log in output.glob("*/raw-turns.jsonl"):
        try:
            for line in raw_log.read_text(encoding="utf-8").splitlines():
                turn = json.loads(line)
                if turn.get("event") == "generation_completed":
                    detail["generated_tokens_in_durable_logs"] += turn.get("generated_tokens", 0)
        except (OSError, json.JSONDecodeError):
            detail["raw_log_parse_error"] = str(raw_log)
    with detail_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(detail, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    summary_path = output / "summary.json"
    if not summary_path.exists():
        summary = {
            "schema_version": 1,
            "protocol_id": protocol_id,
            "status": "FAILED",
            "failure_classification": "supervisor_or_harness_exception",
            "exception_record": detail_path.name,
            "generated_tokens": detail["generated_tokens_in_durable_logs"],
            "optimizer_steps": 0,
            "heldout_tasks_accessed": False,
        }
        with summary_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    manifest_name = (
        "evidence-manifest.json" if not (output / "evidence-manifest.json").exists()
        else "post-exception-evidence-manifest.json"
    )
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {
            "evidence-manifest.json", "post-exception-evidence-manifest.json"
        }:
            files[path.relative_to(output).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "schema_version": 1,
        "protocol_sha256": _sha256(BUNDLE / "protocol.json"),
        "files": files,
    }
    with (output / manifest_name).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _unknown = parser.parse_known_args()
    output = args.output.resolve()
    protocol_path = BUNDLE / "protocol.json"
    sidecar_path = BUNDLE / "protocol.sha256"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if _sha256(protocol_path) != sidecar_path.read_text(encoding="ascii").split()[0]:
        raise RuntimeError("agent-v3 protocol SHA-256 sidecar mismatch")
    resources = protocol.get("resources", {})
    if (protocol.get("status") != "FROZEN_NOT_RUN"
            or resources.get("docker_context") != "colima"
            or resources.get("max_wall_seconds") != protocol.get("agent", {}).get("wall_seconds")):
        raise RuntimeError("agent-v3 execution context or resource limits are inconsistent")
    preflight_path = ROOT / protocol["runtime"]["docker_context_preflight_path"]
    if _sha256(preflight_path) != protocol["runtime"]["docker_context_preflight_sha256"]:
        raise RuntimeError("Docker context infrastructure preflight hash mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight.get("status") != "COMPLETED"
            or preflight.get("docker_context") != "colima"
            or preflight.get("image") != protocol["grader"].get("image")
            or preflight.get("task_prompt_loaded") is not False
            or preflight.get("model_loaded") is not False):
        raise RuntimeError("Docker infrastructure preflight did not pass its frozen checks")

    os.environ["DOCKER_CONTEXT"] = "colima"
    harness.BUNDLE = BUNDLE
    harness.TASK_ROOT = BUNDLE / "task"
    harness.PROTOCOL_PATH = protocol_path
    harness.PROTOCOL_SHA_PATH = sidecar_path
    setattr(harness, "tempfile", _TemporaryDirectoryProxy(tempfile.TemporaryDirectory, output))
    original_monitor = harness.run_monitored

    def safe_monitor(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return original_monitor(*args, **kwargs)
        except BaseException as exc:
            terminate_processes_from_traceback(exc.__traceback__)
            raise

    harness.run_monitored = safe_monitor
    try:
        return harness.main()
    except BaseException as exc:
        stopped = terminate_processes_from_traceback(exc.__traceback__)
        if output.exists():
            _record_exception(output, protocol["protocol_id"], exc, stopped)
        print(f"agent-v3 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
