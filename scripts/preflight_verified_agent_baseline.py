#!/usr/bin/env python3
"""Record a read-only preflight for the verified coding-agent baseline.

This command never loads a model, starts the agent, grades a patch, or executes
candidate code. It records whether prerequisites are present and fails closed
while VARE lacks a reviewed isolated evaluator launcher.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/verified_coding_agent_baseline_v3_preflight.json"
V2_PROTOCOL = ROOT / "protocols/verified_coding_agent_baseline_v2.json"
DEFAULT_MODEL = (
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_docker() -> dict[str, Any]:
    executable = shutil.which("docker")
    if executable is None:
        return {"cli_available": False, "daemon_available": False}
    try:
        contexts = subprocess.run(
            [executable, "context", "ls", "--format", "{{.Name}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired:
        return {"cli_available": True, "daemon_available": False, "probe": "context_list_timeout"}
    names = contexts.stdout.splitlines() if contexts.returncode == 0 else ["default"]
    probes = []
    for name in dict.fromkeys(names):
        if not name:
            continue
        try:
            result = subprocess.run(
                [executable, "--context", name, "info", "--format", "{{.ServerVersion}}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            probes.append({"context": name, "daemon_available": False, "probe": "timeout"})
            continue
        probes.append({
            "context": name,
            "daemon_available": result.returncode == 0 and bool(result.stdout.strip()),
            "server_version": result.stdout.strip() if result.returncode == 0 else None,
        })
    available = next((probe for probe in probes if probe["daemon_available"]), None)
    return {
        "cli_available": True,
        "daemon_available": available is not None,
        "contexts": probes,
        "working_context": available["context"] if available else None,
    }


def collect_preflight(
    *,
    root: Path = ROOT,
    model_path: Path = DEFAULT_MODEL,
    docker_probe: Callable[[], dict[str, Any]] = _probe_docker,
) -> dict[str, Any]:
    protocol = json.loads((root / PROTOCOL.relative_to(ROOT)).read_text(encoding="utf-8"))
    previous = json.loads((root / V2_PROTOCOL.relative_to(ROOT)).read_text(encoding="utf-8"))
    task_root = root / "benchmarks/historical/rvl_behavior_policy_parity"
    task_files = {
        "TASK.md": task_root / "TASK.md",
        "task.json": task_root / "task.json",
        "protocol.lock.json": task_root / "protocol.lock.json",
        "grader.py": task_root / "evaluator/grade.py",
    }
    task_hashes = {
        name: _sha256(path) if path.is_file() else None for name, path in task_files.items()
    }
    task_matches = task_hashes == previous["task"]["hashes"]
    expected_files = protocol["model"]["files"]
    observed_files: dict[str, str | None] = {}
    for name in expected_files:
        file_path = model_path / name
        observed_files[name] = _sha256(file_path) if file_path.is_file() else None
    model_matches = observed_files == expected_files

    versions = {name: _package_version(name) for name in protocol["preflight"]["required_packages"]}
    python_ok = sys.version_info >= (3, 11)
    packages_ok = all(version is not None for version in versions.values())
    docker = docker_probe()

    blockers: list[str] = []
    if not python_ok:
        blockers.append("Python 3.11 or newer is unavailable")
    if not packages_ok:
        blockers.append("one or more pinned agent runtime packages are unavailable")
    if not model_matches:
        blockers.append("the pinned local model snapshot is missing or has mismatched files")
    if not task_matches:
        blockers.append("the historical task inputs do not match the v2 protocol snapshot")
    if not docker.get("daemon_available", False):
        blockers.append("Docker daemon is unavailable")
    blockers.append("VARE has no reviewed isolated evaluator launcher for this task")

    # This protocol is intentionally a readiness record, never a capability run.
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "BLOCKED",
        "stage": "trusted_execution_preflight",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "task_id": protocol["task_id"],
            "task_revision": protocol["task_revision"],
            "task_protocol_id": previous["protocol_id"],
            "task_files_sha256": task_hashes,
            "task_inputs_match_protocol": task_matches,
            "model_repository": protocol["model"]["repository"],
            "model_snapshot": protocol["model"]["snapshot"],
        },
        "environment": {
            "python": platform.python_version(),
            "python_minimum_met": python_ok,
            "packages": versions,
            "model_files_sha256": observed_files,
            "model_snapshot_matches_protocol": model_matches,
            "docker": docker,
            "isolation_adapter": "unavailable",
        },
        "attempt": {
            "model_loaded": False,
            "agent_started": False,
            "grader_started": False,
            "candidate_code_executed": False,
        },
        "measurements": None,
        "blockers": blockers,
        "claim_limit": "Preflight only; no model or coding-agent outcome was measured.",
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite existing evidence: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/verified-coding-agent-baseline-v3/preflight.json",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    report = collect_preflight(model_path=args.model_path.expanduser())
    try:
        write_report(args.output, report)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "blockers": report["blockers"]}, indent=2))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
