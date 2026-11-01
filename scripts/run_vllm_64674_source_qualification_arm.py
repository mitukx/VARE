#!/usr/bin/env python3
"""Run one hash-locked vLLM #64674 source-qualification arm in Docker."""
from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from vare.container_isolation import DockerIsolationRunner, IsolationLimits  # noqa: E402

MANIFEST_PATH = ROOT / "results/mlsys-fresh-vllm-64674-source-qualification-v1/input-manifest.json"
MANIFEST_SHA256 = "fabb07c0d6c8aa499448ff1d8b596e67f49af0a9849e9866871ae5799c82927a"
OUTPUT_ROOT = ROOT / "results/mlsys-fresh-vllm-64674-source-qualification-v2/arms"
ARM_NAMES = {"baseline", "wrong-mutant", "upstream-fixed"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _expected_outcome(arm: str, result: dict[str, Any]) -> tuple[bool, str]:
    if arm == "upstream-fixed":
        passed = (
            result.get("passed") is True
            and result.get("returncode") == 0
            and result.get("stdout") == "2 vLLM #64674 public regression contracts passed\n"
            and result.get("stderr") == ""
        )
        return passed, "fixed source passes both public regression contracts"
    output = str(result.get("stderr", ""))
    intended_failure = (
        result.get("passed") is False
        and result.get("returncode") == 1
        and result.get("stdout") == ""
        and "AssertionError" in output
        and "full_decodes(lines=10) == full_decodes(lines=100)" in output
        and output.count("Traceback (most recent call last)") == 1
    )
    return intended_failure, "baseline/mutant fails only the frozen reparse-count assertion"


def _run_public_tests(workspace: Path, image: str, expected_test_hash: str) -> dict[str, Any]:
    """Run the immutable public overlay with the candidate workspace mounted read-only."""
    test_file = workspace / "test_public.py"
    if not test_file.is_file() or sha256(test_file) != expected_test_hash:
        return {"passed": False, "error": "public test file is missing or changed"}
    limits = IsolationLimits(
        image=image,
        timeout_s=45,
        memory_mb=512,
        cpus=1.0,
        pids_limit=32,
        max_output_bytes=20_000,
    )
    runner = DockerIsolationRunner()
    argv = runner.candidate_argv(
        limits,
        name="vare-agent-" + secrets.token_hex(12),
        allowed_workspace_root=workspace.parent,
        workspace=workspace,
        command=["python", "/workspace/test_public.py"],
        readonly_workspace=True,
    )
    started = time.perf_counter()
    result = asyncio.run(runner.execute(argv, limits))
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "isolation": "docker-network-none-read-only-root-read-only-workspace-no-grader-mount",
    }


def run_arm(arm: str, workspace: Path) -> dict[str, Any]:
    if arm not in ARM_NAMES:
        raise ValueError("unknown qualification arm")
    if os.environ.get("DOCKER_CONTEXT") != "colima":
        raise RuntimeError("DOCKER_CONTEXT=colima is required")
    if sha256(MANIFEST_PATH) != MANIFEST_SHA256:
        raise RuntimeError("input manifest differs from the frozen v2 identity")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    workspace = workspace.resolve(strict=True)
    expected_workspace = (ROOT / f"work/mlsys-fresh-vllm-64674-source-qualification/qualification/{arm}").resolve(strict=True)
    if workspace != expected_workspace or workspace.is_symlink():
        raise ValueError("workspace differs from the frozen arm path")
    expected = manifest["arms"][arm]["workspace_files"]
    workspace_files = sorted(
        path.relative_to(workspace).as_posix()
        for path in workspace.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    if workspace_files != sorted(expected):
        raise ValueError("arm workspace file set differs from the frozen input manifest")
    if any(path.is_symlink() for path in workspace.rglob("*")):
        raise ValueError("arm workspace contains a symlink")
    actual: dict[str, str] = {}
    for relative, expected_sha in expected.items():
        path = workspace / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"arm input is missing or not a regular file: {relative}")
        actual[relative] = sha256(path)
        if actual[relative] != expected_sha:
            raise ValueError(f"arm input hash mismatch: {relative}")
    public_test_sha = manifest["public_test"]["sha256"]
    if actual.get("test_public.py") != public_test_sha:
        raise ValueError("public test hash differs from frozen test identity")

    output_dir = OUTPUT_ROOT / arm
    output_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.perf_counter()
    public_result = _run_public_tests(
        workspace,
        manifest["container_image"],
        public_test_sha,
    )
    elapsed = time.perf_counter() - start
    expected_pass, expected_claim = _expected_outcome(arm, public_result)
    record = {
        "schema_version": 1,
        "arm": arm,
        "candidate": "vllm-64674-responses-custom-tool-stream-decode",
        "started_at_utc": started,
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 6),
        "workspace_path": str(workspace),
        "workspace_file_hashes": actual,
        "test_public_sha256": public_test_sha,
        "container_image": manifest["container_image"],
        "expected_outcome": expected_claim,
        "expected_outcome_observed": expected_pass,
        "public_test_result": public_result,
    }
    _json_write(output_dir / "result.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARM_NAMES))
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_arm(args.arm, args.workspace)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"arm": args.arm, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["expected_outcome_observed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
