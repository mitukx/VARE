#!/usr/bin/env python3
"""Run agent-v2 with a pinned Docker context while reusing the v1 harness."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_mlx_lm_1912_qwen25_agent_v1 as harness  # noqa: E402

BUNDLE = ROOT / "results/mlsys-repo-task-mlx-lm-1912-qwen25-3b-agent-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    protocol_path = BUNDLE / "protocol.json"
    sidecar_path = BUNDLE / "protocol.sha256"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = sidecar_path.read_text(encoding="ascii").split()[0]
    if _sha256(protocol_path) != expected:
        raise RuntimeError("agent-v2 protocol SHA-256 sidecar mismatch")
    if protocol.get("status") != "FROZEN_NOT_RUN":
        raise RuntimeError("agent-v2 protocol is not frozen in its pre-run state")
    context = protocol.get("resources", {}).get("docker_context")
    if context != "colima":
        raise RuntimeError("agent-v2 requires the frozen Colima Docker context")
    preflight_path = ROOT / protocol["runtime"]["docker_context_preflight_path"]
    if _sha256(preflight_path) != protocol["runtime"]["docker_context_preflight_sha256"]:
        raise RuntimeError("Docker context infrastructure preflight hash mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight.get("status") != "COMPLETED"
            or preflight.get("docker_context") != context
            or preflight.get("image") != protocol["grader"].get("image")
            or preflight.get("task_prompt_loaded") is not False
            or preflight.get("model_loaded") is not False):
        raise RuntimeError("Docker infrastructure preflight did not pass its frozen checks")

    # DockerIsolationRunner snapshots these variables when it is constructed.
    os.environ["DOCKER_CONTEXT"] = context
    harness.BUNDLE = BUNDLE
    harness.TASK_ROOT = BUNDLE / "task"
    harness.PROTOCOL_PATH = protocol_path
    harness.PROTOCOL_SHA_PATH = sidecar_path
    return harness.main()


if __name__ == "__main__":
    raise SystemExit(main())
