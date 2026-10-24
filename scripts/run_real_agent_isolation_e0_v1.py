#!/usr/bin/env python3
"""Run the existing VARE E0 boundary probes against the frozen task image."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.container_isolation import IsolationLimits
import verify_docker_isolation_boundary_v1 as probes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(output: Path, image: str, context: str) -> dict:
    os.environ["DOCKER_CONTEXT"] = context
    probes.CONTEXT = context
    probes._limits = lambda image=image, timeout=15.0, output=16_000: IsolationLimits(
        image=image, timeout_s=timeout, memory_mb=probes.MEMORY_MB,
        cpus=probes.CPUS, pids_limit=probes.PIDS_LIMIT,
        uid=probes.UID, gid=probes.GID, max_output_bytes=output,
    )
    environment = probes._environment(context, image)
    if ("seccomp" not in environment["docker_security_options"]
            or environment["colima"].get("driver") != "macOS Virtualization.Framework"
            or environment["colima"].get("runtime") != "docker"
            or environment["colima"].get("kubernetes")):
        raise RuntimeError("Colima/Docker security preconditions did not pass")

    candidate_root = output / "candidate-root"
    workspace = candidate_root / "workspace"
    grader_root = output / "grader-root"
    workspace.mkdir(parents=True)
    grader_root.mkdir(parents=True)
    host_canary = output / "synthetic-host-canary.txt"
    host_canary.write_text("vare-synthetic-host-canary-v1", encoding="utf-8")

    runner = probes.DockerIsolationRunner()
    active = await probes._observe_active_container(runner, candidate_root, workspace)
    candidate = await probes._candidate_probe(
        runner, candidate_root, workspace, host_canary,
        repetitions=probes.REPLICATES,
    )
    lifecycle = await probes._lifecycle_probe(
        runner, candidate_root, workspace, repetitions=probes.REPLICATES,
    )
    grader = []
    for replicate in range(probes.REPLICATES):
        canary = "vare-synthetic-grader-canary-" + secrets.token_hex(16)
        (grader_root / "synthetic-hidden-canary.txt").write_text(canary, encoding="utf-8")
        observation = await probes._grader_probe(
            runner, candidate_root, workspace, grader_root, canary, image=image,
        )
        observation["replicate"] = replicate + 1
        grader.append(observation)

    result = {
        "schema_version": 1,
        "protocol_id": "real-agent-candidate-isolation-e0-v1",
        "image": image,
        "environment": environment,
        "replicates": probes.REPLICATES,
        "active_container": active,
        "candidate_probe": candidate,
        "lifecycle": lifecycle,
        "grader_probe": grader,
        "claim_limit": "E0 runtime configuration and synthetic boundary evidence only; not a hostile-code sandbox certification or coding capability result",
    }
    return result


def accepted(result: dict) -> bool:
    active = result["active_container"]
    candidate = result["candidate_probe"]
    lifecycle = result["lifecycle"]
    grader = result["grader_probe"]
    return all([
        active["running_when_inspected"], active["network_mode"] == "none",
        active["readonly_rootfs"], "ALL" in active["cap_drop"],
        active["image_id"] == result["environment"]["image_id"],
        active["left_after_normal_exit"] == [],
        all(
            row["returncode"] == 0
            and not row["observation"]["host_canary_readable"]
            and not row["observation"]["host_canary_writable"]
            and not row["observation"]["network_reachable"]
            and not row["observation"]["docker_socket_present"]
            and not row["observation"]["rootfs_writable"]
            and row["observation"]["workspace_writable"]
            for row in candidate
        ),
        all(row["expected_timeout"] and not row["new_containers_after"]
            for row in lifecycle["timeout"]),
        all(row["cancelled"] and not row["new_containers_after"]
            for row in lifecycle["cancellation"]),
        all(row["overflowed"] and not row["new_containers_after"]
            for row in lifecycle["output_overflow"]),
        all(not row["canary_visible"] and row["network_mode_none"]
            and row["readonly_rootfs"] and row["readonly_mounts"] for row in grader),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--context", default="colima")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite evidence directory: {output}")
    if not args.image.startswith("sha256:") or len(args.image) != 71:
        parser.error("image must be a pinned local sha256 image ID")
    output.mkdir(parents=True)
    try:
        result = asyncio.run(run(output, args.image, args.context))
        result["accepted"] = accepted(result)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        (output / "observations.json").write_bytes(encoded)
        manifest = {
            "observations_sha256": hashlib.sha256(encoded).hexdigest(),
            "probe_runner_sha256": sha256(Path(probes.__file__)),
            "isolation_module_sha256": sha256(ROOT / "src/vare/container_isolation.py"),
            "protocol_sha256": sha256(ROOT / "protocols/real_agent_interface_v1.json"),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(json.dumps({"accepted": result["accepted"], "observations_sha256": manifest["observations_sha256"]}))
        return 0 if result["accepted"] else 1
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({
            "exception": type(exc).__name__, "message": str(exc),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
