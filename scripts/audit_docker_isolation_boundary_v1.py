#!/usr/bin/env python3
"""Recompute Docker boundary decisions from the retained raw observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(root: Path) -> dict[str, Any]:
    observations_path = root / "observations.json"
    observations_bytes = observations_path.read_bytes()
    observations = json.loads(observations_bytes)
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = json.loads((root / "protocol.lock.json").read_text())
    manifest_ok = (
        manifest.get("observations_sha256") == sha256(observations_bytes)
        and manifest.get("protocol_sha256") == sha256(
            (root / "protocol.lock.json").read_bytes()
        )
    )

    limits = observations["limits"]
    expected_cpu = "50000 100000"
    candidate_checks = []
    for item in observations["candidate_probe"]:
        probe = item["observation"]
        candidate_checks.append(
            item["returncode"] == 0
            and probe["host_canary_readable"] is False
            and probe["host_canary_writable"] is False
            and observations["host_canary"]["unchanged"] is True
            and probe["network_reachable"] is False
            and probe["default_route_present"] is False
            and probe["uid"] == limits["uid"]
            and probe["docker_socket_present"] is False
            and probe["cap_eff"] == "0000000000000000"
            and probe["no_new_privs"] == "1"
            and probe["rootfs_writable"] is False
            and probe["workspace_writable"] is True
            and probe["pids_limit_enforced"] is True
            and probe["pids_max"] == str(limits["pids_limit"])
            and probe["memory_max"] == str(limits["memory_mb"] * 1024 * 1024)
            and probe["cpu_max"] == expected_cpu
            and not item["new_managed_containers_after"]
        )

    active = observations["active_container"]
    expected_image = observations["environment"]["image_reference"]
    active_configuration_ok = (
        active["running_when_inspected"] is True
        and active["configured_image"] == expected_image
        and active["user"] == f"{limits['uid']}:{limits['gid']}"
        and active["network_mode"] == "none"
        and active["readonly_rootfs"] is True
        and "ALL" in active["cap_drop"]
        and "no-new-privileges:true" in active["security_opt"]
        and active["memory_bytes"] == limits["memory_mb"] * 1024 * 1024
        and active["nano_cpus"] == int(limits["cpus"] * 1_000_000_000)
        and active["pids_limit"] == limits["pids_limit"]
        and active["mounts"] == [
            {"destination": "/workspace", "rw": True, "type": "bind"}
        ]
        and not active["left_after_normal_exit"]
    )

    lifecycle = observations["lifecycle"]
    lifecycle_results = {
        "timeout": all(
            item["expected_timeout"] and not item["new_containers_after"]
            for item in lifecycle["timeout"]
        ),
        "cancellation": all(
            item["cancelled"] and not item["new_containers_after"]
            for item in lifecycle["cancellation"]
        ),
        "output_overflow": all(
            item["overflowed"] and not item["new_containers_after"]
            for item in lifecycle["output_overflow"]
        ),
    }
    grader_probes = observations["grader_probe"]
    canary_reads = sum(bool(item["canary_visible"]) for item in grader_probes)
    grader_confidentiality = (
        "FAIL_SYNTHETIC_CANARY_VISIBLE" if canary_reads else "NOT_DEMONSTRATED"
    )

    return {
        "protocol_id": protocol["protocol_id"],
        "manifest_ok": manifest_ok,
        "candidate_probe_repetitions": len(candidate_checks),
        "candidate_containment_passes": sum(candidate_checks),
        "candidate_containment_pass": bool(candidate_checks) and all(candidate_checks),
        "active_container_configuration_pass": active_configuration_ok,
        "lifecycle": lifecycle_results,
        "grader_canary_reads": canary_reads,
        "grader_probe_repetitions": len(grader_probes),
        "grader_confidentiality": grader_confidentiality,
        "live_candidate_execution_eligible": False,
        "overall": "FAIL_CONFIDENTIALITY_BOUNDARY" if canary_reads else "BLOCKED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_dir", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.evidence_dir.expanduser().resolve())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"audit failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["manifest_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
