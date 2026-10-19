#!/usr/bin/env python3
"""Recompute the synthetic trusted-scoring boundary decision from evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def audit(evidence_dir: Path) -> dict[str, Any]:
    obs_path = evidence_dir / "observations.json"
    lock_path = evidence_dir / "protocol.lock.json"
    manifest = json.loads((evidence_dir / "manifest.json").read_text())
    observations_raw = obs_path.read_bytes()
    lock_raw = lock_path.read_bytes()
    observations = json.loads(observations_raw)
    lock = json.loads(lock_raw)
    repo_root = Path(__file__).resolve().parents[1]
    probe = observations["probe"]
    mounts = probe["candidate_launch_mounts"]
    canary = probe["canary_probe"]
    positive = probe["positive_control"]
    negative = probe["forged_score_control"]
    manifest_ok = (
        manifest.get("observations_sha256") == _sha(observations_raw)
        and manifest.get("protocol_sha256") == _sha(lock_raw)
        and manifest.get("runner_script_sha256") == lock.get("runner_script_sha256")
        and manifest.get("implementation_sha256") == lock.get("implementation_sha256")
        and all(
            (repo_root / relative).is_file()
            and _sha((repo_root / relative).read_bytes()) == digest
            for relative, digest in lock.get("implementation_sha256", {}).items()
        )
        and _sha((repo_root / "scripts/verify_trusted_scoring_boundary_v1.py").read_bytes())
        == lock.get("runner_script_sha256")
        and _sha(Path(__file__).read_bytes()) == lock.get("auditor_script_sha256")
    )
    mount_ok = (
        len(mounts) == 1
        and ",target=/workspace,readonly" in mounts[0]
        and "/grader" not in " ".join(mounts)
        and "/operator-private-grader" not in " ".join(mounts)
    )
    canary_hidden = (
        canary.get("returncode") == 0
        and canary.get("candidate_can_read_canary") is False
        and canary.get("candidate_canary_sha256") is None
        and canary.get("response") == {"grader_canary_sha256": None, "value": 4}
    )
    good_control_ok = (
        positive.get("passed") is True
        and positive.get("score") == 1.0
        and len(positive.get("case_results", [])) == 2
        and all(item.get("passed") is True and item.get("returncode") == 0
                for item in positive["case_results"])
    )
    forged_control_rejected = (
        negative.get("passed") is False
        and negative.get("score") == 0.0
        and len(negative.get("case_results", [])) == 1
        and negative["case_results"][0].get("passed") is False
        and negative["case_results"][0].get("failure") == "wrong_json_response"
    )
    cleanup_ok = probe.get("no_new_candidate_containers") is True
    checks = {
        "manifest_ok": manifest_ok,
        "candidate_mount_only_readonly_workspace": mount_ok,
        "synthetic_grader_canary_unreadable": canary_hidden,
        "known_good_control_passes": good_control_ok,
        "forged_score_control_rejected": forged_control_rejected,
        "candidate_containers_cleaned": cleanup_ok,
    }
    passed = all(checks.values())
    return {
        "protocol_id": lock.get("protocol_id"),
        "checks": checks,
        "overall": "PASS_E0_LIMITED_JSON_IO_BOUNDARY" if passed else "FAIL_OR_BLOCKED",
        "live_untrusted_repository_tests_eligible": False,
        "claim_limit": "This synthetic test supports only the bounded JSON-I/O mode and does not prove a general repository grader or sandbox security certification.",
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
    return 0 if result["overall"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
