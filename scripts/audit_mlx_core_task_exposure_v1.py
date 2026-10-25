#!/usr/bin/env python3
"""Search the frozen PR #129 tree for prior local task-family exposure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_REVISION = "4920c249f11d20dc0846ab8966066b1b5d4600e8"
SEARCH_PATHS = ("benchmarks", "docs", "protocols", "results", "scripts")
PATTERNS = {
    "mlx-4729-negative-take": (
        r"(^|[^A-Za-z0-9])#[[:space:]]*4729([^0-9]|$)|"
        r"mlx[-_ ]4729[-_]negative[-_]take|"
        r"negative integer index(es)? (to|in) mx\.take"
    ),
    "mlx-4730-normal-shape": (
        r"(^|[^A-Za-z0-9])#[[:space:]]*4730([^0-9]|$)|"
        r"mlx[-_ ]4730[-_]normal[-_]shape|"
        r"normal (loc[/ ]scale|location[/ ]scale) shape"
    ),
}


def git(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "-c", "core.useReplaceRefs=false", *args],
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("rev-parse", f"{REFERENCE_REVISION}^{{commit}}").decode().strip() != REFERENCE_REVISION:
        raise RuntimeError("pinned exposure-audit revision is unavailable")
    matches = {}
    for task_id, pattern in PATTERNS.items():
        result = subprocess.run(
            ["git", "-C", str(ROOT), "-c", "core.useReplaceRefs=false", "grep",
             "-n", "-I", "-i", "-E", pattern, REFERENCE_REVISION, "--", *SEARCH_PATHS],
            capture_output=True, text=True,
        )
        if result.returncode not in {0, 1}:
            raise RuntimeError(f"git grep failed for {task_id}: {result.stderr}")
        matches[task_id] = result.stdout.splitlines()
    report = {
        "schema_version": 1,
        "auditor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_revision": REFERENCE_REVISION,
        "search_paths": list(SEARCH_PATHS),
        "search_patterns": PATTERNS,
        "matches": matches,
        "local_prior_vare_model_or_agent_exposure": (
            "NO_MATCH_IN_PINNED_LOCAL_EVIDENCE" if not any(matches.values())
            else "MATCHES_REQUIRE_ADJUDICATION"
        ),
        "external_pretraining_or_recognition": "UNKNOWN_NOT_AUDITED",
        "scope": "Tracked VARE benchmark, protocol, documentation, script and result files at PR #129 head only. No claim about external models, public code pretraining, other repositories, untracked files, or prior task exposure outside this tree.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    raw = json.dumps(report, indent=2, sort_keys=True).encode() + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({
        "status": report["local_prior_vare_model_or_agent_exposure"],
        "total_matches": sum(len(rows) for rows in matches.values()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }, sort_keys=True))
    return 0 if not any(matches.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
