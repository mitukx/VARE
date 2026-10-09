#!/usr/bin/env python3
"""Run and independently audit the frozen synthetic preference study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed (%d): %s\n%s\n%s"
            % (result.returncode, " ".join(command), result.stdout, result.stderr)
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("command did not return JSON: " + " ".join(command)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("command returned a non-object JSON value")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new bundle directory; the script never overwrites evidence")
    args = parser.parse_args()
    bundle = args.output.expanduser().resolve()
    record_path = bundle.with_name(bundle.name + "-reproduction.json")
    independent_path = bundle.with_name(bundle.name + "-independent-audit.json")
    if bundle.exists() or record_path.exists() or independent_path.exists():
        parser.error("bundle and sidecar paths must not already exist")

    try:
        status = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
        )
        if status.strip():
            raise RuntimeError("reproduction requires a clean checkout")
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        runner = run_json([
            sys.executable, str(ROOT / "scripts/run_synthetic_preference_robustness.py"),
            "--output", str(bundle),
        ])
        auditor = run_json([
            sys.executable, str(ROOT / "scripts/audit_synthetic_preference_robustness.py"),
            str(bundle),
        ])
        independent = run_json([
            sys.executable,
            str(ROOT / "scripts/verify_synthetic_preference_robustness_bundle_independent.py"),
            str(bundle), "--output", str(independent_path),
        ])
        if auditor.get("status") != "verified" or independent.get("status") != "pass":
            raise RuntimeError("one or more bundle audits did not pass")
        record = {
            "status": "reproduced_and_audited",
            "study": "vare-synthetic-preference-robustness-v1",
            "git_head": head,
            "python_version": sys.version,
            "bundle_path": str(bundle),
            "bundle_manifest_sha256": hashlib.sha256(
                (bundle / "manifest.json").read_bytes()
            ).hexdigest(),
            "fresh_run_summary": runner,
            "shared_auditor": auditor,
            "independent_verifier": independent,
            "external_human_reproduction": False,
            "interpretation": (
                "Fresh run from the frozen v1 protocol using this Python runtime. "
                "The independent verifier replays retained designs; it does not "
                "regenerate Gaussian contexts or sampled action pairs. This is not "
                "outside-person reproduction or language-model capability evidence."
            ),
        }
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(json.dumps({
            "status": record["status"],
            "git_head": head,
            "bundle": str(bundle),
            "record": str(record_path),
            "acceptance_passed": runner.get("acceptance_passed"),
            "independent_audit": independent.get("status"),
            "external_human_reproduction": False,
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print("reproduction failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
