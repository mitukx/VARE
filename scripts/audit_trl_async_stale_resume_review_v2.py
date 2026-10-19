#!/usr/bin/env python3
"""Audit replay-suffix v2 evidence with strict stage exit-code typing."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import reproduce_trl_async_stale_resume_review_v2 as replay_suffix_v2  # noqa: E402

EXPECTED_STAGE_NAMES = (
    "baseline-failing-test",
    "corrected-targeted-test",
    "corrected-full-test-file",
)


def _validate_bundle_log_references(bundle: Path) -> Path:
    bundle = bundle.resolve(strict=True)
    for directory, dirnames, filenames in os.walk(bundle, topdown=True, followlinks=False):
        for name in (*dirnames, *filenames):
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(
                    f"bundle contains a symbolic link: {path.relative_to(bundle).as_posix()}"
                )

    summary = replay_suffix_v2.read_json(bundle / "summary.json")
    runs = summary.get("runs")
    if not isinstance(runs, list) or [
        run.get("name") if isinstance(run, dict) else None for run in runs
    ] != list(EXPECTED_STAGE_NAMES):
        raise ValueError("run inventory mismatch")
    for record in runs:
        name = record["name"]
        if record.get("log") != f"{name}.log":
            raise ValueError(f"stage log path must match its retained filename: {name}")
        stage_record = replay_suffix_v2.read_json(bundle / f"{name}.json")
        for retained_record in (record, stage_record):
            outcome = (
                retained_record.get("pytest_outcome")
                if isinstance(retained_record, dict)
                else None
            )
            if not isinstance(outcome, dict) or any(
                type(count) is not int or count < 0 for count in outcome.values()
            ):
                raise ValueError(
                    f"pytest outcome counts must be nonnegative integers: {name}"
                )
    return bundle


def audit_bundle(bundle: Path, protocol_path: Path | None = None) -> dict[str, Any]:
    """Run the frozen auditor with in-bundle logs and exact integer exit codes."""
    bundle = _validate_bundle_log_references(bundle)
    audited = replay_suffix_v2.audit_bundle(
        bundle,
        protocol_path or replay_suffix_v2.PROTOCOL_PATH,
    )
    runs = audited.get("runs")
    if not isinstance(runs, list):
        raise ValueError("audited stage inventory is not a list")
    for record in runs:
        if not isinstance(record, dict) or type(record.get("exit_code")) is not int:
            stage = record.get("name", "unknown") if isinstance(record, dict) else "unknown"
            raise ValueError(f"stage exit code must be an integer: {stage}")
    return audited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="retained replay-suffix evidence directory")
    parser.add_argument("--protocol", type=Path, help="frozen protocol lock; defaults to v2")
    args = parser.parse_args()
    try:
        audited = audit_bundle(
            args.bundle.expanduser().resolve(),
            args.protocol.expanduser().resolve() if args.protocol else None,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "protocol_id": audited.get("protocol_id"),
                "run_count": len(audited.get("runs", [])),
                "all_checks_passed": audited.get("all_checks_passed"),
            },
            indent=2,
        )
    )
    return 0 if audited.get("all_checks_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
