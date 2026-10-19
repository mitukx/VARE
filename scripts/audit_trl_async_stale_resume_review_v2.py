#!/usr/bin/env python3
"""Audit replay-suffix v2 evidence with strict stage exit-code typing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from scripts import reproduce_trl_async_stale_resume_review_v2 as replay_suffix_v2


def audit_bundle(bundle: Path, protocol_path: Path | None = None) -> dict[str, Any]:
    """Run the frozen auditor, then reject bool/float exit-code aliases."""
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
