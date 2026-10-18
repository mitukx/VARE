#!/usr/bin/env python3
"""Verify hashes and recompute the retained TRL stale-resume observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reproduce_trl_async_stale_resume_v1 import audit_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="reproduction output directory")
    args = parser.parse_args()
    try:
        summary = audit_bundle(args.bundle.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "protocol_id": summary["protocol_id"],
                "run_count": summary["run_count"],
                "all_checks_passed": summary["all_checks_passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
