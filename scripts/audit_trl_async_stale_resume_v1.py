#!/usr/bin/env python3
"""Verify hashes and recompute the retained TRL stale-resume observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import reproduce_trl_async_stale_resume_v1 as stale_resume  # noqa: E402


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _read_strict_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def audit_bundle(bundle: Path) -> dict[str, object]:
    """Reject ambiguous JSON before the frozen v1/v4 auditor reads evidence."""
    _read_strict_json(bundle / "manifest.json")
    summary = _read_strict_json(bundle / "summary.json")
    if isinstance(summary, dict):
        protocol_id = summary.get("protocol_id")
        lock_path = (
            stale_resume.LOCK_PATHS.get(protocol_id)
            if isinstance(protocol_id, str)
            else None
        )
        if lock_path is not None:
            _read_strict_json(lock_path)
    return stale_resume.audit_bundle(bundle)


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
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
