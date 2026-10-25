"""Safely write a report for the retained sustained-crash recovery audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_output_safety import (  # noqa: E402
    preflight_new_output,
    write_new_output,
)
from scripts.audit_sustained_crash_recovery import audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crash-output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = preflight_new_output(
        args.output,
        [args.crash_output, args.fixture, args.source_run,
         args.source_audit, args.protocol],
    )
    report = audit(args.crash_output, args.fixture, args.source_run,
                   args.source_audit, args.protocol)
    write_new_output(output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
