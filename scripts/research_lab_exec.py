#!/usr/bin/env python3
"""Exercise the ResearchGym execution backend on its own.

This is not the operator CLI. ``vare research run|status|resume`` belongs to
the controller slice. The process exit code reports whether this script
produced a run view, not whether the candidate was accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vare.research_lab import ExperimentRunner, load_spec  # noqa: E402
from vare.research_lab.jsonutil import canonical_json  # noqa: E402


def _runner(args: argparse.Namespace) -> ExperimentRunner:
    return ExperimentRunner(args.store, args.runs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--spec", type=Path, required=True)
    for name in ("execute", "resume", "status", "cancel"):
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        runner = _runner(args)
        if args.command == "prepare":
            view = runner.prepare(load_spec(args.spec))
        elif args.command == "execute":
            view = runner.execute(args.run_id)
        elif args.command == "resume":
            view = runner.resume(args.run_id)
        elif args.command == "cancel":
            view = runner.cancel(args.run_id)
        else:
            view = runner.status(args.run_id)
    except (OSError, ValueError, RuntimeError) as exc:
        print(canonical_json({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 1
    json.dump(view.to_dict(), sys.stdout, sort_keys=True, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
