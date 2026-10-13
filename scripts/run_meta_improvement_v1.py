"""Run frozen-seed CPU contextual improvement-value benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

from vare.meta_improvement import canonical_summary_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--budget", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 1000:
        parser.error("--seeds must be in 1..1000")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")
    report = canonical_summary_json(range(args.seed_start, args.seed_start + args.seeds), budget=args.budget)
    # Research results are written only on explicit invocation, never imported.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output} ({args.seeds} matched seeds)")


if __name__ == "__main__":
    main()
