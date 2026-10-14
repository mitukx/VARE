#!/usr/bin/env python3
"""Reward-signal ablation and a short promotion-gate loop.

``--plan`` prints the step count that fits the 2.5 hour budget. It does not
import PyTorch. ``--cpu-smoke`` runs a tiny policy and does not download a
model. The GPU run requires CUDA and writes JSONL that a later process can
resume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minigrpo.flagship import plan_report, run_cpu_smoke, run_gpu


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Flagship reward ablation")
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--output", default="")
    parser.add_argument("--throughput", default="")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args(argv)
    throughput = Path(args.throughput) if args.throughput else None
    if args.plan:
        print(json.dumps(plan_report(args.mode, throughput), indent=2, sort_keys=True))
        return 0
    if not args.output:
        raise SystemExit("--output is required")
    output = Path(args.output)
    if args.cpu_smoke:
        summary = run_cpu_smoke(output)
        print(json.dumps({"heldout_rows": len(summary["heldout"]), "rsi_rows": len(summary["rsi"])}, sort_keys=True))
        return 0
    run_gpu(args.mode, output, throughput)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
