"""MLSys policy comparison entry point.

With no flags, this process refuses. It does not load a model, start a
container, or apply an update. ``--synthetic-train-dry-run`` runs the labeled
mock-policy loop on the frozen train split only. ``--power-analysis`` prints
the design calculation and does not grade tasks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOT_RUN = 3


def _refuse() -> int:
    protocol = json.loads(
        (ROOT / "protocols" / "mlsys_coding_policy_comparison_v1.json").read_text(
            encoding="utf-8"
        )
    )
    reasons = []
    if protocol["authorization"]["status"] != "authorized":
        reasons.append("authorization status is not authorized")
    clearance = ROOT / protocol["authorization"]["required_clearance_file"]
    if not clearance.is_file():
        reasons.append("isolation clearance file is absent")
    if protocol["runner"]["implemented"] is not True:
        reasons.append("comparison runner is not implemented")
    reasons.append(
        "this revision does not load a model, start a container, or apply an update"
    )
    print(
        json.dumps(
            {
                "protocol_id": protocol["protocol_id"],
                "reasons": reasons,
                "status": "not_run",
            },
            indent=2,
        )
    )
    return NOT_RUN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLSys coding policy comparison")
    parser.add_argument("--synthetic-train-dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--power-analysis", action="store_true")
    parser.add_argument("--skip-performance-proxy", action="store_true")
    args = parser.parse_args(argv)
    if not args.synthetic_train_dry_run and not args.power_analysis:
        return _refuse()
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    from vare.mlsys_coding_comparison import power_analysis, run_synthetic_dry_run

    if args.power_analysis and not args.synthetic_train_dry_run:
        print(json.dumps(power_analysis(), indent=2, sort_keys=True))
        return 0
    if args.output is None:
        print("synthetic dry-run requires --output", file=sys.stderr)
        return 2
    summary = run_synthetic_dry_run(
        args.output, performance_proxy=not args.skip_performance_proxy
    )
    if args.power_analysis:
        summary["power_analysis"] = power_analysis()
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "capability_claim": False,
                "real_grading_allowed": summary["real_grading"]["allowed"],
                "status": summary["status"],
                "wall_seconds": summary["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
