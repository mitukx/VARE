"""Refuse the MLSys policy comparison.

This revision records the comparison protocol and does not implement it.
The process does not load a model, start a container, or apply an update.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols" / "mlsys_coding_policy_comparison_v1.json"
NOT_RUN = 3


def main() -> int:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    reasons = []
    if protocol["authorization"]["status"] != "authorized":
        reasons.append("authorization status is not authorized")
    clearance = ROOT / protocol["authorization"]["required_clearance_file"]
    if not clearance.is_file():
        reasons.append("isolation clearance file is absent")
    if protocol["runner"]["implemented"] is not True:
        reasons.append("comparison runner is not implemented")
    reasons.append("this revision does not load a model, start a container, or apply an update")
    print(json.dumps({"protocol_id": protocol["protocol_id"], "reasons": reasons, "status": "not_run"}, indent=2))
    return NOT_RUN


if __name__ == "__main__":
    raise SystemExit(main())
