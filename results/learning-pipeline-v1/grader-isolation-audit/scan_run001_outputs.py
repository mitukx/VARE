#!/usr/bin/env python3
"""Apply the isolated grader's blocking static patterns to run-001 candidates.

Reads the added lines of every applied edit (``final_diff``) in the run-001
train rollout log (hash-only; pass its path) and the three committed held-out
graded logs. Reports how many candidates would have been quarantined. This
only re-reads already-graded outputs; nothing is executed or regenerated.

    python scan_run001_outputs.py --rollouts PATH/rollouts/graded.jsonl --out static_scan_run001.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src"), str(HERE.parents[2])]
from scripts.learning_pipeline_isolated_grading import BLOCKING_PATTERNS  # noqa: E402

ROLLOUTS_SHA256 = "cb18ed4af65a4f542008a49b6a42d8c8ee3b848cf0775ea292e47f63a9e121b3"


def scan(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    applied = [r for r in rows if r.get("final_diff")]
    flagged = []
    for r in applied:
        added = "\n".join(ln[1:] for ln in r["final_diff"].splitlines()
                          if ln.startswith("+") and not ln.startswith("+++"))
        hits = [name for name, pat in BLOCKING_PATTERNS if re.search(pat, added)]
        if hits:
            flagged.append({"instance_id": r["instance_id"], "sample_index": r["sample_index"],
                            "reward": r["reward"], "patterns": hits})
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(rows),
            "applied_edits": len(applied), "trusted_pass": sum(r["reward"] for r in rows),
            "flagged": len(flagged), "flagged_and_trusted_pass": sum(f["reward"] for f in flagged),
            "flagged_rows": flagged}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run = HERE.parent / "run-001"
    out = {"patterns": [n for n, _ in BLOCKING_PATTERNS], "logs": {}}
    r = scan(a.rollouts)
    if r["sha256"] != ROLLOUTS_SHA256:
        raise SystemExit("rollout log hash differs from run-001 RESULTS.md")
    out["logs"]["rollouts/graded.jsonl"] = r
    for arm in ("no_update", "sft_reference", "rft_self"):
        out["logs"][f"eval/{arm}/graded.jsonl"] = scan(run / "eval" / arm / "graded.jsonl")
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("rows", "applied_edits", "trusted_pass", "flagged")}
                      for k, v in out["logs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
