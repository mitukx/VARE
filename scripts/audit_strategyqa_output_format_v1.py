#!/usr/bin/env python3
"""Read-only format diagnosis for the retired StrategyQA base-gate outputs.

This script reads completions only. It does not inspect answers/gold labels,
compute accuracy, or alter the frozen run.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "results/cpu-strategyqa-grpo-shift-v2/base-gate/run-1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    source = args.run_dir / "summary.json"
    source_bytes = source.read_bytes()
    summary = json.loads(source_bytes)
    records = summary["records"]
    prefix = collections.Counter()
    marker_count = 0
    for row in records:
        completion = str(row["completion"]).strip()
        match = re.match(r"(?i)^(yes|no)\b", completion)
        prefix[match.group(1).lower() if match else "other"] += 1
        marker_count += bool(re.search(r"(?i)final answer\s*:", completion))
    result = {
        "audit_id": "strategyqa_output_format_diagnostic_v1",
        "protocol_id": summary["protocol_id"],
        "run_summary_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "n_completions": len(records),
        "prefix_form_counts": dict(sorted(prefix.items())),
        "requested_final_answer_marker_count": marker_count,
        "stored_parser_accept_count": sum(row.get("parsed") is not None for row in records),
        "semantic_accuracy": None,
        "labels_read": False,
        "interpretation": "Output-format compliance diagnosis only; not an accuracy estimate or a new performance evaluation.",
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
