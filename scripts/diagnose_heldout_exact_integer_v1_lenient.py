#!/usr/bin/env python3
"""Re-score the retained v1 samples with the lenient extractor.

This writes a post-hoc diagnostic. It does not change the v1 gate decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vare.answer_scoring import score_output, summarize_two_tier  # noqa: E402
from vare.heldout_exact_integer import load_task_pack  # noqa: E402

PACK = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
RESULT = ROOT / "results/heldout-exact-integer-base-gate-v1/m1-mlx-5d36336"
OUTPUT = RESULT / "posthoc-lenient-diagnostic.json"


def diagnostic() -> dict[str, object]:
    tasks = {row["task_id"]: row for row in load_task_pack(PACK.read_text(encoding="utf-8"))}
    samples = [
        json.loads(line)
        for line in (RESULT / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    scores = [score_output(tasks[row["task_id"]]["expression"], str(row["output"])) for row in samples]
    body = summarize_two_tier(scores)
    body["gate_decision_unchanged"] = summary["decision"]
    body["label"] = "post-hoc diagnostic; not a base-gate result"
    body["lenient_rule"] = "last \\\\boxed{integer}, otherwise the last integer token"
    body["protocol_id"] = summary["protocol_id"]
    body["source_samples"] = "results/heldout-exact-integer-base-gate-v1/m1-mlx-5d36336/samples.jsonl"
    body["strict_rule"] = "last non-empty line full-matches ^####\\s+(-?\\d+)\\s*$"
    return body


def main() -> int:
    payload = json.dumps(diagnostic(), indent=2, sort_keys=True) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
