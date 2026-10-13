from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vare.answer_scoring import lenient_integer, score_output, summarize_two_tier
from vare.heldout_exact_integer import load_task_pack, parse_final_integer

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/heldout-exact-integer-base-gate-v1/m1-mlx-5d36336"
DIAGNOSTIC = RESULT / "posthoc-lenient-diagnostic.json"
PACK = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"


def test_lenient_rule_is_last_boxed_or_last_integer():
    assert parse_final_integer("working\n#### 12") == 12
    assert lenient_integer("working\n#### 12") == 12
    assert parse_final_integer("value 7") is None
    assert lenient_integer("value 7") == 7
    assert lenient_integer("see \\boxed{4} then \\boxed{9}") == 9
    assert lenient_integer("no digits") is None
    scored = score_output("3 * 4 + 2", "#### 14")
    assert scored["strict_format"] and scored["strict_correct"] and scored["lenient_correct"]
    summary = summarize_two_tier(
        [
            {"strict_format": True, "strict_correct": False, "lenient_extracted": True, "lenient_correct": False},
            {"strict_format": False, "strict_correct": False, "lenient_extracted": True, "lenient_correct": True},
        ]
    )
    assert summary["format_compliance_rate"] == 0.5
    assert summary["conditional_accuracy"] == 0.0
    assert summary["lenient_conditional_accuracy"] == 0.5
    assert summary["gate_result"] is False
    assert summary["posthoc"] is True


def test_v1_lenient_rescore_is_posthoc_and_leaves_the_gate():
    recorded = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    assert hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest() == (
        "c5ed2375ed9957228ab2bb789559bffc994066df0eeda14c52da12632939be32"
    )
    tasks = {row["task_id"]: row for row in load_task_pack(PACK.read_text(encoding="utf-8"))}
    rows = [json.loads(line) for line in (RESULT / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    recomputed = summarize_two_tier(
        [score_output(tasks[row["task_id"]]["expression"], str(row["output"])) for row in rows]
    )
    for key, value in recomputed.items():
        assert recorded[key] == value
    summary = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "stop_zero"
    assert recorded["gate_decision_unchanged"] == "stop_zero"
    assert recorded["label"].startswith("post-hoc")
    assert recorded["format_compliant"] == 3
    assert recorded["conditional_accuracy"] == 0.0
    assert recorded["lenient_correct"] == 19
