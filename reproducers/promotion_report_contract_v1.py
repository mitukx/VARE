#!/usr/bin/env python3
"""Reproduce VARE promotion of an under-supported evaluation report.

Loads the pre-fix PromotionGate from the pinned parent revision, compares it
with the current implementation, and checks both against a small independent
report-contract oracle using only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from statistics import fmean

from vare.config import PromotionConfig
from vare.promotion import PromotionGate
from vare.types import EvaluationReport


BASELINE_REVISION = "2572ac7"
ROOT = Path(__file__).resolve().parents[1]
CONFIG = PromotionConfig(
    min_primary_gain=0.01,
    min_eval_examples=64,
    paired_confidence_gate=True,
    min_paired_examples=1,
    paired_bootstrap_samples=200,
    paired_alpha=0.05,
)


def _report(policy_id: str, primary: float, rows: dict[str, float], n: int) -> EvaluationReport:
    return EvaluationReport(
        policy_id=policy_id,
        primary=primary,
        n=n,
        metadata={"per_task_scores": rows},
    )


def _baseline_gate():
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE_REVISION}:src/vare/promotion.py"],
        cwd=ROOT,
        text=True,
    )
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    source = source.replace("from .config import", "from vare.config import")
    source = source.replace("from .types import", "from vare.types import")
    namespace = {"__name__": "vare_promotion_baseline"}
    exec(compile(source, f"{BASELINE_REVISION}:src/vare/promotion.py", "exec"), namespace)
    return namespace["PromotionGate"](CONFIG), source_hash


def _independent_contract(report: EvaluationReport) -> bool:
    """Oracle: the retained paired rows must cover n and reproduce primary."""
    scores = report.metadata.get("per_task_scores")
    if not isinstance(scores, dict) or not scores or len(scores) != report.n:
        return False
    if any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for key, value in scores.items()
    ):
        return False
    return math.isclose(fmean(float(value) for value in scores.values()), report.primary,
                        rel_tol=1e-9, abs_tol=1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    # Adversarial fixture: 64 is declared but only one task is retained.
    incumbent = _report("incumbent", 0.50, {"task-0": 0.50}, n=64)
    candidate = _report("candidate", 0.70, {"task-0": 0.70}, n=64)
    old_gate, baseline_source_hash = _baseline_gate()
    before = old_gate.decide(incumbent, candidate)
    after = PromotionGate(CONFIG).decide(incumbent, candidate)

    # Negative control: complete 64-row evidence with matching aggregates.
    old_rows = {f"task-{i}": 0.50 for i in range(64)}
    new_rows = {f"task-{i}": 0.70 for i in range(64)}
    complete_incumbent = _report("incumbent", 0.50, old_rows, n=64)
    complete_candidate = _report("candidate", 0.70, new_rows, n=64)
    complete = PromotionGate(CONFIG).decide(complete_incumbent, complete_candidate)

    # Negative control: all rows are present but primary disagrees with them.
    inconsistent = _report("candidate", 0.80, new_rows, n=64)
    inconsistent_decision = PromotionGate(CONFIG).decide(complete_incumbent, inconsistent)

    result = {
        "reproducer": "promotion_report_contract_v1",
        "baseline_revision": BASELINE_REVISION,
        "baseline_promotion_source_sha256": baseline_source_hash,
        "contract": "if per_task_scores are present, they must contain exactly n rows and their mean must equal primary within 1e-9",
        "adversarial_case": {
            "declared_n": incumbent.n,
            "retained_rows": len(incumbent.metadata["per_task_scores"]),
            "coverage_fraction": len(incumbent.metadata["per_task_scores"]) / incumbent.n,
            "independent_oracle_accepts": _independent_contract(incumbent) and _independent_contract(candidate),
            "before_fix": {
                "accepted": before.accepted,
                "paired_n": before.paired_n,
                "reasons": list(before.reasons),
            },
            "after_fix": {
                "accepted": after.accepted,
                "paired_n": after.paired_n,
                "reasons": list(after.reasons),
            },
        },
        "negative_controls": {
            "complete_consistent_evidence": {
                "oracle_accepts_both_reports": _independent_contract(complete_incumbent) and _independent_contract(complete_candidate),
                "gate_accepts": complete.accepted,
                "paired_n": complete.paired_n,
            },
            "primary_disagrees_with_task_mean": {
                "oracle_accepts_candidate": _independent_contract(inconsistent),
                "gate_accepts": inconsistent_decision.accepted,
                "reasons": list(inconsistent_decision.reasons),
            },
        },
        "claim_boundary": "This verifies the VARE promotion-report contract with synthetic EvaluationReport objects. It is not a model update or capability result.",
    }
    if before.accepted is not True or before.paired_n != 1:
        raise AssertionError("pinned pre-fix implementation did not reproduce the defect")
    if after.accepted or "invalid_evaluation_metrics" not in after.reasons:
        raise AssertionError("fixed implementation accepted incomplete evidence")
    if not complete.accepted or complete.paired_n != 64:
        raise AssertionError("complete-evidence negative control failed")
    if inconsistent_decision.accepted or _independent_contract(inconsistent):
        raise AssertionError("aggregate-mismatch negative control failed")
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
