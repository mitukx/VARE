#!/usr/bin/env python3
"""Check promotion behavior for finite controls and non-finite reports."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    sys.path.insert(0, str(workspace / "src"))

    from vare.config import PromotionConfig
    from vare.promotion import PromotionGate
    from vare.types import EvaluationReport

    config = PromotionConfig(
        min_primary_gain=0.01,
        max_slice_regression=0.03,
        max_relative_cost_increase=0.20,
        max_verifier_disagreement=0.20,
        min_eval_examples=10,
        paired_confidence_gate=False,
        min_paired_examples=1,
        paired_bootstrap_samples=80,
    )
    gate = PromotionGate(config)
    incumbent = EvaluationReport("incumbent", 0.70, {"hard": 0.60}, cost=1.0,
                                 verifier_disagreement=0.02, n=100)
    candidate = EvaluationReport("candidate", 0.73, {"hard": 0.60}, cost=1.0,
                                 verifier_disagreement=0.02, n=100)

    cases = []

    def run(case_id: str, old: EvaluationReport, new: EvaluationReport,
            *, paired: bool = False) -> None:
        selected = gate
        if paired:
            selected = PromotionGate(PromotionConfig(
                min_primary_gain=0.01,
                min_eval_examples=10,
                paired_confidence_gate=True,
                min_paired_examples=1,
                paired_bootstrap_samples=80,
            ))
        decision = selected.decide(old, new)
        cases.append({
            "case_id": case_id,
            "accepted": decision.accepted,
            "reasons": list(decision.reasons),
            "decision_metrics_finite": all(
                math.isfinite(value) for value in (
                    decision.primary_gain, decision.worst_slice_regression,
                    *(() if decision.paired_gain is None else (decision.paired_gain,)),
                    *(() if decision.paired_lcb is None else (decision.paired_lcb,)),
                )
            ),
        })

    run("finite-positive-control", incumbent, candidate)
    run("candidate-primary-nan", incumbent,
        EvaluationReport("candidate", math.nan, {"hard": 0.60}, cost=1.0,
                         verifier_disagreement=0.02, n=100))
    run("candidate-primary-positive-infinity", incumbent,
        EvaluationReport("candidate", math.inf, {"hard": 0.60}, cost=1.0,
                         verifier_disagreement=0.02, n=100))
    run("incumbent-primary-nan", EvaluationReport("incumbent", math.nan,
        {"hard": 0.60}, cost=1.0, verifier_disagreement=0.02, n=100), candidate)
    run("candidate-slice-nan", incumbent,
        EvaluationReport("candidate", 0.73, {"hard": math.nan}, cost=1.0,
                         verifier_disagreement=0.02, n=100))
    run("candidate-cost-nan", incumbent,
        EvaluationReport("candidate", 0.73, {"hard": 0.60}, cost=math.nan,
                         verifier_disagreement=0.02, n=100))
    run("candidate-disagreement-nan", incumbent,
        EvaluationReport("candidate", 0.73, {"hard": 0.60}, cost=1.0,
                         verifier_disagreement=math.nan, n=100))
    run("paired-task-score-nan", EvaluationReport(
        "incumbent", 0.70, n=10, metadata={"per_task_scores": {"task": 0.60}}),
        EvaluationReport("candidate", 0.73, n=10,
                         metadata={"per_task_scores": {"task": math.nan}}),
        paired=True)

    expected = {"finite-positive-control": True}
    expected.update({case["case_id"]: False for case in cases if case["case_id"] != "finite-positive-control"})
    for case in cases:
        case["expected_accepted"] = expected[case["case_id"]]
        if not case["expected_accepted"]:
            case["expected_reason"] = "invalid_evaluation_metrics"
        case["passed"] = (
            case["accepted"] == case["expected_accepted"]
            and case["decision_metrics_finite"]
            and (case["expected_accepted"]
                 or case["expected_reason"] in case["reasons"])
        )
    result = {
        "schema_version": 1,
        "task_id": "vare-promotion-gate-nonfinite-metrics-v1",
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
        "claim_limit": "A deterministic CPU promotion-gate input validation check; not model-learning or general evaluator-validity evidence.",
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
