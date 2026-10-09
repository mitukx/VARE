#!/usr/bin/env python3
"""Independent integer/rational audit of the async GRPO production-loss result."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/trl_async_grpo_truncated_support_v1.lock.json"
P = (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5))
ADVANTAGES = (Fraction(0), Fraction(0), Fraction(1))
Q_CASES = {
    "identity_control": (Fraction(1, 2), Fraction(3, 10), Fraction(1, 5)),
    "top_p_truncation": (Fraction(5, 8), Fraction(3, 8), Fraction(0)),
}
TOL = 1e-6


def _close(actual, expected):
    return math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= TOL


def audit(result: dict) -> dict:
    lock = json.loads(LOCK.read_text())
    errors = []
    if result.get("protocol_id") != lock["protocol_id"]:
        errors.append("protocol id mismatch")
    if result.get("status") != "pass":
        errors.append("primary runner did not pass")

    raw_target_gradient = P[2] * ADVANTAGES[2] * (1 - P[2])
    if not _close(result.get("raw_target_update_direction_theta", float("nan")), raw_target_gradient):
        errors.append("raw target score gradient mismatch")

    expected_action_sets = {
        "identity_control": (0, 1, 2),
        "top_p_truncation": (0, 1),
    }
    reconstructed = {}
    for case_name, q in Q_CASES.items():
        case = result.get("cases", {}).get(case_name)
        if case is None:
            errors.append(f"missing case: {case_name}")
            continue
        support = expected_action_sets[case_name]
        if tuple(case.get("support", ())) != support:
            errors.append(f"support mismatch: {case_name}")
        expected_retained_mass = sum((P[i] for i in support), Fraction(0))
        if not _close(case.get("retained_raw_mass", float("nan")), expected_retained_mass):
            errors.append(f"retained mass mismatch: {case_name}")
        observed = case.get("per_action", [])
        if [item.get("action") for item in observed] != list(support):
            errors.append(f"action enumeration mismatch: {case_name}")

        expected_ratio_mean = Fraction(0)
        expected_direction = Fraction(0)
        action_audit = []
        for action in support:
            item = observed[support.index(action)] if len(observed) == len(support) else {}
            expected_ratio = P[action] / q[action]
            score = Fraction(int(action == 2)) - P[2]
            expected_update_direction = expected_ratio * ADVANTAGES[action] * score
            expected_loss = -expected_ratio * ADVANTAGES[action]
            expected_kl_metric = float(expected_ratio - 1) - math.log(float(expected_ratio))
            expected_clip_region = float(
                (expected_ratio < Fraction(3, 4) and ADVANTAGES[action] < 0)
                or (expected_ratio > Fraction(5, 4) and ADVANTAGES[action] > 0)
            )
            expected_ratio_mean += q[action] * expected_ratio
            expected_direction += q[action] * expected_update_direction
            for key, value in (
                ("q_probability", q[action]),
                ("advantage", ADVANTAGES[action]),
                ("production_ratio_metric", expected_ratio),
                ("update_direction_theta", expected_update_direction),
                ("loss_gradient_theta", -expected_update_direction),
                ("loss", expected_loss),
                ("production_kl_metric", expected_kl_metric),
                ("production_clip_region_metric", expected_clip_region),
            ):
                if not _close(item.get(key, float("nan")), value):
                    errors.append(f"{case_name} action {action}: {key} mismatch")
            action_audit.append(
                {
                    "action": action,
                    "p": str(P[action]),
                    "q": str(q[action]),
                    "p_over_q": str(expected_ratio),
                    "score_theta": str(score),
                    "expected_update_direction": str(expected_update_direction),
                }
            )

        if not _close(case.get("q_weighted_ratio", float("nan")), expected_ratio_mean):
            errors.append(f"q-weighted ratio mismatch: {case_name}")
        if not _close(case.get("q_weighted_update_direction_theta", float("nan")), expected_direction):
            errors.append(f"q-weighted update direction mismatch: {case_name}")
        reconstructed[case_name] = {
            "q_weighted_ratio": str(expected_ratio_mean),
            "q_weighted_update_direction_theta": str(expected_direction),
            "actions": action_audit,
        }

    expected_predictions = {
        "identity_control": {
            "q_weighted_ratio": Fraction(1),
            "q_weighted_update_direction_theta": raw_target_gradient,
        },
        "top_p_truncation": {
            "q_weighted_ratio": Fraction(4, 5),
            "q_weighted_update_direction_theta": Fraction(0),
        },
    }
    for name, metrics in expected_predictions.items():
        case = result.get("cases", {}).get(name, {})
        for key, expected in metrics.items():
            if not _close(case.get(key, float("nan")), expected):
                errors.append(f"frozen prediction mismatch: {name}/{key}")

    return {
        "protocol_id": lock["protocol_id"],
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "arithmetic": "fractions.Fraction with integer numerators/denominators; no primary-runner imports",
        "raw_target_gradient": str(raw_target_gradient),
        "reconstructed": reconstructed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked = audit(json.loads(args.result.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checked, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checked, indent=2, sort_keys=True))
    return 0 if checked["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
