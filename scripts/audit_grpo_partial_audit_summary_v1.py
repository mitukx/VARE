#!/usr/bin/env python3
"""Post-hoc independent checks of reported summaries and confidence intervals."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/grpo-partial-audit-estimator-v1/run-1/exact-enumeration.json"
OUT = ROOT / "results/grpo-partial-audit-estimator-v1/run-1/summary-aggregation-audit.json"


def quad(obj):
    return Fraction(obj["rational"]), Fraction(obj["sqrt3_coefficient"])


def add(x, y):
    return x[0] + y[0], x[1] + y[1]


def mul(x, y):
    return x[0] * y[0] + 3 * x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def value(x):
    return float(x[0]) + float(x[1]) * math.sqrt(3)


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    d = 1 + z * z / count
    c = (p + z * z / (2 * count)) / d
    h = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return [max(0, c - h), min(1, c + h)]


def main():
    record = json.loads(RUN.read_text())
    rows = record["laws"]
    checks = {
        "runner_reported_exact_checks_pass": record["exact_checks_passed"] is True,
        "all_1560_fixed_budget_ratios_recompute": len(rows) == 1560,
        "per_audit_size_counts_match": True,
        "wilson_intervals_recompute": True,
        "policy_weighted_two_label_bound_is_exact": True,
        "all_direction_trials_are_accounted_for": True,
    }
    grouped = {}
    for m in range(1, 5):
        subset = [r for r in rows if r["audit_size"] == m and r["cost_adjusted_mse_ratio_to_full"] is not None]
        ratios = []
        for row in subset:
            fixed = quad(row["fixed_12_label_budget_mse_exact"])
            full = quad(row["full_audit_fixed_budget_mse_exact"])
            ratio = value(fixed) / value(full)
            if abs(ratio - row["cost_adjusted_mse_ratio_to_full"]) > 1e-12:
                checks["all_1560_fixed_budget_ratios_recompute"] = False
            ratios.append(ratio)
        grouped[str(m)] = {
            "defined_count": len(ratios),
            "below_one_count": sum(x < 1 for x in ratios),
            "equal_one_count": sum(x == 1 for x in ratios),
            "above_one_count": sum(x > 1 for x in ratios),
            "median": sorted(ratios)[len(ratios) // 2] if ratios else None,
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
        }
    checks["per_audit_size_counts_match"] = grouped == record["summary"]["per_audit_size"]

    total_trials = 0
    for row in record["finite_budget_direction_check"]:
        by_seed = row["by_seed"]
        failures = sum(item["wrong_or_zero_direction"] for item in by_seed)
        trials = sum(item["repetitions"] for item in by_seed)
        total_trials += trials
        interval = wilson(failures, trials)
        checks["wilson_intervals_recompute"] &= (
            failures == row["wrong_or_zero_direction"]
            and trials == row["repetitions"]
            and abs(failures / trials - row["rate"]) < 1e-15
            and max(abs(a - b) for a, b in zip(interval, row["wilson_95_interval"])) < 1e-12
            and len(by_seed) == 6
        )
    checks["all_direction_trials_are_accounted_for"] &= total_trials == 196608

    witness = record["two_label_nonidentifiability"]
    gap = quad(witness["conditional_update_gap_exact"])
    lower = quad(witness["minimax_absolute_bias_lower_bound_exact"])
    checks["policy_weighted_two_label_bound_is_exact"] &= (
        lower == (gap[0] / 2, gap[1] / 2)
        and lower == (Fraction(-3, 16), Fraction(1, 8))
    )
    action_probability = Fraction(27, 256)
    policy_gap = (gap[0] * action_probability, gap[1] * action_probability)
    policy_lower = (lower[0] * action_probability, lower[1] * action_probability)
    result = {
        "protocol_id": record["protocol_id"],
        "classification": "posthoc_summary_and_interval_integrity_check",
        "same_host_audit": True,
        "checks": checks,
        "recomputed_summary": grouped,
        "direction_trial_total": total_trials,
        "two_label_witness": {
            "policy_weighted_gap_exact": {"rational": str(policy_gap[0]), "sqrt3_coefficient": str(policy_gap[1])},
            "policy_weighted_gap_decimal": value(policy_gap),
            "policy_weighted_minimax_abs_bias_lower_bound_exact": {"rational": str(policy_lower[0]), "sqrt3_coefficient": str(policy_lower[1])},
            "policy_weighted_minimax_abs_bias_lower_bound_decimal": value(policy_lower),
        },
        "input_sha256": hashlib.sha256(RUN.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitation": "This checks arithmetic aggregation of retained outputs; it is a same-host post-hoc check, not external reproduction.",
    }
    if not all(checks.values()):
        raise SystemExit(f"summary audit failed: {checks}")
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
