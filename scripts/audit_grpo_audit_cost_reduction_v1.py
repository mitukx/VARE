#!/usr/bin/env python3
"""Independent multinomial audit of the cost-reduction sensitivity bundle."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, getcontext
from fractions import Fraction
from math import factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/grpo_audit_cost_reduction_v1.lock.json"
RAW = ROOT / "results/grpo-audit-cost-reduction-v1/run-1/raw.json"
OUT = ROOT / "results/grpo-audit-cost-reduction-v1/run-1/independent_audit.json"
LOCK_SHA256 = "14b57024783e3e560c0ee3ca5f9d9215c61c3c7737290740b44cd6704d7e13c2"
EPS = Fraction(1, 2)
PI = Fraction(27, 256)


def direct_tv(n: int, r: Fraction) -> Fraction:
    """Enumerate positive-parity, negative-parity, and silent transcript counts."""
    plus = r * (1 + EPS) / 2
    minus = r * (1 - EPS) / 2
    silent = 1 - r
    total = Fraction(0)
    for p in range(n + 1):
        for m in range(n - p + 1):
            z = n - p - m
            multiplicity = Fraction(factorial(n), factorial(p) * factorial(m) * factorial(z))
            world_plus = multiplicity * plus**p * minus**m * silent**z
            world_minus = multiplicity * minus**p * plus**m * silent**z
            total += abs(world_plus - world_minus) / 2
    return total


def main() -> None:
    digest = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    if digest != LOCK_SHA256:
        raise SystemExit("protocol hash mismatch")
    raw = json.loads(RAW.read_text())
    if raw["protocol_sha256"] != digest:
        raise SystemExit("raw result is not bound to the frozen lock")
    getcontext().prec = 60
    alpha = Decimal(3) / 8 - Decimal(3).sqrt() / 4
    alpha2 = alpha * alpha
    theta2 = (Decimal(PI.numerator) / Decimal(PI.denominator) * alpha / 2) ** 2
    checked = 0
    max_risk_delta = 0.0
    max_mse_delta = 0.0
    for row in raw["matched_total_cost_designs"]:
        n = row["audited_groups"]
        rate = Fraction(row["informative_rate_per_audited_group"])
        risk = float((1 - direct_tv(n, rate)) / 2) if n else 0.5
        max_risk_delta = max(max_risk_delta, abs(risk - row["exact_optimal_sign_error"]))
        if max_risk_delta > 1e-14:
            raise SystemExit(f"risk mismatch for {row['design']}")
        if n:
            mu = Fraction(row["behavior_mass"])
            inclusion = Fraction(rate if "stratified" in row["design"] else
                                 (Fraction(1) if row["labels_per_audited_group"] == 4 else Fraction(1, 4)))
            # Target non-stratified sampling has an action ratio in its second moment;
            # behavior sampling has pi^2/(mu*e); stratified sampling has pi^2/e.
            if row["design"] == "target_uniform_triple":
                second = Decimal(PI.numerator) / Decimal(PI.denominator) * alpha2 / Decimal(inclusion.numerator) * Decimal(inclusion.denominator)
            elif row["design"].startswith("target_") and not row["design"].startswith("target_action_stratified"):
                second = Decimal(PI.numerator) / Decimal(PI.denominator) * alpha2
            elif "stratified" in row["design"]:
                second = (Decimal(PI.numerator) / Decimal(PI.denominator)) ** 2 * alpha2 / (Decimal(inclusion.numerator) / Decimal(inclusion.denominator))
            else:
                second = (Decimal(PI.numerator) / Decimal(PI.denominator)) ** 2 * alpha2 / (Decimal(mu.numerator) / Decimal(mu.denominator)) / (Decimal(inclusion.numerator) / Decimal(inclusion.denominator))
            expected_mse = float((second - theta2) / n)
            actual_mse = row["active_component_ht_mse_stationary"]
            max_mse_delta = max(max_mse_delta, abs(expected_mse - actual_mse))
            if max_mse_delta > 2e-14:
                raise SystemExit(f"MSE mismatch for {row['design']}: {expected_mse} != {actual_mse}")
        checked += 1

    for row in raw["conditional_shift_sensitivity"]["rows"]:
        expected_bias = float(PI) * float(alpha) * (row["behavior_parity_mean"] - row["target_parity_mean"])
        if abs(expected_bias - row["transport_bias"]) > 1e-15:
            raise SystemExit("conditional transport bias mismatch")

    result = {
        "status": "PASS",
        "protocol_sha256": digest,
        "checks": {
            "matched_total_cost_rows": checked,
            "max_exact_sign_risk_difference": max_risk_delta,
            "max_independent_ht_mse_difference": max_mse_delta,
            "decimal_walsh_coefficient": str(alpha),
            "transport_bias_identity": "verified for all frozen delta rows",
            "independence": "Direct positive/negative/silent multinomial sum; does not import or call the runner TV routine.",
        },
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
