#!/usr/bin/env python3
"""Independent exact audit of the policy-shift GRPO sign-risk analysis."""

from __future__ import annotations

import hashlib
import itertools
import json
from decimal import Decimal, getcontext
from fractions import Fraction
from math import comb, factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_policy_shift_audit_risk_v1.lock.json"
RAW = ROOT / "results/grpo-policy-shift-audit-risk-v1/run-1/raw.json"
OUT = ROOT / "results/grpo-policy-shift-audit-risk-v1/run-1/independent_audit.json"
FROZEN_PROTOCOL_SHA256 = "470a1f0cda6d5e7a5e46a4bdbe9bb09d718ab583c2c4bf4c34b98a0766c6496a"
N_LABELS = 144
EPSILON = Fraction(1, 2)
P_BEHAVIOR = Fraction(1, 4)
P_TARGET = Fraction(3, 4)
ACTION = (1, 1, 1, 0)
T = (0, 1, 2)


def centered_population_advantage(y: tuple[int, ...]) -> tuple[Decimal, ...]:
    total = sum(y)
    if total == 0 or total == 4:
        return (Decimal(0),) * 4
    mu = Decimal(total) / 4
    variance = mu * (1 - mu)
    scale = Decimal(1) / variance.sqrt()
    return tuple((Decimal(bit) - mu) * scale for bit in y)


def independent_truth_table_coefficient() -> Decimal:
    accumulated = Decimal(0)
    vectors = itertools.product((0, 1), repeat=4)
    for y in vectors:
        advantage = centered_population_advantage(y)
        f = sum((Decimal(a) * adv for a, adv in zip(ACTION, advantage)), Decimal(0))
        parity = -1 if sum(y[i] for i in T) % 2 else 1
        accumulated += f * parity
    return accumulated / 16


def mass(action: tuple[int, ...], p: Fraction) -> Fraction:
    return p ** sum(action) * (1 - p) ** (4 - sum(action))


def multinomial_tv(n: int, r: Fraction) -> Fraction:
    """Directly sum the plus/minus/missing multinomial transcript laws."""
    p = (1 + EPSILON) / 2
    q = (1 - EPSILON) / 2
    informative_plus = r * p
    informative_minus = r * q
    silent = 1 - r
    total_variation = Fraction(0)
    for plus_count in range(n + 1):
        for minus_count in range(n - plus_count + 1):
            silent_count = n - plus_count - minus_count
            multiplicity = Fraction(
                factorial(n),
                factorial(plus_count) * factorial(minus_count) * factorial(silent_count),
            )
            under_plus = (
                multiplicity
                * informative_plus**plus_count
                * informative_minus**minus_count
                * silent**silent_count
            )
            under_minus = (
                multiplicity
                * informative_minus**plus_count
                * informative_plus**minus_count
                * silent**silent_count
            )
            total_variation += abs(under_plus - under_minus) / 2
    return total_variation


def main() -> None:
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if protocol_hash != FROZEN_PROTOCOL_SHA256:
        raise SystemExit("protocol hash mismatch")
    raw = json.loads(RAW.read_text())
    if raw["protocol_sha256"] != protocol_hash:
        raise SystemExit("runner output is not bound to the frozen protocol")

    getcontext().prec = 60
    alpha = independent_truth_table_coefficient()
    expected_alpha = Decimal(3) / 8 - Decimal(3).sqrt() / 4
    if abs(alpha - expected_alpha) > Decimal("1e-50"):
        raise SystemExit(f"truth-table coefficient mismatch: {alpha} vs {expected_alpha}")

    mu = mass(ACTION, P_BEHAVIOR)
    pi = mass(ACTION, P_TARGET)
    rates = {
        "item": (144, Fraction(0)),
        "pair": (72, Fraction(0)),
        "behavior_full": (36, mu),
        "behavior_uniform_triple": (48, mu / 4),
        "behavior_target_stratum_triple_oracle": (48, Fraction(1)),
        "fresh_target_uniform_triple": (48, pi / 4),
        "fresh_target_stratum_triple_oracle": (48, Fraction(1)),
    }
    audited = {}
    for name, (n, event_rate) in rates.items():
        estimate = multinomial_tv(n, event_rate)
        reported = raw["designs"][[r["design"] for r in raw["designs"]].index(name)]
        if reported["transcript_total_variation_exact"] != str(estimate):
            # Equivalent rational values can have different numerator/denominator forms;
            # compare exact fractions after parsing both strings.
            if Fraction(reported["transcript_total_variation_exact"]) != estimate:
                raise SystemExit(f"total-variation mismatch for {name}")
        risk = (1 - estimate) / 2
        if abs(float(risk) - reported["optimal_minimax_wrong_sign_probability"]) > 1e-14:
            raise SystemExit(f"sign-risk mismatch for {name}")
        audited[name] = {
            "n_groups": n,
            "informative_record_probability": str(event_rate),
            "transcript_total_variation": float(estimate),
            "optimal_minimax_wrong_sign_probability": float(risk),
        }

    coefficient_sq = alpha * alpha
    target_mass_decimal = Decimal(pi.numerator) / Decimal(pi.denominator)
    signal_sq = target_mass_decimal**2 * coefficient_sq * Decimal("0.25")
    expected_mse = {
        "item": signal_sq,
        "pair": signal_sq,
        "behavior_full": (
            (Decimal(pi.numerator) / Decimal(pi.denominator)) ** 2
            * coefficient_sq
            / (Decimal(mu.numerator) / Decimal(mu.denominator))
            - signal_sq
        ) / 36,
        "behavior_uniform_triple": (
            (Decimal(pi.numerator) / Decimal(pi.denominator)) ** 2
            * coefficient_sq
            / (Decimal(mu.numerator) / Decimal(mu.denominator) / 4)
            - signal_sq
        ) / 48,
        "behavior_target_stratum_triple_oracle": (
            (Decimal(pi.numerator) / Decimal(pi.denominator)) ** 2
            * coefficient_sq
            - signal_sq
        ) / 48,
        "fresh_target_uniform_triple": (
            coefficient_sq
            * target_mass_decimal
            / Decimal("0.25")
            - signal_sq
        ) / 48,
        "fresh_target_stratum_triple_oracle": (
            (Decimal(pi.numerator) / Decimal(pi.denominator)) ** 2
            * coefficient_sq
            - signal_sq
        ) / 48,
    }
    for design, value in expected_mse.items():
        reported = raw["designs"][[r["design"] for r in raw["designs"]].index(design)]
        if abs(float(value) - reported["active_component_estimator_mse"]) > 1e-14:
            raise SystemExit(f"MSE mismatch for {design}: {value}")

    result = {
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "checks": {
            "independent_decimal_truth_table_coefficient_matches": str(expected_alpha),
            "behavior_action_mass": str(mu),
            "target_action_mass": str(pi),
            "target_to_behavior_ratio": str(pi / mu),
            "target_sign_magnitudes_equal": str(abs(alpha) * Decimal(pi.numerator) / Decimal(pi.denominator) / 2),
            "transcript_tv_and_minimax_sign_risk": audited,
            "active_component_mse": {name: float(value) for name, value in expected_mse.items()},
        },
        "independence": "Truth-table coefficient uses Decimal direct enumeration; transcript TV uses an explicit multinomial sum, separate from the runner's binomial mixture decomposition.",
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
