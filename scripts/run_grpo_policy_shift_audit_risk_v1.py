#!/usr/bin/env python3
"""Exact finite-budget sign-risk analysis for partial GRPO group audits."""

from __future__ import annotations

import hashlib
import itertools
import json
from fractions import Fraction
from math import comb, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_policy_shift_audit_risk_v1.lock.json"
OUT = ROOT / "results/grpo-policy-shift-audit-risk-v1/run-1"
FROZEN_PROTOCOL_SHA256 = "470a1f0cda6d5e7a5e46a4bdbe9bb09d718ab583c2c4bf4c34b98a0766c6496a"
BITS = tuple(itertools.product((0, 1), repeat=4))
TARGET_ACTION = (1, 1, 1, 0)
ACTIVE_SET = (0, 1, 2)
EPSILON = Fraction(1, 2)
BEHAVIOR_P = Fraction(1, 4)
TARGET_P = Fraction(3, 4)
BUDGET = 144


class Surd:
    """Exact a + b*sqrt(3) arithmetic for the reward standardization."""

    def __init__(self, rational: Fraction = Fraction(0), radical: Fraction = Fraction(0)):
        self.rational = Fraction(rational)
        self.radical = Fraction(radical)

    def __add__(self, other: Surd) -> Surd:
        return Surd(self.rational + other.rational, self.radical + other.radical)

    def __mul__(self, other: Surd | Fraction | int) -> Surd:
        if isinstance(other, Surd):
            return Surd(
                self.rational * other.rational + 3 * self.radical * other.radical,
                self.rational * other.radical + self.radical * other.rational,
            )
        scalar = Fraction(other)
        return Surd(self.rational * scalar, self.radical * scalar)

    __rmul__ = __mul__

    def __truediv__(self, denominator: Fraction | int) -> Surd:
        return self * (1 / Fraction(denominator))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Surd)
            and self.rational == other.rational
            and self.radical == other.radical
        )

    def as_dict(self) -> dict[str, str]:
        return {"rational": str(self.rational), "sqrt3_coefficient": str(self.radical)}

    def value(self) -> float:
        return float(self.rational) + float(self.radical) * sqrt(3)


ZERO = Surd()


def advantage(reward: tuple[int, ...]) -> tuple[Surd, ...]:
    count = sum(reward)
    if count in (0, 4):
        return (ZERO,) * 4
    if count == 1:
        high, low = Surd(radical=1), Surd(radical=Fraction(-1, 3))
    elif count == 2:
        high, low = Surd(1), Surd(-1)
    else:
        high, low = Surd(radical=Fraction(1, 3)), Surd(radical=-1)
    return tuple(high if bit else low for bit in reward)


def update(action: tuple[int, ...], reward: tuple[int, ...]) -> Surd:
    # Sum_i A_i(a_i-p) is independent of p because sum_i A_i is exactly zero.
    return sum((Fraction(a) * score for a, score in zip(action, advantage(reward))), ZERO)


def character(reward: tuple[int, ...], subset: tuple[int, ...]) -> int:
    return -1 if sum(reward[index] for index in subset) % 2 else 1


def coefficient(action: tuple[int, ...], subset: tuple[int, ...]) -> Surd:
    total = sum(
        (update(action, reward) * character(reward, subset) for reward in BITS), ZERO
    )
    return total / 16


def action_probability(action: tuple[int, ...], p: Fraction) -> Fraction:
    return p ** sum(action) * (1 - p) ** (4 - sum(action))


def binomial_tv(n: int, informative_rate: Fraction, epsilon: Fraction) -> Fraction:
    """TV of n records where an informative record reveals noisy parity."""
    p_plus = (1 + epsilon) / 2
    p_minus = (1 - epsilon) / 2
    result = Fraction(0)
    for informative_count in range(n + 1):
        count_probability = (
            Fraction(comb(n, informative_count))
            * informative_rate**informative_count
            * (1 - informative_rate) ** (n - informative_count)
        )
        conditional_tv = Fraction(0)
        for plus_count in range(informative_count + 1):
            plus_law = (
                Fraction(comb(informative_count, plus_count))
                * p_plus**plus_count
                * p_minus ** (informative_count - plus_count)
            )
            minus_law = (
                Fraction(comb(informative_count, plus_count))
                * p_minus**plus_count
                * p_plus ** (informative_count - plus_count)
            )
            conditional_tv += abs(plus_law - minus_law) / 2
        result += count_probability * conditional_tv
    return result


def estimator_mse(
    *,
    n: int,
    alpha: float,
    target_action_mass: Fraction,
    behavior_action_mass: Fraction | None,
    audit_inclusion: Fraction,
    source_is_target: bool = False,
    conditional_action_stratum: bool = False,
    zero_estimator: bool = False,
) -> float:
    signal = float(target_action_mass) * alpha * float(EPSILON)
    if zero_estimator:
        return signal**2
    if conditional_action_stratum:
        second_moment = float(target_action_mass) ** 2 * alpha**2
    elif source_is_target:
        second_moment = alpha**2 * float(target_action_mass) / float(audit_inclusion)
    else:
        assert behavior_action_mass is not None
        second_moment = (
            float(target_action_mass) ** 2
            * alpha**2
            / (float(behavior_action_mass) * float(audit_inclusion))
        )
    variance_one = second_moment - signal**2
    return variance_one / n


def main() -> None:
    protocol_bytes = PROTOCOL.read_bytes()
    digest = hashlib.sha256(protocol_bytes).hexdigest()
    if FROZEN_PROTOCOL_SHA256 and digest != FROZEN_PROTOCOL_SHA256:
        raise SystemExit("protocol lock hash differs from the version frozen in this runner")

    alpha_exact = coefficient(TARGET_ACTION, ACTIVE_SET)
    expected_alpha = Surd(Fraction(3, 8), Fraction(-1, 4))
    if alpha_exact != expected_alpha:
        raise SystemExit(f"unexpected Walsh coefficient: {alpha_exact.as_dict()!r}")
    alpha = alpha_exact.value()

    behavior_mass = action_probability(TARGET_ACTION, BEHAVIOR_P)
    target_mass = action_probability(TARGET_ACTION, TARGET_P)
    designs = [
        ("item", BUDGET, Fraction(0), "behavior", True, False, BUDGET),
        ("pair", BUDGET // 2, Fraction(0), "behavior", True, False, BUDGET // 2),
        ("behavior_full", BUDGET // 4, Fraction(1), "behavior", False, False, BUDGET // 4),
        ("behavior_uniform_triple", BUDGET // 3, Fraction(1, 4), "behavior", False, False, BUDGET // 3),
        ("behavior_target_stratum_triple_oracle", BUDGET // 3, Fraction(1), "behavior_stratum", False, False, BUDGET // 3),
        ("fresh_target_uniform_triple", BUDGET // 3, Fraction(1, 4), "target", False, True, BUDGET // 3),
        ("fresh_target_stratum_triple_oracle", BUDGET // 3, Fraction(1), "target_stratum", False, True, BUDGET // 3),
    ]

    rows = []
    for name, n, inclusion, source, zero, target_source, rollout_groups in designs:
        if name in ("item", "pair"):
            info_rate = Fraction(0)
        elif source == "behavior":
            info_rate = behavior_mass * inclusion
        elif source == "behavior_stratum":
            info_rate = Fraction(1)
        elif source == "target":
            info_rate = target_mass * inclusion
        else:
            info_rate = Fraction(1)
        tv = binomial_tv(n, info_rate, EPSILON)
        if source == "behavior_stratum":
            expected_source_groups = Fraction(n, 1) / behavior_mass
        elif source == "target_stratum":
            expected_source_groups = Fraction(n, 1) / target_mass
        else:
            expected_source_groups = Fraction(rollout_groups)
        mse = estimator_mse(
            n=n,
            alpha=alpha,
            target_action_mass=target_mass,
            behavior_action_mass=behavior_mass,
            audit_inclusion=inclusion or Fraction(1),
            source_is_target=target_source,
            conditional_action_stratum="stratum" in source,
            zero_estimator=zero,
        )
        rows.append(
            {
                "design": name,
                "clean_label_calls": BUDGET,
                "groups_in_estimator": n,
                "informative_record_probability": str(info_rate),
                "transcript_total_variation_exact": str(tv),
                "transcript_total_variation": float(tv),
                "optimal_minimax_wrong_sign_probability": float((1 - tv) / 2),
                "active_component_estimator_mse": mse,
                "expected_source_groups": float(expected_source_groups),
                "target_action_observation_mass": str(target_mass),
                "behavior_action_observation_mass": str(behavior_mass),
                "assumption_or_status": (
                    "oracle allocation; conditions on known adversarial action and subset"
                    if "oracle" in name
                    else "ordinary design"
                ),
            }
        )

    signal = float(target_mass) * alpha * float(EPSILON)
    output = {
        "protocol_sha256": digest,
        "estimand": {
            "active_walsh_coefficient_exact": alpha_exact.as_dict(),
            "active_walsh_coefficient": alpha,
            "epsilon": str(EPSILON),
            "target_update_mean_world_plus": signal,
            "target_update_mean_world_minus": -signal,
            "absolute_target_update_mean": abs(signal),
            "behavior_action_mass": str(behavior_mass),
            "target_action_mass": str(target_mass),
            "target_to_behavior_action_ratio": str(target_mass / behavior_mass),
        },
        "designs": rows,
        "limitations": [
            "Exact two-world synthetic analysis; no model, optimizer, or real verifier.",
            "The MSE is for the active Walsh-component estimator under the frozen parity pair.",
            "Oracle-stratum designs know the action and subset carrying the hidden signal.",
            "Action importance weighting assumes P(Y|A) is stationary across policies.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
