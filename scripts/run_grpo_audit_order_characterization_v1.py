#!/usr/bin/env python3
"""Exact Walsh-degree characterization for one finite clipped-GRPO update."""

from __future__ import annotations

import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_audit_order_characterization_v1.lock.json"
OUT = ROOT / "results/grpo-audit-order-characterization-v1/run-1"
FROZEN_PROTOCOL_SHA256 = "1286ce726d0d574c50dae794345e3df688c165d5cb7117c06880f080027542d1"
BITS = tuple(itertools.product((0, 1), repeat=4))
TARGET_ACTION = (0, 0, 0, 1)
TARGET_SET = (0, 1, 2)
EPSILON = Fraction(1, 2)


class Quad:
    """Exact a + b*sqrt(3) arithmetic used by the finite GRPO estimator."""

    def __init__(self, rational: Fraction = Fraction(0), sqrt3: Fraction = Fraction(0)):
        self.rational = Fraction(rational)
        self.sqrt3 = Fraction(sqrt3)

    def __add__(self, other: Quad) -> Quad:
        return Quad(self.rational + other.rational, self.sqrt3 + other.sqrt3)

    def __sub__(self, other: Quad) -> Quad:
        return self + (-other)

    def __mul__(self, scalar: Fraction | int) -> Quad:
        scalar = Fraction(scalar)
        return Quad(self.rational * scalar, self.sqrt3 * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Fraction | int) -> Quad:
        return self * (1 / Fraction(scalar))

    def __neg__(self) -> Quad:
        return self * -1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Quad) and self.rational == other.rational and self.sqrt3 == other.sqrt3

    def as_dict(self) -> dict[str, str]:
        return {"rational": str(self.rational), "sqrt3_coefficient": str(self.sqrt3)}

    def value(self) -> float:
        return float(self.rational) + float(self.sqrt3) * (3**0.5)


ZERO = Quad()


def advantage(reward: tuple[int, ...]) -> tuple[Quad, ...]:
    positives = sum(reward)
    if positives in (0, 4):
        return (ZERO,) * 4
    if positives == 1:
        high, low = Quad(sqrt3=1), Quad(sqrt3=Fraction(-1, 3))
    elif positives == 2:
        high, low = Quad(1), Quad(-1)
    else:
        high, low = Quad(sqrt3=Fraction(1, 3)), Quad(sqrt3=-1)
    return tuple(high if bit else low for bit in reward)


def update(action: tuple[int, ...], reward: tuple[int, ...]) -> Quad:
    """Local score gradient at ratio one; sum_i A_i(a_i-p), p=1/4."""
    return sum((adv * (Fraction(bit) - Fraction(1, 4))
                for adv, bit in zip(advantage(reward), action)), ZERO)


def character(reward: tuple[int, ...], subset: tuple[int, ...]) -> int:
    return -1 if sum(reward[index] for index in subset) % 2 else 1


def coefficients(action: tuple[int, ...]) -> dict[tuple[int, ...], Quad]:
    values = {reward: update(action, reward) for reward in BITS}
    result = {}
    for size in range(5):
        for subset in itertools.combinations(range(4), size):
            result[subset] = sum(
                (values[reward] * character(reward, subset) for reward in BITS), ZERO
            ) / 16
    return result


def reward_law(sign: int) -> dict[tuple[int, ...], Fraction]:
    """P_sign(y)=uniform(y)*(1+sign*epsilon*chi_TARGET_SET(y))."""
    return {
        reward: Fraction(1, 16) * (1 + sign * EPSILON * character(reward, TARGET_SET))
        for reward in BITS
    }


def marginal(law: dict[tuple[int, ...], Fraction], subset: tuple[int, ...]) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for reward, probability in law.items():
        key = tuple(reward[index] for index in subset)
        result[key] = result.get(key, Fraction(0)) + probability
    return result


def total_variation(left: dict, right: dict) -> Fraction:
    return sum((abs(left.get(key, 0) - right.get(key, 0)) for key in left.keys() | right.keys()), Fraction(0)) / 2


def action_probability(action: tuple[int, ...]) -> Fraction:
    return Fraction(1, 4) ** sum(action) * Fraction(3, 4) ** (4 - sum(action))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    if FROZEN_PROTOCOL_SHA256 and sha256(PROTOCOL) != FROZEN_PROTOCOL_SHA256:
        raise SystemExit("protocol lock hash differs from the version frozen in this runner")

    per_action = {action: coefficients(action) for action in BITS}
    active_degrees = {
        "".join(map(str, action)): max((len(subset) for subset, coefficient in coeffs.items()
                                        if coefficient != ZERO), default=-1)
        for action, coeffs in per_action.items()
    }
    alpha = per_action[TARGET_ACTION][TARGET_SET]
    plus = reward_law(+1)
    minus = reward_law(-1)
    marginal_differences = {}
    for size in range(3):
        for subset in itertools.combinations(range(4), size):
            marginal_differences["".join(map(str, subset))] = str(
                total_variation(marginal(plus, subset), marginal(minus, subset))
            )
    conditional_gap = sum((update(TARGET_ACTION, reward) * probability
                           for reward, probability in plus.items()), ZERO) - sum(
        (update(TARGET_ACTION, reward) * probability for reward, probability in minus.items()), ZERO
    )
    unconditional_gap = conditional_gap * action_probability(TARGET_ACTION)
    subset3_tv = total_variation(marginal(plus, TARGET_SET), marginal(minus, TARGET_SET))
    exact_checks = {
        "all_degree_four_coefficients_are_zero": all(
            per_action[action][subset] == ZERO
            for action in BITS for subset in itertools.combinations(range(4), 4)
        ),
        "all_constant_action_updates_are_zero": all(
            active_degrees["".join(map(str, action))] == -1
            for action in ((0, 0, 0, 0), (1, 1, 1, 1))
        ),
        "all_nonconstant_action_vectors_have_degree_three": all(
            active_degrees["".join(map(str, action))] == 3
            for action in BITS if action not in ((0, 0, 0, 0), (1, 1, 1, 1))
        ),
        "target_degree_three_coefficient_matches_exact_value": alpha == Quad(Fraction(-3, 8), Fraction(1, 4)),
        "laws_are_nonnegative_and_normalized": min(plus.values()) >= 0 and min(minus.values()) >= 0 and sum(plus.values()) == sum(minus.values()) == 1,
        "all_marginals_up_to_order_two_are_identical": all(value == "0" for value in marginal_differences.values()),
        "conditional_gap_matches_character_coefficient": conditional_gap == alpha * (2 * EPSILON),
        "unconditional_gap_matches_action_probability": unconditional_gap == conditional_gap * action_probability(TARGET_ACTION),
        "order_three_observation_distinguishes_worlds": subset3_tv > 0,
    }
    if not all(exact_checks.values()):
        raise SystemExit(f"frozen exact decision rule failed: {exact_checks}")

    result = {
        "protocol_id": protocol["protocol_id"],
        "classification": "exact_finite_identifiability_characterization",
        "model_or_training_run": False,
        "selection_disclosure": protocol["selection_disclosure"],
        "theorem": protocol["theorem_to_verify"],
        "walsh_degree_by_action_vector": active_degrees,
        "target": {
            "action_vector": "".join(map(str, TARGET_ACTION)),
            "character_subset": list(TARGET_SET),
            "epsilon": str(EPSILON),
            "alpha_exact": alpha.as_dict(),
            "alpha_decimal": alpha.value(),
            "conditional_update_gap_exact": conditional_gap.as_dict(),
            "conditional_update_gap_decimal": conditional_gap.value(),
            "action_probability": str(action_probability(TARGET_ACTION)),
            "unconditional_update_gap_exact": unconditional_gap.as_dict(),
            "unconditional_update_gap_decimal": unconditional_gap.value(),
            "order_at_most_two_marginal_tv": marginal_differences,
            "selected_order_three_marginal_tv": str(subset3_tv),
        },
        "exact_checks": exact_checks,
        "sources": {
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": sha256(PROTOCOL),
            "runner": str(Path(__file__).relative_to(ROOT)),
            "runner_sha256": sha256(Path(__file__)),
        },
        "interpretation_limit": "This exact result applies to one finite four-member binary-reward local clipped-GRPO estimand. It does not establish prevalence, audit cost effectiveness, finite-sample performance, model capability, or novelty.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exact-characterization.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
