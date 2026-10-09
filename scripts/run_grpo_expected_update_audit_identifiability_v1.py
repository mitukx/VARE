#!/usr/bin/env python3
"""Exact finite enumeration for the frozen GRPO audit witness (stdlib only)."""

from __future__ import annotations

import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_expected_update_audit_identifiability_v1.lock.json"
OUT = ROOT / "results/grpo-expected-update-audit-identifiability-v1/run-1"
BITS = tuple(itertools.product((0, 1), repeat=4))
P = Fraction(1, 4)
TARGET = (0, 0, 0, 1)
FROZEN_PROTOCOL_SHA256 = "81ffcf079ddfda4eca07c1dc6e32f70159ec70d2dc4fbcf9a7785c07d3e22f25"


class Quad:
    """Exact a + b*sqrt(3) arithmetic for GRPO's four-member binary rewards."""

    def __init__(self, rational: Fraction = Fraction(0), sqrt3: Fraction = Fraction(0)):
        self.rational = Fraction(rational)
        self.sqrt3 = Fraction(sqrt3)

    def __add__(self, other: Quad) -> Quad:
        return Quad(self.rational + other.rational, self.sqrt3 + other.sqrt3)

    def __mul__(self, scalar: Fraction | int) -> Quad:
        scalar = Fraction(scalar)
        return Quad(self.rational * scalar, self.sqrt3 * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Fraction | int) -> Quad:
        return self * (1 / Fraction(scalar))

    def as_dict(self) -> dict[str, str]:
        return {"rational": str(self.rational), "sqrt3_coefficient": str(self.sqrt3)}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Quad) and self.rational == other.rational and self.sqrt3 == other.sqrt3

    def value(self) -> float:
        return float(self.rational) + float(self.sqrt3) * (3**0.5)


ZERO = Quad()


def advantage(reward: tuple[int, ...]) -> tuple[Quad, ...]:
    """Population-standardized binary advantage; zero for a constant group."""
    positives = sum(reward)
    if positives in (0, 4):
        return (ZERO, ZERO, ZERO, ZERO)
    if positives == 1:
        high, low = Quad(sqrt3=1), Quad(sqrt3=Fraction(-1, 3))
    elif positives == 2:
        high, low = Quad(1), Quad(-1)
    else:
        high, low = Quad(sqrt3=Fraction(1, 3)), Quad(sqrt3=-1)
    return tuple(high if bit else low for bit in reward)


def group_gradient(action: tuple[int, ...], reward: tuple[int, ...]) -> Quad:
    # Sum(A_i)=0, so sum_i A_i(a_i-p) == sum_i A_i*a_i.
    return sum((adv * bit for adv, bit in zip(advantage(reward), action)), ZERO)


def action_probability(action: tuple[int, ...]) -> Fraction:
    return P ** sum(action) * (1 - P) ** (4 - sum(action))


def reward_law(action: tuple[int, ...], world: str) -> dict[tuple[int, ...], Fraction]:
    if action == TARGET:
        if world == "A":
            return {
                (0, 0, 0, 0): Fraction(1, 5),
                (0, 0, 0, 1): Fraction(1, 5),
                (0, 0, 1, 1): Fraction(1, 5),
                (0, 1, 0, 1): Fraction(1, 5),
                (1, 0, 0, 1): Fraction(1, 5),
            }
        return {
            (0, 0, 0, 1): Fraction(7, 10),
            (0, 1, 1, 0): Fraction(1, 10),
            (1, 0, 1, 0): Fraction(1, 10),
            (1, 1, 0, 1): Fraction(1, 10),
        }
    law: dict[tuple[int, ...], Fraction] = {}
    for flips in BITS:
        reward = tuple(a ^ flip for a, flip in zip(action, flips))
        law[reward] = Fraction(1, 5) ** sum(flips) * Fraction(4, 5) ** (4 - sum(flips))
    return law


def summarize_world(world: str) -> tuple[Quad, dict[tuple[tuple[int, ...], int, int], Fraction], dict[tuple[tuple[int, ...], tuple[int, ...]], Fraction]]:
    expected_gradient = ZERO
    item_law: dict[tuple[tuple[int, ...], int, int], Fraction] = {}
    group_law: dict[tuple[tuple[int, ...], tuple[int, ...]], Fraction] = {}
    for action in BITS:
        pa = action_probability(action)
        for reward, py in reward_law(action, world).items():
            joint = pa * py
            expected_gradient += group_gradient(action, reward) * joint
            group_law[(action, reward)] = joint
            for index, label in enumerate(reward):
                key = (action, index, label)
                item_law[key] = item_law.get(key, Fraction(0)) + joint
    return expected_gradient, item_law, group_law


def total_variation(left: dict, right: dict) -> Fraction:
    keys = left.keys() | right.keys()
    return sum((abs(left.get(key, 0) - right.get(key, 0)) for key in keys), Fraction(0)) / 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(PROTOCOL) != FROZEN_PROTOCOL_SHA256:
        raise SystemExit("protocol lock hash differs from the version frozen in this runner")
    grad_a, item_a, group_a = summarize_world("A")
    grad_b, item_b, group_b = summarize_world("B")
    item_tv = total_variation(item_a, item_b)
    group_tv = total_variation(group_a, group_b)
    cond_a = sum((group_gradient(TARGET, reward) * py for reward, py in reward_law(TARGET, "A").items()), ZERO)
    cond_b = sum((group_gradient(TARGET, reward) * py for reward, py in reward_law(TARGET, "B").items()), ZERO)
    expected_gap = grad_b + grad_a * -1
    event_probability = action_probability(TARGET)
    conditional_marginals_ok = all(
        sum((py for reward, py in reward_law(action, world).items() if reward[index] == action[index]), Fraction(0))
        == Fraction(4, 5)
        for world in ("A", "B")
        for action in BITS
        for index in range(4)
    )
    exact_checks = {
        "item_observation_law_identical": item_tv == 0,
        "every_action_conditioned_item_marginal_matches": conditional_marginals_ok,
        "expected_gradient_differs": expected_gap.value() > 0,
        "full_group_law_differs": group_tv > 0,
        "gradient_gap_matches_target_event": expected_gap == (cond_b + cond_a * -1) * event_probability,
        "all_probability_mass_is_one": sum(action_probability(a) for a in BITS) == 1,
    }
    if not all(exact_checks.values()):
        raise SystemExit(f"frozen decision rule failed: {exact_checks}")

    result = {
        "protocol_id": protocol["protocol_id"],
        "classification": "exact_synthetic_identifiability_witness",
        "selection_disclosure": protocol["selection_disclosure"],
        "model_or_training_run": False,
        "policy": {"group_size": 4, "bernoulli_probability": "1/4", "clip_epsilon": 0.2},
        "worlds": {
            "A_expected_group_gradient": grad_a.as_dict(),
            "A_expected_group_gradient_decimal": grad_a.value(),
            "B_expected_group_gradient": grad_b.as_dict(),
            "B_expected_group_gradient_decimal": grad_b.value(),
            "B_minus_A_gradient": expected_gap.as_dict(),
            "B_minus_A_gradient_decimal": expected_gap.value(),
            "target_action_conditional_A": cond_a.as_dict(),
            "target_action_conditional_A_decimal": cond_a.value(),
            "target_action_conditional_B": cond_b.as_dict(),
            "target_action_conditional_B_decimal": cond_b.value(),
        },
        "audit_laws": {
            "one_item_full_action_context_tv": str(item_tv),
            "one_item_full_action_context_tv_decimal": float(item_tv),
            "full_group_observation_tv": str(group_tv),
            "full_group_observation_tv_decimal": float(group_tv),
            "target_action_probability": str(event_probability),
            "target_action_probability_decimal": float(event_probability),
        },
        "exact_checks": exact_checks,
        "sources": {
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": sha256(PROTOCOL),
            "runner": str(Path(__file__).relative_to(ROOT)),
            "runner_sha256": sha256(Path(__file__)),
        },
        "interpretation_limit": "A finite synthetic law proves an existence gap only. It does not estimate how often this occurs with real policy/verifier pairs, demonstrate a trained policy change, or establish task capability.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exact-enumeration.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
