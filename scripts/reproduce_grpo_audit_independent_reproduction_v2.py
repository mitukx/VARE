#!/usr/bin/env python3
"""Independently enumerate the frozen finite GRPO audit-order witness."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "protocols/grpo_audit_independent_reproduction_v2.lock.json"


@dataclass(frozen=True)
class Exact:
    """An element r+s*sqrt(3), with both coefficients rational."""

    rational: Fraction = Fraction(0)
    radical: Fraction = Fraction(0)

    def __add__(self, other: Exact) -> Exact:
        return Exact(self.rational + other.rational, self.radical + other.radical)

    def __sub__(self, other: Exact) -> Exact:
        return Exact(self.rational - other.rational, self.radical - other.radical)

    def __mul__(self, scalar: Fraction | int) -> Exact:
        factor = Fraction(scalar)
        return Exact(self.rational * factor, self.radical * factor)

    def __rmul__(self, scalar: Fraction | int) -> Exact:
        return self * scalar

    def __truediv__(self, scalar: Fraction | int) -> Exact:
        divisor = Fraction(scalar)
        return Exact(self.rational / divisor, self.radical / divisor)

    def is_zero(self) -> bool:
        return self.rational == 0 and self.radical == 0

    def as_json(self) -> dict[str, str]:
        return {
            "rational": str(self.rational),
            "sqrt3_coefficient": str(self.radical),
        }


ZERO = Exact()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vectors(width: int) -> Iterable[tuple[int, ...]]:
    return itertools.product((0, 1), repeat=width)


def bits_as_text(bits: tuple[int, ...]) -> str:
    return "".join(str(bit) for bit in bits)


def exact_advantage(rewards: tuple[int, ...]) -> tuple[Exact, ...]:
    count = sum(rewards)
    if count == 0 or count == len(rewards):
        return tuple(ZERO for _ in rewards)

    # (y_i - mean(y)) / population_std(y) simplifies to
    # (4*y_i-count)/sqrt(count*(4-count)) for a four-member group.
    if count in (1, 3):
        numerator_by_bit = {0: -count, 1: 4 - count}
        return tuple(
            Exact(radical=Fraction(numerator_by_bit[bit], 3)) for bit in rewards
        )

    if count == 2:
        return tuple(Exact(Fraction(4 * bit - count, 2)) for bit in rewards)

    raise AssertionError("the four-member binary reward count is outside 0..4")


def score_update(actions: tuple[int, ...], rewards: tuple[int, ...]) -> Exact:
    advantage = exact_advantage(rewards)
    total = ZERO
    for action, item_advantage in zip(actions, advantage, strict=True):
        total += item_advantage * Fraction(4 * action - 1, 4)
    return total


def subsets(width: int) -> Iterable[tuple[int, ...]]:
    for size in range(width + 1):
        yield from itertools.combinations(range(width), size)


def character(rewards: tuple[int, ...], coordinates: tuple[int, ...]) -> int:
    return -1 if sum(rewards[index] for index in coordinates) % 2 else 1


def walsh_coefficients(
    actions: tuple[int, ...], reward_space: tuple[tuple[int, ...], ...]
) -> dict[tuple[int, ...], Exact]:
    coefficients: dict[tuple[int, ...], Exact] = {}
    for coordinates in subsets(len(actions)):
        weighted_total = ZERO
        for rewards in reward_space:
            weighted_total += score_update(actions, rewards) * character(
                rewards, coordinates
            )
        coefficients[coordinates] = weighted_total / len(reward_space)
    return coefficients


def marginal_table(
    law: dict[tuple[int, ...], Fraction], coordinates: tuple[int, ...]
) -> dict[str, str]:
    table: dict[str, Fraction] = {}
    for rewards, mass in law.items():
        key = bits_as_text(tuple(rewards[index] for index in coordinates))
        table[key] = table.get(key, Fraction(0)) + mass
    return {key: str(table[key]) for key in sorted(table)}


def total_variation(left: dict[str, str], right: dict[str, str]) -> Fraction:
    return (
        sum(
            (
                abs(Fraction(left.get(key, "0")) - Fraction(right.get(key, "0")))
                for key in set(left) | set(right)
            ),
            Fraction(0),
        )
        / 2
    )


def mean_update(
    law: dict[tuple[int, ...], Fraction],
    actions: tuple[int, ...],
) -> Exact:
    total = ZERO
    for rewards, mass in law.items():
        total += score_update(actions, rewards) * mass
    return total


def law_pair(
    reward_space: tuple[tuple[int, ...], ...],
    coordinates: tuple[int, ...],
    epsilon: Fraction,
) -> tuple[dict[tuple[int, ...], Fraction], dict[tuple[int, ...], Fraction]]:
    left: dict[tuple[int, ...], Fraction] = {}
    right: dict[tuple[int, ...], Fraction] = {}
    for rewards in reward_space:
        parity = Fraction(character(rewards, coordinates))
        base = Fraction(1, len(reward_space))
        left[rewards] = base * (1 + epsilon * parity)
        right[rewards] = base * (1 - epsilon * parity)
    return left, right


def verify_input_hashes(lock: dict[str, object]) -> dict[str, str]:
    sources = lock["source_artifacts"]
    assert isinstance(sources, dict)
    actual: dict[str, str] = {}
    for item in sources.values():
        assert isinstance(item, dict)
        path = ROOT / str(item["path"])
        digest = sha256(path)
        assert digest == item["sha256"], f"source hash mismatch: {path}"
        actual[str(item["path"])] = digest
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock_bytes = LOCK_PATH.read_bytes()
    lock = json.loads(lock_bytes)
    source_hashes = verify_input_hashes(lock)

    action_space = tuple(vectors(4))
    reward_space = tuple(vectors(4))
    degrees: dict[str, int] = {}
    all_coefficients: dict[str, dict[str, dict[str, str]]] = {}
    degree_four_is_zero = True
    nonconstant_actions_have_degree_three = True
    constant_actions_have_zero_function = True
    for actions in action_space:
        coefficients = walsh_coefficients(actions, reward_space)
        support = [
            size
            for term, value in coefficients.items()
            if not value.is_zero()
            for size in (len(term),)
        ]
        degrees[bits_as_text(actions)] = max(support, default=-1)
        degree_four_is_zero &= all(
            value.is_zero() for term, value in coefficients.items() if len(term) == 4
        )
        if actions in ((0, 0, 0, 0), (1, 1, 1, 1)):
            constant_actions_have_zero_function &= all(
                value.is_zero() for value in coefficients.values()
            )
        else:
            nonconstant_actions_have_degree_three &= degrees[bits_as_text(actions)] == 3
        all_coefficients[bits_as_text(actions)] = {
            bits_as_text(term): value.as_json() for term, value in coefficients.items()
        }

    target_action = (0, 0, 0, 1)
    target_term = (0, 1, 2)
    target_coefficients = walsh_coefficients(target_action, reward_space)
    alpha = target_coefficients[target_term]
    epsilon = Fraction(1, 2)
    law_plus, law_minus = law_pair(reward_space, target_term, epsilon)
    normalization_plus = sum(law_plus.values(), Fraction(0))
    normalization_minus = sum(law_minus.values(), Fraction(0))
    minimum_mass = min((*law_plus.values(), *law_minus.values()))

    small_marginal_tvs: dict[str, str] = {}
    for term in subsets(4):
        if len(term) <= 2:
            small_marginal_tvs[bits_as_text(term)] = str(
                total_variation(
                    marginal_table(law_plus, term), marginal_table(law_minus, term)
                )
            )
    target_tv = total_variation(
        marginal_table(law_plus, target_term), marginal_table(law_minus, target_term)
    )
    gap = mean_update(law_plus, target_action) - mean_update(law_minus, target_action)
    action_probability = Fraction(1, 4) ** sum(target_action) * Fraction(3, 4) ** (
        4 - sum(target_action)
    )
    unconditional_gap = gap * action_probability
    expected_conditional_gap = alpha * (2 * epsilon)

    checks = {
        "all_action_vectors_enumerated": len(action_space) == 16,
        "constant_actions_have_zero_update_function": constant_actions_have_zero_function,
        "all_degree_four_coefficients_vanish": degree_four_is_zero,
        "every_nonconstant_action_has_degree_three": nonconstant_actions_have_degree_three,
        "target_coefficient_exact": alpha == Exact(Fraction(-3, 8), Fraction(1, 4)),
        "pair_laws_are_normalized": normalization_plus == normalization_minus == 1,
        "pair_laws_are_nonnegative": minimum_mass >= 0,
        "all_marginals_through_order_two_match": all(
            Fraction(value) == 0 for value in small_marginal_tvs.values()
        ),
        "selected_order_three_tv_is_one_half": target_tv == Fraction(1, 2),
        "conditional_mean_gap_matches_exact_coefficient": gap
        == expected_conditional_gap,
        "action_probability_is_27_over_256": action_probability == Fraction(27, 256),
        "unconditional_mean_gap_exact": unconditional_gap
        == Exact(Fraction(-81, 2048), Fraction(27, 1024)),
    }
    assert all(checks.values()), json.dumps(checks, sort_keys=True)

    runner_hash = sha256(Path(__file__).resolve())
    result = {
        "protocol_id": lock["protocol_id"],
        "classification": lock["classification"],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": "Python standard library only",
            "model_or_training_run": False,
        },
        "enumeration": {
            "action_vectors": len(action_space),
            "reward_vectors_per_action": len(reward_space),
            "walsh_degree_by_action_vector": degrees,
            "target_action_all_coefficients": all_coefficients["0001"],
        },
        "target_pair": {
            "action_vector": "0001",
            "character_subset": list(target_term),
            "epsilon": str(epsilon),
            "alpha_exact": alpha.as_json(),
            "probability_of_action": str(action_probability),
            "conditional_update_gap_exact": gap.as_json(),
            "unconditional_update_gap_exact": unconditional_gap.as_json(),
            "minimum_probability_mass": str(minimum_mass),
            "normalization_plus": str(normalization_plus),
            "normalization_minus": str(normalization_minus),
            "marginal_tv_by_subset_order_le_two": small_marginal_tvs,
            "selected_order_three_marginal_tv": str(target_tv),
        },
        "checks": checks,
        "sources": {
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "runner_sha256": runner_hash,
            "input_artifact_sha256": source_hashes,
        },
        "interpretation_limit": (
            "Same-host exact reproduction of one finite four-member binary-reward local "
            "GRPO score update; no external replication, novelty finding, audit-cost "
            "result, model learning, task success, or capability claim."
        ),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite retained output: {args.output}")
    args.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
