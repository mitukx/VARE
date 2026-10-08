#!/usr/bin/env python3
"""Independent exact replay for the finite GRPO audit-order characterization."""

from __future__ import annotations

import hashlib
import itertools
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_audit_order_characterization_v1.lock.json"
RUN = ROOT / "results/grpo-audit-order-characterization-v1/run-1/exact-characterization.json"
OUT = ROOT / "results/grpo-audit-order-characterization-v1/run-1/independent-audit.json"
LOCK_SHA256 = "1286ce726d0d574c50dae794345e3df688c165d5cb7117c06880f080027542d1"
VECTORS = tuple(itertools.product((0, 1), repeat=4))
TARGET = (0, 0, 0, 1)
TRIPLE = (0, 1, 2)
EPS = Fraction(1, 2)


def add(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return left[0] + right[0], left[1] + right[1]


def scale(value: tuple[Fraction, Fraction], scalar: Fraction | int) -> tuple[Fraction, Fraction]:
    scalar = Fraction(scalar)
    return value[0] * scalar, value[1] * scalar


def zero() -> tuple[Fraction, Fraction]:
    return Fraction(0), Fraction(0)


def label_weights(y: tuple[int, ...]) -> tuple[tuple[Fraction, Fraction], ...]:
    count = sum(y)
    if count in (0, 4):
        return ((Fraction(0), Fraction(0)),) * 4
    if count == 1:
        good, bad = (Fraction(0), Fraction(1)), (Fraction(0), Fraction(-1, 3))
    elif count == 2:
        good, bad = (Fraction(1), Fraction(0)), (Fraction(-1), Fraction(0))
    else:
        good, bad = (Fraction(0), Fraction(1, 3)), (Fraction(0), Fraction(-1))
    return tuple(good if bit else bad for bit in y)


def score(action: tuple[int, ...], y: tuple[int, ...]) -> tuple[Fraction, Fraction]:
    total = zero()
    for bit, weight in zip(action, label_weights(y)):
        total = add(total, scale(weight, Fraction(4 * bit - 1, 4)))
    return total


def chi(y: tuple[int, ...], subset: tuple[int, ...]) -> int:
    return (-1) ** sum(y[i] for i in subset)


def coeff(action: tuple[int, ...], subset: tuple[int, ...]) -> tuple[Fraction, Fraction]:
    total = zero()
    for y in VECTORS:
        total = add(total, scale(score(action, y), chi(y, subset)))
    return scale(total, Fraction(1, 16))


def pmf(sign: int) -> dict[tuple[int, ...], Fraction]:
    return {y: Fraction(1, 16) * (1 + sign * EPS * chi(y, TRIPLE)) for y in VECTORS}


def project(law: dict[tuple[int, ...], Fraction], subset: tuple[int, ...]) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for y, mass in law.items():
        key = tuple(y[i] for i in subset)
        result[key] = result.get(key, Fraction(0)) + mass
    return result


def tv(left: dict, right: dict) -> Fraction:
    return sum((abs(left.get(k, 0) - right.get(k, 0)) for k in left.keys() | right.keys()), Fraction(0)) / 2


def expected(action: tuple[int, ...], law: dict[tuple[int, ...], Fraction]) -> tuple[Fraction, Fraction]:
    total = zero()
    for y, probability in law.items():
        total = add(total, scale(score(action, y), probability))
    return total


def pair_text(value: tuple[Fraction, Fraction]) -> dict[str, str]:
    return {"rational": str(value[0]), "sqrt3_coefficient": str(value[1])}


def main() -> None:
    lock_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if lock_hash != LOCK_SHA256:
        raise SystemExit("protocol lock hash mismatch")
    record = json.loads(RUN.read_text())
    plus, minus = pmf(+1), pmf(-1)
    degree_by_action = {}
    for action in VECTORS:
        values = [coeff(action, s) for size in range(5) for s in itertools.combinations(range(4), size)]
        active = [size for size in range(5) for s in itertools.combinations(range(4), size)
                  if coeff(action, s) != zero()]
        degree_by_action["".join(map(str, action))] = max(active, default=-1)

    target_coeff = coeff(TARGET, TRIPLE)
    plus_mean = expected(TARGET, plus)
    minus_mean = expected(TARGET, minus)
    conditional_gap = add(plus_mean, scale(minus_mean, -1))
    p_action = Fraction(27, 256)
    population_gap = scale(conditional_gap, p_action)
    low_order_tvs = {
        "".join(map(str, subset)): str(tv(project(plus, subset), project(minus, subset)))
        for size in range(3) for subset in itertools.combinations(range(4), size)
    }
    triple_tv = tv(project(plus, TRIPLE), project(minus, TRIPLE))
    checks = {
        "lock_hash_matches": lock_hash == record["sources"]["protocol_sha256"],
        "actionwise_walsh_degrees_match_runner": degree_by_action == record["walsh_degree_by_action_vector"],
        "fourth_order_coefficients_vanish": all(
            coeff(action, subset) == zero() for action in VECTORS for subset in itertools.combinations(range(4), 4)
        ),
        "all_nonconstant_action_updates_have_degree_three": all(
            degree_by_action["".join(map(str, action))] == 3
            for action in VECTORS if action not in ((0, 0, 0, 0), (1, 1, 1, 1))
        ),
        "target_coefficient_is_exact": pair_text(target_coeff) == record["target"]["alpha_exact"],
        "all_one_and_two_way_marginals_match": all(value == "0" for value in low_order_tvs.values()),
        "conditional_mean_gap_is_exact": pair_text(conditional_gap) == record["target"]["conditional_update_gap_exact"],
        "unconditional_mean_gap_is_exact": pair_text(population_gap) == record["target"]["unconditional_update_gap_exact"],
        "triple_observation_has_positive_tv": str(triple_tv) == record["target"]["selected_order_three_marginal_tv"] and triple_tv > 0,
    }
    if not all(checks.values()):
        raise SystemExit(f"independent audit failed: {checks}")
    result = {
        "protocol_id": record["protocol_id"],
        "auditor": Path(__file__).name,
        "independent_implementation": True,
        "same_host_audit": True,
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": lock_hash,
        "checks": checks,
        "scope": "Independent exact same-host replay of the finite truth table; not external reproduction or a novelty review.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
