#!/usr/bin/env python3
"""Separate direct-enumeration audit of the retained GRPO witness bundle."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/grpo-expected-update-audit-identifiability-v1/run-1"
BITS = tuple(itertools.product((0, 1), repeat=4))
SPECIAL = tuple(map(int, "0001"))
LOCK = ROOT / "protocols/grpo_expected_update_audit_identifiability_v1.lock.json"
EXPECTED_LOCK_SHA256 = "81ffcf079ddfda4eca07c1dc6e32f70159ec70d2dc4fbcf9a7785c07d3e22f25"


def sample_law(a: tuple[int, ...], world: str) -> dict[tuple[int, ...], Fraction]:
    special_A = {
        tuple(map(int, "0000")): Fraction(1, 5),
        tuple(map(int, "0001")): Fraction(1, 5),
        tuple(map(int, "0011")): Fraction(1, 5),
        tuple(map(int, "0101")): Fraction(1, 5),
        tuple(map(int, "1001")): Fraction(1, 5),
    }
    special_B = {
        tuple(map(int, "0001")): Fraction(7, 10),
        tuple(map(int, "0110")): Fraction(1, 10),
        tuple(map(int, "1010")): Fraction(1, 10),
        tuple(map(int, "1101")): Fraction(1, 10),
    }
    if a == SPECIAL:
        return special_A if world == "A" else special_B
    out: dict[tuple[int, ...], Fraction] = {}
    for y in BITS:
        errors = sum(left != right for left, right in zip(a, y))
        out[y] = Fraction(1, 5) ** errors * Fraction(4, 5) ** (4 - errors)
    return out


def standardized_scores(y: tuple[int, ...]) -> tuple[float, ...]:
    center = sum(y) / 4
    variance = sum((value - center) ** 2 for value in y) / 4
    if variance == 0:
        return (0.0,) * 4
    spread = math.sqrt(variance)
    return tuple((value - center) / spread for value in y)


def probabilities(world: str):
    group = {}
    single = {}
    expected_gradient = 0.0
    for a in BITS:
        pa = Fraction(1, 4) ** sum(a) * Fraction(3, 4) ** (4 - sum(a))
        for y, py in sample_law(a, world).items():
            mass = pa * py
            group[a, y] = mass
            s = standardized_scores(y)
            expected_gradient += float(mass) * sum(si * (ai - 0.25) for si, ai in zip(s, a))
            for i in range(4):
                single[a, i, y[i]] = single.get((a, i, y[i]), Fraction()) + mass
    return expected_gradient, single, group


def tv(left: dict, right: dict) -> Fraction:
    return sum((abs(left.get(k, 0) - right.get(k, 0)) for k in left.keys() | right.keys()), Fraction()) / 2


def main() -> None:
    recorded = json.loads((OUT / "exact-enumeration.json").read_text())
    ga, ia, fa = probabilities("A")
    gb, ib, fb = probabilities("B")
    item_tv = tv(ia, ib)
    group_tv = tv(fa, fb)
    marginal_ok = all(
        sum(prob for y, prob in sample_law(a, w).items() if y[i] == a[i]) == Fraction(4, 5)
        for a in BITS
        for i in range(4)
        for w in ("A", "B")
    )
    checks = {
        "frozen_protocol_hash_matches": hashlib.sha256(LOCK.read_bytes()).hexdigest() == EXPECTED_LOCK_SHA256,
        "all_conditional_item_marginals_are_0_8": marginal_ok,
        "one_item_law_is_identical": item_tv == 0,
        "full_group_law_has_positive_tv": group_tv > 0,
        "gradient_matches_exact_bundle_A": abs(ga - recorded["worlds"]["A_expected_group_gradient_decimal"]) < 1e-12,
        "gradient_matches_exact_bundle_B": abs(gb - recorded["worlds"]["B_expected_group_gradient_decimal"]) < 1e-12,
        "item_tv_matches_exact_bundle": float(item_tv) == recorded["audit_laws"]["one_item_full_action_context_tv_decimal"],
        "group_tv_matches_exact_bundle": float(group_tv) == recorded["audit_laws"]["full_group_observation_tv_decimal"],
        "expected_gradient_gap_is_nonzero": abs(gb - ga) > 0,
    }
    if not all(checks.values()):
        raise SystemExit(f"bundle audit failed: {checks}")
    this = Path(__file__)
    evidence = {
        "classification": "same_host_separately_implemented_bundle_audit",
        "independent_external_reproduction": False,
        "world_A_gradient_decimal": ga,
        "world_B_gradient_decimal": gb,
        "world_B_minus_A_gradient_decimal": gb - ga,
        "item_full_context_tv": str(item_tv),
        "full_group_tv": str(group_tv),
        "checks": checks,
        "source_sha256": hashlib.sha256(this.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256((OUT / "exact-enumeration.json").read_bytes()).hexdigest(),
    }
    (OUT / "independent-audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
