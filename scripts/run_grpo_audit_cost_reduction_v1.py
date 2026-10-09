#!/usr/bin/env python3
"""Exact matched-total-cost audit design sensitivity; standard library only."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/grpo_audit_cost_reduction_v1.lock.json"
OUT = ROOT / "results/grpo-audit-cost-reduction-v1/run-1/raw.json"
LOCK_SHA256 = "14b57024783e3e560c0ee3ca5f9d9215c61c3c7737290740b44cd6704d7e13c2"
BUDGET = 144
PI = Fraction(27, 256)
OVERLAPS = (Fraction(1), Fraction(3), Fraction(9))
ROLLOUT_COSTS = (0, 1)
EPS = Fraction(1, 2)
ALPHA = 3 / 8 - 3**0.5 / 4


def tv(n: int, rate: Fraction) -> Fraction:
    """Exact TV for n draws with silent outcomes and a biased parity bit."""
    plus, minus = (1 + EPS) / 2, (1 - EPS) / 2
    total = Fraction(0)
    for k in range(n + 1):
        pk = comb(n, k) * rate**k * (1 - rate) ** (n - k)
        conditional = Fraction(0)
        for j in range(k + 1):
            lp = comb(k, j) * plus**j * minus ** (k - j)
            lm = comb(k, j) * minus**j * plus ** (k - j)
            conditional += abs(lp - lm) / 2
        total += pk * conditional
    return total


def design_row(name: str, *, mass: Fraction, behavior_mass: Fraction,
               inclusion: Fraction, labels: int,
               rollout_ratio: int, budget: int, stratified: bool,
               oracle_subset: bool = False) -> dict[str, object]:
    cost_per_audited = (Fraction(rollout_ratio, 1) / mass if stratified else Fraction(rollout_ratio)) + labels
    n = int(Fraction(budget, 1) // cost_per_audited)
    event = inclusion if stratified else mass * inclusion
    if name == "target_uniform_triple":
        second_moment = PI * ALPHA**2 / float(inclusion)
    elif stratified:
        second_moment = PI**2 * ALPHA**2 / float(inclusion)
    else:
        second_moment = PI**2 * ALPHA**2 / (float(mass * inclusion))
    theta_sq = (float(PI) * ALPHA * float(EPS)) ** 2
    mse = ((second_moment - theta_sq) / n) if n else None
    return {
        "design": name,
        "behavior_mass": str(behavior_mass),
        "sampling_source_action_mass": str(mass),
        "target_mass": str(PI),
        "behavior_to_target_mass_ratio": str(PI / behavior_mass),
        "rollout_cost_in_label_call_units": rollout_ratio,
        "total_budget_units": budget,
        "expected_cost_per_audited_group": float(cost_per_audited),
        "audited_groups": n,
        "clean_label_calls": n * labels,
        "expected_generated_groups": float(Fraction(n, 1) / mass) if stratified else n,
        "informative_rate_per_audited_group": str(event),
        "exact_optimal_sign_error": float((1 - tv(n, event)) / 2) if n else 0.5,
        "active_component_ht_mse_stationary": mse,
        "labels_per_audited_group": labels,
        "oracle_knows_active_triple": oracle_subset,
    }


def main() -> None:
    lock_hash = hashlib.sha256(LOCK.read_bytes()).hexdigest()
    if lock_hash != LOCK_SHA256:
        raise SystemExit("protocol lock hash mismatch")
    designs = []
    for ratio in OVERLAPS:
        mu = PI / ratio
        for rollout in ROLLOUT_COSTS:
            settings = [
                ("behavior_uniform_triple", mu, Fraction(1, 4), 3, False, False),
                ("behavior_full_group", mu, Fraction(1), 4, False, False),
                ("behavior_action_stratified_uniform_triple", mu, Fraction(1, 4), 3, True, False),
                ("target_uniform_triple", PI, Fraction(1, 4), 3, False, False),
                ("target_action_stratified_uniform_triple", PI, Fraction(1, 4), 3, True, False),
                ("target_action_stratified_oracle_triple", PI, Fraction(1), 3, True, True),
                ("target_action_stratified_full_group", PI, Fraction(1), 4, True, False),
            ]
            for name, mass, incl, k, strat, oracle in settings:
                # Behavior-stratified rows are valid only when the known action has positive mass.
                designs.append(design_row(name, mass=mass, behavior_mass=mu,
                                          inclusion=incl, labels=k,
                                          rollout_ratio=rollout, budget=BUDGET,
                                          stratified=strat, oracle_subset=oracle))

    deltas = []
    behavior_mean = 0.5
    for delta in (Fraction(-3, 4), Fraction(-1, 2), Fraction(0), Fraction(1, 4), Fraction(1, 2)):
        target_mean = behavior_mean + float(delta)
        target_mean = max(-1.0, min(1.0, target_mean))
        target_theta = float(PI) * ALPHA * target_mean
        transported_theta = float(PI) * ALPHA * behavior_mean
        deltas.append({
            "target_parity_mean_shift_delta": str(delta),
            "behavior_parity_mean": behavior_mean,
            "target_parity_mean": target_mean,
            "target_update_mean": target_theta,
            "behavior_transport_estimate_mean": transported_theta,
            "transport_bias": transported_theta - target_theta,
            "sign_reversed_vs_behavior_transport": target_theta * transported_theta < 0,
        })

    result = {
        "protocol_sha256": lock_hash,
        "estimand": {
            "active_walsh_coefficient": ALPHA,
            "target_action_mass": str(PI),
            "conditional_parity_signal": str(EPS),
            "total_budget": BUDGET,
            "budget_units": "one clean-label call; rollout group cost is 0 or 1 label-call equivalent",
        },
        "matched_total_cost_designs": designs,
        "conditional_shift_sensitivity": {
            "identity": "transport_bias = pi(a*) * alpha_T(a*) * (m_behavior - m_target)",
            "rows": deltas,
            "assumption_boundary": "The stationary rows assume conditional transport; the delta rows vary conditional verifier/task error separately from action mass.",
        },
        "adaptive_proxy_reduction": {
            "condition": "Pre-audit proxy observes only strict subsets whose distributions are identical across the frozen active-triple worlds.",
            "mutual_information_hypothesis_proxy": 0,
            "consequence": "Any allocation chosen from that proxy alone has no information about the active triple; under symmetric costs its expected inclusion is the uniform design. Adaptation after clean triple outcomes is a sequential experimental-design problem and is not represented as a new method here.",
        },
        "limitations": [
            "The exact sign error is the optimal two-world test, not the error of a particular GRPO optimizer.",
            "Total generation cost uses expected rejection cost for action-stratified groups and does not claim measured wall time.",
            "An importance-weighted estimator uses the same transcript as its sampling design; weighting does not change the optimal transcript sign risk.",
            "No real verifier, trained model, or capability outcome is evaluated.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
