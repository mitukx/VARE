#!/usr/bin/env python3
"""Reproduce the frozen two-outcome selective-audit gradient counterexample."""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction as F
from pathlib import Path


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def selected_mass(theta: float, theta0: float, *, differentiate_selection: bool) -> float:
    p = sigmoid(theta)
    # The ablation evaluates the same local inclusion probabilities but freezes
    # their derivative, which is exactly the stop-gradient comparison.
    s_a = 0.5 if not differentiate_selection else sigmoid(-4.0 * (theta - theta0))
    s_b = 0.5 if not differentiate_selection else 1.0 - s_a
    return p * s_a / (p * s_a + (1.0 - p) * s_b)


def outer_score(theta: float, theta0: float, *, differentiate_selection: bool) -> float:
    p = sigmoid(theta)
    q = selected_mass(theta, theta0, differentiate_selection=differentiate_selection)
    # The strictly convex selected-data squared-loss optimum is phi*=2q-1.
    phi = 2.0 * q - 1.0
    return (2.0 * p - 1.0) * phi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # Exact analytic derivatives at p=7/10, s_A=s_B=1/2, p'=21/100,
    # s_A'=-1, s_B'=+1.
    p, p_dot, s_a, s_a_dot, s_b, s_b_dot = map(
        F, ("7/10", "21/100", "1/2", "-1", "1/2", "1")
    )
    numerator = p * s_a
    normalizer = numerator + (1 - p) * s_b
    numerator_dot = p_dot * s_a + p * s_a_dot
    normalizer_dot = numerator_dot - p_dot * s_b + (1 - p) * s_b_dot
    q = numerator / normalizer
    q_dot_full = (numerator_dot * normalizer - numerator * normalizer_dot) / normalizer**2

    # Setting s'_A=s'_B=0 leaves dq/dtheta=dp/dtheta at the equal-propensity point.
    q_dot_stopped = p_dot
    phi = 2 * q - 1
    phi_dot_full = 2 * q_dot_full
    phi_dot_stopped = 2 * q_dot_stopped
    j_dot_full = 2 * p_dot * phi + (2 * p - 1) * phi_dot_full
    j_dot_stopped = 2 * p_dot * phi + (2 * p - 1) * phi_dot_stopped

    theta0 = math.log(7.0 / 3.0)
    h = 1e-5
    q_fd_full = (
        selected_mass(theta0 + h, theta0, differentiate_selection=True)
        - selected_mass(theta0 - h, theta0, differentiate_selection=True)
    ) / (2.0 * h)
    q_fd_stopped = (
        selected_mass(theta0 + h, theta0, differentiate_selection=False)
        - selected_mass(theta0 - h, theta0, differentiate_selection=False)
    ) / (2.0 * h)
    j_fd_full = (
        outer_score(theta0 + h, theta0, differentiate_selection=True)
        - outer_score(theta0 - h, theta0, differentiate_selection=True)
    ) / (2.0 * h)
    j_fd_stopped = (
        outer_score(theta0 + h, theta0, differentiate_selection=False)
        - outer_score(theta0 - h, theta0, differentiate_selection=False)
    ) / (2.0 * h)

    # Normalized IPW exactly removes audit selection: E_Q[f/s]/E_Q[1/s]=E_P[f].
    q_ipw = (q / s_a) / (q / s_a + (1 - q) / s_b)
    result = {
        "protocol": "policy_dependent_audit_selection_gradient_v1",
        "exact": {
            "p_A": str(p),
            "selected_q_A": str(q),
            "dq_dtheta_full": str(q_dot_full),
            "dq_dtheta_stop_selection": str(q_dot_stopped),
            "verifier_phi": str(phi),
            "dphi_dtheta_full": str(phi_dot_full),
            "dphi_dtheta_stop_selection": str(phi_dot_stopped),
            "outer_gradient_full": str(j_dot_full),
            "outer_gradient_stop_selection": str(j_dot_stopped),
            "normalized_ipw_q_A": str(q_ipw),
        },
        "finite_difference": {
            "step": h,
            "dq_dtheta_full": q_fd_full,
            "dq_dtheta_stop_selection": q_fd_stopped,
            "outer_gradient_full": j_fd_full,
            "outer_gradient_stop_selection": j_fd_stopped,
        },
        "checks": {
            "full_gradient_reverses_stop_selection": j_dot_full * j_dot_stopped < 0,
            "ipw_recovers_policy_mass": q_ipw == p,
            "finite_difference_matches_exact": (
                abs(q_fd_full - float(q_dot_full)) < 1e-8
                and abs(q_fd_stopped - float(q_dot_stopped)) < 1e-8
                and abs(j_fd_full - float(j_dot_full)) < 1e-8
                and abs(j_fd_stopped - float(j_dot_stopped)) < 1e-8
            ),
        },
    }
    if not all(result["checks"].values()):
        raise SystemExit(json.dumps(result, indent=2, sort_keys=True))
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
