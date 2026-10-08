#!/usr/bin/env python3
"""Exact finite variance/cost analysis for the frozen 3-of-4 GRPO audit."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import bisect
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_partial_audit_estimator_v1.json"
LOCK = ROOT / "protocols/grpo_partial_audit_estimator_v1.lock.json"
OUT = ROOT / "results/grpo-partial-audit-estimator-v1/run-1"
BITS = tuple(itertools.product((0, 1), repeat=4))
SUBSETS = {m: tuple(itertools.combinations(range(4), m)) for m in range(1, 5)}
P_ACTION = Fraction(1, 4)


class Quad:
    """Exact element of Q(sqrt(3)), represented as rational + sqrt3*rational."""

    def __init__(self, rational: Fraction | int = 0, sqrt3: Fraction | int = 0):
        self.r = Fraction(rational)
        self.s = Fraction(sqrt3)

    def __add__(self, other: Quad) -> Quad:
        return Quad(self.r + other.r, self.s + other.s)

    def __sub__(self, other: Quad) -> Quad:
        return self + (-other)

    def __neg__(self) -> Quad:
        return Quad(-self.r, -self.s)

    def __mul__(self, other: Quad | Fraction | int) -> Quad:
        if not isinstance(other, Quad):
            other = Quad(other)
        return Quad(self.r * other.r + 3 * self.s * other.s,
                    self.r * other.s + self.s * other.r)

    __rmul__ = __mul__

    def __truediv__(self, other: Quad | Fraction | int) -> Quad:
        if not isinstance(other, Quad):
            other = Quad(other)
        denominator = other.r * other.r - 3 * other.s * other.s
        if denominator == 0:
            raise ZeroDivisionError
        conjugate = Quad(other.r, -other.s)
        return self * conjugate * (1 / denominator)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Quad) and self.r == other.r and self.s == other.s

    def exact(self) -> dict[str, str]:
        return {"rational": str(self.r), "sqrt3_coefficient": str(self.s)}

    def value(self) -> float:
        return float(self.r) + float(self.s) * (3.0 ** 0.5)


ZERO = Quad()
PI = {
    1: {0: Fraction(1), 1: Fraction(1, 4)},
    2: {0: Fraction(1), 1: Fraction(1, 2), 2: Fraction(1, 6)},
    3: {0: Fraction(1), 1: Fraction(3, 4), 2: Fraction(1, 2), 3: Fraction(1, 4)},
    4: {0: Fraction(1), 1: Fraction(1), 2: Fraction(1), 3: Fraction(1)},
}


def advantage(y: tuple[int, ...]) -> tuple[Quad, ...]:
    count = sum(y)
    if count in (0, 4):
        return (ZERO,) * 4
    if count == 1:
        high, low = Quad(0, 1), Quad(0, Fraction(-1, 3))
    elif count == 2:
        high, low = Quad(1), Quad(-1)
    else:
        high, low = Quad(0, Fraction(1, 3)), Quad(0, -1)
    return tuple(high if bit else low for bit in y)


def update(a: tuple[int, ...], y: tuple[int, ...]) -> Quad:
    return sum((v * (Fraction(bit) - P_ACTION) for bit, v in zip(a, advantage(y))), ZERO)


def chi(y: tuple[int, ...], subset: tuple[int, ...]) -> int:
    return -1 if sum(y[i] for i in subset) % 2 else 1


def coeffs(a: tuple[int, ...]) -> dict[tuple[int, ...], Quad]:
    return {
        t: sum((update(a, y) * chi(y, t) for y in BITS), ZERO) / 16
        for size in range(5) for t in itertools.combinations(range(4), size)
    }


def partial(a: tuple[int, ...], y: tuple[int, ...], subset: tuple[int, ...], alpha: dict) -> Quad:
    m = len(subset)
    return sum((alpha[t] * chi(y, t) / PI[m][len(t)]
                for size in range(m + 1) for t in itertools.combinations(subset, size)), ZERO)


def action_prob(a: tuple[int, ...]) -> Fraction:
    return Fraction(1, 4) ** sum(a) * Fraction(3, 4) ** (4 - sum(a))


def law_independent(q: Fraction) -> dict[tuple[int, ...], Fraction]:
    return {y: q ** sum(y) * (1 - q) ** (4 - sum(y)) for y in BITS}


def law_common(q: Fraction) -> dict[tuple[int, ...], Fraction]:
    return {y: (1 - q if y == (0, 0, 0, 0) else q if y == (1, 1, 1, 1) else Fraction(0)) for y in BITS}


def law_parity(subset: tuple[int, ...], eps: Fraction, sign: int) -> dict[tuple[int, ...], Fraction]:
    return {y: Fraction(1, 16) * (1 + sign * eps * chi(y, subset)) for y in BITS}


def make_laws():
    laws = []
    for q in (Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)):
        laws.append((f"independent_q={q}", {a: law_independent(q) for a in BITS}))
    for q in (Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)):
        laws.append((f"common_mode_q={q}", {a: law_common(q) for a in BITS}))
    uniform = {y: Fraction(1, 16) for y in BITS}
    for target in BITS:
        for subset in SUBSETS[3]:
            for eps in (Fraction(1, 4), Fraction(1, 2), Fraction(1)):
                for sign in (-1, 1):
                    conditional = {a: uniform for a in BITS}
                    conditional[target] = law_parity(subset, eps, sign)
                    key = f"parity_a={''.join(map(str,target))}_T={''.join(map(str,subset))}_eps={eps}_s={sign:+d}"
                    laws.append((key, conditional))
    return laws


def moments(conditional: dict[tuple[int, ...], dict[tuple[int, ...], Fraction]], alphas: dict, m: int):
    mean_full, second_full = ZERO, ZERO
    mean_est, second_est = ZERO, ZERO
    conditional_noise = ZERO
    pointwise_ok = True
    total_mass = Fraction(0)
    for a in BITS:
        pa = action_prob(a)
        alpha = alphas[a]
        for y, py in conditional[a].items():
            if not py:
                continue
            mass = pa * py
            total_mass += mass
            g = update(a, y)
            estimates = [partial(a, y, s, alpha) for s in SUBSETS[m]]
            pointwise_ok = pointwise_ok and (m != 3 or sum(estimates, ZERO) / len(estimates) == g)
            mean_full += mass * g
            second_full += mass * (g * g)
            est_mean_given_y = sum(estimates, ZERO) / len(estimates)
            conditional_noise += mass * sum(((v - est_mean_given_y) * (v - est_mean_given_y) for v in estimates), ZERO) / len(estimates)
            for value in estimates:
                mean_est += mass * value / len(estimates)
                second_est += mass * (value * value) / len(estimates)
    var_full = second_full - mean_full * mean_full
    var_est = second_est - mean_est * mean_est
    bias = mean_est - mean_full
    mse_est = var_est + bias * bias
    mse_full = var_full
    return {
        "mass": total_mass, "mean_full": mean_full, "mean_est": mean_est,
        "bias": bias, "mse": mse_est, "mse_full": mse_full,
        "var_full": var_full, "var_est": var_est,
        "conditional_noise": conditional_noise, "pointwise_ok": pointwise_ok,
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outcome_distribution(conditional, alphas, m):
    values, cumulative, running = [], [], 0.0
    for a in BITS:
        pa = float(action_prob(a))
        for y, py in conditional[a].items():
            if py == 0:
                continue
            for subset in SUBSETS[m]:
                running += pa * float(py) / len(SUBSETS[m])
                values.append(partial(a, y, subset, alphas[a]).value())
                cumulative.append(running)
    cumulative[-1] = 1.0
    return values, cumulative


def wilson(k: int, n: int) -> list[float]:
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def direction_experiment(alphas, law_map):
    seeds = [20261009, 20261010, 20261011, 20261012, 20261013, 20261014]
    repetitions = 4096
    records = []
    for sign_index, sign in enumerate((-1, 1)):
        law_id = f"parity_a=0001_T=012_eps=1/2_s={sign:+d}"
        conditional = law_map[law_id]
        true = moments(conditional, alphas, 4)["mean_full"].value()
        for arm_index, m in enumerate(range(1, 5), start=1):
            values, cdf = outcome_distribution(conditional, alphas, m)
            n_groups = 12 // m
            by_seed = []
            for seed in seeds:
                rng = random.Random(seed + 1000003 * arm_index + 17 * sign_index)
                errors = 0
                for _ in range(repetitions):
                    estimate = 0.0
                    for _ in range(n_groups):
                        draw = rng.random()
                        j = bisect.bisect_left(cdf, draw)
                        estimate += values[min(j, len(values) - 1)]
                    estimate /= n_groups
                    if (true > 0 and estimate <= 0) or (true < 0 and estimate >= 0):
                        errors += 1
                by_seed.append({"seed": seed, "wrong_or_zero_direction": errors,
                                "repetitions": repetitions, "rate": errors / repetitions})
            total_errors = sum(row["wrong_or_zero_direction"] for row in by_seed)
            total_n = repetitions * len(seeds)
            records.append({
                "law_id": law_id,
                "audit_size": m,
                "label_budget": 12,
                "groups_per_replicate": n_groups,
                "true_mean_decimal": true,
                "wrong_or_zero_direction": total_errors,
                "repetitions": total_n,
                "rate": total_errors / total_n,
                "wilson_95_interval": wilson(total_errors, total_n),
                "by_seed": by_seed,
            })
    return records


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    lock = json.loads(LOCK.read_text())
    expected = {
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": sha(Path(__file__)),
        "auditor_sha256": sha(ROOT / "scripts/audit_grpo_partial_audit_estimator_v1.py"),
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise SystemExit("frozen protocol or script hash differs from protocol lock")
    alphas = {a: coeffs(a) for a in BITS}
    rows = []
    exact_all = True
    law_map = dict(make_laws())
    for name, conditional in law_map.items():
        for audit_size in range(1, 5):
            m = moments(conditional, alphas, audit_size)
            exact_all &= m["mass"] == 1
            if audit_size in (3, 4):
                exact_all &= m["pointwise_ok"] and m["mean_full"] == m["mean_est"]
            decomp_ok = audit_size != 3 or m["var_est"] == m["var_full"] + m["conditional_noise"]
            exact_all &= decomp_ok and m["conditional_noise"].value() >= -1e-12
            groups_at_budget = 12 // audit_size
            fixed_budget_mse = m["var_est"] / groups_at_budget + m["bias"] * m["bias"]
            full_budget_mse = m["var_full"] / 3
            ratio = None if full_budget_mse == ZERO else fixed_budget_mse.value() / full_budget_mse.value()
            asymptotic_ratio = None if m["var_full"] == ZERO else Fraction(3, 4) * m["var_est"].value() / m["var_full"].value()
            rows.append({
                "law_id": name,
                "audit_size": audit_size,
                "labels_per_group": audit_size,
                "subset_choices": len(SUBSETS[audit_size]),
                "walsh_terms_evaluated_per_group": 2 ** audit_size,
                "exact_mass": str(m["mass"]),
                "bias_exact": m["bias"].exact(),
                "variance_exact": m["var_est"].exact(),
                "mse_exact": m["mse"].exact(),
                "fixed_12_label_budget_mse_exact": fixed_budget_mse.exact(),
                "full_audit_fixed_budget_mse_exact": full_budget_mse.exact(),
                "full_audit_mse_exact": m["mse_full"].exact(),
                "unbiased_exact": m["mean_full"] == m["mean_est"],
                "pointwise_subset_average_exact": m["pointwise_ok"] if audit_size == 3 else None,
                "variance_decomposition_exact": decomp_ok,
                "audit_randomization_variance_exact": m["conditional_noise"].exact() if audit_size == 3 else None,
                "mse_decimal": m["mse"].value(),
                "fixed_12_label_budget_mse_decimal": fixed_budget_mse.value(),
                "cost_adjusted_mse_ratio_to_full": ratio,
                "asymptotic_variance_per_label_ratio_to_full": asymptotic_ratio,
            })

    # The pair witnesses are observationally identical for every <=2-coordinate view.
    target, triple, eps = (0, 0, 0, 1), (0, 1, 2), Fraction(1, 2)
    plus, minus = law_parity(triple, eps, 1), law_parity(triple, eps, -1)
    alpha = coeffs(target)[triple]
    gap = sum((update(target, y) * (plus[y] - minus[y]) for y in BITS), ZERO)
    exact_all &= gap == 2 * eps * alpha
    lower = eps * alpha

    defined = [r["cost_adjusted_mse_ratio_to_full"] for r in rows if r["cost_adjusted_mse_ratio_to_full"] is not None]
    ratio_by_m = {}
    for audit_size in range(1, 5):
        group = [r["cost_adjusted_mse_ratio_to_full"] for r in rows
                 if r["audit_size"] == audit_size and r["cost_adjusted_mse_ratio_to_full"] is not None]
        ratio_by_m[str(audit_size)] = {
            "defined_count": len(group), "below_one_count": sum(x < 1 for x in group),
            "equal_one_count": sum(x == 1 for x in group), "above_one_count": sum(x > 1 for x in group),
            "median": sorted(group)[len(group) // 2] if group else None,
            "min": min(group) if group else None, "max": max(group) if group else None,
        }
    summary = {
        "law_count": len(law_map),
        "law_by_audit_comparison_count": len(rows),
        "per_audit_size": ratio_by_m,
        "defined_ratio_count": len(defined),
        "ratio_below_one_count": sum(r < 1 for r in defined),
        "ratio_equal_one_count": sum(r == 1 for r in defined),
        "ratio_above_one_count": sum(r > 1 for r in defined),
        "ratio_min": min(defined) if defined else None,
        "ratio_median": sorted(defined)[len(defined) // 2] if defined else None,
        "ratio_max": max(defined) if defined else None,
    }
    result = {
        "protocol_id": protocol["protocol_id"],
        "classification": "exact_finite_sample_cost_analysis",
        "model_or_training_run": False,
        "exact_checks_passed": exact_all,
        "summary": summary,
        "two_label_nonidentifiability": {
            "target_action": "0001", "parity_subset": "012", "epsilon": str(eps),
            "conditional_update_gap_exact": gap.exact(),
            "minimax_absolute_bias_lower_bound_exact": lower.exact(),
            "minimax_absolute_bias_lower_bound_decimal": lower.value(),
            "bound_reason": "The two worlds have the same observation law for every audit design using at most two labels; for any common estimator, the larger absolute bias is at least half the target-mean gap.",
        },
        "finite_budget_direction_check": direction_experiment(alphas, law_map),
        "laws": rows,
        "sources": {"protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__))},
        "interpretation_limit": protocol["claim_limits"],
    }
    if not exact_all:
        raise SystemExit("one or more frozen exact checks failed")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exact-enumeration.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "laws"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
