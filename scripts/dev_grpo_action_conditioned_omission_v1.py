#!/usr/bin/env python3
"""Exact fixed-budget comparison of action-conditioned GRPO label omissions."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path
from statistics import fmean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_action_conditioned_omission_v1.lock.json"
BASE = ROOT / "scripts/run_grpo_partial_audit_estimator_v1.py"
OUT = ROOT / "results/grpo-action-conditioned-omission-v1/run-1"
BITS = tuple(itertools.product((0, 1), repeat=4))
SUBSETS = {m: tuple(itertools.combinations(range(4), m)) for m in range(1, 5)}
QGRID = tuple(Fraction(x, 4) for x in (0, 1, 2, 3, 4))
FAMILY_NAMES = ("random_joint", "independent", "common_shock", "single_stratum_parity")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quad_float(value: Any) -> float:
    return float(value.r) + float(value.s) * math.sqrt(3.0)


def load_base():
    spec = importlib.util.spec_from_file_location("frozen_grpo_partial_audit_v1", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned base estimator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def action_probability(a: tuple[int, ...]) -> Fraction:
    return Fraction(1, 4) ** sum(a) * Fraction(3, 4) ** (4 - sum(a))


def chi(y: tuple[int, ...], subset: tuple[int, ...]) -> int:
    return -1 if sum(y[i] for i in subset) % 2 else 1


def q_structured(a: tuple[int, ...], alpha: dict[tuple[int, ...], float]) -> tuple[float, ...]:
    raw = [abs(alpha[tuple(i for i in range(4) if i != j)]) for j in range(4)]
    total = sum(raw)
    if total == 0:
        return (0.25,) * 4
    return tuple(1.0 / 16.0 + 0.75 * value / total for value in raw)


def estimate(a: tuple[int, ...], y: tuple[int, ...], subset: tuple[int, ...], alpha: dict,
             inclusion: dict[tuple[int, ...], float]) -> float:
    return sum(alpha[t] * chi(y, t) / inclusion[t]
               for size in range(len(subset) + 1)
               for t in itertools.combinations(subset, size))


def inclusion_uniform(m: int) -> dict[tuple[int, ...], float]:
    return {t: float(Fraction(math.comb(4 - len(t), m - len(t)), math.comb(4, m)))
            for size in range(m + 1) for t in itertools.combinations(range(4), size)}


def inclusion_omission(q: tuple[float, ...]) -> dict[tuple[int, ...], float]:
    return {t: sum(q[j] for j in range(4) if j not in t)
            for size in range(4) for t in itertools.combinations(range(4), size)}


def probability_row(rng: random.Random, family: str, a: tuple[int, ...], target_a: tuple[int, ...] | None,
                    target_t: tuple[int, ...] | None, parity_eps: Fraction | None,
                    parity_sign: int | None) -> tuple[Fraction, ...]:
    if family == "random_joint":
        weights = [rng.randint(0, 8) for _ in BITS]
        if not any(weights):
            weights = [1] * len(BITS)
        denominator = sum(weights)
        return tuple(Fraction(w, denominator) for w in weights)
    if family == "independent":
        rates = [rng.choice(QGRID) for _ in range(4)]
        return tuple(mathprod_fraction([rates[i] if y[i] else 1 - rates[i] for i in range(4)]) for y in BITS)
    if family == "common_shock":
        mix = rng.choice((Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)))
        shared_one = rng.choice((Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)))
        rates = [rng.choice(QGRID) for _ in range(4)]
        result = []
        for y in BITS:
            independent = mathprod_fraction([rates[i] if y[i] else 1 - rates[i] for i in range(4)])
            common = (1 - shared_one) if y == (0, 0, 0, 0) else shared_one if y == (1, 1, 1, 1) else Fraction(0)
            result.append((1 - mix) * independent + mix * common)
        return tuple(result)
    if family == "single_stratum_parity":
        if a != target_a:
            return tuple(Fraction(1, 16) for _ in BITS)
        return tuple(Fraction(1, 16) * (1 + parity_sign * parity_eps * chi(y, target_t)) for y in BITS)
    raise ValueError(f"unknown law family {family}")


def mathprod_fraction(values) -> Fraction:
    result = Fraction(1)
    for value in values:
        result *= value
    return result


def generate_panel(seed: int, law_count: int) -> list[dict[str, Any]]:
    if law_count != 128:
        raise ValueError("law_count differs from the frozen protocol")
    rng = random.Random(seed)
    panel = []
    for family_index, family in enumerate(FAMILY_NAMES):
        for within_family in range(32):
            target_a = target_t = parity_eps = parity_sign = None
            if family == "single_stratum_parity":
                target_a = rng.choice(BITS)
                target_t = rng.choice(SUBSETS[3] + SUBSETS[4])
                parity_eps = rng.choice((Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)))
                parity_sign = rng.choice((-1, 1))
            rows = [
                [probability_row(rng, family, a, target_a, target_t, parity_eps, parity_sign)
                 for a in BITS]
                for _ in range(1)
            ][0]
            panel.append({
                "law_id": f"{family_index:02d}-{within_family:02d}",
                "family": family,
                "conditional_probabilities": [
                    [[str(p.numerator), str(p.denominator)] for p in row]
                    for row in rows
                ],
                "parameters": {
                    "target_action": "".join(map(str, target_a)) if target_a is not None else None,
                    "target_subset": list(target_t) if target_t is not None else None,
                    "parity_epsilon": str(parity_eps) if parity_eps is not None else None,
                    "parity_sign": parity_sign,
                },
            })
    return panel


def metrics_for_law(law: dict[str, Any], base) -> dict[str, Any]:
    probs = [[Fraction(int(n), int(d)) for n, d in row] for row in law["conditional_probabilities"]]
    arms = {name: {"mean": 0.0, "second": 0.0} for name in ("m1", "m2", "m3_uniform", "m3_structured", "m4_full")}
    true_mean = true_second = 0.0
    pointwise_ok = True
    for ai, a in enumerate(BITS):
        pa = float(action_probability(a))
        alpha_exact = base.coeffs(a)
        alpha = {t: quad_float(value) for t, value in alpha_exact.items()}
        q = q_structured(a, alpha)
        q_inc = inclusion_omission(q)
        uni3_inc = inclusion_omission((0.25,) * 4)
        for yi, y in enumerate(BITS):
            py = float(probs[ai][yi])
            if py == 0:
                continue
            weight = pa * py
            g = quad_float(base.update(a, y))
            true_mean += weight * g
            true_second += weight * g * g
            for m in (1, 2):
                inc = inclusion_uniform(m)
                values = [estimate(a, y, subset, alpha, inc) for subset in SUBSETS[m]]
                mean_hat = sum(values) / len(values)
                second_hat = sum(value * value for value in values) / len(values)
                arms[f"m{m}"]["mean"] += weight * mean_hat
                arms[f"m{m}"]["second"] += weight * second_hat
            structured_values, uniform_values = [], []
            for j in range(4):
                subset = tuple(i for i in range(4) if i != j)
                structured_values.append(estimate(a, y, subset, alpha, q_inc))
                uniform_values.append(estimate(a, y, subset, alpha, uni3_inc))
            structured_mean = sum(q[j] * structured_values[j] for j in range(4))
            structured_second = sum(q[j] * structured_values[j] ** 2 for j in range(4))
            uniform_mean = sum(0.25 * value for value in uniform_values)
            uniform_second = sum(0.25 * value * value for value in uniform_values)
            arms["m3_structured"]["mean"] += weight * structured_mean
            arms["m3_structured"]["second"] += weight * structured_second
            arms["m3_uniform"]["mean"] += weight * uniform_mean
            arms["m3_uniform"]["second"] += weight * uniform_second
            arms["m4_full"]["mean"] += weight * g
            arms["m4_full"]["second"] += weight * g * g
            pointwise_ok = pointwise_ok and abs(structured_mean - g) <= 1e-10 and abs(uniform_mean - g) <= 1e-10
    true_var = max(0.0, true_second - true_mean * true_mean)
    budgets = {"m1": 12, "m2": 6, "m3_uniform": 4, "m3_structured": 4, "m4_full": 3}
    rows = {}
    for name, moments in arms.items():
        bias = moments["mean"] - true_mean
        variance = max(0.0, moments["second"] - moments["mean"] ** 2)
        mse = variance / budgets[name] + bias * bias
        rows[name] = {"mean": moments["mean"], "bias": bias, "variance": variance, "fixed_budget_mse": mse}
    full = rows["m4_full"]["fixed_budget_mse"]
    for row in rows.values():
        row["ratio_to_full"] = row["fixed_budget_mse"] / full if full > 1e-20 else None
    return {"law_id": law["law_id"], "family": law["family"], "true_mean": true_mean,
            "true_variance": true_var, "arms": rows, "pointwise_m3_unbiased": pointwise_ok,
            "zero_full_mse": full <= 1e-20}


def draw_weighted(rng: random.Random, probs: tuple[float, ...]) -> int:
    target = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probs):
        cumulative += probability
        if target < cumulative:
            return index
    return len(probs) - 1


def direction_stress(base, seed: int, replicates: int) -> list[dict[str, Any]]:
    results = []
    target_a = (0, 0, 1, 0)
    target_t = (0, 1, 3)
    eps = Fraction(1, 2)
    for sign in (-1, 1):
        conditional = []
        for a in BITS:
            if a == target_a:
                conditional.append([Fraction(1, 16) * (1 + sign * eps * chi(y, target_t)) for y in BITS])
            else:
                conditional.append([Fraction(1, 16) for _ in BITS])
        true = 0.0
        for ai, a in enumerate(BITS):
            pa = float(action_probability(a))
            for yi, y in enumerate(BITS):
                true += pa * float(conditional[ai][yi]) * quad_float(base.update(a, y))
        counts = {"full": 0, "uniform": 0, "structured": 0}
        paired = {"uniform_minus_full": [], "structured_minus_full": []}
        for seed_index in range(6):
            run_seed = seed + seed_index
            rng_world = random.Random(run_seed + 91 * (sign + 1))
            rng_uniform = random.Random(run_seed + 100003 + sign)
            rng_structured = random.Random(run_seed + 200003 + sign)
            per_seed = {name: 0 for name in counts}
            per_seed_diff = {key: [] for key in paired}
            for _ in range(replicates):
                groups = []
                for _group in range(4):
                    a = BITS[draw_weighted(rng_world, tuple(float(action_probability(x)) for x in BITS))]
                    ai = BITS.index(a)
                    y = BITS[draw_weighted(rng_world, tuple(float(x) for x in conditional[ai]))]
                    groups.append((a, y))
                full_est = sum(quad_float(base.update(a, y)) for a, y in groups[:3]) / 3
                uniform_terms, structured_terms = [], []
                for a, y in groups:
                    alpha = {t: quad_float(v) for t, v in base.coeffs(a).items()}
                    j_uniform = rng_uniform.randrange(4)
                    subset_u = tuple(i for i in range(4) if i != j_uniform)
                    uniform_terms.append(estimate(a, y, subset_u, alpha, inclusion_omission((0.25,) * 4)))
                    q = q_structured(a, alpha)
                    draw = rng_structured.random()
                    j = 3
                    cumulative = 0.0
                    for j0, probability in enumerate(q):
                        cumulative += probability
                        if draw < cumulative:
                            j = j0
                            break
                    subset_s = tuple(i for i in range(4) if i != j)
                    structured_terms.append(estimate(a, y, subset_s, alpha, inclusion_omission(q)))
                uniform_est = sum(uniform_terms) / 4
                structured_est = sum(structured_terms) / 4
                err = {
                    "full": int(full_est == 0 or (full_est > 0) != (true > 0)),
                    "uniform": int(uniform_est == 0 or (uniform_est > 0) != (true > 0)),
                    "structured": int(structured_est == 0 or (structured_est > 0) != (true > 0)),
                }
                for name in counts:
                    counts[name] += err[name]
                    per_seed[name] += err[name]
                for name in paired:
                    lhs, rhs = name.split("_minus_")
                    delta = err[lhs] - err[rhs]
                    paired[name].append(delta)
                    per_seed_diff[name].append(delta)
            results.append({"sign": sign, "seed": run_seed, "true_mean": true,
                            "replicates": replicates,
                            "errors": per_seed,
                            "error_rates": {k: v / replicates for k, v in per_seed.items()},
                            "paired_excess_error_mean": {k: sum(v) / replicates for k, v in per_seed_diff.items()}})
        pooled = {name: counts[name] / (replicates * 6) for name in counts}
        paired_stats = {}
        for key, values in paired.items():
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
            se = math.sqrt(variance / len(values))
            paired_stats[key] = {"mean_difference": mean, "normal_95_interval": [mean - 1.96 * se, mean + 1.96 * se]}
        results.append({"sign": sign, "pooled": pooled, "paired": paired_stats,
                        "replicates": replicates * 6, "true_mean": true})
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    base = load_base()
    panel = generate_panel(args.seed, 128)
    rows = [metrics_for_law(law, base) for law in panel]
    nonzero = [row for row in rows if not row["zero_full_mse"]]
    structured = [row["arms"]["m3_structured"]["ratio_to_full"] for row in nonzero]
    uniform = [row["arms"]["m3_uniform"]["ratio_to_full"] for row in nonzero]
    structured_uniform = [row["arms"]["m3_structured"]["fixed_budget_mse"] / row["arms"]["m3_uniform"]["fixed_budget_mse"] for row in nonzero]
    panel_path = args.output_dir / "law_panel.json"
    panel_path.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "status": "exploratory-development-only-not-preregistered",
        "seed": args.seed,
        "law_count": len(panel),
        "nondegenerate_laws": len(nonzero),
        "source_script_sha256": sha(Path(__file__)),
        "base_estimator_sha256": sha(BASE),
        "law_panel_sha256": sha(panel_path),
        "structured_mse_ratio_full_median": median(structured),
        "structured_mse_ratio_full_mean": fmean(structured),
        "structured_beats_full_count": sum(x < 1 for x in structured),
        "structured_10pct_below_full_count": sum(x <= .9 for x in structured),
        "uniform_mse_ratio_full_median": median(uniform),
        "structured_beats_uniform_count": sum(x < y for x,y in zip(structured,uniform,strict=True)),
        "structured_uniform_ratio_median": median(structured_uniform),
        "per_law": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k != "per_law"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
