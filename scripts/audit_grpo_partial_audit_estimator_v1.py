#!/usr/bin/env python3
"""Independent exact replay of the partial-audit estimator result bundle."""

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
LOCK = ROOT / "protocols/grpo_partial_audit_estimator_v1_implementation_r2.lock.json"
ERRATUM = ROOT / "protocols/grpo_partial_audit_estimator_v1_implementation_erratum_1.json"
ERRATUM_2 = ROOT / "protocols/grpo_partial_audit_estimator_v1_implementation_erratum_2.json"
RUN = ROOT / "results/grpo-partial-audit-estimator-v1/run-1/exact-enumeration.json"
OUT = ROOT / "results/grpo-partial-audit-estimator-v1/run-1/independent-audit.json"
V = tuple(itertools.product((0, 1), repeat=4))
SETS = {m: tuple(itertools.combinations(range(4), m)) for m in range(1, 5)}
PIS = {
    1: (Fraction(1), Fraction(1, 4)),
    2: (Fraction(1), Fraction(1, 2), Fraction(1, 6)),
    3: (Fraction(1), Fraction(3, 4), Fraction(1, 2), Fraction(1, 4)),
    4: (Fraction(1), Fraction(1), Fraction(1), Fraction(1), Fraction(1)),
}
Pair = tuple[Fraction, Fraction]
Z: Pair = (Fraction(0), Fraction(0))


def add(x: Pair, y: Pair) -> Pair:
    return x[0] + y[0], x[1] + y[1]


def neg(x: Pair) -> Pair:
    return -x[0], -x[1]


def sub(x: Pair, y: Pair) -> Pair:
    return add(x, neg(y))


def scale(x: Pair, q: Fraction | int) -> Pair:
    q = Fraction(q)
    return x[0] * q, x[1] * q


def mul(x: Pair, y: Pair) -> Pair:
    return x[0] * y[0] + 3 * x[1] * y[1], x[0] * y[1] + x[1] * y[0]


def val(x: Pair) -> float:
    return float(x[0]) + float(x[1]) * (3.0 ** 0.5)


def exact(x: Pair) -> dict[str, str]:
    return {"rational": str(x[0]), "sqrt3_coefficient": str(x[1])}


def adv(y: tuple[int, ...]) -> tuple[Pair, ...]:
    k = sum(y)
    if k in (0, 4):
        return (Z,) * 4
    if k == 1:
        hi, lo = (Fraction(0), Fraction(1)), (Fraction(0), Fraction(-1, 3))
    elif k == 2:
        hi, lo = (Fraction(1), Fraction(0)), (Fraction(-1), Fraction(0))
    else:
        hi, lo = (Fraction(0), Fraction(1, 3)), (Fraction(0), Fraction(-1))
    return tuple(hi if bit else lo for bit in y)


def score(a: tuple[int, ...], y: tuple[int, ...]) -> Pair:
    total = Z
    for ai, vi in zip(a, adv(y)):
        total = add(total, scale(vi, Fraction(ai) - Fraction(1, 4)))
    return total


def character(y: tuple[int, ...], t: tuple[int, ...]) -> int:
    return -1 if sum(y[i] for i in t) % 2 else 1


def basis(a: tuple[int, ...]) -> dict[tuple[int, ...], Pair]:
    out = {}
    for k in range(5):
        for t in itertools.combinations(range(4), k):
            s = Z
            for y in V:
                s = add(s, scale(score(a, y), character(y, t)))
            out[t] = scale(s, Fraction(1, 16))
    return out


def law(q: Fraction, kind: str):
    if kind == "independent":
        return {y: q ** sum(y) * (1 - q) ** (4 - sum(y)) for y in V}
    return {y: (1 - q if y == (0, 0, 0, 0) else q if y == (1, 1, 1, 1) else Fraction(0)) for y in V}


def parity(t: tuple[int, ...], eps: Fraction, sign: int):
    return {y: Fraction(1, 16) * (1 + sign * eps * character(y, t)) for y in V}


def make_laws():
    for kind in ("independent", "common_mode"):
        for q in (Fraction(1, 5), Fraction(1, 2), Fraction(4, 5)):
            dist = law(q, "common" if kind == "common_mode" else kind)
            yield f"{kind}_q={q}", {a: dist for a in V}
    uniform = {y: Fraction(1, 16) for y in V}
    for target in V:
        for t in SETS[3]:
            for eps in (Fraction(1, 4), Fraction(1, 2), Fraction(1)):
                for sign in (-1, 1):
                    dists = {a: uniform for a in V}
                    dists[target] = parity(t, eps, sign)
                    yield f"parity_a={''.join(map(str,target))}_T={''.join(map(str,t))}_eps={eps}_s={sign:+d}", dists


def sample_estimate(a, y, subset, alphas):
    m = len(subset)
    out = Z
    for k in range(m + 1):
        for t in itertools.combinations(subset, k):
            out = add(out, scale(alphas[t], Fraction(character(y, t), 1) / PIS[m][k]))
    return out


def precompute_observations(alphas):
    scores = {(a, y): score(a, y) for a in V for y in V}
    estimates = {}
    for m in range(1, 5):
        estimates[m] = {}
        for a in V:
            for y in V:
                samples = [sample_estimate(a, y, subset, alphas[a]) for subset in SETS[m]]
                average = scale(sum_pairs(samples), Fraction(1, len(samples)))
                second = scale(sum_pairs([mul(x, x) for x in samples]), Fraction(1, len(samples)))
                estimates[m][(a, y)] = {"samples": samples, "mean": average, "second": second}
    return scores, estimates


def calculate(dists, scores, estimates_by_m, m):
    mean_g, mean_h, eg2, eh2, subset_noise, mass_sum = Z, Z, Z, Z, Z, Fraction(0)
    pointwise = True
    for a in V:
        pa = Fraction(1, 4) ** sum(a) * Fraction(3, 4) ** (4 - sum(a))
        for y, py in dists[a].items():
            if py == 0:
                continue
            w = pa * py
            mass_sum += w
            g = scores[(a, y)]
            cell = estimates_by_m[m][(a, y)]
            avg = cell["mean"]
            pointwise &= m not in (3, 4) or avg == g
            mean_g = add(mean_g, scale(g, w))
            eg2 = add(eg2, scale(mul(g, g), w))
            mean_h = add(mean_h, scale(avg, w))
            eh2 = add(eh2, scale(cell["second"], w))
            conditional_var = sub(cell["second"], mul(avg, avg))
            subset_noise = add(subset_noise, scale(conditional_var, w))
    var_g = sub(eg2, mul(mean_g, mean_g))
    var_h = sub(eh2, mul(mean_h, mean_h))
    bias = sub(mean_h, mean_g)
    mse = add(var_h, mul(bias, bias))
    return {"mass": mass_sum, "pointwise": pointwise, "mean_g": mean_g, "mean_h": mean_h,
            "variance_g": var_g, "variance_h": var_h, "bias": bias, "mse": mse,
            "noise": subset_noise}


def sum_pairs(xs):
    total = Z
    for x in xs:
        total = add(total, x)
    return total


def independent_direction_replay(dists, scores, estimates_by_m, m, sign, seeds, repetitions):
    outcomes, cumulative, running = [], [], 0.0
    for a in V:
        p_a = Fraction(1, 4) ** sum(a) * Fraction(3, 4) ** (4 - sum(a))
        for y, p_y in dists[a].items():
            if p_y == 0:
                continue
            for subset_index, subset in enumerate(SETS[m]):
                running += float(p_a * p_y) / len(SETS[m])
                outcomes.append(val(estimates_by_m[m][(a, y)]["samples"][subset_index]))
                cumulative.append(running)
    cumulative[-1] = 1.0
    mean = Z
    for a in V:
        for y, probability in dists[a].items():
            mean = add(mean, scale(scores[(a, y)], Fraction(1, 4) ** sum(a) * Fraction(3, 4) ** (4 - sum(a)) * probability))
    true_value = val(mean)
    sign_index = 0 if sign == -1 else 1
    counts = {}
    for arm_index, audit_size in enumerate(range(1, 5), start=1):
        if audit_size != m:
            continue
        for seed in seeds:
            rng = random.Random(seed + 1000003 * arm_index + 17 * sign_index)
            errors = 0
            groups = 12 // audit_size
            for _ in range(repetitions):
                estimate = 0.0
                for _ in range(groups):
                    j = bisect.bisect_left(cumulative, rng.random())
                    estimate += outcomes[min(j, len(outcomes) - 1)]
                estimate /= groups
                if (true_value > 0 and estimate <= 0) or (true_value < 0 and estimate >= 0):
                    errors += 1
            counts[seed] = errors
    return counts


def main() -> None:
    lock = json.loads(LOCK.read_text())
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    hashes = {"protocol_sha256": protocol_hash,
              "runner_sha256": hashlib.sha256((ROOT / "scripts/run_grpo_partial_audit_estimator_v1.py").read_bytes()).hexdigest(),
              "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    hashes["erratum_sha256"] = hashlib.sha256(ERRATUM.read_bytes()).hexdigest()
    hashes["erratum_2_sha256"] = hashlib.sha256(ERRATUM_2.read_bytes()).hexdigest()
    if any(lock.get(key) != value for key, value in hashes.items()):
        raise SystemExit("frozen protocol or script hash mismatch")
    record = json.loads(RUN.read_text())
    coefficients = {a: basis(a) for a in V}
    scores, estimates_by_m = precompute_observations(coefficients)
    observed = {(r["law_id"], r["audit_size"]): r for r in record["laws"]}
    checks = {"frozen_protocol_matches": record["sources"]["protocol_sha256"] == protocol_hash,
              "law_case_count_matches_protocol": len(list(make_laws())) == 390,
              "all_exact_rows_recompute": True,
              "unbiased_three_label_estimators": True,
              "four_label_estimator_is_exact": True,
              "variance_decomposition_for_three_labels": True,
              "finite_budget_direction_counts_recompute": True}
    for name, dists in make_laws():
        for m in range(1, 5):
            c = calculate(dists, scores, estimates_by_m, m)
            row = observed.get((name, m))
            expected_fixed_mse = add(scale(c["variance_h"], Fraction(m, 12)), mul(c["bias"], c["bias"]))
            checks["all_exact_rows_recompute"] &= (
                row is not None and row["bias_exact"] == exact(c["bias"])
                and row["variance_exact"] == exact(c["variance_h"])
                and row["mse_exact"] == exact(c["mse"])
                and row["fixed_12_label_budget_mse_exact"] == exact(expected_fixed_mse)
            )
            if m == 3:
                checks["unbiased_three_label_estimators"] &= c["mean_h"] == c["mean_g"] and c["pointwise"]
                checks["variance_decomposition_for_three_labels"] &= c["variance_h"] == add(c["variance_g"], c["noise"])
            if m == 4:
                checks["four_label_estimator_is_exact"] &= c["bias"] == Z and c["variance_h"] == c["variance_g"] and c["pointwise"]
    for row in record["finite_budget_direction_check"]:
        law_id = row["law_id"]
        sign = -1 if law_id.endswith("s=-1") else 1
        dists = dict(make_laws())[law_id]
        counts = independent_direction_replay(dists, scores, estimates_by_m, row["audit_size"], sign,
                                               [x["seed"] for x in row["by_seed"]],
                                               row["by_seed"][0]["repetitions"])
        checks["finite_budget_direction_counts_recompute"] &= all(
            counts[x["seed"]] == x["wrong_or_zero_direction"] for x in row["by_seed"])
    if not all(checks.values()):
        raise SystemExit(f"independent exact audit failed: {checks}")
    result = {"protocol_id": record["protocol_id"], "auditor": Path(__file__).name,
              "independent_implementation": True, "same_host_audit": True,
              "checks": checks, "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "protocol_sha256": protocol_hash,
              "scope": "Independent exact same-host recomputation using rational coefficient pairs; not outside reproduction or a novelty review."}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
