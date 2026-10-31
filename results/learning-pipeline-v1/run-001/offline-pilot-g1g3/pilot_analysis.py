#!/usr/bin/env python3
"""Offline pilot analysis for G1/G3 (novelty_design.md §5.2 and §6, revised).

Reads a VARE learning-pipeline rollout log (`rollouts/graded.jsonl`, schema
`vare.learning_pipeline.rollout/v1`) and replays budgeted trusted audits
without the model or any test execution. Linux box only; needs numpy.

  python3 pilot_analysis.py --graded rollouts/graded.jsonl --out-dir out/

Labels
  trusted t : every requested test in verifiers.trusted_full_v1.tests is PASSED (empty -> 0)
  cheap  c  : nat = every test in verifiers.cheap_f2p_v1.tests PASSED (separate F2P-only run);
              syn(rho0, rho1) = t flipped with known FP rate rho0 and FN rate rho1
Methods (trusted budget = 10% of rollouts, in expectation)
  B2 cheap_only    r = c
  B3 audit_only    uniform audits; only audited samples are trained on (r = t)
  B4 uniform_ht    uniform audits, r = c + a (t - c) / pi          (G1 alone)
  B5 uniform_rate  uniform audits; audited r = t, otherwise backward-corrected
                   c using rho estimated from the audits (S47-style)
  B6 adaptive_ht   phase 1: uniform pi_min; phase 2: pi2 proportional to
                   sensitivity * sqrt(u(1-u)) / cost, with u estimated leaving
                   out the sample's own group; HT with the combined inclusion
                   probability (G1 x G3)
  B6_no_s          B6 with sensitivity set to 1 (pre-registered ablation)
Primary metric (pre-registration v2): policy-gradient alignment
  cos(g_hat, g*) under a linear proxy: the GRPO update is sum_i A_i * grad log pi_i,
  and with the per-rollout score vectors treated as orthonormal (no model needed),
  cos(g_hat, g*) = <A_hat, A*> / (|A_hat| |A*|) over the whole rollout set.
  It is invariant to positive rescaling, so shrinking advantages toward 0 cannot
  raise it, and an all-zero estimate is defined as cos = 0.
  Expected over audit/flip seeds; CI from a bootstrap over task groups x seeds.
Secondary: MSE (pre-registration v1 primary; rewards shrinkage), MSE relative to the
zero-advantage baseline, sign accuracy on rollouts with A* != 0, RFT false accepts.
v1 -> v2 change was made before any real graded.jsonl was opened (see PREREG).

All thresholds are proposed values from novelty_design.md (labelled
【提案値】 there); they are not derived from data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCHEMA = "vare.learning_pipeline.rollout/v1"
CHEAP = "cheap_f2p_v1"
TRUSTED = "trusted_full_v1"
EPS = 1e-8

PREREG = {
    "version": "g1g3-offline-pilot-v2",
    "supersedes": "g1g3-offline-pilot-v1 sha256 90dd5a9e05b15880d1a2744e9956e6569df795a329924723d8b985ec47069bdd",
    "change_note": "primary metric changed from advantage MSE (minimised by shrinking advantages to 0; audit-only won on synthetic data) to linear-proxy gradient alignment cos(A_hat, A*); made before any real rollout log was opened",
    "primary_metric": "E_seed[cos(A_hat, A*)] pooled over all rollouts; cos := 0 if A_hat == 0",
    "secondary_metrics": ["mse", "mse_over_zero_baseline", "sign_accuracy_nonzero_A*", "rft_false_accept_rate"],
    "budget_fraction": 0.10,
    "pi_min": 0.05,
    "syn_settings": [[0.1, 0.1], [0.3, 0.1], [0.1, 0.3]],
    "n_seeds": 200,
    "n_boot": 2000,
    "ci": 0.95,
    "min_informative_groups": 8,
    "advantage": "(r - mean_g) / (std_g + 1e-8), population std, zero for constant groups",
    "rft_max_per_task": 2,
    "g1_supported": "dcos_G1 = cos(B4)-cos(B2): CI lower > 0 in all syn settings and in nat",
    "g1_rejected": "dcos_G1 CI upper < 0 in syn (0.1, 0.1)",
    "g1_weak_novelty": "cos(B4)-cos(B5) CI contains 0 in nat (reported as a flag only)",
    "g3_supported": "rel = ((1-cos B6)-(1-cos B4))/(1-cos B4) <= -0.10 (point) with CI upper < 0 in >= 2 syn settings, mean false-accept(B6) <= false-accept(B4) + 0.02, and rel(B6_no_s) > rel(B6) (point; ablation shrinks the gain)",
    "g3_rejected": "not supported and, in >= 2 syn settings, rel CI lower > -0.05 and CI upper >= 0",
    "g3_rel_threshold": -0.10,
    "g3_reject_margin": -0.05,
    "m3_tolerance": 0.02,
    "otherwise": "inconclusive",
}
PREREG_SHA256 = hashlib.sha256(json.dumps(PREREG, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ loading
class InputError(ValueError):
    pass


def all_passed(tests) -> int:
    return int(bool(tests) and all(v == "PASSED" for v in tests.values()))


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("schema") != SCHEMA:
                raise InputError(f"line {n}: schema {r.get('schema')!r} != {SCHEMA}")
            # Held-out or evaluation rows must never enter this analysis.
            if r.get("split") != "train" or r.get("stage") != "rollout":
                raise InputError(f"line {n}: split={r.get('split')!r} stage={r.get('stage')!r}; "
                                 "only train-split rollout rows are allowed")
            rows.append(r)
    if not rows:
        raise InputError("no rows")
    return rows


def build(rows: list[dict]) -> dict:
    """Group rows, derive labels and costs. Returns arrays sorted by group."""
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r)
    dropped = {g: len(v) for g, v in by_group.items() if len(v) < 2}
    groups = sorted(g for g in by_group if g not in dropped)
    keys = set()
    recs = []
    for gi, g in enumerate(groups):
        members = sorted(by_group[g], key=lambda r: r["sample_index"])
        ck = {(m["checkpoint_id"], m["checkpoint_step"]) for m in members}
        if len(ck) != 1:
            raise InputError(f"group {g} mixes checkpoints {ck}")
        for m in members:
            k = (g, m["sample_index"])
            if k in keys:
                raise InputError(f"duplicate sample {k}")
            keys.add(k)
            v = m.get("verifiers", {})
            tt = v.get(TRUSTED, {}).get("tests") or {}
            ct = v.get(CHEAP, {}).get("tests") or {}
            f2p_from_trusted = {t: tt[t] for t in ct if t in tt}
            recs.append(dict(
                gi=gi, group_id=g, instance_id=m["instance_id"], task_group=m.get("task_group"),
                sample_index=m["sample_index"], sha=m.get("output_sha256"),
                t=all_passed(tt), c=all_passed(ct),
                c_derived=all_passed(f2p_from_trusted) if ct else 0,
                reward=m.get("reward"), cheap_passed=bool(v.get(CHEAP, {}).get("passed")),
                cost_t=float(v.get(TRUSTED, {}).get("seconds") or 0.0),
                cost_c=float(v.get(CHEAP, {}).get("seconds") or 0.0),
            ))
    if not recs:
        raise InputError("no group with >= 2 samples")
    a = lambda k, dt=float: np.array([r[k] for r in recs], dtype=dt)
    gidx = a("gi", int)
    # Cost model for allocation: median trusted seconds of the sample's repository
    # (known from logs before an audit is chosen); 1.0 when no cost is logged.
    by_repo = defaultdict(list)
    for r in recs:
        by_repo[r["task_group"]].append(r["cost_t"])
    repo_cost = {k: float(np.median(v)) for k, v in by_repo.items()}
    kappa = np.array([repo_cost[r["task_group"]] or 1.0 for r in recs])
    return dict(recs=recs, gidx=gidx, n_groups=len(groups), t=a("t"), c_nat=a("c"),
                c_derived=a("c_derived"), kappa=np.maximum(kappa, 1e-3),
                dropped_groups=dropped, group_ids=groups)


# --------------------------------------------------------------- advantages
def advantages(r: np.ndarray, gidx: np.ndarray, n_groups: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Group-relative advantage; members with mask=False are excluded (adv 0)."""
    w = np.ones_like(r) if mask is None else mask.astype(float)
    cnt = np.bincount(gidx, weights=w, minlength=n_groups)
    s1 = np.bincount(gidx, weights=w * r, minlength=n_groups)
    s2 = np.bincount(gidx, weights=w * r * r, minlength=n_groups)
    mean = np.divide(s1, cnt, out=np.zeros(n_groups), where=cnt > 0)
    var = np.divide(s2, cnt, out=np.zeros(n_groups), where=cnt > 0) - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    adv = (r - mean[gidx]) / (std[gidx] + EPS)
    adv[std[gidx] < 1e-12] = 0.0
    return adv * w


def sensitivity(c: np.ndarray, gidx: np.ndarray, n_groups: int) -> np.ndarray:
    """L2 change of the group's advantage vector when sample i's cheap label flips."""
    base = advantages(c, gidx, n_groups)
    s = np.zeros_like(c)
    members = defaultdict(list)
    for i, g in enumerate(gidx):
        members[g].append(i)
    for g, idx in members.items():
        idx = np.array(idx)
        cg = c[idx]
        bg = base[idx]
        sub = np.zeros(len(idx), dtype=int)
        for j in range(len(idx)):
            f = cg.copy()
            f[j] = 1.0 - f[j]
            s[idx[j]] = float(np.linalg.norm(advantages(f, sub, 1) - bg))
    return s


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    """Primary metric kernel: cos(x, y); 0 when either vector is all zero."""
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / den) if den > 0 else 0.0


# ------------------------------------------------------------------ methods
def ht(c, t, audited, p):
    return c + audited * (t - c) / p


def rate_estimates(c, t, audited):
    """Beta(1,1) posterior means of rho0 = P(c=1|t=0), rho1 = P(c=0|t=1)."""
    a = audited.astype(bool)
    t0, t1 = a & (t == 0), a & (t == 1)
    rho0 = (np.sum(c[t0] == 1) + 1) / (t0.sum() + 2)
    rho1 = (np.sum(c[t1] == 0) + 1) / (t1.sum() + 2)
    return float(rho0), float(rho1)


def backward(c, rho0, rho1):
    den = max(1.0 - rho0 - rho1, 0.05)
    return np.where(c == 1, (1.0 - rho0) / den, -rho0 / den)


def loo_error_rate(c, t, audited, gidx, n_groups):
    """u_i = P(t != c | c_i) from phase-1 audits outside sample i's own group."""
    a = audited.astype(float)
    u = np.zeros_like(c)
    for cv in (0.0, 1.0):
        sel = a * (c == cv)
        err = sel * (t != c)
        n_g = np.bincount(gidx, weights=sel, minlength=n_groups)
        e_g = np.bincount(gidx, weights=err, minlength=n_groups)
        n_loo = n_g.sum() - n_g[gidx]
        e_loo = e_g.sum() - e_g[gidx]
        u = np.where(c == cv, (e_loo + 1.0) / (n_loo + 2.0), u)
    return u


def phase2_probs(score, budget, pi_min):
    """pi2 = min(1, lam*score) with sum of 1-(1-pi_min)(1-pi2) == budget (bisection)."""
    n = len(score)
    target = budget - n * pi_min
    if target <= 0 or score.sum() <= 0:
        return np.zeros(n)
    tot = lambda lam: float(np.sum((1 - pi_min) * np.minimum(1.0, lam * score)))
    lo, hi = 0.0, 1.0
    while tot(hi) < target and hi < 1e12:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if tot(mid) < target else (lo, mid)
    return np.minimum(1.0, hi * score)


def run_methods(c, t, gidx, n_groups, kappa, rng, budget_frac, pi_min):
    n = len(c)
    B = budget_frac * n
    out = {}
    out["B2_cheap_only"] = (c.copy(), None)
    pu = B / n
    a = (rng.random(n) < pu).astype(float)
    out["B3_audit_only"] = (t * a, a.astype(bool))
    out["B4_uniform_ht"] = (ht(c, t, a, pu), None)
    rho0, rho1 = rate_estimates(c, t, a)
    out["B5_uniform_rate"] = (np.where(a == 1, t, backward(c, rho0, rho1)), None)
    # B6: phase 1 uniform at pi_min, phase 2 adaptive (same random stream per arm design)
    a1 = (rng.random(n) < pi_min).astype(float)
    u = loo_error_rate(c, t, a1, gidx, n_groups)
    s = sensitivity(c, gidx, n_groups)
    unc = np.sqrt(u * (1 - u))
    v2 = rng.random(n)
    for name, sens in (("B6_adaptive_ht", s), ("B6_no_s", np.ones(n))):
        pi2 = phase2_probs(sens * unc / kappa, B, pi_min)
        p = 1.0 - (1.0 - pi_min) * (1.0 - pi2)
        aud = np.maximum(a1, (v2 < pi2).astype(float))
        out[name] = (ht(c, t, aud, p), None)
    return out


# -------------------------------------------------------------------- RFT
def accepted_purity(r_hat, t, recs, max_per_task, mask=None):
    """Admit r_hat >= 1 (audit-only: audited & t=1), at most k distinct outputs per task."""
    seen = defaultdict(set)
    acc = []
    for i in sorted(range(len(recs)), key=lambda i: (recs[i]["instance_id"], recs[i]["sample_index"])):
        if mask is not None and not mask[i]:
            continue
        if r_hat[i] < 1.0 - 1e-9:
            continue
        key = recs[i]["instance_id"]
        sha = recs[i]["sha"] or str(i)
        if sha in seen[key] or len(seen[key]) >= max_per_task:
            continue
        seen[key].add(sha)
        acc.append(i)
    if not acc:
        return float("nan"), 0
    return float(np.mean(t[acc])), len(acc)


# ------------------------------------------------------------------ analysis
def informative_groups(t, c, gidx, n_groups) -> int:
    def mixed(x):
        lo = np.full(n_groups, np.inf)
        hi = np.full(n_groups, -np.inf)
        np.minimum.at(lo, gidx, x)
        np.maximum.at(hi, gidx, x)
        return hi > lo
    return int(np.sum(mixed(t) | mixed(c)))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    m = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(m - h, 4), round(m + h, 4)]


def analyse_setting(data, setting, prereg, rng):
    t, gidx, G, kappa, recs = data["t"], data["gidx"], data["n_groups"], data["kappa"], data["recs"]
    n = len(t)
    S = prereg["n_seeds"]
    a_star = advantages(t, gidx, G)
    names = ["B2_cheap_only", "B3_audit_only", "B4_uniform_ht", "B5_uniform_rate", "B6_adaptive_ht", "B6_no_s"]
    sse = {m: np.zeros((S, G)) for m in names}
    dot = {m: np.zeros((S, G)) for m in names}
    hh = {m: np.zeros((S, G)) for m in names}
    sgn = {m: np.zeros((S, G)) for m in names}
    ss = np.bincount(gidx, weights=a_star ** 2, minlength=G)
    nz = a_star != 0
    n_nz = int(nz.sum())
    far = {m: [] for m in names}
    nacc = {m: [] for m in names}
    for s in range(S):
        if setting == "nat":
            c = data["c_nat"]
        else:
            rho0, rho1 = setting
            flip = rng.random(n)
            c = np.where(t == 1, (flip >= rho1).astype(float), (flip < rho0).astype(float))
        res = run_methods(c, t, gidx, G, kappa, rng, prereg["budget_fraction"], prereg["pi_min"])
        for m, (r, mask) in res.items():
            adv = advantages(r, gidx, G, mask)
            sse[m][s] = np.bincount(gidx, weights=(adv - a_star) ** 2, minlength=G)
            dot[m][s] = np.bincount(gidx, weights=adv * a_star, minlength=G)
            hh[m][s] = np.bincount(gidx, weights=adv * adv, minlength=G)
            sgn[m][s] = np.bincount(gidx, weights=(nz & (np.sign(adv) == np.sign(a_star))).astype(float), minlength=G)
            pur, k = accepted_purity(r, t, recs, prereg["rft_max_per_task"], mask)
            far[m].append(1 - pur if k else float("nan"))
            nacc[m].append(k)
    sizes = np.bincount(gidx, minlength=G).astype(float)
    B = prereg["n_boot"]
    alpha = (1 - prereg["ci"]) / 2
    wg = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(B)])  # (B, G)
    ws = rng.integers(0, S, (B, S))
    nb = wg @ sizes

    def boot_mse(m):
        tot = sse[m] @ wg.T  # (S, B)
        return np.take_along_axis(tot.T, ws, axis=1).mean(axis=1) / nb

    ss_b = wg @ ss  # (B,)

    def boot_cos(m):
        d = (dot[m] @ wg.T)  # (S, B)
        h = (hh[m] @ wg.T)
        den = np.sqrt(h * ss_b[None, :])
        cs = np.divide(d, den, out=np.zeros_like(d), where=den > 0)
        return np.take_along_axis(cs.T, ws, axis=1).mean(axis=1)  # (B,)

    cos_b = {m: boot_cos(m) for m in names}
    ss_tot = float(ss.sum())

    def cos_point(m):
        d, h = dot[m].sum(axis=1), hh[m].sum(axis=1)
        den = np.sqrt(h * ss_tot)
        return float(np.mean(np.divide(d, den, out=np.zeros_like(d), where=den > 0)))

    cosp = {m: cos_point(m) for m in names}
    point = {m: float(sse[m].sum(axis=1).mean() / n) for m in names}

    def summ(d, p):
        lo, hi = np.quantile(d, [alpha, 1 - alpha])
        return {"point": round(float(p), 6), "ci": [round(float(lo), 6), round(float(hi), 6)]}

    def dcos(m1, m2):
        return summ(cos_b[m1] - cos_b[m2], cosp[m1] - cosp[m2])

    def rel_mis(m1, m2):
        """Relative change of misalignment 1-cos of m1 vs m2 (negative = m1 better)."""
        d = ((1 - cos_b[m1]) - (1 - cos_b[m2])) / np.maximum(1 - cos_b[m2], 1e-12)
        p = ((1 - cosp[m1]) - (1 - cosp[m2])) / max(1 - cosp[m2], 1e-12)
        return summ(d, p)

    zero_mse = float(np.sum(a_star ** 2) / n)
    return {
        "cos": {m: round(v, 6) for m, v in cosp.items()},
        "cos_ci": {m: [round(float(x), 6) for x in np.quantile(cos_b[m], [alpha, 1 - alpha])] for m in names},
        "dcos_G1_B4_minus_B2": dcos("B4_uniform_ht", "B2_cheap_only"),
        "dcos_B4_minus_B5": dcos("B4_uniform_ht", "B5_uniform_rate"),
        "dcos_B4_minus_B3": dcos("B4_uniform_ht", "B3_audit_only"),
        "rel_G3_B6_vs_B4": rel_mis("B6_adaptive_ht", "B4_uniform_ht"),
        "rel_G3_B6_no_s_vs_B4": rel_mis("B6_no_s", "B4_uniform_ht"),
        "secondary": {
            "mse": {m: round(v, 6) for m, v in point.items()},
            "mse_zero_advantage_reference": round(zero_mse, 6),
            "mse_over_zero_baseline": {m: (round(v / zero_mse, 4) if zero_mse > 0 else None) for m, v in point.items()},
            "sign_accuracy_nonzero_A*": {m: (round(float(sgn[m].sum(axis=1).mean() / n_nz), 4) if n_nz else None)
                                         for m in names},
        },
        "rft_false_accept_rate_mean": {m: (None if np.all(np.isnan(v)) else round(float(np.nanmean(v)), 4))
                                       for m, v in far.items()},
        "rft_accepted_mean": {m: round(float(np.mean(v)), 2) for m, v in nacc.items()},
    }


def verdicts(per, prereg, abort):
    if abort:
        reason = "informative groups below the pre-registered minimum"
        return {"G1": "inconclusive", "G3": "inconclusive", "reason": reason}
    syn = [k for k in per if k != "nat"]
    g1_sup = all(per[k]["dcos_G1_B4_minus_B2"]["ci"][0] > 0 for k in syn + ["nat"])
    base = "syn_0.1_0.1"
    g1_rej = base in per and per[base]["dcos_G1_B4_minus_B2"]["ci"][1] < 0
    G1 = "supported" if g1_sup else "rejected" if g1_rej else "inconclusive"
    sup_n = 0
    rej_n = 0
    for k in syn:
        r = per[k]["rel_G3_B6_vs_B4"]
        if r["point"] <= prereg["g3_rel_threshold"] and r["ci"][1] < 0:
            sup_n += 1
        if r["ci"][0] > prereg["g3_reject_margin"] and r["ci"][1] >= 0:
            rej_n += 1
    far6 = np.nanmean([per[k]["rft_false_accept_rate_mean"]["B6_adaptive_ht"] or 0 for k in syn])
    far4 = np.nanmean([per[k]["rft_false_accept_rate_mean"]["B4_uniform_ht"] or 0 for k in syn])
    m3_ok = far6 <= far4 + prereg["m3_tolerance"]
    abl_ok = np.mean([per[k]["rel_G3_B6_no_s_vs_B4"]["point"] > per[k]["rel_G3_B6_vs_B4"]["point"]
                      for k in syn]) >= 0.5
    g3_sup = sup_n >= 2 and m3_ok and abl_ok
    G3 = "supported" if g3_sup else "rejected" if rej_n >= 2 else "inconclusive"
    weak = per.get("nat", {}).get("dcos_B4_minus_B5", {}).get("ci", [None, None])
    flag = weak[0] is not None and weak[0] <= 0 <= weak[1]
    return {"G1": G1, "G3": G3, "g1_weak_novelty_flag": bool(flag),
            "g3_settings_supported": sup_n, "g3_settings_rejecting": rej_n,
            "g3_m3_ok": bool(m3_ok), "g3_ablation_ok": bool(abl_ok)}


def analyse(path: Path, prereg=PREREG, seed=20261031) -> dict:
    raw = Path(path).read_bytes()
    rows = load_rows(path)
    data = build(rows)
    t, c, gidx, G = data["t"], data["c_nat"], data["gidx"], data["n_groups"]
    inf = informative_groups(t, c, gidx, G)
    abort = inf < prereg["min_informative_groups"]
    fp = int(np.sum((c == 1) & (t == 0)))
    fn = int(np.sum((c == 0) & (t == 1)))
    n0, n1 = int(np.sum(t == 0)), int(np.sum(t == 1))
    recs = data["recs"]
    summary = {
        "prereg_sha256": PREREG_SHA256,
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "n_rollouts": len(t), "n_groups": G, "group_sizes": sorted(set(np.bincount(gidx).tolist())),
        "dropped_groups_lt2": len(data["dropped_groups"]),
        "n_repos": len({r["task_group"] for r in recs}),
        "trusted_pass": n1, "cheap_nat_pass": int(c.sum()),
        "nat_false_pos": fp, "nat_false_neg": fn,
        "nat_rho0": round(fp / n0, 4) if n0 else None, "nat_rho0_wilson95": wilson(fp, n0),
        "nat_rho1": round(fn / n1, 4) if n1 else None, "nat_rho1_wilson95": wilson(fn, n1),
        "cheap_direct_vs_derived_disagree": int(np.sum(c != data["c_derived"])),
        "reward_vs_trusted_tests_mismatch": int(sum(1 for r in recs if r["reward"] is not None and int(r["reward"]) != r["t"])),
        "cheap_passed_vs_tests_mismatch": int(sum(1 for r in recs if r["cheap_passed"] != bool(r["c"]))),
        "informative_groups": inf, "abort": bool(abort),
        "kappa_trusted_median_s": round(float(np.median(data["kappa"])), 3),
    }
    per = {}
    if not abort:
        rng = np.random.default_rng(seed)
        for st in prereg["syn_settings"]:
            per[f"syn_{st[0]}_{st[1]}"] = analyse_setting(data, tuple(st), prereg, rng)
        per["nat"] = analyse_setting(data, "nat", prereg, rng)
    return {"summary": summary, "settings": per, "verdict": verdicts(per, prereg, abort), "prereg": prereg}


def to_markdown(res: dict) -> str:
    s, v = res["summary"], res["verdict"]
    L = ["# G1/G3 offline pilot verdict", "",
         f"- G1 (budgeted HT correction): **{v['G1']}**",
         f"- G3 (sensitivity-based allocation): **{v['G3']}**", "",
         f"- rollouts {s['n_rollouts']}, groups {s['n_groups']}, informative groups {s['informative_groups']}"
         f" (minimum {res['prereg']['min_informative_groups']}){' -> ABORT' if s['abort'] else ''}",
         f"- trusted pass {s['trusted_pass']}, cheap(F2P) pass {s['cheap_nat_pass']}, "
         f"FP {s['nat_false_pos']} (rho0 {s['nat_rho0']}, Wilson95 {s['nat_rho0_wilson95']}), "
         f"FN {s['nat_false_neg']} (rho1 {s['nat_rho1']})",
         f"- prereg sha256 `{s['prereg_sha256'][:16]}`, input sha256 `{s['input_sha256'][:16]}`", ""]
    if res["settings"]:
        L += ["Primary: cos(A_hat, A*) (linear-proxy gradient alignment; scale-invariant).", "",
              "| setting | cos B2 / B3 / B4 / B5 / B6 | dcos G1 (B4-B2) [95% CI] | rel misalign G3 (B6 vs B4) [95% CI] | FA rate B4 / B6 |",
              "|---|---|---|---|---|"]
        for k, r in res["settings"].items():
            d, g, c = r["dcos_G1_B4_minus_B2"], r["rel_G3_B6_vs_B4"], r["cos"]
            fa = r["rft_false_accept_rate_mean"]
            cs = " / ".join(f"{c[m]:.3f}" for m in ("B2_cheap_only", "B3_audit_only", "B4_uniform_ht", "B5_uniform_rate", "B6_adaptive_ht"))
            L.append(f"| {k} | {cs} | {d['point']:.4f} [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}] | "
                     f"{g['point']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}] | {fa['B4_uniform_ht']} / {fa['B6_adaptive_ht']} |")
        L += ["", "Secondary (not used for G1/G3 verdicts): MSE / zero-advantage MSE, B3 vs B4: "
              + "; ".join(f"{k} {r['secondary']['mse_over_zero_baseline']['B3_audit_only']} vs "
                          f"{r['secondary']['mse_over_zero_baseline']['B4_uniform_ht']}" for k, r in res["settings"].items())]
        L += [f"- G1 weak-novelty flag (B4 vs B5 CI contains 0, nat): {v.get('g1_weak_novelty_flag')}"]
    L += ["", f"Pre-registration {res['prereg']['version']} (primary metric changed from MSE before any real log was opened). "
          "Thresholds are pre-registered proposed values (novelty_design.md §5.2); "
          "small samples are expected to read as inconclusive. Offline replay only: no online effect is claimed."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graded", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--n-seeds", type=int, default=None, help="override (smoke tests only; breaks prereg)")
    ap.add_argument("--n-boot", type=int, default=None, help="override (smoke tests only; breaks prereg)")
    a = ap.parse_args(argv)
    prereg = dict(PREREG)
    if a.n_seeds or a.n_boot:
        prereg.update({k: v for k, v in (("n_seeds", a.n_seeds), ("n_boot", a.n_boot)) if v})
        print("WARNING: non-preregistered seed/boot counts", file=sys.stderr)
    try:
        res = analyse(a.graded, prereg, a.seed)
    except InputError as e:
        print(f"input rejected: {e}", file=sys.stderr)
        return 2
    res["summary"]["prereg_overridden"] = prereg != PREREG
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "pilot_result.json").write_text(json.dumps(res, indent=1))
    (a.out_dir / "pilot_verdict.md").write_text(to_markdown(res))
    print(json.dumps(res["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
