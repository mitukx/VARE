#!/usr/bin/env python3
"""Independent re-derivation of the run-001 statistics from the raw graded logs.

Written without importing or copying ``analysis/stats.py``,
``vare.eval_stats`` or ``vare.learning_pipeline.comparison``. Standard library
only. The verdict of each record is recomputed from its per-test statuses;
the ``reward`` field is only compared with it, never used as the source.

Inputs (repository-relative to this directory):
  ../eval/{no_update,sft_reference,rft_self}/graded.jsonl   held-out outputs (already graded)
  ../calibration/admitted.json                              admitted task lists
  ../SHA256SUMS.txt                                         recorded hashes
  --rollouts PATH                                           hash-only train rollout log (optional)
  ../data/rft_self/{train,valid}.provenance.jsonl           RFT dataset provenance

Usage: python reanalyze.py [--rollouts ../rollouts/graded.jsonl] --out reanalysis.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ARMS = ("no_update", "sft_reference", "rft_self")
TRUSTED, CHEAP = "trusted_full_v1", "cheap_f2p_v1"
Z = 1.959963984540054  # standard normal 0.975 quantile


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def recorded_hashes() -> dict[str, str]:
    out = {}
    for line in (RUN / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            out[parts[3]] = parts[0]
    return out


def verdict(rec: dict) -> int:
    """1 iff the trusted verifier executed and every requested test PASSED."""
    v = rec.get("verifiers", {}).get(TRUSTED, {})
    tests = v.get("tests") or {}
    return int(bool(tests) and all(s == "PASSED" for s in tests.values())
               and v.get("f2p_passed") == v.get("f2p_total")
               and v.get("p2p_passed") == v.get("p2p_total"))


def cheap_verdict(rec: dict) -> int:
    tests = rec.get("verifiers", {}).get(CHEAP, {}).get("tests") or {}
    return int(bool(tests) and all(s == "PASSED" for s in tests.values()))


def binom_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n


def mcnemar(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"two_sided_exact": 1.0, "mid_p": 1.0, "one_sided_b_gt_c": 1.0}
    k = min(b, c)
    exact = min(1.0, 2 * binom_cdf(k, n))
    pk = math.comb(n, k) / 2 ** n
    mid = min(1.0, 2 * (binom_cdf(k, n) - pk) + pk)
    one = 1 - binom_cdf(b - 1, n) if b > 0 else 1.0  # P(X >= b), X ~ Bin(n, 1/2)
    return {"two_sided_exact": exact, "mid_p": mid, "one_sided_b_gt_c": one}


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    m, out, running = len(ps), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * ps[k]))
        out[k] = running
    return out


def wilson(x: int, n: int) -> tuple[float, float]:
    p = x / n
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return centre - half, centre + half


def newcombe_paired(a: int, b: int, c: int, d: int) -> dict:
    """Newcombe (1998) method 10 for p_arm - p_base; a=both, b=arm only, c=base only, d=neither."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    out = {}
    for name, corrected in (("phi_uncorrected", False), ("phi_corrected", True)):
        den = (a + b) * (c + d) * (a + c) * (b + d)
        num = a * d - b * c
        if corrected and num > 0:
            num = max(num - n / 2, 0)
        phi = num / math.sqrt(den) if den else 0.0
        diff = p1 - p2
        lo = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
        hi = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
        out[name] = [lo, hi]
    return out


def bootstrap(arm: list[int], base: list[int], *, reps: int = 10000, seed: int = 20261031) -> list[float]:
    rng = random.Random(seed)
    n = len(arm)
    diffs = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(arm[i] - base[i] for i in idx) / n)
    diffs.sort()
    return [diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1]]


def pair(x: dict[str, int], y: dict[str, int], ids: list[str]) -> dict:
    a = sum(x[t] and y[t] for t in ids)
    b = sum(x[t] and not y[t] for t in ids)
    c = sum(y[t] and not x[t] for t in ids)
    d = len(ids) - a - b - c
    return {"both": a, "first_only": b, "second_only": c, "neither": d,
            "first_only_tasks": sorted(t for t in ids if x[t] and not y[t]),
            "second_only_tasks": sorted(t for t in ids if y[t] and not x[t])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sums = recorded_hashes()
    checks: dict[str, object] = {}
    admitted = json.loads((RUN / "calibration" / "admitted.json").read_text())
    heldout, train = list(admitted["heldout"]), list(admitted["train"])
    checks["admitted_sha256_matches"] = sha256(RUN / "calibration" / "admitted.json") == sums.get("calibration/admitted.json")
    checks["n_heldout_admitted"] = len(heldout)
    checks["n_train_admitted"] = len(train)
    checks["train_heldout_ids_disjoint"] = not (set(heldout) & set(train))
    repo_of = lambda t: t.split(".")[0]  # noqa: E731 - mirror name before the commit suffix
    checks["train_heldout_repos_disjoint"] = not ({repo_of(t) for t in heldout} & {repo_of(t) for t in train})

    outcome: dict[str, dict[str, int]] = {}
    arms: dict[str, dict] = {}
    for arm in ARMS:
        p = RUN / "eval" / arm / "graded.jsonl"
        rows = read_jsonl(p)
        k0 = [r for r in rows if r["sample_index"] == 0]
        ids = [r["instance_id"] for r in k0]
        mism = [r["instance_id"] for r in rows if verdict(r) != r["reward"]]
        sha_bad = [r["instance_id"] for r in rows
                   if hashlib.sha256(r["output_text"].encode()).hexdigest() != r["output_sha256"]]
        outcome[arm] = {r["instance_id"]: verdict(r) for r in k0}
        x = sum(outcome[arm].values())
        groups: dict[str, list[int]] = defaultdict(list)
        for r in k0:
            groups[r["task_group"]].append(verdict(r))
        arms[arm] = {
            "file_sha256_matches_SHA256SUMS": sha256(p) == sums.get(f"eval/{arm}/graded.jsonl"),
            "rows": len(rows), "rows_sample_index_0": len(k0),
            "ids_equal_admitted_heldout": sorted(ids) == sorted(heldout) and len(set(ids)) == len(ids),
            "split_values": sorted({r["split"] for r in rows}),
            "reward_vs_recomputed_verdict_mismatches": mism,
            "output_sha256_mismatches": sha_bad,
            "checkpoint_ids": sorted({r["checkpoint_id"] for r in rows}),
            "pass": x, "rate": x / len(k0), "wilson95": list(wilson(x, len(k0))),
            "cheap_pass": sum(cheap_verdict(r) for r in k0),
            "cheap_false_positive": sum(cheap_verdict(r) and not verdict(r) for r in k0),
            "outcomes": dict(Counter(r["outcome"] for r in k0)),
            "finish": dict(Counter(r["finish"] for r in k0)),
            "per_task_group": {g: [sum(v), len(v)] for g, v in sorted(groups.items())},
        }
    comps: dict[str, dict] = {}
    for arm in ("sft_reference", "rft_self"):
        t = pair(outcome[arm], outcome["no_update"], heldout)
        a, b, c, d = t["both"], t["first_only"], t["second_only"], t["neither"]
        av = [outcome[arm][i] for i in heldout]
        bv = [outcome["no_update"][i] for i in heldout]
        comps[arm] = {"table_arm_vs_base": t, "diff": (b - c) / len(heldout), "mcnemar": mcnemar(b, c),
                      "newcombe95": newcombe_paired(a, b, c, d), "bootstrap95_own_rng": bootstrap(av, bv)}
    h = holm({k: v["mcnemar"]["two_sided_exact"] for k, v in comps.items()})
    for k in comps:
        comps[k]["holm"] = h[k]
    t = pair(outcome["rft_self"], outcome["sft_reference"], heldout)
    exploratory = {"rft_vs_sft": {"rft_only": t["first_only"], "sft_only": t["second_only"],
                                  "mcnemar": mcnemar(t["first_only"], t["second_only"])}}

    train_part: dict[str, object] = {}
    if args.rollouts:
        rows = read_jsonl(args.rollouts)
        train_part["sha256_matches_SHA256SUMS"] = sha256(args.rollouts) == sums.get("rollouts/graded.jsonl")
        train_part["samples"] = len(rows)
        train_part["tasks"] = len({r["instance_id"] for r in rows})
        train_part["all_in_admitted_train"] = all(r["instance_id"] in set(train) for r in rows)
        train_part["reward_vs_recomputed_verdict_mismatches"] = sum(verdict(r) != r["reward"] for r in rows)
        train_part["trusted_pass"] = sum(verdict(r) for r in rows)
        g: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            g[r["instance_id"]].append(verdict(r))
        train_part["tasks_with_pass"] = sum(any(v) for v in g.values())
        train_part["tasks_all_pass"] = sum(all(v) for v in g.values())
        train_part["mixed_groups"] = sum(0 < sum(v) < len(v) for v in g.values())
        train_part["cheap_false_positive"] = sum(cheap_verdict(r) and not verdict(r) for r in rows)
        train_part["cheap_false_negative"] = sum(verdict(r) and not cheap_verdict(r) for r in rows)
        # RFT admission restated from the protocol: trusted pass, at most 2 distinct
        # outputs per task, lowest sample_index first.
        chosen = []
        for tid in sorted(g):
            seen: list[str] = []
            for r in sorted((r for r in rows if r["instance_id"] == tid and verdict(r)),
                            key=lambda r: r["sample_index"]):
                if r["output_sha256"] not in seen and len(seen) < 2:
                    seen.append(r["output_sha256"])
                    chosen.append((tid, r["sample_index"], r["output_sha256"]))
        prov = [(p["instance_id"], p["source_sample_index"], p["source_output_sha256"])
                for part in ("train", "valid")
                for p in read_jsonl(RUN / "data" / "rft_self" / f"{part}.provenance.jsonl")]
        train_part["rft_examples_recomputed"] = len(chosen)
        train_part["rft_tasks_recomputed"] = len({c[0] for c in chosen})
        train_part["rft_set_equals_provenance"] = sorted(chosen) == sorted(prov)
        train_part["rft_provenance_rows"] = len(prov)
        train_part["rft_heldout_overlap"] = sorted({c[0] for c in prov} & set(heldout))

    out = {"schema": "run001_independent_reanalysis_v1", "checks": checks, "arms": arms,
           "preregistered_comparisons_vs_no_update": comps, "exploratory": exploratory,
           "train_rollouts": train_part}
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    brief = {a: (arms[a]["pass"], round(arms[a]["rate"], 4)) for a in ARMS}
    print(json.dumps({"pass": brief, "rft": comps["rft_self"]["table_arm_vs_base"]["first_only"],
                      "p": comps["rft_self"]["mcnemar"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
