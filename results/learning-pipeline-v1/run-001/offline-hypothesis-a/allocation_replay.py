#!/usr/bin/env python3
"""Offline replay of trusted-verification allocation (hypothesis A) on run-001 train rollouts.

See PROTOCOL.md. Standard library only. Allocation policies see only label-free fields, or labels
that the replay has revealed after a unit of budget was spent on them.

Usage:
  python allocation_replay.py --rollouts ../rollouts/graded.jsonl --stage dev --out dev.json
  python allocation_replay.py --rollouts ../rollouts/graded.jsonl --stage confirm \
      --proxy <from dev> --variant <from dev> --out confirm.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from typing import Callable

ROLLOUTS_SHA256 = "cb18ed4a"  # prefix check; full value is in ../SHA256SUMS.txt
CAP = 2  # frozen rft.max_examples_per_task
SEEDS = 400
BUDGET_FRACTIONS = (0.10, 0.20, 0.40)
PRIMARY = 0.20
PROXIES = ("agreement", "small_diff", "short")
VARIANTS = ("exploit", "explore")


@dataclass(frozen=True)
class Unit:
    """One trusted run: a distinct applied candidate within a task (label hidden until graded)."""

    task: str
    key: str
    first_index: int
    agreement: int
    diff_lines: int
    stop: bool
    tokens: int
    # hidden
    passed: bool
    output_shas: tuple[str, ...]


def split_of(task: str) -> str:
    return "confirm" if int(hashlib.sha256(f"hypA-split-v1:{task}".encode()).hexdigest(), 16) % 2 else "dev"


def label(r: dict) -> bool:
    v = r["verifiers"]["trusted_full_v1"]
    t = v.get("tests") or {}
    return bool(t) and all(s == "PASSED" for s in t.values())


def build_units(rows: list[dict]) -> tuple[dict[str, list[Unit]], dict[str, list[bool]]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["instance_id"]].append(r)
    units: dict[str, list[Unit]] = {}
    group_labels: dict[str, list[bool]] = {}
    for task, rs in sorted(by_task.items()):
        rs.sort(key=lambda r: r["sample_index"])
        group_labels[task] = [label(r) for r in rs]
        applied = [r for r in rs if r["apply"] == "applied"]
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in applied:
            groups[r["candidate_sha256"]].append(r)
        out = []
        for key, members in groups.items():
            labs = {label(m) for m in members}
            if len(labs) != 1:
                raise SystemExit(f"non-deterministic verdict for identical candidate in {task}")
            m0 = members[0]
            diff = m0.get("final_diff") or ""
            out.append(Unit(
                task=task, key=key, first_index=m0["sample_index"], agreement=len(members),
                diff_lines=sum(1 for ln in diff.splitlines()
                               if ln[:1] in "+-" and not ln.startswith(("+++", "---"))),
                stop=m0["finish"] == "stop", tokens=m0["completion_tokens"],
                passed=labs.pop(), output_shas=tuple(sorted({m["output_sha256"] for m in members})),
            ))
        units[task] = out
    return units, group_labels


def confidence(u: Unit, proxy: str) -> tuple:
    if proxy == "agreement":
        return (-u.agreement, u.first_index)
    if proxy == "small_diff":
        return (u.diff_lines, u.first_index)
    if proxy == "short":
        return (not u.stop, u.tokens, u.first_index)
    raise ValueError(proxy)


Policy = Callable[[dict[str, list[Unit]], int, random.Random], list[Unit]]


def _shuffled(xs: list, rng: random.Random) -> list:
    xs = list(xs)
    rng.shuffle(xs)
    return xs


def p_random(units, budget, rng):
    return _shuffled([u for us in units.values() for u in us], rng)[:budget]


def p_uniform(units, budget, rng):
    queues = {t: _shuffled(us, rng) for t, us in units.items()}
    order = _shuffled(list(queues), rng)
    out: list[Unit] = []
    while len(out) < budget and any(queues.values()):
        for t in order:
            if queues[t] and len(out) < budget:
                out.append(queues[t].pop(0))
    return out


def p_deterministic(units, budget, rng):
    return [u for t in sorted(units) for u in sorted(units[t], key=lambda u: u.first_index)][:budget]


def make_confidence(proxy: str) -> Policy:
    def p(units, budget, rng):
        allu = _shuffled([u for us in units.values() for u in us], rng)
        return sorted(allu, key=lambda u: confidence(u, proxy))[:budget]
    return p


def p_uncertainty(units, budget, rng):
    order = sorted(_shuffled(list(units), rng), key=lambda t: -len(units[t]))
    return [u for t in order for u in _shuffled(units[t], rng)][:budget]


def make_selective(proxy: str, variant: str) -> Policy:
    def p(units, budget, rng):
        queues = {t: sorted(_shuffled(us, rng), key=lambda u: confidence(u, proxy)) for t, us in units.items()}
        found: dict[str, set[str]] = defaultdict(set)
        out: list[Unit] = []

        def take(t: str) -> None:
            u = queues[t].pop(0)
            out.append(u)
            if u.passed:  # revealed after spending budget
                found[t].update(u.output_shas)

        first = sorted(_shuffled([t for t in queues if queues[t]], rng),
                       key=lambda t: confidence(queues[t][0], proxy))
        for t in first:
            if len(out) >= budget:
                return out
            take(t)
        while len(out) < budget:
            open_tasks = [t for t in queues if queues[t] and len(found[t]) < CAP]
            if not open_tasks:
                break
            has = 0 if variant == "exploit" else 1
            open_tasks = sorted(_shuffled(open_tasks, rng),
                                key=lambda t: ((has if found[t] else 1 - has), confidence(queues[t][0], proxy)))
            take(open_tasks[0])
        return out
    return p


def yield_of(graded: list[Unit]) -> dict:
    found: dict[str, set[str]] = defaultdict(set)
    for u in graded:
        if u.passed:
            found[u.task].update(u.output_shas)
    return {"rft": sum(min(CAP, len(s)) for s in found.values()), "tasks": len(found)}


def mixed_resolved(graded: list[Unit], units, group_labels) -> int:
    seen = {(u.task, u.key) for u in graded}
    n = 0
    for t, us in units.items():
        labs = group_labels[t]
        if all((t, u.key) in seen for u in us) and 0 < sum(labs) < len(labs):
            n += 1
    return n


def evaluate(units, group_labels, policy: Policy, budget: int, seeds: int = SEEDS) -> dict:
    res = []
    for s in range(seeds):
        g = policy(units, budget, random.Random(s))
        assert len(g) <= budget
        y = yield_of(g)
        y["mixed_resolved"] = mixed_resolved(g, units, group_labels)
        res.append(y)
    return {k: mean(r[k] for r in res) for k in res[0]}


def full_info(units, group_labels) -> dict:
    allu = [u for us in units.values() for u in us]
    y = yield_of(allu)
    y["mixed_groups"] = sum(0 < sum(v) < len(v) for v in group_labels.values())
    y["units"] = len(allu)
    return y


def baselines(proxy: str) -> dict[str, Policy]:
    return {"random": p_random, "uniform": p_uniform, "deterministic": p_deterministic,
            "confidence": make_confidence(proxy), "uncertainty": p_uncertainty}


def budgets(n_units: int) -> dict[str, int]:
    return {f"{f:.2f}": max(1, round(f * n_units)) for f in BUDGET_FRACTIONS}


def run_stage(units, group_labels, proxy_list, variant_list) -> dict:
    full = full_info(units, group_labels)
    out = {"full_information": full, "budgets": budgets(full["units"]), "results": {}}
    for bname, b in out["budgets"].items():
        row = {}
        for proxy in proxy_list:
            for name, pol in baselines(proxy).items():
                if name != "confidence" and proxy != proxy_list[0]:
                    continue
                key = f"confidence[{proxy}]" if name == "confidence" else name
                row[key] = evaluate(units, group_labels, pol, b)
            for v in variant_list:
                row[f"selective[{proxy},{v}]"] = evaluate(units, group_labels, make_selective(proxy, v), b)
        for k in row:
            row[k]["rft_recall"] = row[k]["rft"] / full["rft"] if full["rft"] else None
        out["results"][bname] = row
    return out


def bootstrap_delta(units, group_labels, proxy, variant, budget_frac, reps, seeds) -> dict:
    tasks = sorted(units)
    rng = random.Random(20261031)
    deltas = []
    for _ in range(reps):
        pick = [rng.choice(tasks) for _ in tasks]
        bu: dict[str, list[Unit]] = {}
        bl: dict[str, list[bool]] = {}
        for i, t in enumerate(pick):
            nt = f"{t}#{i}"
            bu[nt] = [Unit(**{**u.__dict__, "task": nt}) for u in units[t]]
            bl[nt] = group_labels[t]
        full = full_info(bu, bl)
        if full["rft"] == 0:
            deltas.append(0.0)
            continue
        b = max(1, round(budget_frac * full["units"]))
        sel = evaluate(bu, bl, make_selective(proxy, variant), b, seeds)["rft"]
        best = max(evaluate(bu, bl, pol, b, seeds)["rft"] for pol in baselines(proxy).values())
        deltas.append((sel - best) / full["rft"])
    deltas.sort()
    return {"reps": reps, "seeds_per_rep": seeds, "ci95": [deltas[int(0.025 * reps)], deltas[int(0.975 * reps) - 1]]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--stage", choices=("dev", "confirm"), required=True)
    ap.add_argument("--proxy", choices=PROXIES)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    digest = hashlib.sha256(a.rollouts.read_bytes()).hexdigest()
    if not digest.startswith(ROLLOUTS_SHA256):
        raise SystemExit("rollout log hash mismatch")
    rows = [json.loads(x) for x in a.rollouts.read_text().splitlines() if x.strip()]
    if any(r["split"] != "train" for r in rows):
        raise SystemExit("non-train record in rollout log")
    units, labels = build_units(rows)
    part = {t: us for t, us in units.items() if split_of(t) == a.stage}
    plabels = {t: labels[t] for t in part}
    rec: dict = {"schema": "hypothesis_a_offline_replay_v1", "stage": a.stage, "rollouts_sha256": digest,
                 "tasks": len(part), "seeds": SEEDS}
    # Regime C context: does the cheap verifier ever disagree with the trusted label?
    rec["regime_C"] = {
        "cheap_trusted_disagreements": sum(
            bool((r["verifiers"]["cheap_f2p_v1"].get("tests") or {}) and all(
                s == "PASSED" for s in r["verifiers"]["cheap_f2p_v1"]["tests"].values())) != label(r)
            for r in rows if split_of(r["instance_id"]) == a.stage),
    }
    if a.stage == "dev":
        rec.update(run_stage(part, plabels, list(PROXIES), list(VARIANTS)))
    else:
        if not (a.proxy and a.variant):
            raise SystemExit("--proxy and --variant (selected on dev) are required for confirm")
        rec["selected_on_dev"] = {"proxy": a.proxy, "variant": a.variant}
        rec.update(run_stage(part, plabels, [a.proxy], [a.variant]))
        prim = rec["results"][f"{PRIMARY:.2f}"]
        sel = prim[f"selective[{a.proxy},{a.variant}]"]["rft_recall"]
        best_name = max((k for k in prim if not k.startswith("selective")), key=lambda k: prim[k]["rft_recall"])
        delta = sel - prim[best_name]["rft_recall"]
        bs = bootstrap_delta(part, plabels, a.proxy, a.variant, PRIMARY, reps=2000, seeds=40)
        lo, hi = bs["ci95"]
        verdict = ("supported_offline" if delta >= 0.10 and lo > 0
                   else "not_supported" if hi < 0.10 else "inconclusive")
        rec["decision"] = {"primary_budget": PRIMARY, "selective_recall": sel, "best_baseline": best_name,
                           "best_baseline_recall": prim[best_name]["rft_recall"], "delta": delta,
                           "bootstrap": bs, "verdict": verdict}
    cost = {"generation_seconds": sum(r["generation_seconds"] for r in rows),
            "trusted_seconds": sum(r["verifiers"]["trusted_full_v1"].get("seconds") or 0 for r in rows),
            "cheap_seconds": sum(r["verifiers"]["cheap_f2p_v1"].get("seconds") or 0 for r in rows)}
    cost["trusted_share_of_generation_plus_grading"] = cost["trusted_seconds"] / sum(cost.values())
    rec["cost_all_160"] = cost
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
