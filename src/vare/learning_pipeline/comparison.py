"""Paired held-out comparison of trained arms against the no-update arm.

Primary metric: per-task trusted pass (``trusted_full_v1``), one greedy
sample per task, identical budgets. Tests: exact McNemar per comparison,
Holm-adjusted over the family of trained-vs-base comparisons, paired
bootstrap interval on the pass-rate difference. The decision rule is frozen
in the protocol (``success_criteria``).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from vare.eval_stats import mcnemar_exact, mcnemar_power, paired_bootstrap_interval


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def paired(base: Mapping[str, int], arm: Mapping[str, int], task_ids: Sequence[str]) -> dict[str, Any]:
    left = [int(base[t]) for t in task_ids]
    right = [int(arm[t]) for t in task_ids]
    b = sum(1 for l, r in zip(left, right) if l == 0 and r == 1)  # arm fixes, base did not
    c = sum(1 for l, r in zip(left, right) if l == 1 and r == 0)
    return {
        "n": len(task_ids),
        "base_pass": sum(left),
        "arm_pass": sum(right),
        "base_rate": sum(left) / len(left),
        "arm_rate": sum(right) / len(right),
        "arm_only": b,
        "base_only": c,
        "mcnemar": mcnemar_exact(b, c),
        "bootstrap": paired_bootstrap_interval(left, right, samples=10000, seed=0),
    }


def decide(outcomes: Mapping[str, Mapping[str, int]], task_ids: Sequence[str],
           criteria: Mapping[str, Any], *, base_arm: str = "no_update") -> dict[str, Any]:
    comps = {arm: paired(outcomes[base_arm], outcomes[arm], task_ids)
             for arm in outcomes if arm != base_arm}
    adj = holm({arm: c["mcnemar"]["p_value"] for arm, c in comps.items()})
    alpha = criteria["alpha"]
    min_lift = criteria["min_absolute_lift"]
    for arm, c in comps.items():
        c["holm_p"] = adj[arm]
        lift = c["arm_rate"] - c["base_rate"]
        if c["holm_p"] < alpha and lift >= min_lift and c["arm_only"] > c["base_only"]:
            verdict = "improved"
        elif c["holm_p"] < alpha and lift < 0:
            verdict = "regressed"
        else:
            verdict = "no_detectable_change"
        c["verdict"] = verdict
    return {"comparisons": comps, "alpha": alpha, "min_absolute_lift": min_lift,
            "multiplicity": "holm", "n_tasks": len(task_ids)}


def power_table(n: int, scenarios: Sequence[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"n": n, "p_arm_only": p10, "p_base_only": p01, "power": round(mcnemar_power(n, p10, p01), 3)}
            for p10, p01 in scenarios]
