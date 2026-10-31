"""Post-hoc descriptive breakdown of the group-informativeness study (exploratory).

Reads the frozen protocol and this directory's samples.jsonl; writes analysis.json.
Descriptive only: small n, no significance claims.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def prior_class(e: dict) -> str:
    if e["k"] == 0:
        return "prior_all_zero"
    if e["k"] == e["n"]:
        return "prior_all_one"
    return "prior_mixed"


def main() -> None:
    proto = json.loads((ROOT / "protocols/grpo_group_informativeness_v1.json").read_text())
    per_task = proto["prior"]["per_task"]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in (HERE / "samples.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert r["split"] == "train"
        groups[(r["condition"], r["instance_id"])].append(r)

    def is_mixed(rs: list[float]) -> bool:
        return 0 < sum(1 for x in rs if x > 0) < len(rs)

    strata: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    g8_halves = [0, 0]
    pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for (cond, iid), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["sample_index"])
        rewards = [float(r["reward"]) for r in rows]
        cls = prior_class(per_task[iid])
        cell = strata[cond][cls]
        cell[0] += int(is_mixed(rewards))
        cell[1] += 1
        if cond == "uniform_g8":
            for half in (rewards[:4], rewards[4:]):
                g8_halves[0] += int(is_mixed(half))
                g8_halves[1] += 1
        pooled[iid][0] += sum(1 for x in rewards if x > 0)
        pooled[iid][1] += len(rewards)

    out_strata = {
        cond: {
            cls: {"mixed": k, "groups": n, "fraction": round(k / n, 4), "wilson95": wilson(k, n)}
            for cls, (k, n) in sorted(cells.items())
        }
        for cond, cells in sorted(strata.items())
    }
    # Prior class vs. what this study observed, pooling all study generations per task.
    flips: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for iid, (k, n) in pooled.items():
        obs = "obs_all_zero" if k == 0 else "obs_all_one" if k == n else "obs_mixed"
        flips[prior_class(per_task[iid])][obs] += 1
    result = {
        "label": "exploratory / diagnostic; Train tasks only; descriptive post-hoc breakdown",
        "mixed_fraction_by_condition_and_prior_class": out_strata,
        "uniform_g8_split_into_two_g4_halves": {
            "mixed": g8_halves[0],
            "groups": g8_halves[1],
            "fraction": round(g8_halves[0] / g8_halves[1], 4),
            "wilson95": wilson(*g8_halves),
        },
        "prior_class_vs_pooled_study_outcome": {k: dict(v) for k, v in sorted(flips.items())},
        "tasks_observed": len(pooled),
    }
    (HERE / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
