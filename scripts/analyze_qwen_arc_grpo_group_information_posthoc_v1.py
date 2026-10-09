#!/usr/bin/env python3
"""Outcome-informed diagnostic of reward-information coverage in retained ARC runs."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/qwen-arc-grpo-sft-comparison-v3/run-3/summary.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--permutations", type=int, default=50000)
    args = ap.parse_args()
    raw = SOURCE.read_bytes()
    d = json.loads(raw)
    seeds = sorted(d["rollouts_by_seed"])
    by_qid: dict[str, list[list[int]]] = collections.defaultdict(list)
    per_seed = {}
    total_success = 0
    total_groups = 0
    mixed = all_wrong = all_correct = 0
    positive_in_all_correct_groups = 0
    group_reward_histogram = collections.Counter()
    for seed in seeds:
        groups = d["rollouts_by_seed"][seed]
        hist = collections.Counter()
        for group in groups:
            rewards = [int(float(m["reward"])) for m in group["members"]]
            if len(rewards) != 4 or any(y not in (0, 1) for y in rewards):
                raise ValueError("unexpected group size or nonbinary reward")
            qid = group["task_id"]
            by_qid[qid].append(rewards)
            k = sum(rewards)
            hist[k] += 1
            group_reward_histogram[str(k)] += 1
            total_success += k
            total_groups += 1
            mixed += int(0 < k < 4)
            all_wrong += int(k == 0)
            all_correct += int(k == 4)
            positive_in_all_correct_groups += k if k == 4 else 0
        per_seed[seed] = {"groups": len(groups), "reward_sum_histogram": dict(sorted(hist.items()))}
    if len(by_qid) != 32 or any(len(v) != 3 for v in by_qid.values()):
        raise ValueError("expected the same 32 train prompts in three seeds")

    # Conditional randomization: preserve each prompt's 12 observed binary outcomes,
    # then reassign them to three groups of four. This tests whether the seed-defined
    # grouping has unusually low/high reward diversity beyond prompt-level difficulty.
    rng = random.Random(20261009)
    null_mixed = []
    for _ in range(args.permutations):
        count = 0
        for seed_groups in by_qid.values():
            ys = [y for group in seed_groups for y in group]
            rng.shuffle(ys)
            count += sum(0 < sum(ys[i:i + 4]) < 4 for i in (0, 4, 8))
        null_mixed.append(count)
    null_mixed.sort()
    null_mean = sum(null_mixed) / len(null_mixed)
    null_sd = math.sqrt(sum((x - null_mean) ** 2 for x in null_mixed) / len(null_mixed))
    obs_dist = abs(mixed - null_mean)
    p_two_sided = (1 + sum(abs(x - null_mean) >= obs_dist for x in null_mixed)) / (len(null_mixed) + 1)

    base = {r["task_id"]: bool(r["exact"]) for r in d["base_eval"]}
    evaluation = {}
    for arm in ("grpo", "success_trace_sft"):
        evaluation[arm] = {}
        for seed in seeds:
            rows = d["arms"][arm][seed]["eval"]
            pred = {r["task_id"]: bool(r["exact"]) for r in rows}
            gains = sum(not base[q] and pred[q] for q in base)
            regressions = sum(base[q] and not pred[q] for q in base)
            evaluation[arm][seed] = {
                "correct": sum(pred.values()),
                "gains_vs_base": gains,
                "regressions_vs_base": regressions,
                "unchanged_vs_base": len(base) - gains - regressions,
            }
    out = {
        "analysis_id": "qwen_arc_grpo_group_information_posthoc_v1",
        "classification": "outcome-informed descriptive diagnostic; not confirmatory and not a new frozen experiment",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_protocol": d["protocol_id"],
        "n_seeds": len(seeds),
        "n_groups": total_groups,
        "group_size": 4,
        "sampled_successes": total_success,
        "sampled_success_rate": total_success / (4 * total_groups),
        "group_reward_sum_histogram": dict(sorted(group_reward_histogram.items(), key=lambda kv: int(kv[0]))),
        "mixed_groups": mixed,
        "all_wrong_groups": all_wrong,
        "all_correct_groups": all_correct,
        "fraction_groups_with_nonzero_centered_binary_advantage": mixed / total_groups,
        "successful_completions_in_all_correct_groups_with_zero_centered_advantage": positive_in_all_correct_groups,
        "conditional_randomization": {
            "null": "within each prompt, uniformly shuffle its 12 observed rewards and re-split into three groups of four",
            "permutations": args.permutations,
            "seed": 20261009,
            "observed_mixed_groups": mixed,
            "null_mean": null_mean,
            "null_sd": null_sd,
            "null_95_percentile_interval": [null_mixed[int(0.025 * len(null_mixed))], null_mixed[int(0.975 * len(null_mixed))]],
            "two_sided_randomization_p": p_two_sided,
            "interpretation_limit": "This is a post-hoc exchangeability diagnostic, not a causal test of why held-out accuracy changed."
        },
        "per_seed": per_seed,
        "heldout_transitions_vs_base": evaluation,
        "heldout_base_correct": sum(base.values()),
        "heldout_n": len(base),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
