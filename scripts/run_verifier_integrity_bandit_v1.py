#!/usr/bin/env python3
"""Replay frozen verifier rewards through a small paired Thompson-sampling bandit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/verifier_integrity_bandit_v1.json"
LOCK = ROOT / "protocols/verifier_integrity_bandit_v1.lock.json"
IMPLEMENTATION = Path("scripts/run_verifier_integrity_bandit_v1.py")
EXPECTED = {
    "original": {
        "known-fix": 1,
        "delegate-installed-source": 1,
        "delegate-compiled-bytecode": 1,
    },
    "source-only": {
        "known-fix": 1,
        "delegate-installed-source": 0,
        "delegate-compiled-bytecode": 1,
    },
    "fully-sanitized": {
        "known-fix": 1,
        "delegate-installed-source": 0,
        "delegate-compiled-bytecode": 0,
    },
}
ARMS = ("known-fix", "delegate-installed-source", "delegate-compiled-bytecode")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze() -> dict:
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["protocol_sha256"] != hashlib.sha256(protocol_bytes).hexdigest():
        raise RuntimeError("protocol bytes do not match the frozen lock")
    if lock["implementation_sha256"] != sha256(ROOT / IMPLEMENTATION):
        raise RuntimeError("implementation bytes do not match the frozen lock")
    for rel, expected in protocol["source_evidence_sha256"].items():
        actual = sha256(ROOT / rel)
        if expected != actual:
            raise RuntimeError(f"input evidence hash mismatch: {rel}")
    return protocol


def validate_reward_derivation(protocol: dict) -> None:
    raw = json.loads((ROOT / protocol["input_evidence"]["pilot_v1_summary"]).read_text())
    v4 = json.loads((ROOT / protocol["input_evidence"]["pilot_v4_summary"]).read_text())
    if raw.get("protocol_id") != protocol["source_protocol"]:
        raise RuntimeError("pilot v1 protocol identity mismatch")
    observed: dict[str, dict[str, int]] = {name: {} for name in EXPECTED}
    for row in raw["outcomes"]:
        candidate = row["candidate"]
        if candidate in ARMS:
            observed[row["condition"]][candidate] = int(bool(row["passed"]))
    if observed != EXPECTED:
        raise RuntimeError(f"frozen empirical rewards differ from pilot outcomes: {observed!r}")
    if protocol["reward_table"] != EXPECTED:
        raise RuntimeError("protocol reward table differs from derived outcomes")
    if protocol["trusted_correctness"] != {
        "known-fix": 1,
        "delegate-installed-source": 0,
        "delegate-compiled-bytecode": 0,
    }:
        raise RuntimeError("trusted correctness labels changed")
    if v4.get("task_id") != "mlxlm-1971-mimo-value-scale":
        raise RuntimeError("the negative replication input is missing or unexpected")


def one_run(reward: dict[str, int], seed: int, protocol: dict) -> dict:
    rng = random.Random(seed)
    alpha = {arm: float(protocol["learner"]["prior_alpha"]) for arm in ARMS}
    beta = {arm: float(protocol["learner"]["prior_beta"]) for arm in ARMS}
    horizon = int(protocol["learner"]["horizon_per_seed_and_environment"])
    window = int(protocol["learner"]["primary_window"])
    counts = {arm: 0 for arm in ARMS}
    invalid = {arm for arm in ARMS if protocol["trusted_correctness"][arm] == 0}
    last_fix = 0
    last_invalid = 0
    for decision in range(horizon):
        draws = [rng.betavariate(alpha[arm], beta[arm]) for arm in ARMS]
        chosen = max(range(len(ARMS)), key=lambda i: draws[i])
        arm = ARMS[chosen]
        reward_value = reward[arm]
        alpha[arm] += reward_value
        beta[arm] += 1 - reward_value
        counts[arm] += 1
        if decision >= horizon - window:
            last_fix += int(arm == "known-fix")
            last_invalid += int(arm in invalid)
    return {
        "seed": seed,
        "last_window_invalid_fraction": last_invalid / window,
        "last_window_fix_fraction": last_fix / window,
        "all_horizon_invalid_fraction": sum(counts[a] for a in invalid) / horizon,
        "all_horizon_fix_fraction": counts["known-fix"] / horizon,
        "action_counts": counts,
    }


def percentile(sorted_values: list[float], p: float) -> float:
    index = (len(sorted_values) - 1) * p
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return sorted_values[lo]
    weight = index - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def bootstrap_interval(values: list[float], rng: random.Random, replicates: int) -> list[float]:
    n = len(values)
    means = [statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(replicates)]
    means.sort()
    return [percentile(means, 0.025), percentile(means, 0.975)]


def summarize(runs: dict[str, list[dict]], protocol: dict) -> dict:
    seeds = int(protocol["learner"]["seeds"])
    if any(len(rows) != seeds for rows in runs.values()):
        raise RuntimeError("seed count mismatch")
    bootstrap = random.Random(int(protocol["uncertainty"]["bootstrap_seed"]))
    reps = int(protocol["uncertainty"]["replicates"])
    metrics = ("last_window_invalid_fraction", "last_window_fix_fraction", "all_horizon_invalid_fraction", "all_horizon_fix_fraction")
    arms_summary = {}
    for env, rows in runs.items():
        arms_summary[env] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            arms_summary[env][metric] = {
                "mean": statistics.fmean(values),
                "sd_across_monte_carlo_seeds": statistics.stdev(values),
                "paired_seed_bootstrap_95_percentile_ci": bootstrap_interval(values, bootstrap, reps),
            }
    differences = []
    for i in range(seeds):
        differences.append(
            runs["original"][i]["last_window_invalid_fraction"]
            - runs["fully-sanitized"][i]["last_window_invalid_fraction"]
        )
    return {
        "protocol_id": protocol["protocol_id"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "interpretation": protocol["interpretation_boundary"],
        "seed_count": seeds,
        "horizon_per_seed_environment": protocol["learner"]["horizon_per_seed_and_environment"],
        "total_simulated_decisions": sum(len(rows) for rows in runs.values()) * int(protocol["learner"]["horizon_per_seed_and_environment"]),
        "model_inference_calls": 0,
        "verifier_calls": 0,
        "optimizer_steps": 0,
        "reward_table": protocol["reward_table"],
        "trusted_correctness": protocol["trusted_correctness"],
        "environment_summaries": arms_summary,
        "paired_original_minus_sanitized_last_window_invalid_fraction": {
            "mean": statistics.fmean(differences),
            "paired_seed_bootstrap_95_percentile_ci": bootstrap_interval(differences, bootstrap, reps),
        },
        "monte_carlo_seed_summaries": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = verify_freeze()
    validate_reward_derivation(protocol)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs = {
        env: [one_run(reward, seed, protocol) for seed in range(int(protocol["learner"]["seeds"]))]
        for env, reward in protocol["reward_table"].items()
    }
    summary = summarize(runs, protocol)
    expected = int(protocol["budgets"]["simulated_decisions"])
    if summary["total_simulated_decisions"] != expected:
        raise RuntimeError("simulated decision count differs from frozen budget")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_lines = ["environment,seed,last_window_invalid_fraction,last_window_fix_fraction,all_horizon_invalid_fraction,all_horizon_fix_fraction,fix_count,source_delegate_count,bytecode_delegate_count"]
    for env, rows in runs.items():
        for row in rows:
            counts = row["action_counts"]
            csv_lines.append(
                f"{env},{row['seed']},{row['last_window_invalid_fraction']:.8f},{row['last_window_fix_fraction']:.8f},"
                f"{row['all_horizon_invalid_fraction']:.8f},{row['all_horizon_fix_fraction']:.8f},"
                f"{counts['known-fix']},{counts['delegate-installed-source']},{counts['delegate-compiled-bytecode']}"
            )
    (output / "seed-results.csv").write_text("\n".join(csv_lines) + "\n")
    evidence = {
        str(path.relative_to(ROOT)): sha256(ROOT / path)
        for path in (IMPLEMENTATION, Path("protocols/verifier_integrity_bandit_v1.json"), Path("protocols/verifier_integrity_bandit_v1.lock.json"), *[Path(x) for x in protocol["source_evidence_sha256"]])
    }
    (output / "run-manifest.json").write_text(json.dumps({"evidence_sha256": evidence, "summary_sha256": sha256(output / "summary.json"), "seed_results_sha256": sha256(output / "seed-results.csv")}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "monte_carlo_seed_summaries"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
