#!/usr/bin/env python3
"""Run the frozen synthetic contextual DPO control using only Python stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/synthetic_dpo_cpu_v1.json"
LOCK_PATH = ROOT / "protocols/synthetic_dpo_cpu_v1.lock.json"
DIM = 8
N_ACTIONS = 4


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol() -> tuple[dict[str, Any], str]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    observed = sha256_bytes(canonical_json(spec))
    frozen = dict(lock)
    recorded = frozen.pop("sha256", None)
    if recorded != observed or canonical_json(frozen) != canonical_json(spec):
        raise ValueError("synthetic DPO protocol differs from its frozen lock")
    return spec, observed


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def softmax(logits: list[float]) -> list[float]:
    peak = max(logits)
    values = [math.exp(value - peak) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def teacher_weights() -> list[list[float]]:
    return [
        [math.sin(0.7 * (action + 1) * (feature + 1))
         + 0.5 * math.cos(0.31 * (action + 2) * (feature + 1))
         for feature in range(DIM)]
        for action in range(N_ACTIONS)
    ]


TEACHER = teacher_weights()


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def teacher_utility(context: list[float], action: int) -> float:
    return dot(context, TEACHER[action])


def logits(theta: list[list[float]], context: list[float]) -> list[float]:
    return [dot(theta[action], context) for action in range(N_ACTIONS)]


def log_prob_difference(theta: list[list[float]], context: list[float],
                        chosen: int, rejected: int) -> float:
    # The shared log-normalizer cancels between actions for the same context.
    values = logits(theta, context)
    return values[chosen] - values[rejected]


def objective(theta: list[list[float]], examples: list[dict[str, Any]], beta: float) -> float:
    if not examples:
        raise ValueError("objective requires examples")
    return sum(
        softplus(-beta * log_prob_difference(theta, item["x"], item["chosen"], item["rejected"]))
        for item in examples
    ) / len(examples)


def gradient(theta: list[list[float]], examples: list[dict[str, Any]], beta: float) -> list[list[float]]:
    if not examples:
        raise ValueError("gradient requires examples")
    result = [[0.0] * DIM for _ in range(N_ACTIONS)]
    scale = beta / len(examples)
    for item in examples:
        chosen, rejected, context = item["chosen"], item["rejected"], item["x"]
        margin = beta * log_prob_difference(theta, context, chosen, rejected)
        coefficient = -scale * sigmoid(-margin)
        for index, value in enumerate(context):
            result[chosen][index] += coefficient * value
            result[rejected][index] -= coefficient * value
    return result


def generate_examples(rng: random.Random, context_count: int, pairs_per_context: int) -> list[dict[str, Any]]:
    examples = []
    for _ in range(context_count):
        context = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        for _ in range(pairs_per_context):
            left, right = rng.sample(range(N_ACTIONS), 2)
            probability_left = sigmoid(teacher_utility(context, left) - teacher_utility(context, right))
            chosen = left if rng.random() < probability_left else right
            rejected = right if chosen == left else left
            examples.append({"x": context[:], "a": left, "b": right,
                             "chosen": chosen, "rejected": rejected})
    return examples


def train(theta: list[list[float]], examples: list[dict[str, Any]],
          beta: float, learning_rate: float, updates: int) -> list[list[float]]:
    values = [row[:] for row in theta]
    for _ in range(updates):
        grad = gradient(values, examples, beta)
        for action in range(N_ACTIONS):
            for feature in range(DIM):
                values[action][feature] -= learning_rate * grad[action][feature]
    return values


def finite_difference_check(spec: dict[str, Any]) -> dict[str, float | bool]:
    epsilon = spec["gradient_check"]["epsilon"]
    theta = [[(action - 1.2) * (feature + 0.7) * 0.013
              for feature in range(DIM)] for action in range(N_ACTIONS)]
    examples = [
        {"x": [0.2, -0.5, 0.1, 0.7, -0.3, 0.4, 0.8, -0.2], "chosen": 0, "rejected": 2},
        {"x": [-0.4, 0.3, 0.9, -0.1, 0.5, -0.8, 0.2, 0.6], "chosen": 3, "rejected": 1},
        {"x": [0.7, 0.1, -0.2, 0.4, 0.9, 0.3, -0.5, -0.6], "chosen": 2, "rejected": 0},
    ]
    beta = 1.0
    analytic = gradient(theta, examples, beta)
    max_absolute = 0.0
    max_relative = 0.0
    for action in range(N_ACTIONS):
        for feature in range(DIM):
            plus = [row[:] for row in theta]
            minus = [row[:] for row in theta]
            plus[action][feature] += epsilon
            minus[action][feature] -= epsilon
            numeric = (objective(plus, examples, beta) - objective(minus, examples, beta)) / (2.0 * epsilon)
            absolute = abs(numeric - analytic[action][feature])
            relative = absolute / max(1e-12, abs(numeric), abs(analytic[action][feature]))
            max_absolute = max(max_absolute, absolute)
            max_relative = max(max_relative, relative)
    passed = max_absolute <= spec["gradient_check"]["absolute_tolerance"] and max_relative <= spec["gradient_check"]["relative_tolerance"]
    return {"passed": passed, "max_absolute_error": max_absolute, "max_relative_error": max_relative}


def evaluate(theta: list[list[float]], examples: list[dict[str, Any]]) -> dict[str, float]:
    nlls, correct, briers, kls, utilities = [], 0, [], [], []
    for item in examples:
        margin = log_prob_difference(theta, item["x"], item["chosen"], item["rejected"])
        probability = sigmoid(margin)
        nlls.append(softplus(-margin))
        correct += 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        briers.append((probability - 1.0) ** 2)
        probs = softmax(logits(theta, item["x"]))
        kls.append(sum(p * math.log(p * N_ACTIONS) for p in probs if p > 0))
        utilities.append(sum(probs[action] * teacher_utility(item["x"], action)
                             for action in range(N_ACTIONS)))
    return {
        "heldout_preference_nll": statistics.fmean(nlls),
        "heldout_preference_accuracy": correct / len(examples),
        "heldout_brier": statistics.fmean(briers),
        "mean_kl_to_uniform": statistics.fmean(kls),
        "mean_teacher_expected_utility": statistics.fmean(utilities),
    }


def parameter_delta(theta: list[list[float]]) -> float:
    return math.sqrt(sum(value * value for row in theta for value in row))


def bootstrap_interval(values: list[float], resamples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(values, k=len(values))) for _ in range(resamples)]
    means.sort()
    return [means[math.floor(0.025 * (resamples - 1))], means[math.floor(0.975 * (resamples - 1))]]


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest(root: Path) -> None:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[path.relative_to(root).as_posix()] = sha256_file(path)
    json_write(root / "manifest.json", {"schema_version": 1, "files": files})


def run(output: Path) -> dict[str, Any]:
    spec, protocol_hash = verify_protocol()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory: {output}")
    output.mkdir(parents=True)
    (output / "protocol.json").write_bytes(SPEC_PATH.read_bytes())
    (output / "protocol.lock.json").write_bytes(LOCK_PATH.read_bytes())
    source_dir = output / "source"
    source_dir.mkdir()
    (source_dir / "run_synthetic_dpo.py").write_bytes(Path(__file__).read_bytes())
    gradient_check = finite_difference_check(spec)
    json_write(output / "gradient-check.json", gradient_check)
    if not gradient_check["passed"]:
        raise ArithmeticError(f"objective gradient check failed: {gradient_check}")
    spec_sha = sha256_file(SPEC_PATH)
    lock_sha = sha256_file(LOCK_PATH)
    script_sha = sha256_file(Path(__file__))
    seeds = spec["seeds"]
    all_results: dict[str, list[dict[str, Any]]] = {arm: [] for arm in spec["arms"]}
    paired_improvements = []
    started_total = time.perf_counter()
    seed_raw_root = output / "seeds"
    seed_raw_root.mkdir()
    for seed in seeds:
        started = time.perf_counter()
        rng = random.Random(seed)
        train_data = generate_examples(rng, 64, 8)
        heldout_data = generate_examples(rng, 128, 2)
        base_theta = [[0.0] * DIM for _ in range(N_ACTIONS)]
        baseline = evaluate(base_theta, heldout_data)
        arm_results = {"reference": {**baseline, "parameter_l2_delta": 0.0, "updates": 0}}
        clean_labels = [item["chosen"] for item in train_data]
        noisy_labels = clean_labels[:]
        noise_rng = random.Random(seed + 200000)
        for index in range(len(noisy_labels)):
            if noise_rng.random() < 0.2:
                item = train_data[index]
                noisy_labels[index] = item["rejected"] if noisy_labels[index] == item["chosen"] else item["chosen"]
        shuffled_labels = clean_labels[:]
        random.Random(seed + 100000).shuffle(shuffled_labels)
        for arm, labels in (("clean_dpo", clean_labels), ("flip_20pct", noisy_labels), ("shuffle_labels", shuffled_labels)):
            examples = []
            for item, label in zip(train_data, labels):
                examples.append({**item, "chosen": label,
                                 "rejected": item["b"] if label == item["a"] else item["a"]})
            theta = train(base_theta, examples, 1.0, 0.2, 200)
            arm_results[arm] = {**evaluate(theta, heldout_data),
                                "parameter_l2_delta": parameter_delta(theta), "updates": 200}
        paired_improvements.append(baseline["heldout_preference_nll"] - arm_results["clean_dpo"]["heldout_preference_nll"])
        arm_seconds = time.perf_counter() - started
        for arm, result in arm_results.items():
            all_results[arm].append({"seed": seed, **result, "seed_wall_seconds": arm_seconds})
        json_write(seed_raw_root / f"seed-{seed}.json", {
            "seed": seed, "train_examples": train_data, "heldout_examples": heldout_data,
            "metrics": arm_results, "seed_wall_seconds": arm_seconds,
        })
    improvements = paired_improvements
    ci = bootstrap_interval(improvements, 10000, 20261008)
    mean_improvement = statistics.fmean(improvements)
    clean_mean_kl = statistics.fmean(item["mean_kl_to_uniform"] for item in all_results["clean_dpo"])
    improved_seeds = sum(value > 0.0 for value in improvements)
    acceptance = spec["metrics"]["acceptance"]
    accepted = (mean_improvement >= acceptance["mean_clean_nll_improvement_at_least_nats_per_pair"]
                and ci[0] > acceptance["paired_bootstrap_lower_bound_above"]
                and improved_seeds >= acceptance["minimum_seeds_with_clean_nll_improvement"]
                and clean_mean_kl <= acceptance["maximum_mean_clean_policy_kl"])
    summary = {
        "schema_version": 1,
        "protocol_id": spec["protocol_id"],
        "protocol_canonical_sha256": protocol_hash,
        "protocol_file_sha256": spec_sha,
        "protocol_lock_sha256": lock_sha,
        "source_script_sha256": script_sha,
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_rss_unit": "bytes_on_macos_kib_on_linux_or_other_posix; raw value only",
        "wall_seconds": time.perf_counter() - started_total,
        "seed_count": len(seeds),
        "gradient_check": gradient_check,
        "arms": all_results,
        "clean_dpo_vs_reference_nll_improvement_per_seed": improvements,
        "clean_dpo_vs_reference_nll_improvement_mean": mean_improvement,
        "clean_dpo_vs_reference_nll_improvement_paired_bootstrap_95pct": ci,
        "clean_dpo_improved_seed_count": improved_seeds,
        "clean_dpo_mean_kl_to_uniform": clean_mean_kl,
        "acceptance_passed": accepted,
    }
    json_write(output / "summary.json", summary)
    write_manifest(output)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are never overwritten")
    args = parser.parse_args()
    try:
        summary = run(args.output.expanduser().resolve())
    except Exception as exc:
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "complete", "acceptance_passed": summary["acceptance_passed"],
                      "output": str(args.output.expanduser().resolve()),
                      "mean_clean_nll_improvement": summary["clean_dpo_vs_reference_nll_improvement_mean"],
                      "ci95": summary["clean_dpo_vs_reference_nll_improvement_paired_bootstrap_95pct"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
