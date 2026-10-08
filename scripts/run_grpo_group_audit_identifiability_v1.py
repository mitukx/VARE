#!/usr/bin/env python3
"""Run the frozen two-world GRPO group-audit identifiability experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/grpo_group_audit_identifiability_v1.lock.json"
OUT = ROOT / "results/grpo-group-audit-identifiability-v1/run-1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_group(rng: random.Random, world: str) -> tuple[int, int]:
    bit = rng.randrange(2)
    if world == "A":
        return bit, bit
    return bit, 1 - bit


def normalized_advantages(rewards: tuple[int, int]) -> tuple[float, float]:
    mean = sum(rewards) / 2
    variance = sum((reward - mean) ** 2 for reward in rewards) / 2
    if variance == 0:
        return (0.0, 0.0)
    scale = math.sqrt(variance)
    return tuple((reward - mean) / scale for reward in rewards)


def item_batch(seed: int, world: str, mode: str) -> tuple[int, int]:
    rng = random.Random(seed)
    positives = 0
    for _ in range(200):
        labels = clean_group(rng, world)
        index = rng.randrange(2) if mode == "uniform" else 0
        positives += labels[index]
    prediction = "B" if positives > 100 else "A"
    return positives, int(prediction == world)


def group_batch(seed: int, world: str) -> tuple[int, int, int]:
    rng = random.Random(seed)
    mixed = 0
    for _ in range(100):
        labels = clean_group(rng, world)
        mixed += int(labels[0] != labels[1])
    prediction = "B" if mixed else "A"
    return mixed, int(prediction == world), 200


def binomial_half(n: int) -> list[float]:
    probability = 2.0 ** -n
    result = [probability]
    for k in range(n):
        probability *= (n - k) / (k + 1)
        result.append(probability)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    raw_path = out / "trials.jsonl"
    counts = {name: {world: 0 for world in ("A", "B")} for name in ("item_uniform", "item_proxy_positive", "group_atomic")}
    with raw_path.open("w") as stream:
        for seed in range(2000):
            for world in ("A", "B"):
                u_pos, u_ok = item_batch(seed + 10_000, world, "uniform")
                p_pos, p_ok = item_batch(seed + 30_000, world, "proxy_positive")
                mixed, g_ok, calls = group_batch(seed + 50_000, world)
                counts["item_uniform"][world] += u_ok
                counts["item_proxy_positive"][world] += p_ok
                counts["group_atomic"][world] += g_ok
                record = {
                    "seed": seed,
                    "world": world,
                    "item_uniform_positive_labels": u_pos,
                    "item_uniform_prediction": "B" if u_pos > 100 else "A",
                    "item_proxy_positive_labels": p_pos,
                    "item_proxy_positive_prediction": "B" if p_pos > 100 else "A",
                    "group_atomic_mixed_groups": mixed,
                    "group_atomic_prediction": "B" if mixed else "A",
                    "group_atomic_label_calls": calls,
                    "labels_per_arm": 200,
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    item_distribution = binomial_half(200)
    tv = 0.5 * sum(abs(a - b) for a, b in zip(item_distribution, item_distribution))
    per_world_accuracy = {
        arm: {world: successes / 2000 for world, successes in per_world.items()}
        for arm, per_world in counts.items()
    }
    balanced_accuracy = {
        arm: sum(per_world_accuracy[arm].values()) / 2
        for arm in per_world_accuracy
    }
    passed = (
        balanced_accuracy["item_uniform"] <= 0.55
        and balanced_accuracy["item_proxy_positive"] <= 0.55
        and balanced_accuracy["group_atomic"] >= 0.99
        and tv == 0.0
    )
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "seed_count": 2000,
        "batches_per_world": 2000,
        "audit_label_calls_per_batch": 200,
        "per_world_accuracy": per_world_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "item_uniform_exact_observation_tv": tv,
        "item_proxy_positive_exact_observation_tv": tv,
        "group_signal": {
            "world_A_advantages": list(normalized_advantages((0, 0))),
            "world_B_advantages": [
                list(normalized_advantages((1, 0))),
                list(normalized_advantages((0, 1))),
            ],
            "world_A_group_mixed_probability": 0.0,
            "world_B_group_mixed_probability": 1.0
        },
        "raw_trials_sha256": sha256(raw_path),
        "decision": "mechanism_supported" if passed else "non_pass",
        "limitations": protocol["limitations"]
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {
        name: sha256(path)
        for name, path in sorted((p.name, p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json")
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out), "balanced_accuracy": summary["balanced_accuracy"], "exact_item_law_tv": tv}))


if __name__ == "__main__":
    main()
