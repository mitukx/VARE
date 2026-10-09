#!/usr/bin/env python3
"""Recompute a post-hoc response-length diagnostic for HH DPO v2."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values) if values else None


def pearson(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must have the same non-zero length")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator else None


def audit_manifest(bundle):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path, expected in manifest["files"].items():
        path = bundle / relative_path
        if not path.is_file() or sha256(path) != expected:
            raise ValueError("source bundle manifest mismatch: %s" % relative_path)
    return manifest_path, manifest


def analyze(bundle):
    manifest_path, manifest = audit_manifest(bundle)
    summary_path = bundle / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock_path = Path("protocols/cpu_hh_human_dpo_development_v2.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if summary["protocol_sha256"] != lock["sha256"]:
        raise ValueError("source summary does not match the frozen v2 protocol")

    selected = summary["development_selection"]
    base_rows = summary["base_per_prompt"]
    seeds = sorted(summary["arms"]["dpo"])
    if not seeds or len(selected) != len(base_rows):
        raise ValueError("source prompt inventories are incomplete")
    if any(len(summary["arms"]["dpo"][seed]["per_prompt"]) != len(selected) for seed in seeds):
        raise ValueError("DPO prompt inventories do not match selection")

    rows = []
    for index, selected_row in enumerate(selected):
        length_delta = selected_row["chosen_tokens"] - selected_row["rejected_tokens"]
        dpo_rows = [summary["arms"]["dpo"][seed]["per_prompt"][index] for seed in seeds]
        rows.append(
            {
                "length_delta": length_delta,
                "base_accuracy": base_rows[index]["accuracy"],
                "dpo_accuracy": mean([row["accuracy"] for row in dpo_rows]),
                "base_margin": base_rows[index]["margin"],
                "dpo_margin": mean([row["margin"] for row in dpo_rows]),
            }
        )

    groups = {
        "chosen_longer": lambda delta: delta > 0,
        "equal_length": lambda delta: delta == 0,
        "chosen_shorter": lambda delta: delta < 0,
    }
    grouped = {}
    for name, predicate in groups.items():
        subset = [row for row in rows if predicate(row["length_delta"])]
        grouped[name] = {
            "n": len(subset),
            "base_pair_accuracy": mean([row["base_accuracy"] for row in subset]),
            "dpo_pair_accuracy_mean_across_seeds": mean([row["dpo_accuracy"] for row in subset]),
            "mean_chosen_minus_rejected_tokens": mean([row["length_delta"] for row in subset]),
            "base_margin_mean": mean([row["base_margin"] for row in subset]),
            "dpo_margin_mean_across_seeds": mean([row["dpo_margin"] for row in subset]),
        }

    length_accuracy = mean(
        [0.5 if row["length_delta"] == 0 else float(row["length_delta"] > 0) for row in rows]
    )
    diagnostic = {
        "analysis": "post-hoc descriptive response-length diagnostic; non-confirmatory",
        "n_prompts": len(rows),
        "n_dpo_seeds": len(seeds),
        "length_only_pair_accuracy_recomputed": length_accuracy,
        "length_only_pair_accuracy_in_run_summary": summary["base_metrics"]["length_only_pair_accuracy"],
        "groups": grouped,
        "pearson_r_length_delta_vs_base_raw_sequence_margin": pearson(
            [row["length_delta"] for row in rows], [row["base_margin"] for row in rows]
        ),
        "pearson_r_length_delta_vs_dpo_mean_raw_sequence_margin": pearson(
            [row["length_delta"] for row in rows], [row["dpo_margin"] for row in rows]
        ),
        "pearson_r_length_delta_vs_dpo_minus_base_margin": pearson(
            [row["length_delta"] for row in rows],
            [row["dpo_margin"] - row["base_margin"] for row in rows],
        ),
        "frozen_decision_unchanged": summary["decision"],
        "interpretation_limit": "Post-hoc analysis of the already-scored development cohort; no hypothesis, metric, gate, or decision is changed. Association is descriptive and not causal.",
        "source": {
            "protocol_sha256": summary["protocol_sha256"],
            "manifest_sha256": sha256(manifest_path),
            "summary_sha256": sha256(summary_path),
            "bundle_manifest_entries_verified": len(manifest["files"]),
        },
    }
    return diagnostic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("results/cpu-hh-human-preference-dpo-v2/development/run-1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cpu-hh-human-preference-dpo-v2/analysis/length-diagnostic-v1.json"),
    )
    args = parser.parse_args()
    diagnostic = analyze(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostic, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
