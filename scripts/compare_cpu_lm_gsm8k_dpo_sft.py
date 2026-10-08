#!/usr/bin/env python3
"""Compare paired, audited DPO and SFT development runs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_run(path: Path, expected_method: str):
    audit = read_json(path / "audit.json")
    if audit.get("audit") != "passed":
        raise ValueError(f"{path}: offline audit did not pass")
    spec = read_json(path / "protocol.snapshot.json")
    lock = read_json(path / "protocol.lock.snapshot.json")
    lock_hash = lock.pop("sha256", None)
    if lock_hash != hashlib.sha256(canonical(spec)).hexdigest() or canonical(lock) != canonical(spec):
        raise ValueError(f"{path}: protocol snapshot differs from lock")
    if spec["learner"].get("method") != expected_method:
        raise ValueError(f"{path}: expected {expected_method} protocol")
    data = read_json(path / "development.json")
    summary = data["summary"]
    if summary.get("training_method") != expected_method:
        raise ValueError(f"{path}: recorded training method differs")
    return spec, data


def question_success(run):
    summary = run["summary"]
    rows = summary["selected_adapters_and_generation"]
    seeds = summary["per_seed"]
    if [item["seed"] for item in rows] != [item["seed"] for item in seeds]:
        raise ValueError("selected generation seeds differ from training seeds")
    matrix = []
    for item in rows:
        generated = item["generated_validation"]
        matrix.append([bool(row["exact_match"]) for row in generated])
    if not matrix or len({len(row) for row in matrix}) != 1:
        raise ValueError("selected generation matrix is empty or ragged")
    return [statistics.fmean(float(matrix[seed][question]) for seed in range(len(matrix)))
            for question in range(len(matrix[0]))]


def bootstrap_mean_interval(values, seed=57391, samples=20000):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return [means[int(0.025 * samples)], means[min(samples - 1, int(0.975 * samples))]]


def compare(dpo_dir: Path, sft_dir: Path, anchor_dir: Path | None = None):
    expected_methods = ["dpo", "sft"] + (["dpo_sft_anchor"] if anchor_dir else [])
    directories = [dpo_dir, sft_dir] + ([anchor_dir] if anchor_dir else [])
    checked = [checked_run(path, method) for path, method in zip(directories, expected_methods)]
    specs = [item[0] for item in checked]
    runs = [item[1] for item in checked]
    normalized_specs = [copy.deepcopy(spec) for spec in specs]
    for spec in normalized_specs:
        spec.pop("protocol_id", None)
        spec.pop("purpose", None)
        spec["learner"].pop("method", None)
        spec["learner"].pop("objective", None)
        spec["learner"].pop("sft_anchor_weight", None)
    if any(canonical(spec) != canonical(normalized_specs[0]) for spec in normalized_specs[1:]):
        raise ValueError("method locks differ beyond the declared method/objective fields")
    reference_data = runs[0]
    sft_data = runs[1]
    for method, data in zip(expected_methods[1:], runs[1:]):
        for key in ("train_examples", "validation_examples"):
            if reference_data[key] != data[key]:
                raise ValueError(f"paired runs do not share identical {key}: {method}")
        for key in ("base_training_generation", "base_validation_generation"):
            if reference_data["summary"][key] != data["summary"][key]:
                raise ValueError(f"paired runs have different {key}: {method}")

    def arm_summary(data):
        summary = data["summary"]
        epoch = str(summary["selected_epochs"])
        metrics = summary["checkpoint_summary"][epoch]
        return {
            "selected_epoch": summary["selected_epochs"],
            "mean_exact_match": summary["selected_mean_exact_match_accuracy"],
            "decision": summary["decision"],
            "validation_preference_nll": metrics["mean_validation_dpo_preference_nll"],
            "mean_token_kl": metrics["mean_validation_token_kl_to_base"],
        }

    per_arm = {method: arm_summary(data) for method, data in zip(expected_methods, runs)}
    paired = {}
    for method, data in zip(expected_methods, runs):
        if method == "sft":
            continue
        differences = [left - right for left, right in zip(question_success(data), question_success(sft_data))]
        paired[f"{method}_minus_sft"] = {
            "mean_exact_match_difference": statistics.fmean(differences),
            "question_bootstrap_95_percent_interval": bootstrap_mean_interval(differences),
            "questions_better_tied_worse": {
                "better": sum(value > 0 for value in differences),
                "tied": sum(value == 0 for value in differences),
                "worse": sum(value < 0 for value in differences),
            },
        }
    return {
        "comparison": "matched_dpo_sft_and_optional_anchor_development" if anchor_dir else "matched_dpo_vs_sft_development",
        "validation_questions": len(runs[0]["validation_examples"]),
        "seeds": specs[0]["learner"]["seeds"],
        "base_exact_match": reference_data["summary"]["base_validation_exact_match_accuracy"],
        "arms": per_arm,
        "paired_question_comparisons": paired,
        "claim_boundary": "Development-only comparison. The same validation questions select each arm's checkpoint; intervals are descriptive and are not independent confirmation or capability claims.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dpo_run", type=Path)
    parser.add_argument("sft_run", type=Path)
    parser.add_argument("--anchor-run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.dpo_run.resolve(), args.sft_run.resolve(), args.anchor_run.resolve() if args.anchor_run else None)
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
