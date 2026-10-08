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


def compare(dpo_dir: Path, sft_dir: Path):
    dpo_spec, dpo = checked_run(dpo_dir, "dpo")
    sft_spec, sft = checked_run(sft_dir, "sft")
    normalized_dpo, normalized_sft = copy.deepcopy(dpo_spec), copy.deepcopy(sft_spec)
    for spec in (normalized_dpo, normalized_sft):
        spec.pop("protocol_id", None)
        spec.pop("purpose", None)
        spec["learner"].pop("method", None)
        spec["learner"].pop("objective", None)
    if canonical(normalized_dpo) != canonical(normalized_sft):
        raise ValueError("DPO and SFT locks differ beyond the declared method/objective fields")
    for key in ("train_examples", "validation_examples"):
        if dpo[key] != sft[key]:
            raise ValueError(f"paired runs do not share identical {key}")
    for key in ("base_training_generation", "base_validation_generation"):
        if dpo["summary"][key] != sft["summary"][key]:
            raise ValueError(f"paired runs have different {key}")

    dpo_values = question_success(dpo)
    sft_values = question_success(sft)
    differences = [left - right for left, right in zip(dpo_values, sft_values)]
    dpo_summary, sft_summary = dpo["summary"], sft["summary"]
    return {
        "comparison": "matched_dpo_vs_sft_development",
        "validation_questions": len(differences),
        "seeds": dpo_spec["learner"]["seeds"],
        "base_exact_match": dpo_summary["base_validation_exact_match_accuracy"],
        "dpo": {
            "selected_epoch": dpo_summary["selected_epochs"],
            "mean_exact_match": dpo_summary["selected_mean_exact_match_accuracy"],
            "decision": dpo_summary["decision"],
            "validation_preference_nll": dpo_summary["checkpoint_summary"][str(dpo_summary["selected_epochs"])]["mean_validation_dpo_preference_nll"],
            "mean_token_kl": dpo_summary["checkpoint_summary"][str(dpo_summary["selected_epochs"])]["mean_validation_token_kl_to_base"],
        },
        "sft": {
            "selected_epoch": sft_summary["selected_epochs"],
            "mean_exact_match": sft_summary["selected_mean_exact_match_accuracy"],
            "decision": sft_summary["decision"],
            "validation_preference_nll": sft_summary["checkpoint_summary"][str(sft_summary["selected_epochs"])]["mean_validation_dpo_preference_nll"],
            "mean_token_kl": sft_summary["checkpoint_summary"][str(sft_summary["selected_epochs"])]["mean_validation_token_kl_to_base"],
        },
        "paired_question_mean_exact_match_difference_dpo_minus_sft": statistics.fmean(differences),
        "paired_question_bootstrap_95_percent_interval": bootstrap_mean_interval(differences),
        "claim_boundary": "Development-only comparison. The same validation questions select each arm's checkpoint; the paired interval is descriptive and is not independent confirmation or a capability claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dpo_run", type=Path)
    parser.add_argument("sft_run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.dpo_run.resolve(), args.sft_run.resolve())
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
