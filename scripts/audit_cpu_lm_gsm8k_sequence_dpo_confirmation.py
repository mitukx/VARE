#!/usr/bin/env python3
"""Audit hashes, paired metrics, bootstrap, and fixed confirmation decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def nll(relative_margin, beta):
    x = beta * relative_margin
    return max(-x, 0.0) + math.log1p(math.exp(-abs(x)))


def close(a, b, tol=1e-10):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def verify_manifest(out):
    manifest = read(out / "manifest.json")
    names = {p.relative_to(out).as_posix() for p in out.rglob("*")
             if p.is_file() and p.name not in ("manifest.json", "audit.json")}
    if names != set(manifest["files"]):
        raise ValueError("manifest inventory differs")
    for name, digest in manifest["files"].items():
        if sha256(out / name) != digest:
            raise ValueError("artifact hash mismatch: " + name)
    return sha256(out / "manifest.json")


def audit(out):
    source_manifest_hash = verify_manifest(out)
    spec = read(out / "protocol.snapshot.json")
    lock = read(out / "protocol.lock.snapshot.json")
    lock_hash = lock.pop("sha256", None)
    protocol_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_hash != protocol_hash or canonical(lock) != canonical(spec):
        raise ValueError("protocol snapshot does not match its committed lock")
    bundle = read(out / "confirmation.json")
    result = bundle["summary"]
    if result["protocol_sha256"] != protocol_hash:
        raise ValueError("result protocol hash differs")
    if len(bundle["update_examples"]) != 128 or len(bundle["heldout_examples"]) != 512:
        raise ValueError("confirmation row counts differ")
    update_hashes = {row["question_sha256"] for row in bundle["update_examples"]}
    heldout_hashes = {row["question_sha256"] for row in bundle["heldout_examples"]}
    if len(update_hashes) != 128 or len(heldout_hashes) != 512 or update_hashes & heldout_hashes:
        raise ValueError("update/held-out row overlap or duplicates")

    tokenized = read(out / "tokenized_examples.json")
    for name, rows in (("update", bundle["update_examples"]), ("heldout", bundle["heldout_examples"])):
        encoded = tokenized[name]
        if len(encoded) != len(rows):
            raise ValueError("tokenized example count differs")
        for row, candidate in zip(rows, encoded):
            if row["dataset_index"] != candidate["dataset_index"]:
                raise ValueError("tokenized dataset index differs")
            for choice in ("chosen", "rejected"):
                ids, response = candidate[choice]["input_ids"], candidate[choice]["response_ids"]
                if not response or ids[-len(response):] != response:
                    raise ValueError("completion token suffix is invalid")

    beta = spec["learner"]["beta"]
    seeds = spec["learner"]["seeds"]
    seed_deltas = []
    updated_accuracy = []
    for expected_seed, item in zip(seeds, result["seed_results"]):
        if item["seed"] != expected_seed or item["update_epochs"] != 4:
            raise ValueError("seed or frozen epoch differs")
        metrics = item["preference"]
        margins = metrics["relative_preference_margins"]
        if len(margins) != 512:
            raise ValueError("held-out per-example margin count differs")
        delta = statistics.fmean(nll(margin, beta) - nll(0.0, beta) for margin in margins)
        seed_deltas.append(delta)
        ref = item["adapter_artifact"]
        adapter = out / ref["path"]
        if not adapter.is_file() or sha256(adapter) != ref["sha256"]:
            raise ValueError("adapter checksum differs")
        import numpy as np
        with np.load(adapter, allow_pickle=False) as arrays:
            if set(arrays.files) != {"A", "B"} or not all(np.isfinite(arrays[k]).all() for k in ("A", "B")):
                raise ValueError("adapter arrays invalid")
        answers = item["generated_answers"]
        if len(answers) != 512:
            raise ValueError("generated answer count differs")
        accuracy = statistics.fmean(float(row["exact_match"]) for row in answers)
        if not close(accuracy, item["generation_exact_match_accuracy"]):
            raise ValueError("per-seed exact-match summary differs")
        updated_accuracy.append(accuracy)

    if len(result["base_heldout_generation"]) != 512:
        raise ValueError("base generation count differs")
    base_accuracy = statistics.fmean(float(row["exact_match"]) for row in result["base_heldout_generation"])
    if not close(base_accuracy, result["base_exact_match_accuracy"]):
        raise ValueError("base exact-match summary differs")
    bootstrap = result["paired_question_bootstrap"]
    per_question = [statistics.fmean(nll(result["seed_results"][s]["preference"]["relative_preference_margins"][i], beta)-nll(0.0,beta)
                                       for s in range(len(seeds))) for i in range(512)]
    observed = statistics.fmean(per_question)
    rng = random.Random(bootstrap["seed"])
    draws = sorted(statistics.fmean(per_question[rng.randrange(512)] for _ in range(512))
                   for _ in range(bootstrap["replicates"]))
    lower, upper = draws[int(0.025 * bootstrap["replicates"])], draws[int(0.975 * bootstrap["replicates"]) - 1]
    if not (close(observed, bootstrap["mean_nll_change_updated_minus_base"]) and
            close(lower, bootstrap["ci_lower"]) and close(upper, bootstrap["ci_upper"])):
        raise ValueError("paired bootstrap summary cannot be reconstructed")
    if len(bootstrap["per_question_mean_deltas"]) != 512 or not all(
            close(a, b) for a, b in zip(per_question, bootstrap["per_question_mean_deltas"])):
        raise ValueError("paired bootstrap inputs differ")

    mean_kl = statistics.fmean(item["preference"]["mean_full_vocab_token_kl_to_base"] for item in result["seed_results"])
    mean_accuracy = statistics.fmean(updated_accuracy)
    two_improve = sum(value < 0 for value in seed_deltas) >= 2
    two_not_worse = sum(value >= base_accuracy for value in updated_accuracy) >= 2
    passed = upper < 0 and two_improve and mean_kl <= 0.5 and mean_accuracy >= base_accuracy and two_not_worse
    decision = "sequence_dpo_confirmation_pass" if passed else "sequence_dpo_confirmation_non_pass"
    if result["decision"] != decision:
        raise ValueError("decision differs from frozen confirmation rule")
    for name, expected in (("per_seed_nll_change", seed_deltas),):
        if len(result[name]) != len(expected) or not all(close(a, b) for a, b in zip(result[name], expected)):
            raise ValueError(name + " differs from reconstructed metrics")
    if not close(mean_kl, result["mean_updated_token_kl"]) or not close(mean_accuracy, result["mean_updated_exact_match_accuracy"]):
        raise ValueError("aggregate metrics differ")

    report = {"audit": "passed", "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_hash,
              "source_manifest_sha256": source_manifest_hash, "update_rows": 128, "heldout_rows": 512,
              "per_seed_nll_change": seed_deltas,
              "paired_nll_change": observed, "paired_nll_ci_95": [lower, upper],
              "base_exact_match_accuracy": base_accuracy, "updated_exact_match_by_seed": updated_accuracy,
              "mean_kl": mean_kl, "decision_recomputed": decision}
    (out / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_directory.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
