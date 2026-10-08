#!/usr/bin/env python3
"""Offline integrity and metric audit for a cpu_lm_dpo_head_v1 result bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

from run_cpu_lm_dpo_head import (
    LOCK_PATH, SPEC_PATH, bootstrap_paired, canonical, generate_examples,
    kl_to_base, preference_metrics, sha256_bytes, sha256_file,
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = load(root / "manifest.json")
    listed = manifest.get("files", {})
    observed_paths = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    if set(listed) != observed_paths:
        raise ValueError("manifest file set differs from bundle contents")
    for relative, expected in listed.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"hash mismatch: {relative}")

    protocol = load(root / "protocol.snapshot.json")
    lock = load(root / "protocol.lock.snapshot.json")
    locked_hash = lock.pop("sha256", None)
    protocol_hash = sha256_bytes(canonical(protocol))
    if locked_hash != protocol_hash or canonical(lock) != canonical(protocol):
        raise ValueError("bundle protocol does not match its frozen lock")
    live_spec = load(SPEC_PATH)
    live_lock = load(LOCK_PATH)
    if canonical(live_spec) != canonical(protocol) or live_lock.get("sha256") != protocol_hash:
        raise ValueError("bundle protocol differs from current repository lock")

    failure_path = root / "failure.json"
    if failure_path.is_file():
        if (root / "summary.json").exists() or (root / "seed_records.json").exists():
            raise ValueError("failed bundle must not contain success metrics")
        failure = load(failure_path)
        if "exception_type" not in failure or "message" not in failure:
            raise ValueError("failure record is incomplete")
        return {"status": "failed_attempt_preserved", "bundle": str(root),
                "exception_type": failure["exception_type"], "message": failure["message"],
                "elapsed_seconds": failure.get("elapsed_seconds"),
                "peak_rss_bytes": failure.get("peak_rss_bytes"),
                "files_verified": len(listed),
                "interpretation": "No model-learning or held-out result is claimed from this failed attempt."}

    summary = load(root / "summary.json")
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError("summary protocol hash mismatch")
    if summary["runner_sha256"] != sha256_file(root / "runner.snapshot.py"):
        raise ValueError("summary runner hash mismatch")
    if summary["model_revision"] != protocol["model"]["revision"]:
        raise ValueError("model revision mismatch")
    if summary["device"] != "cpu" or not summary["network_disabled"] or summary["paid_compute"]:
        raise ValueError("recorded runtime violates the frozen compute mode")

    raw = load(root / "seed_records.json")
    expected_seeds = [str(seed) for seed in protocol["learner"]["seeds"]]
    if list(raw) != expected_seeds:
        raise ValueError("seed order/set differs from protocol")
    changes: list[tuple[int, float]] = []
    seed_summaries = []
    for seed_text in expected_seeds:
        seed = int(seed_text)
        record = raw[seed_text]
        train_rows = record["train_examples"]
        heldout_rows = record["heldout_examples"]
        if train_rows != generate_examples(seed, "train", protocol["task"]["train"]["examples_per_seed"]):
            raise ValueError(f"training examples do not reconstruct for seed {seed}")
        if heldout_rows != generate_examples(seed, "heldout", protocol["task"]["heldout"]["examples_per_seed"]):
            raise ValueError(f"held-out examples do not reconstruct for seed {seed}")
        train_operands = {value for row in train_rows for value in (row["left"], row["right"])}
        heldout_operands = {value for row in heldout_rows for value in (row["left"], row["right"])}
        if train_operands & heldout_operands:
            raise ValueError("train and held-out operand domains overlap")
        if any(not math.isfinite(float(x)) for x in record["heldout_base_margins"] + record["heldout_updated_margins"]):
            raise ValueError(f"non-finite held-out margin for seed {seed}")
        signs = [1 if row["correct_label"] == "A" else -1 for row in heldout_rows]
        base_metrics = preference_metrics(record["heldout_base_margins"], signs)
        updated_metrics = preference_metrics(record["heldout_updated_margins"], signs)
        metrics = record["metrics"]
        for key in base_metrics:
            if not close(metrics["heldout_base"][key], base_metrics[key]):
                raise ValueError(f"base held-out metric mismatch for seed {seed}: {key}")
            if not close(metrics["heldout_updated"][key], updated_metrics[key]):
                raise ValueError(f"updated held-out metric mismatch for seed {seed}: {key}")
        kl = kl_to_base(record["heldout_base_margins"], record["heldout_updated_margins"])
        if not close(metrics["heldout_mean_bernoulli_kl_updated_to_base"], kl):
            raise ValueError(f"held-out KL mismatch for seed {seed}")
        delta = statistics.fmean(
            (max(-s * u, 0.0) + math.log1p(math.exp(-abs(s * u)))) -
            (max(-s * b, 0.0) + math.log1p(math.exp(-abs(s * b))))
            for b, u, s in zip(record["heldout_base_margins"], record["heldout_updated_margins"], signs)
        )
        changes.extend((seed, (max(-s * u, 0.0) + math.log1p(math.exp(-abs(s * u)))) -
                        (max(-s * b, 0.0) + math.log1p(math.exp(-abs(s * b)))))
                       for b, u, s in zip(record["heldout_base_margins"], record["heldout_updated_margins"], signs))
        for name, values in record["adapter"].items():
            flat = values if values and not isinstance(values[0], list) else [v for row in values for v in row]
            if any(not math.isfinite(float(value)) for value in flat):
                raise ValueError(f"non-finite adapter value for seed {seed}: {name}")
        if metrics["updates"] != protocol["learner"]["updates"] or metrics["device"] != "cpu":
            raise ValueError(f"update count or device mismatch for seed {seed}")
        seed_summaries.append({"seed": seed, "heldout_nll_change": delta, "kl": kl})

    expected_change = statistics.fmean(value for _, value in changes)
    expected_interval = bootstrap_paired(changes, protocol["metrics"]["paired_bootstrap"]["seed"],
                                         protocol["metrics"]["paired_bootstrap"]["resamples"])
    if not close(summary["mean_heldout_nll_change_updated_minus_base"], expected_change):
        raise ValueError("aggregate held-out NLL change mismatch")
    if any(not close(a, b) for a, b in zip(summary["paired_seed_stratified_bootstrap_95pct"], expected_interval)):
        raise ValueError("aggregate paired bootstrap mismatch")
    all_improve = all(row["heldout_nll_change"] < 0 for row in seed_summaries)
    all_kl_ok = all(row["kl"] <= 0.5 for row in seed_summaries)
    passed = all_improve and expected_interval[1] < 0 and all_kl_ok
    decision = "positive_small_model_preference_result" if passed else "non_pass_or_null"
    if summary["decision"] != decision or summary["all_seeds_improve"] != all_improve or summary["all_seed_kl_within_limit"] != all_kl_ok:
        raise ValueError("decision does not reconstruct from recorded metrics")
    if summary["peak_rss_bytes"] > protocol["compute_limits"]["max_peak_rss_bytes"] or summary["total_wall_seconds"] > protocol["compute_limits"]["max_wall_seconds"]:
        raise ValueError("recorded run exceeded frozen resource limits")
    return {"status": "pass", "bundle": str(root), "decision": decision,
            "mean_heldout_nll_change_updated_minus_base": expected_change,
            "paired_seed_stratified_bootstrap_95pct": expected_interval,
            "per_seed": seed_summaries,
            "audit_limit": "Reconstructs bundle integrity, generated data, recorded-margin metrics and decision. It does not rerun the model forward pass or adapter optimizer; model and runtime provenance rely on recorded hashes and environment.",
            "files_verified": len(listed)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    report = audit(args.bundle)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
