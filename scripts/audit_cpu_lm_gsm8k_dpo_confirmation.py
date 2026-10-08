#!/usr/bin/env python3
"""Offline integrity, data, metric, and decision audit for GSM8K DPO confirmation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any

from gsm8k_preference_task import build_example

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_lm_gsm8k_dpo_confirmation_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_lm_gsm8k_dpo_confirmation_v1.lock.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, tol: float = 1e-9) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tol


def metrics(margins: list[float], signs: list[int]) -> dict[str, float]:
    nll = [max(-s * m, 0.0) + math.log1p(math.exp(-abs(s * m))) for m, s in zip(margins, signs)]
    acc = [1.0 if s * m > 0 else 0.5 if m == 0 else 0.0 for m, s in zip(margins, signs)]
    return {"mean_conditional_preference_nll": statistics.fmean(nll), "preference_accuracy": statistics.fmean(acc)}


def kl_to_base(base: list[float], updated: list[float]) -> float:
    total = 0.0
    for before, after in zip(base, updated):
        p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, before))))
        q = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, after))))
        total += q * math.log(max(q, 1e-300) / max(p, 1e-300))
        total += (1.0 - q) * math.log(max(1.0 - q, 1e-300) / max(1.0 - p, 1e-300))
    return total / len(base)


def bootstrap(values: list[tuple[int, float]], seed: int, resamples: int) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for seed_value, delta in values:
        grouped.setdefault(seed_value, []).append(delta)
    rng = random.Random(seed)
    samples = [statistics.fmean(statistics.fmean(rng.choices(rows, k=len(rows))) for rows in grouped.values())
               for _ in range(resamples)]
    samples.sort()
    return [samples[math.floor(.025 * (resamples - 1))], samples[math.floor(.975 * (resamples - 1))]]


def audit(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest = load(bundle / "manifest.json")
    listed = manifest.get("files", {})
    observed = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if set(listed) != observed:
        raise ValueError("manifest file inventory differs from bundle")
    for name, expected in listed.items():
        if sha256_file(bundle / name) != expected:
            raise ValueError(f"bundle hash mismatch: {name}")

    spec = load(bundle / "protocol.snapshot.json")
    lock = load(bundle / "protocol.lock.snapshot.json")
    expected_hash = lock.pop("sha256", None)
    spec_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if expected_hash != spec_hash or canonical(lock) != canonical(spec):
        raise ValueError("bundle protocol snapshot does not match its lock")
    if canonical(spec) != canonical(load(SPEC_PATH)) or load(LOCK_PATH).get("sha256") != spec_hash:
        raise ValueError("bundle protocol differs from current repository protocol")

    failure_path = bundle / "failure.json"
    if failure_path.exists():
        failure = load(failure_path)
        if (bundle / "summary.json").exists() or (bundle / "seed_records.json").exists():
            raise ValueError("failed bundle also contains result records")
        return {"status": "failed_attempt_preserved", "exception_type": failure.get("exception_type"),
                "message": failure.get("message"), "files_verified": len(listed),
                "interpretation": "No model update or held-out metric is claimed."}

    if sha256_file(bundle / "run_cpu_lm_gsm8k_dpo_confirmation.snapshot.py") != load(bundle / "summary.json")["runner_sha256"]:
        raise ValueError("runner source hash mismatch")
    if sha256_file(bundle / "gsm8k_preference_task.snapshot.py") != load(bundle / "summary.json")["task_helper_sha256"]:
        raise ValueError("task helper source hash mismatch")
    summary = load(bundle / "summary.json")
    if summary["protocol_sha256"] != spec_hash:
        raise ValueError("summary protocol hash mismatch")
    if summary["device"] != "cpu" or summary["paid_compute"] or not summary["network_disabled"]:
        raise ValueError("run violated locked compute mode")
    if summary["runtime"] != spec["runtime"] or summary["model_revision"] != spec["model"]["revision"]:
        raise ValueError("model or runtime provenance mismatch")
    if summary["dataset_train_sha256"] != spec["dataset"]["cached_train_arrow_sha256"] or summary["dataset_test_sha256"] != spec["dataset"]["cached_test_arrow_sha256"]:
        raise ValueError("dataset file hash mismatch")

    examples = load(bundle / "examples.json")
    train_rows, heldout_rows = examples["train"], examples["heldout"]
    if len(train_rows) != spec["dataset"]["train_examples"] or len(heldout_rows) != spec["dataset"]["confirmation_examples"]:
        raise ValueError("example count mismatch")
    for rows, split in ((train_rows, "train"), (heldout_rows, "test")):
        if len({row["question_sha256"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate question hash in {split}")
        for row in rows:
            rebuilt = build_example({"question": row["question"], "answer": row["source_answer"]}, row["dataset_index"], split)
            if rebuilt != row:
                raise ValueError(f"preference pair does not reconstruct: {split} row {row['dataset_index']}")
    train_ids = {r["question_sha256"] for r in train_rows}
    test_ids = {r["question_sha256"] for r in heldout_rows}
    if train_ids & test_ids:
        raise ValueError("train/test question overlap")

    # When the pinned cache is present, independently reconstruct the exact split selection.
    source_check = "embedded source examples only; cached upstream rows unavailable"
    if (DATA_DIR / "gsm8k-train.arrow").is_file() and (DATA_DIR / "gsm8k-test.arrow").is_file():
        if sha256_file(DATA_DIR / "gsm8k-train.arrow") != spec["dataset"]["cached_train_arrow_sha256"] or sha256_file(DATA_DIR / "gsm8k-test.arrow") != spec["dataset"]["cached_test_arrow_sha256"]:
            raise ValueError("local cached dataset hash differs from lock")
        import os
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        from datasets import load_dataset
        from gsm8k_preference_task import make_split
        train_ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="train")
        test_ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="test")
        expected_train = make_split(train_ds, "train", 544)[288:544]
        expected_test = make_split(test_ds, "test", len(test_ds))
        if train_rows != expected_train or heldout_rows != expected_test:
            raise ValueError("recorded examples do not match the pinned cached dataset and selection rule")
        if summary["dataset_train_fingerprint"] != train_ds._fingerprint or summary["dataset_test_fingerprint"] != test_ds._fingerprint:
            raise ValueError("dataset fingerprint mismatch")
        source_check = "pinned local train/test files, hashes, selected rows, and fingerprints verified"

    raw = load(bundle / "seed_records.json")
    expected_seed_keys = [str(seed) for seed in spec["learner"]["seeds"]]
    if list(raw) != expected_seed_keys:
        raise ValueError("seed set/order mismatch")
    signs_train = [1 if r["correct_label"] == "A" else -1 for r in train_rows]
    signs_test = [1 if r["correct_label"] == "A" else -1 for r in heldout_rows]
    all_changes: list[tuple[int, float]] = []
    reconstructed = []
    for seed_text in expected_seed_keys:
        record = raw[seed_text]
        if record["seed"] != int(seed_text):
            raise ValueError("seed record key mismatch")
        for key in ("train_base_margins", "train_updated_margins"):
            if len(record[key]) != len(train_rows) or any(not math.isfinite(float(x)) for x in record[key]):
                raise ValueError(f"invalid {key} for seed {seed_text}")
        for key in ("heldout_base_margins", "heldout_updated_margins"):
            if len(record[key]) != len(heldout_rows) or any(not math.isfinite(float(x)) for x in record[key]):
                raise ValueError(f"invalid {key} for seed {seed_text}")
        metrics_obj = record["metrics"]
        for condition, margins, signs in (("train_base", record["train_base_margins"], signs_train),
                                          ("train_updated", record["train_updated_margins"], signs_train),
                                          ("heldout_base", record["heldout_base_margins"], signs_test),
                                          ("heldout_updated", record["heldout_updated_margins"], signs_test)):
            expected = metrics(margins, signs)
            for name, value in expected.items():
                if not close(metrics_obj[condition][name], value):
                    raise ValueError(f"metric mismatch: seed {seed_text} {condition}.{name}")
        kl = kl_to_base(record["heldout_base_margins"], record["heldout_updated_margins"])
        if not close(metrics_obj["heldout_mean_bernoulli_kl_updated_to_base"], kl):
            raise ValueError(f"heldout KL mismatch: seed {seed_text}")
        deltas = []
        for before, after, sign in zip(record["heldout_base_margins"], record["heldout_updated_margins"], signs_test):
            base_nll = max(-sign * before, 0.0) + math.log1p(math.exp(-abs(before)))
            updated_nll = max(-sign * after, 0.0) + math.log1p(math.exp(-abs(after)))
            deltas.append(updated_nll - base_nll)
            all_changes.append((int(seed_text), updated_nll - base_nll))
        reconstructed.append({"seed": int(seed_text), "heldout_nll_change": statistics.fmean(deltas), "kl": kl})
        for name, matrix in record["adapter"].items():
            if len(matrix) != spec["learner"]["adapter_rank"] and name.startswith("B_"):
                raise ValueError(f"adapter rank mismatch: {seed_text}.{name}")
            if any(not math.isfinite(float(v)) for row in matrix for v in row) if matrix and isinstance(matrix[0], list) else any(not math.isfinite(float(v)) for v in matrix):
                raise ValueError(f"non-finite adapter parameter: {seed_text}.{name}")
        if metrics_obj["updates"] != spec["learner"]["updates"] or metrics_obj["device"] != "cpu":
            raise ValueError(f"update count/device mismatch for seed {seed_text}")

    mean_change = statistics.fmean(delta for _, delta in all_changes)
    interval = bootstrap(all_changes, spec["metrics"]["paired_bootstrap"]["seed"], spec["metrics"]["paired_bootstrap"]["resamples"])
    all_improve = all(row["heldout_nll_change"] < 0 for row in reconstructed)
    all_kl_ok = all(row["kl"] <= 0.5 for row in reconstructed)
    passed = all_improve and interval[1] < 0 and all_kl_ok
    decision = "positive_small_model_preference_result" if passed else "non_pass_or_null"
    if not close(summary["mean_heldout_nll_change_updated_minus_base"], mean_change):
        raise ValueError("aggregate NLL change mismatch")
    if any(not close(a, b) for a, b in zip(summary["paired_seed_stratified_bootstrap_95pct"], interval)):
        raise ValueError("bootstrap interval mismatch")
    if summary["decision"] != decision or summary["all_seeds_improve"] != all_improve or summary["all_seed_kl_within_limit"] != all_kl_ok:
        raise ValueError("decision does not reconstruct")
    if summary["peak_rss_bytes"] > spec["compute_limits"]["max_peak_rss_bytes"] or summary["total_wall_seconds"] > spec["compute_limits"]["max_wall_seconds"]:
        raise ValueError("resource limit exceeded")
    if summary["compute_limits_respected"] is not True:
        raise ValueError("resource-limit flag is not affirmative")
    if MODEL_DIR.is_dir():
        for name, expected in summary["model_file_sha256"].items():
            if sha256_file(MODEL_DIR / name) != expected:
                raise ValueError(f"local model hash mismatch: {name}")
    try:
        bundle_name = bundle.relative_to(ROOT).as_posix()
    except ValueError:
        bundle_name = bundle.name
    return {"status": "pass", "bundle": bundle_name, "decision": decision,
            "mean_heldout_nll_change_updated_minus_base": mean_change,
            "paired_seed_stratified_bootstrap_95pct": interval, "per_seed": reconstructed,
            "source_check": source_check,
            "audit_limit": "Reconstructs data selection when the pinned cache is present, bundle hashes, preference pairs, recorded-margin metrics, KL, bootstrap, and decision. It does not rerun the model forward pass or optimizer; model/runtime integrity relies on recorded hashes and environment.",
            "files_verified": len(listed)}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    report = audit(args.bundle)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
