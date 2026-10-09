#!/usr/bin/env python3
"""Regenerate and audit the frozen CPU preference noise/shift experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/synthetic_preference_robustness_v1.json"
LOCK_PATH = ROOT / "protocols/synthetic_preference_robustness_v1.lock.json"
RUNNER_PATH = ROOT / "scripts/run_synthetic_preference_robustness.py"
AUDITOR_PATH = Path(__file__).resolve()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def close(actual: Any, expected: float, label: str) -> None:
    if (not isinstance(actual, (int, float)) or not math.isfinite(float(actual))
            or not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(f"reconstruction mismatch for {label}: recorded={actual!r} expected={expected!r}")


def load_runner(path: Path):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("vare_preference_robustness_audit_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runner snapshot: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_manifest(bundle: Path) -> dict[str, Any]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest files field is not a mapping")
    expected = set()
    for name, digest in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"unsafe manifest path: {name}")
        path = bundle.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(bundle.resolve()) or not path.is_file():
            raise ValueError(f"missing or escaping file: {name}")
        if sha(path) != digest:
            raise ValueError(f"hash mismatch: {name}")
        expected.add(relative.as_posix())
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
              if path.is_file() and path.name != "manifest.json"}
    if actual != expected:
        raise ValueError(f"manifest inventory mismatch: missing={sorted(expected-actual)}, unlisted={sorted(actual-expected)}")
    return manifest


def verify(bundle: Path) -> dict[str, Any]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock_body = dict(lock)
    recorded_lock_hash = lock_body.pop("sha256", None)
    protocol_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_body != spec or recorded_lock_hash != protocol_hash:
        raise ValueError("checked-in protocol and lock disagree")
    if json.loads((bundle / "protocol.json").read_text()) != spec:
        raise ValueError("bundle protocol snapshot differs from frozen protocol")
    if json.loads((bundle / "protocol.lock.json").read_text()) != lock:
        raise ValueError("bundle protocol lock snapshot differs from frozen lock")
    runner_snapshot = bundle / "source/run_synthetic_preference_robustness.py"
    auditor_snapshot = bundle / "source/audit_synthetic_preference_robustness.py"
    if not runner_snapshot.is_file() or sha(runner_snapshot) != sha(RUNNER_PATH):
        raise ValueError("runner snapshot differs from the frozen runner source")
    if not auditor_snapshot.is_file() or sha(auditor_snapshot) != sha(AUDITOR_PATH):
        raise ValueError("auditor snapshot differs from the checked-in auditor")
    manifest = verify_manifest(bundle)
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    for key, value in (
        ("protocol_id", spec["protocol_id"]),
        ("protocol_canonical_sha256", protocol_hash),
        ("protocol_file_sha256", sha(bundle / "protocol.json")),
        ("protocol_lock_sha256", sha(bundle / "protocol.lock.json")),
        ("source_script_sha256", sha(runner_snapshot)),
        ("audit_script_sha256", sha(auditor_snapshot)),
    ):
        if summary.get(key) != value:
            raise ValueError(f"summary {key} mismatch")

    runner = load_runner(runner_snapshot)
    gradient_check = runner.finite_difference_check(spec)
    if gradient_check != summary.get("gradient_check") or not gradient_check["passed"]:
        raise ValueError("finite-difference gradient check did not reconstruct")
    data = spec["data"]
    learner = spec["learner"]
    base_theta = [[0.0] * runner.DIM for _ in range(runner.N_ACTIONS)]
    reconstructed_by_arm = {
        arm: {"base_teacher": [], "shifted_teacher": []} for arm in spec["arms"]
    }
    primary_improvements = []
    flip_counts_by_rate = {"flip_20pct": [], "flip_40pct": []}
    expected_raw_files = {"protocol.json", "protocol.lock.json", "gradient-check.json",
                          "source/run_synthetic_preference_robustness.py",
                          "source/audit_synthetic_preference_robustness.py", "summary.json"}

    for seed in spec["seeds"]:
        seed_path = bundle / "seeds" / f"seed-{seed}.json"
        expected_raw_files.add(seed_path.relative_to(bundle).as_posix())
        raw = json.loads(seed_path.read_text(encoding="utf-8"))
        train_designs = runner.generate_designs(
            random.Random(seed + data["training_seed_offset"]),
            data["training_contexts"], data["comparisons_per_training_context"])
        train_examples = runner.label_designs(
            train_designs, random.Random(seed + data["base_label_seed_offset"]),
            runner.teacher_utility)
        held_designs = runner.generate_designs(
            random.Random(seed + data["heldout_design_seed_offset"]),
            data["heldout_contexts"], data["comparisons_per_heldout_context"])
        held_base = runner.label_designs(
            held_designs, random.Random(seed + data["heldout_base_label_seed_offset"]),
            runner.teacher_utility)
        held_shifted = runner.label_designs(
            held_designs, random.Random(seed + data["shifted_label_seed_offset"]),
            runner.shifted_teacher_utility)
        if raw.get("seed") != seed:
            raise ValueError(f"seed record mismatch: {seed}")
        for field, expected in (("train_designs", train_designs), ("train_examples", train_examples),
                                ("heldout_designs", held_designs),
                                ("heldout_base_examples", held_base),
                                ("heldout_shifted_examples", held_shifted)):
            if raw.get(field) != expected:
                raise ValueError(f"seed {seed} data regeneration mismatch: {field}")
        training_contexts = {tuple(item["x"]) for item in train_designs}
        heldout_contexts = {tuple(item["x"]) for item in held_designs}
        if training_contexts & heldout_contexts:
            raise ValueError(f"seed {seed} has training/held-out context overlap")
        if ([(item["x"], item["a"], item["b"]) for item in held_base]
                != [(item["x"], item["a"], item["b"]) for item in held_shifted]):
            raise ValueError(f"seed {seed} base/shift evaluation designs are not paired")

        per_arm = raw.get("metrics", {})
        if set(per_arm) != set(spec["arms"]):
            raise ValueError(f"seed {seed} arm inventory mismatch")
        seed_seconds = raw.get("seed_wall_seconds")
        if not isinstance(seed_seconds, (int, float)) or not math.isfinite(seed_seconds) or seed_seconds <= 0:
            raise ValueError(f"seed {seed} runtime is missing or invalid")
        reference = {
            "base_teacher": {**runner.evaluate(base_theta, held_base), "parameter_l2_delta": 0.0, "updates": 0},
            "shifted_teacher": {**runner.evaluate(base_theta, held_shifted, runner.shifted_teacher_utility),
                                 "parameter_l2_delta": 0.0, "updates": 0},
        }
        close(per_arm["reference"].get("train_flip_count"), 0.0, f"seed {seed} reference flip count")
        for condition in ("base_teacher", "shifted_teacher"):
            for metric, value in reference[condition].items():
                close(per_arm["reference"][condition].get(metric), value,
                      f"seed {seed} reference {condition}.{metric}")
            reconstructed_by_arm["reference"][condition].append({
                "seed": seed, **reference[condition], "train_flip_count": 0,
                "train_flip_fraction": 0.0, "seed_wall_seconds": seed_seconds,
            })

        noise_rng = random.Random(seed + data["noise_seed_offset"])
        noise_uniforms = [noise_rng.random() for _ in train_examples]
        for arm, rate in (("clean", 0.0), ("flip_20pct", 0.2), ("flip_40pct", 0.4)):
            examples = []
            flips = 0
            for item, uniform in zip(train_examples, noise_uniforms):
                flip = uniform < rate
                flips += int(flip)
                examples.append({**item,
                                 "chosen": item["rejected"] if flip else item["chosen"],
                                 "rejected": item["chosen"] if flip else item["rejected"]})
            if arm != "clean":
                flip_counts_by_rate[arm].append(flips)
            theta = runner.train(base_theta, examples, learner["beta"],
                                 learner["learning_rate"], learner["updates"])
            metrics_by_condition = {
                "base_teacher": {**runner.evaluate(theta, held_base),
                                 "parameter_l2_delta": runner.parameter_delta(theta),
                                 "updates": learner["updates"]},
                "shifted_teacher": {**runner.evaluate(theta, held_shifted, runner.shifted_teacher_utility),
                                    "parameter_l2_delta": runner.parameter_delta(theta),
                                    "updates": learner["updates"]},
            }
            close(per_arm[arm].get("train_flip_count"), float(flips), f"seed {seed} {arm} flips")
            close(per_arm[arm].get("train_flip_fraction"), flips / len(examples), f"seed {seed} {arm} flip fraction")
            for condition, metrics in metrics_by_condition.items():
                for metric, value in metrics.items():
                    close(per_arm[arm][condition].get(metric), value,
                          f"seed {seed} {arm}.{condition}.{metric}")
                reconstructed_by_arm[arm][condition].append({
                    "seed": seed, **metrics, "train_flip_count": flips,
                    "train_flip_fraction": flips / len(examples),
                    "seed_wall_seconds": seed_seconds,
                })
        primary_improvements.append(
            reference["base_teacher"]["heldout_preference_nll"]
            - per_arm["clean"]["base_teacher"]["heldout_preference_nll"])

    if summary.get("arms") != reconstructed_by_arm:
        raise ValueError("aggregate per-seed arm metrics do not reconstruct")
    if summary.get("clean_vs_reference_base_nll_improvement_per_seed") != primary_improvements:
        raise ValueError("per-seed primary improvements do not reconstruct")
    bootstrap = spec["metrics"]["paired_bootstrap"]
    interval = runner.bootstrap_interval(primary_improvements, bootstrap["resamples"], bootstrap["seed"])
    mean = statistics.fmean(primary_improvements)
    improved = sum(value > 0 for value in primary_improvements)
    mean_kl = statistics.fmean(item["mean_kl_to_uniform"]
                               for item in reconstructed_by_arm["clean"]["base_teacher"])
    acceptance = spec["metrics"]["primary_acceptance"]
    accepted = (mean >= acceptance["mean_nll_improvement_at_least"]
                and interval[0] > acceptance["paired_bootstrap_lower_bound_above"]
                and improved >= acceptance["minimum_seeds_improved"]
                and mean_kl <= acceptance["maximum_mean_clean_policy_kl_to_uniform"])
    for key, value in (
        ("clean_vs_reference_base_nll_improvement_mean", mean),
        ("clean_vs_reference_base_nll_improvement_paired_bootstrap_95pct", interval),
        ("clean_improved_seed_count", improved),
        ("clean_mean_kl_to_uniform", mean_kl),
        ("acceptance_passed", accepted),
    ):
        if summary.get(key) != value and not (isinstance(value, float) and math.isclose(summary.get(key), value, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f"summary {key} does not reconstruct")
    actual_raw_files = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
                        if path.is_file() and path.name != "manifest.json"}
    if actual_raw_files != expected_raw_files:
        raise ValueError(f"unexpected evidence inventory: {sorted(actual_raw_files ^ expected_raw_files)}")
    return {
        "status": "verified",
        "protocol_id": spec["protocol_id"],
        "seed_count": len(spec["seeds"]),
        "reconstructed_arm_condition_seed_metrics": sum(
            len(rows) for conditions in reconstructed_by_arm.values() for rows in conditions.values()),
        "train_and_holdout_data_regenerated": True,
        "heldout_contexts_disjoint_from_training": True,
        "base_and_shifted_evaluation_pairs_matched": True,
        "flip_count_by_rate": {arm: {"minimum": min(values), "maximum": max(values),
                                      "mean": statistics.fmean(values)}
                               for arm, values in flip_counts_by_rate.items()},
        "mean_clean_base_nll_improvement": mean,
        "paired_bootstrap_95pct": interval,
        "mean_clean_kl_to_uniform": mean_kl,
        "acceptance_passed": accepted,
        "manifest_file_count": len(manifest["files"]),
        "claim_limit": spec["claim_boundary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.bundle.expanduser().resolve())
    except Exception as exc:
        print(f"audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
