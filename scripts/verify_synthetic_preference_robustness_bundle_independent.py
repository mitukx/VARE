#!/usr/bin/env python3
"""Independently replay a retained synthetic preference bundle using stdlib only.

This verifier does not import the experiment runner or its auditor. It checks the
bundle manifest, reconstructs labels from retained designs and frozen RNG streams,
re-trains every policy, and recomputes all reported metrics and the primary gate.
It does not regenerate Gaussian contexts/action-pair designs or prove external
human reproduction; see the report for this verifier's scope.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from pathlib import Path
from typing import Any


DIM = 8
ACTION_COUNT = 4
METRIC_TOLERANCE = 2e-10


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def teacher_utility(context: list[float], action: int) -> float:
    return sum(
        context[d] * (
            math.sin(0.7 * (action + 1) * (d + 1))
            + 0.5 * math.cos(0.31 * (action + 2) * (d + 1))
        )
        for d in range(DIM)
    )


def shifted_utility(context: list[float], action: int) -> float:
    shift = (-1.5, -0.5, 0.5, 1.5)[action]
    return teacher_utility(context, action) + 1.25 * context[0] * shift


def check_row(row: dict[str, Any], design: dict[str, Any], expected: tuple[int, int], label: str) -> None:
    if set(row) != {"x", "a", "b", "chosen", "rejected"}:
        raise ValueError("example has unexpected or missing fields")
    if row["x"] != design["x"] or row["a"] != design["a"] or row["b"] != design["b"]:
        raise ValueError(label + " does not match its retained design")
    if (row["chosen"], row["rejected"]) != expected:
        raise ValueError(label + " has an invalid chosen/rejected orientation")
    if row["chosen"] not in (row["a"], row["b"]) or row["rejected"] not in (row["a"], row["b"]):
        raise ValueError(label + " action indices differ from the design pair")


def replay_labels(designs: list[dict[str, Any]], seed: int, offset: int, utility_fn) -> list[dict[str, Any]]:
    rng = random.Random(seed + offset)
    examples = []
    for design in designs:
        p_a = sigmoid(utility_fn(design["x"], design["a"]) - utility_fn(design["x"], design["b"]))
        chosen = design["a"] if rng.random() < p_a else design["b"]
        examples.append({**design, "chosen": chosen, "rejected": design["b"] if chosen == design["a"] else design["a"]})
    return examples


def flipped_examples(clean: list[dict[str, Any]], seed: int, offset: int, threshold: float):
    rng = random.Random(seed + offset)
    uniforms = [rng.random() for _ in clean]
    output, flipped = [], []
    for item, draw in zip(clean, uniforms):
        did_flip = draw < threshold
        flipped.append(did_flip)
        output.append({**item,
                       "chosen": item["rejected"] if did_flip else item["chosen"],
                       "rejected": item["chosen"] if did_flip else item["rejected"]})
    return output, flipped


def dot(left: list[float], right: list[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def margin(theta: list[list[float]], item: dict[str, Any]) -> float:
    return dot(theta[item["chosen"]], item["x"]) - dot(theta[item["rejected"]], item["x"])


def fit_policy(examples: list[dict[str, Any]], beta: float, learning_rate: float, updates: int):
    theta = [[0.0] * DIM for _ in range(ACTION_COUNT)]
    count = len(examples)
    for _ in range(updates):
        gradient = [[0.0] * DIM for _ in range(ACTION_COUNT)]
        for item in examples:
            chosen, rejected, x = item["chosen"], item["rejected"], item["x"]
            coefficient = -beta * sigmoid(-beta * margin(theta, item)) / count
            for d, feature in enumerate(x):
                gradient[chosen][d] += coefficient * feature
                gradient[rejected][d] -= coefficient * feature
        for action in range(ACTION_COUNT):
            for d in range(DIM):
                theta[action][d] -= learning_rate * gradient[action][d]
    return theta


def evaluate(theta: list[list[float]], examples: list[dict[str, Any]], utility_fn) -> dict[str, float]:
    nll, accuracy, brier, kl, expected_utility = [], [], [], [], []
    for item in examples:
        value = margin(theta, item)
        probability = sigmoid(value)
        nll.append(softplus(-value))
        accuracy.append(1.0 if value > 0 else 0.5 if value == 0 else 0.0)
        brier.append((probability - 1.0) ** 2)
        logits = [dot(theta[a], item["x"]) for a in range(ACTION_COUNT)]
        peak = max(logits)
        weights = [math.exp(v - peak) for v in logits]
        denominator = sum(weights)
        policy = [v / denominator for v in weights]
        kl.append(sum(p * math.log(p * ACTION_COUNT) for p in policy if p > 0.0))
        expected_utility.append(sum(policy[a] * utility_fn(item["x"], a) for a in range(ACTION_COUNT)))
    return {
        "heldout_preference_nll": statistics.fmean(nll),
        "heldout_preference_accuracy": statistics.fmean(accuracy),
        "heldout_brier": statistics.fmean(brier),
        "mean_kl_to_uniform": statistics.fmean(kl),
        "mean_teacher_expected_utility": statistics.fmean(expected_utility),
        "parameter_l2_delta": math.sqrt(sum(v * v for row in theta for v in row)),
        "updates": float(0),
    }


def stable_bootstrap_interval(values: list[float], count: int, seed: int) -> list[float]:
    """Percentile bootstrap using Python's specified random() stream directly."""
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(count):
        means.append(statistics.fmean(values[min(int(rng.random() * n), n - 1)] for _ in range(n)))
    means.sort()
    return [means[math.floor(0.025 * (count - 1))], means[math.floor(0.975 * (count - 1))]]


def close(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > METRIC_TOLERANCE:
        raise ValueError("metric mismatch for %s: bundle=%r independent=%r" % (label, actual, expected))


def validate_manifest(bundle: Path) -> dict[str, Any]:
    manifest = read_json(bundle / "manifest.json")
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, dict):
        raise ValueError("unsupported bundle manifest")
    observed = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if observed != set(files):
        raise ValueError("bundle file inventory differs from its manifest")
    for relative, digest in files.items():
        if sha256_file(bundle / relative) != digest:
            raise ValueError("bundle SHA-256 mismatch: " + relative)
    return manifest


def audit(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest = validate_manifest(bundle)
    spec = read_json(bundle / "protocol.json")
    lock = read_json(bundle / "protocol.lock.json")
    locked = dict(lock)
    frozen_hash = locked.pop("sha256", None)
    protocol_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if frozen_hash != protocol_hash or canonical(locked) != canonical(spec):
        raise ValueError("retained protocol does not match its lock")
    if spec.get("protocol_id") != "vare-synthetic-preference-robustness-v1":
        raise ValueError("unsupported protocol id")

    data = spec["data"]
    learner = spec["learner"]
    expected_train = data["training_contexts"] * data["comparisons_per_training_context"]
    expected_eval = data["heldout_contexts"] * data["comparisons_per_heldout_context"]
    seed_metrics = {arm: {condition: [] for condition in ("base_teacher", "shifted_teacher")}
                    for arm in ("reference", "clean", "flip_20pct", "flip_40pct")}
    improvements = []

    for seed in spec["seeds"]:
        record = read_json(bundle / "seeds" / ("seed-%s.json" % seed))
        if record.get("seed") != seed:
            raise ValueError("seed file names and seed records differ")
        td, train = record["train_designs"], record["train_examples"]
        hd, base_labels, shifted_labels = (record["heldout_designs"], record["heldout_base_examples"],
                                           record["heldout_shifted_examples"])
        if not (len(td) == len(train) == expected_train):
            raise ValueError("training design count differs from the frozen protocol")
        if not (len(hd) == len(base_labels) == len(shifted_labels) == expected_eval):
            raise ValueError("held-out design count differs from the frozen protocol")
        for design, row in zip(td, train):
            if (design.get("a"), design.get("b")) == (None, None):
                raise ValueError("training design lacks an action pair")
            check_row(row, design, (row.get("chosen"), row.get("rejected")), "training example")
            if row["chosen"] == row["rejected"]:
                raise ValueError("training example has a tied pair")
            if len(row["x"]) != DIM or not all(math.isfinite(v) for v in row["x"]):
                raise ValueError("invalid training context")
        for design, base, shifted in zip(hd, base_labels, shifted_labels):
            for row, kind in ((base, "base"), (shifted, "shifted")):
                check_row(row, design, (row.get("chosen"), row.get("rejected")), kind + " held-out example")
                if row["chosen"] == row["rejected"]:
                    raise ValueError("held-out example has a tied pair")
                if len(row["x"]) != DIM or not all(math.isfinite(v) for v in row["x"]):
                    raise ValueError("invalid held-out context")
        train_contexts = {tuple(row["x"]) for row in train}
        heldout_contexts = {tuple(row["x"]) for row in base_labels}
        train_counts = {context: sum(tuple(row["x"]) == context for row in train) for context in train_contexts}
        heldout_counts = {context: sum(tuple(row["x"]) == context for row in base_labels)
                          for context in heldout_contexts}
        if len(train_contexts) != data["training_contexts"] or set(train_counts.values()) != {
                data["comparisons_per_training_context"]}:
            raise ValueError("training contexts do not match the frozen group structure")
        if len(heldout_contexts) != data["heldout_contexts"] or set(heldout_counts.values()) != {
                data["comparisons_per_heldout_context"]}:
            raise ValueError("held-out contexts do not match the frozen group structure")
        if train_contexts & heldout_contexts:
            raise ValueError("training and held-out context sets overlap")
        if any(b["x"] != s["x"] or b["a"] != s["a"] or b["b"] != s["b"]
               for b, s in zip(base_labels, shifted_labels)):
            raise ValueError("base and shifted evaluations are not paired")

        # Recreate labels conditional on the retained action-pair designs. This avoids
        # relying on version-sensitive Gaussian generation or action-pair sampling.
        reconstructed_train = replay_labels(td, seed, data["base_label_seed_offset"], teacher_utility)
        for given, expected in zip(train, reconstructed_train):
            if given["chosen"] != expected["chosen"] or given["rejected"] != expected["rejected"]:
                raise ValueError("training labels do not match their frozen teacher/RNG stream")
        reconstructed_base = replay_labels(hd, seed, data["heldout_base_label_seed_offset"], teacher_utility)
        reconstructed_shifted = replay_labels(hd, seed, data["shifted_label_seed_offset"], shifted_utility)
        for given, expected in zip(base_labels, reconstructed_base):
            if given["chosen"] != expected["chosen"]:
                raise ValueError("base held-out labels do not match their frozen teacher/RNG stream")
        for given, expected in zip(shifted_labels, reconstructed_shifted):
            if given["chosen"] != expected["chosen"]:
                raise ValueError("shifted held-out labels do not match their frozen teacher/RNG stream")

        noisy20, mask20 = flipped_examples(train, seed, data["noise_seed_offset"], 0.2)
        noisy40, mask40 = flipped_examples(train, seed, data["noise_seed_offset"], 0.4)
        if any(a and not b for a, b in zip(mask20, mask40)):
            raise ValueError("the frozen nested flip conditions are violated")
        arms = {"reference": None, "clean": train, "flip_20pct": noisy20, "flip_40pct": noisy40}
        for arm, examples in arms.items():
            theta = [[0.0] * DIM for _ in range(ACTION_COUNT)] if examples is None else fit_policy(
                examples, learner["beta"], learner["learning_rate"], learner["updates"])
            for condition, rows, utility_fn in (
                ("base_teacher", base_labels, teacher_utility),
                ("shifted_teacher", shifted_labels, shifted_utility),
            ):
                computed = evaluate(theta, rows, utility_fn)
                recorded = record["metrics"][arm][condition]
                computed["updates"] = 0.0 if arm == "reference" else float(learner["updates"])
                for key, value in computed.items():
                    close(recorded[key], value, "%s/%s/%s/%s" % (seed, arm, condition, key))
                seed_metrics[arm][condition].append({"seed": seed, **computed})
            if arm == "reference":
                flip_count = 0
            elif arm == "clean":
                flip_count = 0
            else:
                flip_count = sum(mask20 if arm == "flip_20pct" else mask40)
            close(record["metrics"][arm]["train_flip_count"], flip_count, "%s/%s/flip_count" % (seed, arm))
            close(record["metrics"][arm]["train_flip_fraction"], flip_count / len(train), "%s/%s/flip_fraction" % (seed, arm))

        improvements.append(seed_metrics["reference"]["base_teacher"][-1]["heldout_preference_nll"]
                            - seed_metrics["clean"]["base_teacher"][-1]["heldout_preference_nll"])

    summary = read_json(bundle / "summary.json")
    mean_gain = statistics.fmean(improvements)
    interval = stable_bootstrap_interval(improvements,
        spec["metrics"]["paired_bootstrap"]["resamples"], spec["metrics"]["paired_bootstrap"]["seed"])
    mean_kl = statistics.fmean(row["mean_kl_to_uniform"] for row in seed_metrics["clean"]["base_teacher"])
    improved = sum(x > 0.0 for x in improvements)
    acceptance = spec["metrics"]["primary_acceptance"]
    passed = (mean_gain >= acceptance["mean_nll_improvement_at_least"]
              and interval[0] > acceptance["paired_bootstrap_lower_bound_above"]
              and improved >= acceptance["minimum_seeds_improved"]
              and mean_kl <= acceptance["maximum_mean_clean_policy_kl_to_uniform"])
    close(summary["clean_vs_reference_base_nll_improvement_mean"], mean_gain, "summary/mean_improvement")
    close(summary["clean_mean_kl_to_uniform"], mean_kl, "summary/mean_kl")
    close(summary["clean_improved_seed_count"], improved, "summary/improved_seed_count")
    close(summary["clean_vs_reference_base_nll_improvement_paired_bootstrap_95pct"][0], interval[0], "summary/bootstrap_low")
    close(summary["clean_vs_reference_base_nll_improvement_paired_bootstrap_95pct"][1], interval[1], "summary/bootstrap_high")
    if summary["acceptance_passed"] is not passed:
        raise ValueError("frozen acceptance decision differs from independent replay")
    if set(summary["arms"]) != set(seed_metrics):
        raise ValueError("summary arm inventory differs")
    for arm, conditions in seed_metrics.items():
        for condition, rows in conditions.items():
            recorded_rows = summary["arms"][arm][condition]
            if len(rows) != len(recorded_rows):
                raise ValueError("summary seed count differs for %s/%s" % (arm, condition))
            for expected_row, recorded_row in zip(rows, recorded_rows):
                for key, value in expected_row.items():
                    if key in recorded_row:
                        close(recorded_row[key], value, "summary/%s/%s/%s/%s" % (arm, condition, expected_row["seed"], key))

    return {
        "status": "pass",
        "verifier": Path(__file__).name,
        "verifier_sha256": sha256_file(Path(__file__)),
        "python_version": sys.version,
        "platform": platform.platform(),
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "protocol_canonical_sha256": protocol_hash,
        "independent_training_and_metric_replay": "pass",
        "label_replay_from_retained_designs": "pass",
        "design_generation_replay": "not_performed; Python random.gauss/sample version behavior is not frozen",
        "external_human_reproduction": False,
        "acceptance_passed": passed,
        "seed_count": len(improvements),
        "mean_clean_nll_improvement": mean_gain,
        "bootstrap_95pct": interval,
        "mean_clean_kl": mean_kl,
        "clean_seeds_improved": improved,
        "files_verified": len(manifest["files"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="retained confirmation bundle directory")
    parser.add_argument("--output", type=Path, help="optional JSON path for a compact audit record")
    args = parser.parse_args()
    try:
        result = audit(args.bundle)
        payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "fail", "error": type(exc).__name__, "message": str(exc)}, indent=2), file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
