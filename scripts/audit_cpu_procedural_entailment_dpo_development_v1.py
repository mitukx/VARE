#!/usr/bin/env python3
"""Reconstruct the frozen synthetic procedural-DPO development bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "e43654607952fa410a2dbc94c6f3fd54c591226a"
PRE_RUN_FILES = {
    "protocols/cpu_procedural_entailment_dpo_development_v1.json": "060bc1d7a684535466074e932dc92dbd4623e098",
    "protocols/cpu_procedural_entailment_dpo_development_v1.lock.json": "da0af614cfa9389d0082d1af485a5e0e305dcab9",
    "protocols/cpu_procedural_entailment_dpo_pilot_hashes_v1.json": "380115df13974ed7f0e9f9c6c421be187534ba4d",
    "scripts/procedural_entailment_task.py": "49864dce1821900f58991469368b3006a69ecf89",
    "scripts/run_cpu_procedural_entailment_dpo_development_v1.py": "5cf17a01de7b0dacb0df743dc8f8a350a8f01266",
}
sys.path.insert(0, str(ROOT / "scripts"))
from procedural_entailment_task import generate_split  # noqa: E402


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected, tolerance=1e-6):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"metric mismatch: {actual!r} != {expected!r}")


def manifest_check(output: Path):
    manifest = read_json(output / "manifest.json")["files"]
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*")
              if p.is_file() and p.name not in ("manifest.json", "audit.json")}
    if set(manifest) != actual:
        raise ValueError("bundle manifest inventory mismatch")
    for name, digest in manifest.items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"bundle file hash mismatch: {name}")


def frozen_source_check(output: Path):
    snapshots = {
        "protocols/cpu_procedural_entailment_dpo_development_v1.json": "protocol.json",
        "protocols/cpu_procedural_entailment_dpo_development_v1.lock.json": "protocol.lock.json",
        "scripts/procedural_entailment_task.py": "task_generator.py",
        "scripts/run_cpu_procedural_entailment_dpo_development_v1.py": "runner.py",
        "protocols/cpu_procedural_entailment_dpo_pilot_hashes_v1.json": "pilot_hashes.json",
    }
    for source, snapshot in snapshots.items():
        blob = subprocess.check_output(["git", "rev-parse", f"{FROZEN_COMMIT}:{source}"], cwd=ROOT, text=True).strip()
        if blob != PRE_RUN_FILES[source]:
            raise ValueError("hard-coded pre-run Git blob id changed")
        if subprocess.check_output(["git", "hash-object", str(output / snapshot)], cwd=ROOT, text=True).strip() != blob:
            raise ValueError(f"{snapshot} differs from source committed before the model run")


def sigmoid(value):
    if value >= 0:
        exp = math.exp(-value)
        return 1 / (1 + exp)
    exp = math.exp(value)
    return exp / (1 + exp)


def reconstructed_metrics(rows, margins, deltas):
    labels = [row["label"] == "Yes" for row in rows]
    logits = [float(m) + float(d) for m, d in zip(margins, deltas)]
    probs = [sigmoid(x) for x in logits]
    preds = [x > 0 for x in logits]
    correct = [int(a == b) for a, b in zip(preds, labels)]
    yes_idx = [i for i, label in enumerate(labels) if label]
    no_idx = [i for i, label in enumerate(labels) if not label]
    yes_recall = sum(correct[i] for i in yes_idx) / len(yes_idx)
    no_recall = sum(correct[i] for i in no_idx) / len(no_idx)
    f1s = []
    for label in (True, False):
        tp = sum(p == label and y == label for p, y in zip(preds, labels))
        fp = sum(p == label and y != label for p, y in zip(preds, labels))
        fn = sum(p != label and y == label for p, y in zip(preds, labels))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    nll = -sum((1.0 if y else 0.0) * math.log(max(p, 1e-15)) +
               (0.0 if y else 1.0) * math.log(max(1 - p, 1e-15))
               for y, p in zip(labels, probs)) / len(rows)
    kl_values = []
    for margin, probability in zip(margins, probs):
        base = min(max(sigmoid(float(margin)), 1e-7), 1 - 1e-7)
        policy = min(max(probability, 1e-7), 1 - 1e-7)
        kl_values.append(policy * math.log(policy / base) +
                         (1 - policy) * math.log((1 - policy) / (1 - base)))
    return {
        "accuracy": sum(correct) / len(correct),
        "balanced_accuracy": (yes_recall + no_recall) / 2,
        "macro_f1": sum(f1s) / 2,
        "binary_nll": nll,
        "mean_conditional_kl_nats_per_action": sum(kl_values) / len(kl_values),
        "per_class": {"Yes": {"support": len(yes_idx), "recall": yes_recall},
                      "No": {"support": len(no_idx), "recall": no_recall}},
        "correct_by_prompt": correct,
    }


def bootstrap_difference(rows, left, right, seed, resamples):
    groups = {label: [i for i, row in enumerate(rows) if row["label"] == label] for label in ("Yes", "No")}
    sizes = {label: len(groups[label]) for label in groups}

    def indices(replicate):
        out = []
        for label in ("Yes", "No"):
            members = groups[label]
            for draw in range(len(members)):
                payload = f"vare-proc-entail-bootstrap-v1\0{seed}\0{replicate}\0{label}\0{draw}".encode()
                pick = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(members)
                out.append(members[pick])
        return out

    def difference(selected):
        yes = selected[:sizes["Yes"]]
        no = selected[sizes["Yes"]:]
        return ((sum(left[i] - right[i] for i in yes) / sizes["Yes"]) +
                (sum(left[i] - right[i] for i in no) / sizes["No"])) / 2

    point = difference(groups["Yes"] + groups["No"])
    values = sorted(difference(indices(i)) for i in range(resamples))
    return {"difference": point, "ci95": [values[int(0.025 * (resamples - 1))],
                                             values[int(0.975 * (resamples - 1))]],
            "resamples": resamples}


def audit(output: Path):
    output = output.resolve()
    manifest_check(output)
    frozen_source_check(output)
    spec = read_json(output / "protocol.json")
    lock = read_json(output / "protocol.lock.json")
    lock_digest = lock.pop("sha256", None)
    spec_digest = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_digest != spec_digest or canonical(lock) != canonical(spec):
        raise ValueError("run protocol and lock differ")
    summary = read_json(output / "summary.json")
    if summary["protocol_sha256"] != spec_digest:
        raise ValueError("summary protocol digest mismatch")
    if summary["decision_stage"] != "development_screening_only":
        raise ValueError("development must remain screening-only")
    pilot_file = output / "pilot_hashes.json"
    pilot_hashes_doc = read_json(pilot_file)
    if sha256_file(pilot_file) != spec["data"]["pilot_hashes_sha256"]:
        raise ValueError("pilot hash denylist differs from protocol")
    pilot_hashes = set(pilot_hashes_doc["prompt_sha256"])
    task_doc = read_json(output / "tasks.json")
    cfg = spec["data"]
    expected_train = generate_split(seed=cfg["training"]["seed"], count=cfg["training"]["count"],
                                    style=cfg["training"]["style"], positive_count=cfg["training"]["positive_count"],
                                    n_nodes=cfg["training"]["nodes"], edge_probability=cfg["training"]["edge_probability"])
    expected_dev = generate_split(seed=cfg["development"]["seed"], count=cfg["development"]["count"],
                                  style=cfg["development"]["style"], positive_count=cfg["development"]["positive_count"],
                                  n_nodes=cfg["development"]["nodes"], edge_probability=cfg["development"]["edge_probability"])
    if task_doc != {"training": expected_train, "development": expected_dev}:
        raise ValueError("saved task prompts or graph proofs do not reconstruct")
    train_hashes = {row["prompt_sha256"] for row in expected_train}
    dev_hashes = {row["prompt_sha256"] for row in expected_dev}
    if train_hashes & dev_hashes or (train_hashes | dev_hashes) & pilot_hashes:
        raise ValueError("training/development prompt separation failed")
    if train_hashes != set(summary["training_prompt_hashes"]) or dev_hashes != set(summary["development_prompt_hashes"]):
        raise ValueError("summary prompt hashes differ from reconstructed rows")

    arrays = np.load(output / "base_features.npz", allow_pickle=False)
    train_x = arrays["train_features"]
    dev_x = arrays["development_features"]
    margins = arrays["development_margins"].astype(float)
    mean, scale = arrays["training_mean"], arrays["training_scale"]
    if not np.allclose(mean, train_x.mean(axis=0), rtol=1e-6, atol=1e-6):
        raise ValueError("stored feature mean is not training-only mean")
    expected_scale = np.maximum(train_x.std(axis=0), 1e-5)
    if not np.allclose(scale, expected_scale, rtol=1e-6, atol=1e-6):
        raise ValueError("stored feature scale is not training-only population standard deviation")
    centered = (dev_x - mean) / scale
    adapters = summary["selected_adapters"]
    seeds = [str(seed) for seed in spec["learner"]["seeds"]]
    row_results = []
    pooled_correct = {arm: [0.0] * len(expected_dev) for arm in ("base", "calibration_only", "context_sft", "context_dpo")}
    base_result = reconstructed_metrics(expected_dev, margins, [0.0] * len(margins))
    for arm in ("calibration_only", "context_sft", "context_dpo"):
        for seed in seeds:
            state = adapters[arm][seed]
            checkpoint_metrics = summary["all_checkpoint_metrics"][arm][seed]
            best_update = max((int(update) for update, values in checkpoint_metrics.items()),
                              key=lambda update: (checkpoint_metrics[str(update)]["balanced_accuracy"], -update))
            if state["updates"] != best_update:
                raise ValueError(f"selected checkpoint differs from locked development rule: {arm}/{seed}")
            delta = ([float(state["bias"])] * len(expected_dev) if state["kind"] == "scalar" else
                     (centered @ np.asarray(state["weight"], dtype=float) + float(state["bias"])).tolist())
            rebuilt = reconstructed_metrics(expected_dev, margins, delta)
            seed_position = seeds.index(seed)
            recorded = summary["per_seed_selected_development"][seed_position][arm]
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "binary_nll", "mean_conditional_kl_nats_per_action"):
                close(rebuilt[key], recorded[key])
            for label in ("Yes", "No"):
                close(rebuilt["per_class"][label]["recall"], recorded["per_class"][label]["recall"])
            pooled_correct[arm] = [a + b / len(seeds) for a, b in zip(pooled_correct[arm], rebuilt["correct_by_prompt"])]
    pooled_correct["base"] = base_result["correct_by_prompt"]
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "binary_nll", "mean_conditional_kl_nats_per_action"):
        close(base_result[key], summary["base_development"][key])

    means = {"base": base_result}
    for arm in ("calibration_only", "context_sft", "context_dpo"):
        rows = [item[arm] for item in summary["per_seed_selected_development"]]
        means[arm] = {key: sum(float(item[key]) for item in rows) / len(rows)
                      for key in ("accuracy", "balanced_accuracy", "macro_f1", "binary_nll", "mean_conditional_kl_nats_per_action")}
        for key, value in means[arm].items():
            close(value, summary["selected_development"][arm][key])
    for right in ("base", "calibration_only", "context_sft"):
        rebuilt = bootstrap_difference(expected_dev, pooled_correct["context_dpo"], pooled_correct[right],
                                       spec["metrics"]["paired_bootstrap"]["seed"],
                                       spec["metrics"]["paired_bootstrap"]["resamples"])
        stored = summary["contrasts"]["context_dpo_minus_" + right]
        close(rebuilt["difference"], stored["difference"])
        close(rebuilt["ci95"][0], stored["ci95"][0])
        close(rebuilt["ci95"][1], stored["ci95"][1])

    decision = summary["development_decision"]
    gate = spec["metrics"]["development_gate"]
    dpo, base, scalar, sft = (means["context_dpo"], means["base"],
                              means["calibration_only"], means["context_sft"])
    contrast_base = summary["contrasts"]["context_dpo_minus_base"]
    contrast_scalar = summary["contrasts"]["context_dpo_minus_calibration_only"]
    contrast_sft = summary["contrasts"]["context_dpo_minus_context_sft"]
    seed_rows = summary["per_seed_selected_development"]
    checks = {
        "base_floor": (base["balanced_accuracy"] >= gate["minimum_base_balanced_accuracy"] and
                       min(summary["base_development"]["per_class"][label]["recall"] for label in ("Yes", "No")) >=
                       gate["minimum_base_recall_per_class"]),
        "dpo_gain_over_base": contrast_base["difference"] >= gate["context_dpo_minus_base_balanced_accuracy_at_least"],
        "dpo_base_ci": contrast_base["ci95"][0] > gate["context_dpo_minus_base_ci_lower_above"],
        "dpo_gain_over_calibration": contrast_scalar["difference"] >= gate["context_dpo_minus_calibration_balanced_accuracy_at_least"],
        "dpo_calibration_ci": contrast_scalar["ci95"][0] > gate["context_dpo_minus_calibration_ci_lower_above"],
        "dpo_noninferior_to_sft": contrast_sft["ci95"][0] > gate["context_dpo_minus_context_sft_ci_lower_above"],
        "seed_consistency_vs_base": sum(row["context_dpo"]["balanced_accuracy"] > row["base"]["balanced_accuracy"]
                                         for row in seed_rows) >= gate["minimum_seeds_better_than_base"],
        "seed_consistency_vs_calibration": sum(row["context_dpo"]["balanced_accuracy"] > row["calibration_only"]["balanced_accuracy"]
                                                for row in seed_rows) >= gate["minimum_seeds_better_than_calibration"],
        "kl": dpo["mean_conditional_kl_nats_per_action"] <= gate["maximum_mean_conditional_kl_nats_per_action"],
        "per_seed_kl": max(row["context_dpo"]["mean_conditional_kl_nats_per_action"] for row in seed_rows) <=
                       gate["maximum_per_seed_conditional_kl_nats_per_action"],
        "data_and_oracle_checks": summary["data_checks"]["pass"],
    }
    if checks != decision["checks"] or all(checks.values()) != decision["pass"]:
        raise ValueError("development gate decision does not reconstruct from locked thresholds")
    return {
        "audit": "pass",
        "frozen_pre_run_commit": FROZEN_COMMIT,
        "manifest_files_verified": len(read_json(output / "manifest.json")["files"]),
        "train_rows_reconstructed": len(expected_train),
        "development_rows_reconstructed": len(expected_dev),
        "pilot_hashes_excluded": len(pilot_hashes),
        "summary_metrics_and_paired_bootstrap_reconstructed": True,
        "development_gate": decision,
        "audit_limit": "Reconstructs prompts, graph proofs, frozen source snapshots, stored-feature adapter metrics, and paired bootstrap. It does not independently rerun the model forward pass or optimizer; model provenance is the recorded local model-file fingerprint.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = audit(args.bundle)
    (args.bundle / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
