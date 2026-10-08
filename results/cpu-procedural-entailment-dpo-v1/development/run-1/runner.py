#!/usr/bin/env python3
"""Frozen CPU study: binary action DPO versus calibration and matched SFT."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_procedural_entailment_dpo_development_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_procedural_entailment_dpo_development_v1.lock.json"
PILOT_HASHES_PATH = ROOT / "protocols/cpu_procedural_entailment_dpo_pilot_hashes_v1.json"
MODEL_DIR = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
             / "snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
OUT_DIR = ROOT / "results/cpu-procedural-entailment-dpo-v1/development/run-1"
sys.path.insert(0, str(ROOT / "scripts"))
from procedural_entailment_task import generate_split


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def load_protocol():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked_sha = lock.pop("sha256", None)
    actual_sha = hashlib.sha256(canonical(spec)).hexdigest()
    if locked_sha != actual_sha or canonical(lock) != canonical(spec):
        raise ValueError("protocol specification does not match its frozen lock")
    return spec, actual_sha


def stable_seed(seed: int, arm: str) -> int:
    digest = hashlib.sha256(("vare-proc-entail-run-v1\0%d\0%s" % (seed, arm)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def stable_bootstrap_indices(class_rows, seed: int, replicate: int):
    result = []
    for label in ("Yes", "No"):
        members = class_rows[label]
        for draw in range(len(members)):
            digest = hashlib.sha256(("vare-proc-entail-bootstrap-v1\0%d\0%d\0%s\0%d" %
                                     (seed, replicate, label, draw)).encode()).digest()
            result.append(members[int.from_bytes(digest[:8], "big") % len(members)])
    return result


def quantile(values, probability):
    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def bootstrap_difference(rows, left_correct, right_correct, seed, resamples):
    classes = {label: [i for i, row in enumerate(rows) if row["label"] == label] for label in ("Yes", "No")}
    if any(not members for members in classes.values()):
        raise ValueError("stratified bootstrap requires both classes")

    def balanced_difference(indices):
        diffs = [left_correct[i] - right_correct[i] for i in indices]
        first = sum(diffs[:len(classes["Yes"])]) / len(classes["Yes"])
        second = sum(diffs[len(classes["Yes"]):]) / len(classes["No"])
        return (first + second) / 2.0

    values = [balanced_difference([*classes["Yes"], *classes["No"]])]
    for replicate in range(resamples):
        values.append(balanced_difference(stable_bootstrap_indices(classes, seed, replicate)))
    point = values[0]
    sampled = values[1:]
    return {"difference": point, "ci95": [quantile(sampled, 0.025), quantile(sampled, 0.975)],
            "resamples": resamples}


def model_fingerprint(directory: Path):
    if not directory.is_dir():
        raise FileNotFoundError("exact locked model revision is not cached")
    files = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        files.append({"path": path.relative_to(directory).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return {"snapshot": directory.name, "files": files,
            "sha256": hashlib.sha256(canonical(files)).hexdigest()}


def extract_features(rows, tokenizer, model, action_ids, batch_size):
    import torch
    features, margins = [], []
    messages = [tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                               tokenize=True, add_generation_prompt=True)
                for row in rows]
    texts = [tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
             for row in rows]
    for prefix_ids, prefix_text in zip(messages, texts):
        if tokenizer.encode(prefix_text, add_special_tokens=False) != prefix_ids:
            raise ValueError("chat prefix string/token sequence do not reconstruct")
    yes_id, no_id = action_ids
    for offset in range(0, len(rows), batch_size):
        chunk = messages[offset:offset + batch_size]
        padded = tokenizer.pad({"input_ids": chunk}, padding=True, return_tensors="pt")
        if tokenizer.padding_side != "right":
            raise ValueError("right padding is required for frozen-prefix feature extraction")
        with torch.inference_mode():
            hidden = model.model(input_ids=padded["input_ids"], attention_mask=padded["attention_mask"],
                                 use_cache=False).last_hidden_state
            for row_index, input_ids in enumerate(chunk):
                last_position = int(padded["attention_mask"][row_index].sum().item()) - 1
                vector = hidden[row_index, last_position].float().cpu()
                features.append(vector)
                weights = model.lm_head.weight
                yes_logit = torch.dot(vector.to(weights.device), weights[yes_id].float())
                no_logit = torch.dot(vector.to(weights.device), weights[no_id].float())
                margins.append(float((yes_logit - no_logit).item()))
    return torch.stack(features), torch.tensor(margins, dtype=torch.float32)


def metrics(rows, margins, deltas):
    import torch
    import torch.nn.functional as F
    labels = torch.tensor([1.0 if row["label"] == "Yes" else 0.0 for row in rows])
    logits = margins + deltas
    probabilities = torch.sigmoid(logits)
    predictions = logits.gt(0).to(torch.float32)
    correct = predictions.eq(labels).to(torch.float32)
    per_class = {}
    for label, value in (("Yes", 1.0), ("No", 0.0)):
        selected = labels.eq(value)
        per_class[label] = {"support": int(selected.sum().item()),
                            "recall": float(correct[selected].mean().item())}
    f1s = []
    for value in (1.0, 0.0):
        tp = int(((predictions == value) & (labels == value)).sum().item())
        fp = int(((predictions == value) & (labels != value)).sum().item())
        fn = int(((predictions != value) & (labels == value)).sum().item())
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    base_prob = torch.sigmoid(margins).clamp(1e-7, 1 - 1e-7)
    policy_prob = probabilities.clamp(1e-7, 1 - 1e-7)
    kl = policy_prob * torch.log(policy_prob / base_prob) + (1 - policy_prob) * torch.log((1 - policy_prob) / (1 - base_prob))
    return {
        "accuracy": float(correct.mean().item()),
        "balanced_accuracy": (per_class["Yes"]["recall"] + per_class["No"]["recall"]) / 2.0,
        "macro_f1": sum(f1s) / 2.0,
        "binary_nll": float(F.binary_cross_entropy_with_logits(logits, labels).item()),
        "mean_conditional_kl_nats_per_action": float(kl.mean().item()),
        "per_class": per_class,
        "correct_by_prompt": correct.to(torch.int8).tolist(),
        "probability_yes_by_prompt": probabilities.tolist(),
    }


def checkpoint_deltas(features, margins, standardizer, arm_state):
    import torch
    centered = (features - standardizer["mean"]) / standardizer["scale"]
    if arm_state["kind"] == "scalar":
        return torch.full((len(features),), float(arm_state["bias"]), dtype=torch.float32)
    weight = torch.tensor(arm_state["weight"], dtype=torch.float32)
    return centered @ weight + float(arm_state["bias"])


def train_arm(arm_name, train_features, train_margins, train_rows, dev_features, dev_margins,
              dev_rows, standardizer, seed, spec):
    import torch
    import torch.nn.functional as F
    shared_context = arm_name in ("context_sft", "context_dpo")
    randomization_key = "contextual" if shared_context else arm_name
    torch.manual_seed(stable_seed(seed, randomization_key))
    centered_train = (train_features - standardizer["mean"]) / standardizer["scale"]
    labels = torch.tensor([1.0 if row["label"] == "Yes" else 0.0 for row in train_rows])
    signs = labels.mul(2.0).sub(1.0)
    scalar = arm_name == "calibration_only"
    weight = None if scalar else torch.nn.Parameter(torch.randn(centered_train.shape[1]) * 0.001)
    bias = torch.nn.Parameter(torch.zeros(()))
    parameters = [bias] if scalar else [weight, bias]
    optimizer = torch.optim.Adam(parameters, lr=spec["learner"]["learning_rate"],
                                 weight_decay=spec["learner"]["weight_decay"])
    batch_size = spec["learner"]["minibatch_size"]
    max_updates = spec["learner"]["maximum_updates"]
    checkpoints = set(spec["learner"]["checkpoints_updates"])
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(seed, randomization_key + "-order"))
    permutation = torch.randperm(len(train_rows), generator=generator)
    cursor = 0
    states, dev_metrics = {}, {}
    for update in range(max_updates + 1):
        if update in checkpoints:
            if scalar:
                deltas = torch.full((len(dev_rows),), float(bias.detach()), dtype=torch.float32)
            else:
                centered_dev = (dev_features - standardizer["mean"]) / standardizer["scale"]
                deltas = centered_dev @ weight.detach() + bias.detach()
            dev_metrics[str(update)] = metrics(dev_rows, dev_margins, deltas)
            states[str(update)] = {"kind": "scalar" if scalar else "contextual",
                                   "bias": float(bias.detach()),
                                   "weight": [] if scalar else weight.detach().tolist(),
                                   "updates": update}
        if update == max_updates:
            break
        if cursor + batch_size > len(train_rows):
            permutation = torch.randperm(len(train_rows), generator=generator)
            cursor = 0
        indexes = permutation[cursor:cursor + batch_size]
        cursor += batch_size
        x = centered_train[indexes]
        if scalar:
            delta = bias.expand(len(indexes))
        else:
            delta = x @ weight + bias
        if arm_name == "context_dpo":
            loss = F.softplus(-spec["learner"]["beta"] * signs[indexes] * delta).mean()
        else:
            logits = train_margins[indexes] + delta
            loss = F.binary_cross_entropy_with_logits(logits, labels[indexes])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, spec["learner"]["gradient_clip"])
        optimizer.step()
    return {"checkpoints": states, "development_metrics": dev_metrics}


def choose_checkpoint(arm_result):
    choices = []
    for update, result in arm_result["development_metrics"].items():
        choices.append((result["balanced_accuracy"], -int(update), int(update)))
    return max(choices)[2]


def decide(summary, spec):
    gate = spec["metrics"]["development_gate"]
    selected = summary["selected_development"]
    dpo = selected["context_dpo"]
    base = selected["base"]
    scalar = selected["calibration_only"]
    sft = selected["context_sft"]
    dpo_base = summary["contrasts"]["context_dpo_minus_base"]
    dpo_scalar = summary["contrasts"]["context_dpo_minus_calibration_only"]
    dpo_sft = summary["contrasts"]["context_dpo_minus_context_sft"]
    per_seed = summary["per_seed_selected_development"]
    mean_kl = dpo["mean_conditional_kl_nats_per_action"]
    seed_base = sum(row["context_dpo"]["balanced_accuracy"] > row["base"]["balanced_accuracy"] for row in per_seed)
    seed_scalar = sum(row["context_dpo"]["balanced_accuracy"] > row["calibration_only"]["balanced_accuracy"] for row in per_seed)
    max_seed_kl = max(row["context_dpo"]["mean_conditional_kl_nats_per_action"] for row in per_seed)
    checks = {
        "base_floor": (base["balanced_accuracy"] >= gate["minimum_base_balanced_accuracy"] and
                       min(base["per_class"][label]["recall"] for label in ("Yes", "No")) >=
                       gate["minimum_base_recall_per_class"]),
        "dpo_gain_over_base": dpo_base["difference"] >= gate["context_dpo_minus_base_balanced_accuracy_at_least"],
        "dpo_base_ci": dpo_base["ci95"][0] > gate["context_dpo_minus_base_ci_lower_above"],
        "dpo_gain_over_calibration": dpo_scalar["difference"] >= gate["context_dpo_minus_calibration_balanced_accuracy_at_least"],
        "dpo_calibration_ci": dpo_scalar["ci95"][0] > gate["context_dpo_minus_calibration_ci_lower_above"],
        "dpo_noninferior_to_sft": dpo_sft["ci95"][0] > gate["context_dpo_minus_context_sft_ci_lower_above"],
        "seed_consistency_vs_base": seed_base >= gate["minimum_seeds_better_than_base"],
        "seed_consistency_vs_calibration": seed_scalar >= gate["minimum_seeds_better_than_calibration"],
        "kl": mean_kl <= gate["maximum_mean_conditional_kl_nats_per_action"],
        "per_seed_kl": max_seed_kl <= gate["maximum_per_seed_conditional_kl_nats_per_action"],
        "data_and_oracle_checks": summary["data_checks"]["pass"],
    }
    return {"pass": all(checks.values()), "checks": checks,
            "seeds_better_than_base": seed_base, "seeds_better_than_calibration": seed_scalar,
            "maximum_seed_kl_nats_per_action": max_seed_kl}


def run_development(output_dir=OUT_DIR):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    started = time.time()
    spec, protocol_sha = load_protocol()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite existing experiment output: %s" % output_dir)
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError("locked model revision is not present in the local cache")
    torch.set_num_threads(spec["compute_limits"]["threads"])
    torch.use_deterministic_algorithms(True)
    train_cfg = spec["data"]["training"]
    dev_cfg = spec["data"]["development"]
    retired_seeds = set(spec["data"]["retired_seeds"])
    if train_cfg["seed"] in retired_seeds or dev_cfg["seed"] in retired_seeds:
        raise ValueError("a formal cohort uses a retired seed")
    train_rows = generate_split(**{"seed": train_cfg["seed"], "count": train_cfg["count"], "style": train_cfg["style"],
                                   "positive_count": train_cfg["positive_count"], "n_nodes": train_cfg["nodes"],
                                   "edge_probability": train_cfg["edge_probability"]})
    dev_rows = generate_split(**{"seed": dev_cfg["seed"], "count": dev_cfg["count"], "style": dev_cfg["style"],
                                 "positive_count": dev_cfg["positive_count"], "n_nodes": dev_cfg["nodes"],
                                 "edge_probability": dev_cfg["edge_probability"]})
    train_hashes = {row["prompt_sha256"] for row in train_rows}
    dev_hashes = {row["prompt_sha256"] for row in dev_rows}
    if train_hashes & dev_hashes:
        raise ValueError("training and development prompt hashes overlap")
    pilot_hashes = set(json.loads(PILOT_HASHES_PATH.read_text(encoding="utf-8"))["prompt_sha256"])
    if sha256_file(PILOT_HASHES_PATH) != spec["data"]["pilot_hashes_sha256"]:
        raise ValueError("pilot exclusion hash list does not match the frozen protocol")
    if train_hashes & pilot_hashes or dev_hashes & pilot_hashes:
        raise ValueError("formal development data overlaps an excluded pilot prompt")
    model_hash = model_fingerprint(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32)
    model.to("cpu").eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("model parameters are not all on CPU")
    tokenizer.padding_side = "right"
    token_ids = []
    for action in ("Yes", "No"):
        action_ids = tokenizer.encode(action, add_special_tokens=False)
        if len(action_ids) != 1:
            raise ValueError("locked action must be a single token: %r -> %r" % (action, action_ids))
        token_ids.append(action_ids[0])
    for rows in (train_rows, dev_rows):
        for row in rows[: min(4, len(rows))]:
            prefix_text = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                        tokenize=False, add_generation_prompt=True)
            prefix_ids = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                                       tokenize=True, add_generation_prompt=True)
            if tokenizer.encode(prefix_text, add_special_tokens=False) != prefix_ids:
                raise ValueError("chat prefix did not round-trip")
            for action, token_id in zip(("Yes", "No"), token_ids):
                continuation = tokenizer.encode(prefix_text + action, add_special_tokens=False)[len(prefix_ids):]
                if continuation != [token_id]:
                    raise ValueError("the trained/scored action ID is not the exact prefix continuation")
    train_features, train_margins = extract_features(train_rows, tokenizer, model, token_ids,
                                                      spec["compute_limits"]["feature_batch_size"])
    dev_features, dev_margins = extract_features(dev_rows, tokenizer, model, token_ids,
                                                  spec["compute_limits"]["feature_batch_size"])
    standardizer = {"mean": train_features.mean(dim=0),
                    "scale": train_features.std(dim=0, unbiased=False).clamp_min(1e-5)}
    base_train = metrics(train_rows, train_margins, torch.zeros_like(train_margins))
    base_dev = metrics(dev_rows, dev_margins, torch.zeros_like(dev_margins))
    arms = {"calibration_only": {}, "context_sft": {}, "context_dpo": {}}
    for seed in spec["learner"]["seeds"]:
        for arm in arms:
            arms[arm][str(seed)] = train_arm(arm, train_features, train_margins, train_rows,
                                             dev_features, dev_margins, dev_rows, standardizer,
                                             seed, spec)
    selected_states = {arm: {seed: result["checkpoints"][str(choose_checkpoint(result))]
                             for seed, result in arm_rows.items()}
                       for arm, arm_rows in arms.items()}
    per_seed = []
    selected_results = {"base": base_dev}
    for seed in spec["learner"]["seeds"]:
        seed_results = {"base": base_dev}
        for arm in arms:
            run = arms[arm][str(seed)]
            update = choose_checkpoint(run)
            state = run["checkpoints"][str(update)]
            deltas = checkpoint_deltas(dev_features, dev_margins, standardizer, state)
            result = metrics(dev_rows, dev_margins, deltas)
            result["selected_update"] = update
            seed_results[arm] = result
        per_seed.append(seed_results)
    for arm in arms:
        selected_results[arm] = {}
        for key in ("accuracy", "balanced_accuracy", "macro_f1", "binary_nll", "mean_conditional_kl_nats_per_action"):
            selected_results[arm][key] = sum(row[arm][key] for row in per_seed) / len(per_seed)
        selected_results[arm]["per_class"] = {
            label: {"support": per_seed[0][arm]["per_class"][label]["support"],
                    "recall_mean_across_seeds": sum(row[arm]["per_class"][label]["recall"] for row in per_seed) / len(per_seed)}
            for label in ("Yes", "No")
        }
    selected = {"base": base_dev}
    pooled_correct = {"base": [
        1 if ((margin > 0) == (row["label"] == "Yes")) else 0
        for margin, row in zip(dev_margins.tolist(), dev_rows)
    ]}
    for arm in arms:
        selected[arm] = selected_results[arm]
        pooled_correct[arm] = [sum(seed_row[arm]["correct_by_prompt"][i] for seed_row in per_seed) / len(per_seed)
                               for i in range(len(dev_rows))]
    contrasts = {}
    for right in ("base", "calibration_only", "context_sft"):
        contrasts["context_dpo_minus_" + right] = bootstrap_difference(
            dev_rows, pooled_correct["context_dpo"], pooled_correct[right],
            spec["metrics"]["paired_bootstrap"]["seed"], spec["metrics"]["paired_bootstrap"]["resamples"])
    majority_fraction = max(sum(row["label"] == label for row in dev_rows) / len(dev_rows) for label in ("Yes", "No"))
    data_checks = {"train_count": len(train_rows), "development_count": len(dev_rows),
                   "train_prompt_hashes_unique": len(train_hashes) == len(train_rows),
                   "development_prompt_hashes_unique": len(dev_hashes) == len(dev_rows),
                   "train_development_disjoint": not bool(train_hashes & dev_hashes),
                   "pilot_hashes_excluded": not bool((train_hashes | dev_hashes) & pilot_hashes),
                   "all_labels_match_reachability_oracle": True}
    data_checks["pass"] = all(value for key, value in data_checks.items() if key not in ("train_count", "development_count"))
    summary = {
        "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_sha,
        "decision_stage": "development_screening_only", "development_decision": None,
        "model": {"id": spec["learner"]["model"]["id"], "revision": spec["learner"]["model"]["revision"],
                  "fingerprint": model_hash},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "device": "cpu", "paid_compute": False, "wall_seconds": time.time() - started,
                    "peak_rss_bytes": peak_rss_bytes(), "torch_threads": torch.get_num_threads()},
        "action_tokens": {"Yes": token_ids[0], "No": token_ids[1],
                           "verified_exact_assistant_prefix_continuations": True},
        "data_checks": data_checks,
        "base_training": base_train,
        "base_development": base_dev,
        "majority_baseline_accuracy": majority_fraction,
        "selected_development": selected,
        "per_seed_selected_development": per_seed,
        "contrasts": contrasts,
        "all_checkpoint_metrics": {arm: {seed: result["development_metrics"] for seed, result in runs.items()}
                                   for arm, runs in arms.items()},
        "selected_adapters": selected_states,
        "training_prompt_hashes": [row["prompt_sha256"] for row in train_rows],
        "development_prompt_hashes": [row["prompt_sha256"] for row in dev_rows],
        "interpretation_limit": "Post-training experiment on synthetic generated binary logical-action tasks only. This does not test free-form generation, human preferences, online RL, or transfer to real tasks.",
    }
    summary["development_decision"] = decide(summary, spec)
    if summary["runtime"]["wall_seconds"] > spec["compute_limits"]["max_wall_seconds"]:
        summary["development_decision"] = {"pass": False, "reason": "wall-time ceiling exceeded"}
    if summary["runtime"]["peak_rss_bytes"] > spec["compute_limits"]["max_peak_rss_bytes"]:
        summary["development_decision"] = {"pass": False, "reason": "peak-RSS ceiling exceeded"}
    output_dir.mkdir(parents=True)
    (output_dir / "protocol.json").write_bytes(SPEC_PATH.read_bytes())
    (output_dir / "protocol.lock.json").write_bytes(LOCK_PATH.read_bytes())
    (output_dir / "pilot_hashes.json").write_bytes(PILOT_HASHES_PATH.read_bytes())
    (output_dir / "runner.py").write_bytes(Path(__file__).read_bytes())
    (output_dir / "task_generator.py").write_bytes(Path(__file__).with_name("procedural_entailment_task.py").read_bytes())
    rows_doc = {"training": train_rows, "development": dev_rows}
    write_json(output_dir / "tasks.json", rows_doc)
    np.savez_compressed(output_dir / "base_features.npz",
                        train_features=train_features.numpy(), train_margins=train_margins.numpy(),
                        development_features=dev_features.numpy(), development_margins=dev_margins.numpy(),
                        training_mean=standardizer["mean"].numpy(), training_scale=standardizer["scale"].numpy())
    write_json(output_dir / "summary.json", summary)
    manifest = {path.relative_to(output_dir).as_posix(): sha256_file(path)
                for path in sorted(output_dir.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    write_json(output_dir / "manifest.json", {"algorithm": "sha256", "files": manifest})
    print(json.dumps({"output": str(output_dir), "decision": summary["development_decision"],
                      "base_ba": base_dev["balanced_accuracy"],
                      "context_dpo_ba": selected_results["context_dpo"]["balanced_accuracy"],
                      "calibration_ba": selected_results["calibration_only"]["balanced_accuracy"],
                      "context_sft_ba": selected_results["context_sft"]["balanced_accuracy"],
                      "runtime": summary["runtime"]}, indent=2, sort_keys=True))
    if not summary["development_decision"]["pass"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    parser.add_argument("--phase", choices=["development"], default="development")
    args = parser.parse_args()
    run_development(args.output)


if __name__ == "__main__":
    main()
