#!/usr/bin/env python3
"""Frozen CPU-only binary-action RLOO development study on BoolQ."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np

try:  # Imports when loaded as scripts.run_cpu_lm_boolq_verifier_rloo_development_v1.
    from .boolq_rloo_task import (
        balanced_accuracy,
        binary_nll,
        build_selected_rows,
        class_weights,
        hash_rank_indices,
        stratified_paired_bootstrap,
        validate_disjoint_hashes,
    )
    from .rloo_binary_objectives import (
        bernoulli_kl_from_logits,
        expected_reward_gradient_self_check,
        exact_expected_reward_loss,
        rloo_policy_loss,
        supervised_loss,
    )
except ImportError:  # Direct CLI execution places scripts/ on sys.path.
    from boolq_rloo_task import (
        balanced_accuracy,
        binary_nll,
        build_selected_rows,
        class_weights,
        hash_rank_indices,
        stratified_paired_bootstrap,
        validate_disjoint_hashes,
    )
    from rloo_binary_objectives import (
        bernoulli_kl_from_logits,
        expected_reward_gradient_self_check,
        exact_expected_reward_loss,
        rloo_policy_loss,
        supervised_loss,
    )


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_lm_boolq_verifier_rloo_development_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_lm_boolq_verifier_rloo_development_v1.lock.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/datasets/google___boolq/default/0.0.0/35b264d03638db9f4ce671b711558bf7ff0f80d5"
ARMS = ("scalar_calibration", "context_sft", "exact_expected_reward", "rloo_k4")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest(output: Path, *, include_audit: bool = False) -> None:
    files = {
        p.relative_to(output).as_posix(): sha256_file(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "manifest.json" and (include_audit or p.name != "audit.json")
    }
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": files})


def load_locked_spec(spec_path: Path = SPEC_PATH, lock_path: Path = LOCK_PATH):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_hash = lock.pop("sha256", None)
    spec_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_hash != spec_hash or canonical(lock) != canonical(spec):
        raise ValueError("RLOO development protocol differs from its lock")
    return spec, spec_hash


def recorded_boolq_hashes() -> set[str]:
    hashes: set[str] = set()
    for version in (16, 17):
        pattern = f"results/cpu-lm-boolq-posttraining-development-v{version}-*/run-*/development.json"
        for path in ROOT.glob(pattern):
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in ("train_examples", "validation_examples"):
                for row in data.get(key, []):
                    digest = row.get("question_sha256")
                    if digest:
                        hashes.add(str(digest))
    return hashes


def check_offline_cache(spec: dict[str, Any]) -> dict[str, str]:
    data = spec["data"]["dataset"]
    train_arrow = DATA_DIR / "boolq-train.arrow"
    validation_arrow = DATA_DIR / "boolq-validation.arrow"
    files = {
        "train_arrow": train_arrow,
        "validation_arrow": validation_arrow,
    }
    expected = {
        "train_arrow": data["train_arrow_sha256"],
        "validation_arrow": data["validation_arrow_sha256"],
    }
    for name, digest in spec["learner"]["model"]["snapshot_file_sha256"].items():
        files[f"model_snapshot/{name}"] = MODEL_DIR / name
        expected[f"model_snapshot/{name}"] = digest
    got = {}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"required offline cached file is missing: {name}")
        got[name] = sha256_file(path)
        if got[name] != expected[name]:
            raise ValueError(f"cached source hash mismatch: {name}")
    if MODEL_DIR.name != spec["learner"]["model"]["revision"]:
        raise ValueError("cached model snapshot revision differs from the lock")
    return got


def verify_token_actions(tokenizer, expected):
    action_ids = []
    for label in ("Yes", "No"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{label} action is not exactly one token")
        action_ids.append(int(ids[0]))
    if action_ids[0] == action_ids[1]:
        raise ValueError("Yes and No action token IDs collide")
    if {"yes": action_ids[0], "no": action_ids[1]} != expected:
        raise ValueError("binary action token IDs differ from the pinned protocol")
    fixture = [{"role": "user", "content": "Tokenizer-only fixture. Is this a tokenization check?"}]
    prefix_ids = tokenizer.apply_chat_template(fixture, tokenize=True, add_generation_prompt=True)
    prefix_text = tokenizer.apply_chat_template(fixture, tokenize=False, add_generation_prompt=True)
    if tokenizer.encode(prefix_text, add_special_tokens=False) != prefix_ids:
        raise ValueError("chat prefix string and token IDs differ")
    for label, token_id in zip(("Yes", "No"), action_ids):
        suffix_ids = tokenizer.encode(label, add_special_tokens=False)
        if tokenizer.encode(prefix_text + label, add_special_tokens=False) != prefix_ids + suffix_ids:
            raise ValueError(f"{label} action crosses the prefix-token boundary")
        decoded = tokenizer.decode(prefix_ids + [token_id], skip_special_tokens=False)
        if not decoded.rstrip().endswith(label):
            raise ValueError(f"decoded action does not end in {label}")
    return {"yes": action_ids[0], "no": action_ids[1]}


def extract_features(rows, tokenizer, model, action_ids, batch_size: int, *, deadline: float, max_wall: float):
    import torch
    import torch.nn.functional as F

    hidden_chunks = []
    margin_chunks = []
    token_counts = []
    yes_idx = action_ids["yes"]
    no_idx = action_ids["no"]
    weight = model.lm_head.weight.index_select(0, torch.tensor([yes_idx, no_idx], dtype=torch.long))
    bias = None
    if model.lm_head.bias is not None:
        bias = model.lm_head.bias.index_select(0, torch.tensor([yes_idx, no_idx], dtype=torch.long))
    for start in range(0, len(rows), batch_size):
        if time.monotonic() >= deadline or time.monotonic() - (deadline - max_wall) >= max_wall:
            raise TimeoutError("campaign exceeded its frozen wall-time budget during feature extraction")
        batch_rows = rows[start:start + batch_size]
        messages = [[{"role": "user", "content": row["user_message"]}] for row in batch_rows]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        ids = encoded["input_ids"]
        attention = encoded["attention_mask"]
        lengths = attention.sum(dim=1)
        if torch.any(lengths > model.config.max_position_embeddings):
            raise ValueError("BoolQ prompt exceeds the pinned model context length")
        last_positions = lengths - 1
        with torch.inference_mode():
            hidden_all = model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
            indices = torch.arange(len(batch_rows), dtype=torch.long)
            hidden = hidden_all[indices, last_positions].to(dtype=torch.float32).contiguous()
            action_logits = F.linear(hidden, weight, bias)
            margins = action_logits[:, 0] - action_logits[:, 1]
        hidden_chunks.append(hidden.cpu())
        margin_chunks.append(margins.float().cpu())
        token_counts.extend(lengths.tolist())
    return {
        "hidden": torch.cat(hidden_chunks).numpy().astype(np.float32, copy=False),
        "base_margin": torch.cat(margin_chunks).numpy().astype(np.float64, copy=False),
        "prompt_tokens": token_counts,
    }


def standardized_features(train_hidden: np.ndarray, development_hidden: np.ndarray):
    mean = train_hidden.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_hidden.std(axis=0, dtype=np.float64).astype(np.float32)
    zero_scale = scale == 0
    scale[zero_scale] = 1.0
    return (train_hidden - mean) / scale, (development_hidden - mean) / scale, mean, scale


def logits_for(features, base_margins, weight, bias):
    if weight is None:
        return base_margins + bias
    return base_margins + features @ weight + bias


def evaluate_policy(features, base_margins, labels, weight, bias, selected_updates: int):
    import torch

    hidden = torch.as_tensor(features, dtype=torch.float32)
    base = torch.as_tensor(base_margins, dtype=torch.float32)
    gold = torch.as_tensor(labels, dtype=torch.float32)
    adapter = None if weight is None else torch.as_tensor(weight, dtype=torch.float32)
    shift = torch.as_tensor(bias, dtype=torch.float32)
    action_margin = logits_for(hidden, base, adapter, shift)
    p_yes = torch.sigmoid(action_margin).numpy().astype(float).tolist()
    # Use the action-margin boundary directly so it matches the auditor even if
    # float32 sigmoid rounds a tiny margin to exactly 0.5.
    predicted = [1 if float(margin) >= 0.0 else 0 for margin in action_margin]
    label_list = [int(v) for v in labels]
    base_yes = torch.sigmoid(base).numpy()
    kl = bernoulli_kl_from_logits(action_margin, base).numpy().astype(float).tolist()
    per_prompt_correct = [float(a == b) for a, b in zip(predicted, label_list)]
    correct_yes = sum(a == 1 and b == 1 for a, b in zip(predicted, label_list))
    correct_no = sum(a == 0 and b == 0 for a, b in zip(predicted, label_list))
    support_yes = sum(label_list)
    support_no = len(label_list) - support_yes
    precision_yes = correct_yes / max(1, sum(predicted))
    precision_no = correct_no / max(1, len(predicted) - sum(predicted))
    f1_yes = 2 * precision_yes * (correct_yes / max(1, support_yes)) / max(1e-12, precision_yes + correct_yes / max(1, support_yes))
    f1_no = 2 * precision_no * (correct_no / max(1, support_no)) / max(1e-12, precision_no + correct_no / max(1, support_no))
    return {
        "selected_updates": int(selected_updates),
        "examples": len(label_list),
        "exact_action_accuracy": sum(per_prompt_correct) / len(per_prompt_correct),
        "balanced_accuracy": balanced_accuracy(predicted, label_list),
        "yes_recall": correct_yes / support_yes,
        "no_recall": correct_no / support_no,
        "macro_f1": (f1_yes + f1_no) / 2.0,
        "yes_support": support_yes,
        "no_support": support_no,
        "action_nll": binary_nll(p_yes, label_list),
        "mean_conditional_bernoulli_kl_nats_per_action": statistics.fmean(kl),
        "parse_rate": 1.0,
        "per_prompt_correct": per_prompt_correct,
        "per_prompt_probability_yes": p_yes,
        "per_prompt_base_probability_yes": base_yes.astype(float).tolist(),
        "per_prompt_kl": kl,
    }


def make_minibatches(size: int, updates: int, minibatch_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed ^ 0x51A7C0DE)
    return rng.integers(0, size, size=(updates, minibatch_size), endpoint=False).astype(int).tolist()


def train_arm(arm: str, seed: int, train_hidden, train_margin, train_labels, weights_by_class,
              minibatches, spec: dict[str, Any], rollouts: list[dict[str, Any]], *,
              deadline: Optional[float] = None, max_peak_rss_bytes: Optional[int] = None):
    import torch

    learner = spec["learner"]
    optimizer_spec = learner["optimizer"]
    beta_kl = float(learner["kl"]["coefficient"])
    feature_dim = int(train_hidden.shape[1])
    device_hidden = torch.as_tensor(train_hidden, dtype=torch.float32)
    device_base = torch.as_tensor(train_margin, dtype=torch.float32)
    device_labels = torch.as_tensor(train_labels, dtype=torch.float32)
    weights_np = np.asarray([weights_by_class[int(y)] for y in train_labels], dtype=np.float32)
    device_weights = torch.as_tensor(weights_np, dtype=torch.float32)

    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    weight = None if arm == "scalar_calibration" else torch.nn.Parameter(torch.zeros(feature_dim, dtype=torch.float32))
    params = [bias] if weight is None else [weight, bias]
    optimizer = torch.optim.Adam(params, lr=float(optimizer_spec["learning_rate"]), weight_decay=float(optimizer_spec["weight_decay"]))
    sample_rng = np.random.default_rng(seed ^ 0xA17E6D4B)
    checkpoints = set(int(x) for x in optimizer_spec["checkpoint_updates"])
    results = []

    def current_logits(indices):
        return logits_for(device_hidden[indices], device_base[indices], weight, bias)

    if 0 in checkpoints:
        results.append({"update": 0, "weight": None if weight is None else weight.detach().numpy().astype(float).tolist(),
                        "bias": float(bias.detach()), "metrics": None})
    for update in range(1, int(optimizer_spec["maximum_updates"]) + 1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("campaign exceeded its frozen wall-time budget during policy updates")
        if max_peak_rss_bytes is not None and rss_bytes() > max_peak_rss_bytes:
            raise MemoryError("campaign exceeded its frozen peak-RSS budget during policy updates")
        indices_list = minibatches[update - 1]
        indices = torch.as_tensor(indices_list, dtype=torch.long)
        policy_logits = current_logits(indices)
        prompt_labels = device_labels[indices]
        prompt_weights = device_weights[indices]
        base_logits = device_base[indices]

        optimizer.zero_grad(set_to_none=True)
        if arm in ("scalar_calibration", "context_sft"):
            loss, _ = supervised_loss(policy_logits, prompt_labels, prompt_weights, base_logits, beta_kl)
        elif arm == "exact_expected_reward":
            loss, _ = exact_expected_reward_loss(policy_logits, prompt_labels, prompt_weights, base_logits, beta_kl)
        elif arm == "rloo_k4":
            p_yes = torch.sigmoid(policy_logits.detach()).cpu().numpy()
            actions_np = sample_rng.binomial(1, p_yes[:, None], size=(len(indices_list), int(learner["rloo"]["group_size"])))
            actions = torch.as_tensor(actions_np, dtype=torch.float32)
            loss, diagnostic = rloo_policy_loss(policy_logits, actions, prompt_labels, prompt_weights, base_logits, beta_kl)
            action_log_probability = actions * torch.nn.functional.logsigmoid(policy_logits.detach()[:, None]) \
                + (1 - actions) * torch.nn.functional.logsigmoid(-policy_logits.detach()[:, None])
            raw_rewards = (actions_np == prompt_labels.detach().cpu().numpy().astype(int)[:, None]).astype(float) \
                * prompt_weights.detach().cpu().numpy().astype(float)[:, None]
            baseline = (raw_rewards.sum(axis=1, keepdims=True) - raw_rewards) / (raw_rewards.shape[1] - 1)
            advantages = raw_rewards - baseline
            rollouts.append({
                "seed": int(seed),
                "update": int(update),
                "train_indices": [int(train_labels_index) for train_labels_index in indices_list],
                "actions": actions_np.astype(int).tolist(),
                "rewards": raw_rewards.tolist(),
                "leave_one_out_advantages": advantages.tolist(),
                "p_yes_before": p_yes.astype(float).tolist(),
                "sampled_action_log_probabilities": action_log_probability.cpu().numpy().astype(float).tolist(),
                "mixed_reward_group_rate": float(diagnostic["mixed_reward_group_rate"].detach()),
                "zero_advantage_group_rate": float(diagnostic["zero_advantage_group_rate"].detach()),
            })
        else:
            raise ValueError(f"unsupported policy arm: {arm}")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {arm} loss at update {update}")
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in params):
            raise FloatingPointError(f"non-finite or missing {arm} gradient at update {update}")
        torch.nn.utils.clip_grad_norm_(params, float(optimizer_spec["gradient_norm_clip"]))
        optimizer.step()
        if update in checkpoints:
            results.append({"update": update,
                            "weight": None if weight is None else weight.detach().cpu().numpy().astype(float).tolist(),
                            "bias": float(bias.detach()),
                            "metrics": None})
    return results


def choose_checkpoints(seed_runs: dict[str, dict[str, list[dict[str, Any]]]], dev_hidden, dev_margin, dev_labels, spec):
    chosen: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, by_seed in seed_runs.items():
        chosen[arm] = {}
        for seed_text, checkpoints in by_seed.items():
            evaluated = []
            for checkpoint in checkpoints:
                metrics = evaluate_policy(
                    dev_hidden, dev_margin, dev_labels,
                    None if checkpoint["weight"] is None else np.asarray(checkpoint["weight"], dtype=np.float32),
                    np.float32(checkpoint["bias"]), int(checkpoint["update"]),
                )
                candidate = dict(checkpoint)
                candidate["metrics"] = metrics
                eligible = metrics["mean_conditional_bernoulli_kl_nats_per_action"] <= spec["learner"]["kl"]["maximum_selected_mean_nats_per_action"]
                candidate["eligible"] = bool(eligible)
                if eligible:
                    evaluated.append(candidate)
            if not evaluated:
                raise RuntimeError(f"no KL-eligible development checkpoint for {arm} seed {seed_text}")
            best = max(evaluated, key=lambda c: (c["metrics"]["balanced_accuracy"], -int(c["update"])))
            chosen[arm][seed_text] = best
    return chosen


def average_seed_correct(rows_by_seed: list[list[float]]) -> list[list[float]]:
    if not rows_by_seed:
        raise ValueError("no seed outcomes")
    length = len(rows_by_seed[0])
    if any(len(values) != length for values in rows_by_seed):
        raise ValueError("per-seed validation outcome lengths differ")
    return [[float(rows_by_seed[seed][i]) for seed in range(len(rows_by_seed))] for i in range(length)]


def gate_and_summary(
    selected: dict[str, dict[str, dict[str, Any]]], base_metrics: dict[str, Any], dev_labels,
    spec, *, gradient_check_passed: bool, resource_limits_passed: bool,
):
    seed_order = [str(s) for s in spec["learner"]["optimizer"]["seeds"]]
    arm_summary = {}
    for arm in ("frozen_base",) + ARMS:
        metrics_by_seed = (
            {seed: base_metrics for seed in seed_order}
            if arm == "frozen_base"
            else {seed: selected[arm][seed]["metrics"] for seed in seed_order}
        )
        arm_summary[arm] = {
            "mean_balanced_accuracy": statistics.fmean(v["balanced_accuracy"] for v in metrics_by_seed.values()),
            "mean_exact_action_accuracy": statistics.fmean(v["exact_action_accuracy"] for v in metrics_by_seed.values()),
            "mean_macro_f1": statistics.fmean(v["macro_f1"] for v in metrics_by_seed.values()),
            "mean_action_nll": statistics.fmean(v["action_nll"] for v in metrics_by_seed.values()),
            "mean_kl": statistics.fmean(v["mean_conditional_bernoulli_kl_nats_per_action"] for v in metrics_by_seed.values()),
            "per_seed_balanced_accuracy": {seed: metrics_by_seed[seed]["balanced_accuracy"] for seed in seed_order},
            "per_seed_mean_kl": {seed: metrics_by_seed[seed]["mean_conditional_bernoulli_kl_nats_per_action"] for seed in seed_order},
            "per_seed_selected_updates": {seed: (0 if arm == "frozen_base" else selected[arm][seed]["update"]) for seed in seed_order},
        }
    primary = selected["rloo_k4"]
    primary_per_prompt = average_seed_correct([primary[s]["metrics"]["per_prompt_correct"] for s in seed_order])
    base_per_prompt = average_seed_correct([base_metrics["per_prompt_correct"] for _ in seed_order])
    labels = [int(x) for x in dev_labels]
    comparison = stratified_paired_bootstrap(
        primary_per_prompt, base_per_prompt, labels,
        resamples=spec["metrics"]["paired_bootstrap"]["resamples"],
        seed=spec["metrics"]["paired_bootstrap"]["seed"],
    )
    baseline = selected["exact_expected_reward"]
    expected_per_prompt = average_seed_correct([baseline[s]["metrics"]["per_prompt_correct"] for s in seed_order])
    rloo_vs_expected = stratified_paired_bootstrap(
        primary_per_prompt, expected_per_prompt, labels,
        resamples=spec["metrics"]["paired_bootstrap"]["resamples"],
        seed=spec["metrics"]["paired_bootstrap"]["seed"] + 1,
    )
    sft_per_prompt = average_seed_correct([selected["context_sft"][s]["metrics"]["per_prompt_correct"] for s in seed_order])
    rloo_vs_sft = stratified_paired_bootstrap(
        primary_per_prompt, sft_per_prompt, labels,
        resamples=spec["metrics"]["paired_bootstrap"]["resamples"],
        seed=spec["metrics"]["paired_bootstrap"]["seed"] + 2,
    )
    mean_base = arm_summary["frozen_base"]["mean_balanced_accuracy"]
    mean_rloo = arm_summary["rloo_k4"]["mean_balanced_accuracy"]
    seed_differences = [primary[s]["metrics"]["balanced_accuracy"] - base_metrics["balanced_accuracy"] for s in seed_order]
    base_floor = base_metrics["balanced_accuracy"] >= spec["metrics"]["development_gate"]["minimum_base_balanced_accuracy"]
    recalls_ok = min(base_metrics["yes_recall"], base_metrics["no_recall"]) >= spec["metrics"]["development_gate"]["minimum_base_recall_per_class"]
    seed_nonworse = sum(d >= 0 for d in seed_differences)
    selected_kl = [primary[s]["metrics"]["mean_conditional_bernoulli_kl_nats_per_action"] for s in seed_order]
    gate = spec["metrics"]["development_gate"]
    decision = {
        "all_provenance_and_prompt_separation_checks_pass": True,
        "base_quality_floor_pass": bool(base_floor and recalls_ok),
        "rloo_mean_gain_threshold_pass": bool(mean_rloo - mean_base >= gate["rloo_minus_base_balanced_accuracy_at_least"]),
        "rloo_paired_ci_pass": bool(comparison["ci95_low"] > gate["rloo_minus_base_ci_lower_above"]),
        "seed_consistency_pass": bool(seed_nonworse >= gate["minimum_seeds_rloo_not_worse_than_base"]),
        "mean_and_per_seed_kl_pass": bool(
            arm_summary["rloo_k4"]["mean_kl"] <= gate["maximum_selected_mean_kl_nats_per_action"]
            and max(selected_kl) <= gate["maximum_selected_per_seed_mean_kl_nats_per_action"]
        ),
        "expected_reward_gradient_checks_pass": bool(gradient_check_passed),
        "all_arms_within_resource_limits": bool(resource_limits_passed),
        "result": "pass" if base_floor and recalls_ok and mean_rloo - mean_base >= gate["rloo_minus_base_balanced_accuracy_at_least"]
        and comparison["ci95_low"] > gate["rloo_minus_base_ci_lower_above"]
        and seed_nonworse >= gate["minimum_seeds_rloo_not_worse_than_base"]
        and arm_summary["rloo_k4"]["mean_kl"] <= gate["maximum_selected_mean_kl_nats_per_action"]
        and max(selected_kl) <= gate["maximum_selected_per_seed_mean_kl_nats_per_action"]
        and gradient_check_passed and resource_limits_passed else "non-pass",
    }
    return {
        "arms": arm_summary,
        "primary_rloo_minus_base_balanced_accuracy": comparison,
        "rloo_minus_exact_expected_reward_balanced_accuracy": rloo_vs_expected,
        "rloo_minus_context_sft_balanced_accuracy": rloo_vs_sft,
        "rloo_seed_balanced_accuracy_differences_vs_base": dict(zip(seed_order, seed_differences)),
        "rloo_seeds_not_worse_than_base": seed_nonworse,
        "development_gate": decision,
    }


def current_git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    except Exception as exc:
        raise RuntimeError("cannot establish source revision") from exc
    if dirty:
        raise RuntimeError("source tree must be clean before the frozen run")
    return {"commit": commit, "working_tree_clean": True}


def run(output: Path, spec_path: Path = SPEC_PATH, lock_path: Path = LOCK_PATH):
    started = time.monotonic()
    spec, protocol_hash = load_locked_spec(spec_path, lock_path)
    source_state = current_git_state()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(lock_path.read_bytes())
    snapshot_files = [
        spec_path,
        lock_path,
        Path(__file__),
        Path(__file__).with_name("boolq_rloo_task.py"),
        Path(__file__).with_name("boolq_sequence_task.py"),
        Path(__file__).with_name("rloo_binary_objectives.py"),
        Path(__file__).with_name("audit_cpu_lm_boolq_verifier_rloo_development_v1.py"),
    ]
    for source in snapshot_files:
        (output / f"source-{source.name}").write_bytes(source.read_bytes())
    (output / "DATASET-NOTICE.md").write_bytes((ROOT / "data/BOOLQ-DATA-NOTICE.md").read_bytes())
    compute = spec["compute_limits"]
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    campaign_deadline = started + float(compute["max_wall_seconds_campaign"])
    try:
        import datasets
        import safetensors
        import tokenizers
        import torch
        import transformers
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runtime_versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "numpy": np.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
        }
        if runtime_versions != {
            key: spec["runtime_lock"][key] for key in runtime_versions
        }:
            raise RuntimeError(f"runtime versions differ from the pre-run lock: {runtime_versions}")
        torch.set_num_threads(int(compute["threads"]))
        objective_checks = expected_reward_gradient_self_check()
        if objective_checks["pass"] is not True:
            raise AssertionError("frozen expected reward gradient self-check failed")
        if torch.cuda.is_initialized():
            raise RuntimeError("the frozen experiment requires CPU-only execution")
        if torch.backends.mps.is_available():
            # Availability does not enable MPS, but the protocol forbids selecting it.
            pass
        cached_hashes = check_offline_cache(spec)
        data_spec = spec["data"]["dataset"]
        train_ds = load_dataset(data_spec["id"], "default", split="train", revision=data_spec["revision"])
        validation_ds = load_dataset(data_spec["id"], "default", split="validation", revision=data_spec["revision"])
        if len(train_ds) != 9427 or len(validation_ds) != 3270:
            raise ValueError("cached BoolQ split sizes differ from the pinned source")
        if train_ds._fingerprint != data_spec["train_fingerprint"] or validation_ds._fingerprint != data_spec["validation_fingerprint"]:
            raise ValueError("cached BoolQ dataset fingerprints differ from the protocol")

        # Rank with prompt fields only. The answer column is loaded only for the selected train and development rows.
        train_indices = hash_rank_indices(train_ds, spec["data"]["training"]["rank_start"], spec["data"]["training"]["rank_stop"])
        development_indices = hash_rank_indices(validation_ds, spec["data"]["development"]["rank_start"], spec["data"]["development"]["rank_stop"])
        confirmation = spec["data"]["confirmation_reservation"]
        confirmation_indices = hash_rank_indices(validation_ds, confirmation["rank_start"], confirmation["rank_stop"])
        train_rows = build_selected_rows(train_ds, train_indices, "train")
        development_rows = build_selected_rows(validation_ds, development_indices, "validation")
        # Confirmation prompts are accessed only as transient hashes for duplicate detection; labels/text are not requested, stored, or scored.
        validation_prompt_view = validation_ds.select_columns(["question", "passage"])
        confirmation_hashes = [
            hashlib.sha256((validation_prompt_view[i]["question"] + "\n" + validation_prompt_view[i]["passage"]).encode("utf-8")).hexdigest()
            for i in confirmation_indices
        ]
        validate_disjoint_hashes({"train": train_rows, "development": development_rows})
        if set(confirmation_hashes) & ({row["question_sha256"] for row in train_rows} | {row["question_sha256"] for row in development_rows}):
            raise ValueError("confirmation prompt hashes overlap train/development cohorts")
        if set(train_rows[i]["question_sha256"] for i in range(len(train_rows))) & recorded_boolq_hashes():
            raise ValueError("new train prompt overlaps an earlier recorded BoolQ v16/v17 prompt")
        if {row["question_sha256"] for row in development_rows} & recorded_boolq_hashes():
            raise ValueError("new development prompt overlaps an earlier recorded BoolQ v16/v17 prompt")
        if set(confirmation_hashes) & recorded_boolq_hashes():
            raise ValueError("reserved confirmation prompt overlaps an earlier recorded BoolQ v16/v17 prompt")
        # The earlier private formatting pilot occupied training ranks 0–23; compare prompt-only hashes.
        pilot_indices = hash_rank_indices(train_ds, 0, 24)
        pilot_view = train_ds.select_columns(["question", "passage"])
        pilot_hashes = {
            hashlib.sha256((pilot_view[i]["question"] + "\n" + pilot_view[i]["passage"]).encode("utf-8")).hexdigest()
            for i in pilot_indices
        }
        if pilot_hashes & {row["question_sha256"] for row in train_rows}:
            raise ValueError("new train prompt overlaps the private BoolQ formatting pilot")
        if pilot_hashes & {row["question_sha256"] for row in development_rows}:
            raise ValueError("new development prompt overlaps the private BoolQ formatting pilot")
        if time.monotonic() >= campaign_deadline:
            raise TimeoutError("campaign exceeded its wall-time budget during dataset preparation")

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        action_ids = verify_token_actions(tokenizer, spec["learner"]["action_token_ids"])
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to("cpu")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model_hashes = cached_hashes
        train_features = extract_features(
            train_rows, tokenizer, model, action_ids, int(compute["feature_batch_size"]),
            deadline=campaign_deadline, max_wall=float(compute["max_wall_seconds_campaign"]),
        )
        development_features = extract_features(
            development_rows, tokenizer, model, action_ids, int(compute["feature_batch_size"]),
            deadline=campaign_deadline, max_wall=float(compute["max_wall_seconds_campaign"]),
        )
        train_y = np.asarray([1 if row["verifier_answer"] == "yes" else 0 for row in train_rows], dtype=np.int64)
        development_y = np.asarray([1 if row["verifier_answer"] == "yes" else 0 for row in development_rows], dtype=np.int64)
        train_x, development_x, train_mean, train_scale = standardized_features(
            train_features["hidden"], development_features["hidden"]
        )
        train_weights = class_weights(train_y.tolist())
        if time.monotonic() >= campaign_deadline or rss_bytes() > int(compute["max_peak_rss_bytes_campaign"]):
            raise RuntimeError("campaign exceeded its frozen CPU resource budget during feature extraction")

        # Save only selected train/development prompts and their features; reserved labels and text are never persisted.
        write_json(output / "cohorts.json", {
            "training": train_rows,
            "development": development_rows,
            "indices": {"training": train_indices, "development": development_indices},
            "prompt_hashes": {
                "training": [row["question_sha256"] for row in train_rows],
                "development": [row["question_sha256"] for row in development_rows],
            },
        })
        np.savez_compressed(
            output / "features.npz",
            train_hidden=train_features["hidden"],
            development_hidden=development_features["hidden"],
            train_standardized=train_x.astype(np.float32),
            development_standardized=development_x.astype(np.float32),
            train_mean=train_mean,
            train_scale=train_scale,
            train_base_margin=train_features["base_margin"],
            development_base_margin=development_features["base_margin"],
            train_labels=train_y,
            development_labels=development_y,
        )
        write_json(output / "feature_metadata.json", {
            "action_token_ids": action_ids,
            "train_prompt_token_counts": train_features["prompt_tokens"],
            "development_prompt_token_counts": development_features["prompt_tokens"],
            "feature_dimension": int(train_features["hidden"].shape[1]),
            "train_class_weights": {str(k): float(v) for k, v in train_weights.items()},
            "normalization": "train-only mean and population standard deviation; zero scales replaced with one",
            "confirmation_access": "prompt-only rank/hash assignment; answer labels, model features, rewards, scores, metrics, and serialized rows were not accessed",
        })
        write_json(output / "input_fingerprints.json", {
            "dataset_revision": data_spec["revision"],
            "cached_dataset_and_model_sha256": model_hashes,
            "dataset_fingerprints": {"train": train_ds._fingerprint, "validation": validation_ds._fingerprint},
            "source": source_state,
        })
        del model

        labels_t = train_y
        seeds = [int(x) for x in spec["learner"]["optimizer"]["seeds"]]
        seed_runs: dict[str, dict[str, list[dict[str, Any]]]] = {arm: {} for arm in ARMS}
        orders: dict[str, list[list[int]]] = {}
        rollouts: list[dict[str, Any]] = []
        arm_wall_seconds = {arm: 0.0 for arm in ARMS}
        for seed in seeds:
            batch_order = make_minibatches(
                len(train_rows),
                int(spec["learner"]["optimizer"]["maximum_updates"]),
                int(spec["learner"]["optimizer"]["minibatch_size"]),
                seed,
            )
            orders[str(seed)] = batch_order
            for arm in ARMS:
                if time.monotonic() >= campaign_deadline or rss_bytes() > int(compute["max_peak_rss_bytes_campaign"]):
                    raise RuntimeError("campaign exceeded its frozen CPU resource budget during policy updates")
                arm_started = time.monotonic()
                checkpoints = train_arm(
                    arm, seed, train_x, train_features["base_margin"], labels_t, train_weights,
                    batch_order, spec, rollouts,
                    deadline=campaign_deadline,
                    max_peak_rss_bytes=int(compute["max_peak_rss_bytes_campaign"]),
                )
                arm_wall_seconds[arm] += time.monotonic() - arm_started
                seed_runs[arm][str(seed)] = checkpoints
                if time.monotonic() - started > float(compute["max_wall_seconds_campaign"]):
                    raise TimeoutError("campaign exceeded its frozen wall-time budget")
        write_json(output / "training_minibatches.json", orders)
        write_json(output / "rloo_rollouts.json", rollouts)

        base_metrics = evaluate_policy(
            development_x, development_features["base_margin"], development_y, None, np.float32(0), 0
        )
        chosen = choose_checkpoints(seed_runs, development_x, development_features["base_margin"], development_y, spec)
        # Add base selected records in the same per-seed format for downstream comparisons.
        campaign_wall_seconds = time.monotonic() - started
        resource_limits_passed = (
            campaign_wall_seconds <= float(compute["max_wall_seconds_campaign"])
            and rss_bytes() <= int(compute["max_peak_rss_bytes_campaign"])
        )
        summary = gate_and_summary(
            chosen, base_metrics, development_y.tolist(), spec,
            gradient_check_passed=bool(objective_checks["pass"]),
            resource_limits_passed=resource_limits_passed,
        )
        # Retain all checkpoint adapters/metrics for selection reconstruction.
        checkpoint_history = {}
        per_seed = {arm: {} for arm in ARMS}
        for arm in ARMS:
            checkpoint_history[arm] = {}
            for seed in [str(s) for s in seeds]:
                history = []
                for checkpoint in seed_runs[arm][seed]:
                    metrics = evaluate_policy(
                        development_x, development_features["base_margin"], development_y,
                        None if checkpoint["weight"] is None else np.asarray(checkpoint["weight"], dtype=np.float32),
                        np.float32(checkpoint["bias"]), int(checkpoint["update"]),
                    )
                    eligible = metrics["mean_conditional_bernoulli_kl_nats_per_action"] <= spec["learner"]["kl"]["maximum_selected_mean_nats_per_action"]
                    history.append({"update": int(checkpoint["update"]), "weight": checkpoint["weight"],
                                    "bias": float(checkpoint["bias"]), "metrics": metrics,
                                    "eligible": bool(eligible),
                                    "optimizer_updates": int(checkpoint["update"]),
                                    "cumulative_rloo_sampled_actions": int(
                                        checkpoint["update"] * spec["learner"]["optimizer"]["minibatch_size"]
                                        * spec["learner"]["rloo"]["group_size"]
                                    ) if arm == "rloo_k4" else 0})
                checkpoint_history[arm][seed] = history
                best = max((x for x in history if x["eligible"]), key=lambda x: (x["metrics"]["balanced_accuracy"], -x["update"]))
                if best["update"] != chosen[arm][seed]["update"]:
                    raise AssertionError("selected checkpoint differs from frozen rule")
                per_seed[arm][seed] = {
                    "selected_updates": best["update"],
                    "selected_metrics": best["metrics"],
                    "selected_weight": best["weight"],
                    "selected_bias": best["bias"],
                    "parameter_delta_l2_from_base": float(math.sqrt(
                        (0.0 if best["weight"] is None else float(np.square(np.asarray(best["weight"], dtype=np.float64)).sum()))
                        + float(best["bias"]) ** 2
                    )),
                    "cumulative_rloo_sampled_actions_at_selected_update": int(
                        best["update"] * spec["learner"]["optimizer"]["minibatch_size"]
                        * spec["learner"]["rloo"]["group_size"]
                    ) if arm == "rloo_k4" else 0,
                }

        rollout_by_seed = {}
        for seed in [str(s) for s in seeds]:
            records = [entry for entry in rollouts if str(entry["seed"]) == seed]
            rewards = [reward for entry in records for group in entry["rewards"] for reward in group]
            actions = [action for entry in records for group in entry["actions"] for action in group]
            rollout_by_seed[seed] = {
                "updates": len(records),
                "prompt_groups": sum(len(entry["actions"]) for entry in records),
                "sampled_actions": len(actions),
                "sampled_yes_actions": sum(actions),
                "sampled_no_actions": len(actions) - sum(actions),
                "mean_verifier_reward": statistics.fmean(rewards) if rewards else None,
                "verifier_reward_variance": statistics.pvariance(rewards) if rewards else None,
                "mixed_reward_group_fraction": statistics.fmean(
                    entry["mixed_reward_group_rate"] for entry in records
                ) if records else None,
                "zero_advantage_group_fraction": statistics.fmean(
                    entry["zero_advantage_group_rate"] for entry in records
                ) if records else None,
            }
        write_json(output / "checkpoint_history.json", checkpoint_history)
        write_json(output / "summary.json", {
            "protocol_id": spec["protocol_id"],
            "protocol_sha256": protocol_hash,
            "source_commit": source_state["commit"],
            "runtime": {
                **runtime_versions,
                "platform": platform.platform(),
                "torch_threads": torch.get_num_threads(), "device": "cpu", "network_offline": True,
            },
            "cohort_counts": {"training": len(train_rows), "development": len(development_rows)},
            "label_support": {
                "training_yes": int(train_y.sum()), "training_no": int(len(train_y) - train_y.sum()),
                "development_yes": int(development_y.sum()), "development_no": int(len(development_y) - development_y.sum()),
            },
            "model_action_token_ids": action_ids,
            "base_metrics": base_metrics,
            "comparison": summary,
            "objective_gradient_checks": objective_checks,
            "per_seed": per_seed,
            "rloo_rollout_diagnostics_by_seed": rollout_by_seed,
            "resource_measurements": {
                "campaign_wall_seconds": campaign_wall_seconds,
                "campaign_peak_rss_bytes": rss_bytes(),
                "cpu_threads": int(compute["threads"]),
                "device": "cpu",
                "network_offline": True,
                "rollout_action_count": len(rollouts) * int(spec["learner"]["rloo"]["group_size"]) * int(spec["learner"]["optimizer"]["minibatch_size"]),
                "optimizer_updates_per_trained_arm_seed": int(spec["learner"]["optimizer"]["maximum_updates"]),
            },
            "arm_wall_seconds": arm_wall_seconds,
            "limitations": [
                "Public BoolQ may have appeared in model pretraining.",
                "Verifier reward is answer-key correctness, not a learned reward model or human preference.",
                "The policy chooses one token from Yes/No only; no continuation is sampled.",
                "Checkpoint selection uses the development cohort; its confidence intervals are screening statistics.",
                "RLOO and exact expected reward have the same expected binary-action reward gradient.",
            ],
        })
        summary_path = output / "summary.json"
        finalized = json.loads(summary_path.read_text(encoding="utf-8"))
        final_wall = time.monotonic() - started
        final_peak = rss_bytes()
        final_resource_pass = (
            final_wall <= float(compute["max_wall_seconds_campaign"])
            and final_peak <= int(compute["max_peak_rss_bytes_campaign"])
        )
        finalized["resource_measurements"]["campaign_wall_seconds"] = final_wall
        finalized["resource_measurements"]["campaign_peak_rss_bytes"] = final_peak
        decision = finalized["comparison"]["development_gate"]
        decision["all_arms_within_resource_limits"] = final_resource_pass
        if not final_resource_pass:
            decision["result"] = "non-pass"
        write_json(summary_path, finalized)
        write_manifest(output)
        if rss_bytes() > int(compute["max_peak_rss_bytes_campaign"]) or time.monotonic() - started > float(compute["max_wall_seconds_campaign"]):
            raise RuntimeError("campaign resource ceiling crossed before finalization")
        subprocess.check_call(
            [sys.executable, str(Path(__file__).with_name("audit_cpu_lm_boolq_verifier_rloo_development_v1.py")), str(output)],
            cwd=ROOT,
        )
        # The auditor verifies the pre-audit manifest before it writes audit.json.
        # Seal the complete accepted bundle after the report exists.
        write_manifest(output, include_audit=True)
    except Exception as exc:
        summary_path = output / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            decision = summary.get("comparison", {}).get("development_gate", {})
            resource_ok = (
                time.monotonic() - started <= float(compute["max_wall_seconds_campaign"])
                and rss_bytes() <= int(compute["max_peak_rss_bytes_campaign"])
            )
            decision["all_arms_within_resource_limits"] = bool(resource_ok)
            decision["all_provenance_and_prompt_separation_checks_pass"] = False
            decision["result"] = "non-pass"
            summary.setdefault("resource_measurements", {})["campaign_wall_seconds_at_finalization"] = time.monotonic() - started
            summary["resource_measurements"]["campaign_peak_rss_bytes_at_finalization"] = rss_bytes()
            summary["finalization_failure"] = {"exception_type": type(exc).__name__, "message": str(exc)}
            write_json(summary_path, summary)
        (output / "audit.json").unlink(missing_ok=True)
        write_json(output / "failure.json", {
            "protocol_id": spec["protocol_id"],
            "source_commit": source_state["commit"],
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": rss_bytes(),
        })
        write_manifest(output)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=SPEC_PATH)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args()
    run(args.output, args.protocol, args.lock)


if __name__ == "__main__":
    main()
