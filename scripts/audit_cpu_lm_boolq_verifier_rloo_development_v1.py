#!/usr/bin/env python3
"""Independently reconstruct a frozen BoolQ binary-action RLOO development bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import statistics
import subprocess
import sys
import time
import platform
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "cpu_lm_boolq_verifier_rloo_development_v1"
SOURCE_MAP = {
    f"protocols/{PROTOCOL}.json": f"source-{PROTOCOL}.json",
    f"protocols/{PROTOCOL}.lock.json": f"source-{PROTOCOL}.lock.json",
    "scripts/run_cpu_lm_boolq_verifier_rloo_development_v1.py": "source-run_cpu_lm_boolq_verifier_rloo_development_v1.py",
    "scripts/boolq_rloo_task.py": "source-boolq_rloo_task.py",
    "scripts/boolq_sequence_task.py": "source-boolq_sequence_task.py",
    "scripts/rloo_binary_objectives.py": "source-rloo_binary_objectives.py",
    "scripts/audit_cpu_lm_boolq_verifier_rloo_development_v1.py": "source-audit_cpu_lm_boolq_verifier_rloo_development_v1.py",
    "data/BOOLQ-DATA-NOTICE.md": "DATASET-NOTICE.md",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, *, tolerance: float = 2e-5) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"reconstructed value differs: {actual!r} != {expected!r}")


def compare_metric(actual: Any, expected: Any, *, key: str) -> None:
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"metric list length differs: {key}")
        if key == "per_prompt_correct":
            if [int(value) for value in actual] != [int(value) for value in expected]:
                raise ValueError(f"per-prompt correctness differs: {key}")
        else:
            for left, right in zip(actual, expected):
                close(left, right)
    elif isinstance(expected, (int, float)):
        close(actual, expected)
    elif actual != expected:
        raise ValueError(f"metric differs: {key}")


def verify_manifest(bundle: Path) -> int:
    manifest = read_json(bundle / "manifest.json").get("files")
    if not isinstance(manifest, dict):
        raise ValueError("manifest.files is missing")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "audit.json"}
    }
    if actual != set(manifest):
        raise ValueError("manifest inventory differs from bundle files")
    for name, digest in manifest.items():
        if sha256_file(bundle / name) != digest:
            raise ValueError(f"manifest hash mismatch: {name}")
    return len(manifest)


def verify_source_snapshots(bundle: Path, commit: str) -> None:
    git_dir = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    if Path(git_dir) != ROOT:
        raise ValueError("auditor root is not the expected repository")
    subprocess.check_output(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT)
    for relative, snapshot_name in SOURCE_MAP.items():
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if (bundle / snapshot_name).read_bytes() != committed:
            raise ValueError(f"pre-run source snapshot differs from recorded commit: {relative}")


def independently_check_rloo_gradient_identity(tolerance: float = 1e-10) -> dict[str, Any]:
    checked = 0
    maximum = 0.0
    group_size = 4
    for logit in (-1.3, 0.37, 1.8):
        p_yes = 1 / (1 + math.exp(-logit))
        for label in (0, 1):
            for reward_weight in (0.6, 1.4):
                exact_gradient = (-1 if label == 1 else 1) * reward_weight * p_yes * (1 - p_yes)
                enumerated = 0.0
                for mask in range(1 << group_size):
                    actions = [(mask >> index) & 1 for index in range(group_size)]
                    probability = math.prod(p_yes if action else 1 - p_yes for action in actions)
                    rewards = [reward_weight if action == label else 0.0 for action in actions]
                    advantages = [
                        reward - (sum(rewards) - reward) / (group_size - 1)
                        for reward in rewards
                    ]
                    sample_gradient = -sum(
                        advantage * (action - p_yes)
                        for advantage, action in zip(advantages, actions)
                    ) / group_size
                    enumerated += probability * sample_gradient
                error = abs(enumerated - exact_gradient)
                maximum = max(maximum, error)
                checked += 1
                if error > tolerance:
                    raise ValueError("independent RLOO expected-gradient enumeration failed")
    return {"cases": checked, "max_abs_error": maximum, "tolerance": tolerance}


def verify_protocol(bundle: Path) -> tuple[dict[str, Any], str]:
    spec = read_json(bundle / "protocol.snapshot.json")
    lock = read_json(bundle / "protocol.lock.snapshot.json")
    lock_digest = lock.pop("sha256", None)
    digest = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_digest != digest or canonical(lock) != canonical(spec):
        raise ValueError("protocol snapshot does not match its lock")
    if spec.get("protocol_id") != "cpu-lm-boolq-verifier-rloo-development-v1":
        raise ValueError("unexpected protocol id")
    return spec, digest


def prompt_hash(question: str, passage: str) -> str:
    return hashlib.sha256((question + "\n" + passage).encode("utf-8")).hexdigest()


def expected_prompt(question: str, passage: str) -> str:
    return (
        "Read the passage and answer the question using the passage only. "
        "Reply with exactly one word: Yes or No.\n\n"
        f"Passage: {passage}\n\nQuestion: {question}"
    )


def verify_cohorts(bundle: Path, spec: dict[str, Any], fingerprints: dict[str, Any]) -> dict[str, int]:
    # Hugging Face rows are sliced to prompt columns before hashing/rank selection.
    from datasets import load_dataset

    data_spec = spec["data"]["dataset"]
    train_ds = load_dataset(data_spec["id"], "default", split="train", revision=data_spec["revision"])
    validation_ds = load_dataset(data_spec["id"], "default", split="validation", revision=data_spec["revision"])
    train_view = train_ds.select_columns(["question", "passage"])
    validation_view = validation_ds.select_columns(["question", "passage"])

    def ranked(view):
        return sorted(
            range(len(view)),
            key=lambda i: (prompt_hash(view[i]["question"], view[i]["passage"]), i),
        )

    train_order = ranked(train_view)
    validation_order = ranked(validation_view)
    intervals = {
        "training": (train_order, spec["data"]["training"]),
        "development": (validation_order, spec["data"]["development"]),
    }
    cohorts = read_json(bundle / "cohorts.json")
    expected_indices = {}
    for cohort_name, (ordering, cfg) in intervals.items():
        indices = ordering[int(cfg["rank_start"]):int(cfg["rank_stop"])]
        rows = cohorts[cohort_name]
        if len(indices) != int(cfg["count"]) or len(rows) != len(indices):
            raise ValueError(f"cohort size differs from the lock: {cohort_name}")
        if [int(row["dataset_index"]) for row in rows] != indices:
            raise ValueError(f"ranked source indices differ: {cohort_name}")
        view = train_view if cohort_name == "training" else validation_view
        labels = train_ds if cohort_name == "training" else validation_ds
        split_name = cfg["split"]
        for row, index in zip(rows, indices):
            source = view[index]
            label = bool(labels[int(index)]["answer"])
            if row["question"] != source["question"] or row["passage"] != source["passage"]:
                raise ValueError(f"selected row text differs from pinned source: {cohort_name}/{index}")
            if row["dataset_split"] != split_name or row["source_answer"] is not label:
                raise ValueError(f"selected row metadata/label differs from source: {cohort_name}/{index}")
            expected_answer = "yes" if label else "no"
            if row["verifier_answer"] != expected_answer:
                raise ValueError(f"verifier label differs from source: {cohort_name}/{index}")
            if row["user_message"] != expected_prompt(source["question"], source["passage"]):
                raise ValueError(f"prompt template differs: {cohort_name}/{index}")
            digest = prompt_hash(source["question"], source["passage"])
            if row["question_sha256"] != digest:
                raise ValueError(f"prompt digest differs: {cohort_name}/{index}")
        expected_indices[cohort_name] = indices

    train_hashes = {row["question_sha256"] for row in cohorts["training"]}
    dev_hashes = {row["question_sha256"] for row in cohorts["development"]}
    if len(train_hashes) != len(cohorts["training"]) or len(dev_hashes) != len(cohorts["development"]):
        raise ValueError("duplicate prompt hash within a selected cohort")
    if train_hashes & dev_hashes:
        raise ValueError("training and development prompt hashes overlap")
    reserve = spec["data"]["confirmation_reservation"]
    reserve_order = validation_order[int(reserve["rank_start"]):int(reserve["rank_stop"])]
    reserve_hashes = {prompt_hash(validation_view[i]["question"], validation_view[i]["passage"]) for i in reserve_order}
    if len(reserve_hashes) != int(reserve["count"]):
        raise ValueError("reserved prompt hashes are not unique")
    if reserve_hashes & (train_hashes | dev_hashes):
        raise ValueError("reserved prompt hashes overlap a scored cohort")
    pilot_hashes = {
        prompt_hash(train_view[i]["question"], train_view[i]["passage"])
        for i in train_order[:24]
    }
    if pilot_hashes & (train_hashes | dev_hashes):
        raise ValueError("selected cohort overlaps the prior 24-row format pilot")
    legacy_hashes = set()
    for version in (16, 17):
        pattern = f"results/cpu-lm-boolq-posttraining-development-v{version}-*/run-*/development.json"
        for path in ROOT.glob(pattern):
            prior = read_json(path)
            for key in ("train_examples", "validation_examples"):
                for row in prior.get(key, []):
                    if row.get("question_sha256"):
                        legacy_hashes.add(str(row["question_sha256"]))
    if (train_hashes | dev_hashes | reserve_hashes) & legacy_hashes:
        raise ValueError("selected or reserved cohort overlaps a retained BoolQ v16/v17 prompt")
    if cohorts["indices"] != expected_indices:
        raise ValueError("cohort index summary differs from reconstructed rank intervals")
    if cohorts["prompt_hashes"] != {
        "training": [row["question_sha256"] for row in cohorts["training"]],
        "development": [row["question_sha256"] for row in cohorts["development"]],
    }:
        raise ValueError("cohort prompt-hash index differs from saved rows")
    if train_ds._fingerprint != fingerprints["dataset_fingerprints"]["train"]:
        raise ValueError("training dataset fingerprint differs")
    if validation_ds._fingerprint != fingerprints["dataset_fingerprints"]["validation"]:
        raise ValueError("validation dataset fingerprint differs")
    return {
        "training": len(train_hashes),
        "development": len(dev_hashes),
        "reserved_prompt_hashes": len(reserve_hashes),
        "prior_format_pilot_hashes_excluded": len(pilot_hashes),
        "prior_v16_v17_hashes_excluded": len(legacy_hashes),
    }


def binary_metrics(logits: np.ndarray, labels: np.ndarray, base_logits: np.ndarray) -> dict[str, Any]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    base_logits = np.asarray(base_logits, dtype=np.float64).reshape(-1)
    if not (len(logits) == len(labels) == len(base_logits)) or not len(labels):
        raise ValueError("metric vectors have inconsistent sizes")
    pred = (logits >= 0).astype(np.int64)
    yes_idx = labels == 1
    no_idx = labels == 0
    if not yes_idx.any() or not no_idx.any():
        raise ValueError("development cohort must contain both labels")
    yes_recall = float((pred[yes_idx] == 1).mean())
    no_recall = float((pred[no_idx] == 0).mean())
    f1_values = []
    for cls in (0, 1):
        tp = int(((pred == cls) & (labels == cls)).sum())
        fp = int(((pred == cls) & (labels != cls)).sum())
        fn = int(((pred != cls) & (labels == cls)).sum())
        f1_values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    probs = np.where(logits >= 0, 1 / (1 + np.exp(-logits)), np.exp(logits) / (1 + np.exp(logits)))
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    nll = -np.where(labels == 1, np.log(probs), np.log1p(-probs)).mean()
    base_p = np.where(base_logits >= 0, 1 / (1 + np.exp(-base_logits)), np.exp(base_logits) / (1 + np.exp(base_logits)))
    base_p = np.clip(base_p, 1e-12, 1 - 1e-12)
    kl = probs * np.log(probs / base_p) + (1 - probs) * np.log((1 - probs) / (1 - base_p))
    correct = (pred == labels).astype(np.float64)
    return {
        "examples": int(len(labels)),
        "exact_action_accuracy": float(correct.mean()),
        "balanced_accuracy": float((yes_recall + no_recall) / 2),
        "yes_recall": yes_recall,
        "no_recall": no_recall,
        "yes_support": int(yes_idx.sum()),
        "no_support": int(no_idx.sum()),
        "macro_f1": float(statistics.fmean(f1_values)),
        "action_nll": float(nll),
        "mean_conditional_bernoulli_kl_nats_per_action": float(kl.mean()),
        "parse_rate": 1.0,
        "per_prompt_correct": correct.astype(int).tolist(),
        "per_prompt_probability_yes": probs.tolist(),
        "per_prompt_base_probability_yes": base_p.tolist(),
        "per_prompt_kl": kl.tolist(),
    }


def paired_bootstrap(a: list[list[float]], b: list[list[float]], labels: list[int], resamples: int, seed: int):
    if len(a) != len(b) or len(a) != len(labels) or not labels:
        raise ValueError("paired bootstrap vectors do not align")
    if any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError("paired bootstrap seed dimensions differ")
    groups = {cls: [i for i, label in enumerate(labels) if int(label) == cls] for cls in (0, 1)}
    if not all(groups.values()):
        raise ValueError("paired bootstrap needs both label classes")

    def score(sampled):
        means = [statistics.fmean(a[i]) - statistics.fmean(b[i]) for i in sampled]
        recalls = [statistics.fmean(means[i] for i in sampled if labels[i] == cls) for cls in (0, 1)]
        return statistics.fmean(recalls)

    observed = score(list(range(len(labels))))
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = []
        for cls in (0, 1):
            sample.extend(rng.choices(groups[cls], k=len(groups[cls])))
        values.append(score(sample))
    values.sort()
    return {
        "difference": observed,
        "ci95_low": values[max(0, int(0.025 * resamples))],
        "ci95_high": values[min(resamples - 1, int(0.975 * resamples))],
    }


def verify_features(bundle: Path, summary: dict[str, Any], spec: dict[str, Any]):
    cohorts = read_json(bundle / "cohorts.json")
    with np.load(bundle / "features.npz", allow_pickle=False) as arrays:
        train_h = arrays["train_hidden"].astype(np.float64)
        dev_h = arrays["development_hidden"].astype(np.float64)
        train_x = arrays["train_standardized"].astype(np.float64)
        dev_x = arrays["development_standardized"].astype(np.float64)
        mean = arrays["train_mean"].astype(np.float64)
        scale = arrays["train_scale"].astype(np.float64)
        train_margin = arrays["train_base_margin"].astype(np.float64)
        dev_margin = arrays["development_base_margin"].astype(np.float64)
        train_y = arrays["train_labels"].astype(np.int64)
        dev_y = arrays["development_labels"].astype(np.int64)
    if train_h.shape[0] != len(cohorts["training"]) or dev_h.shape[0] != len(cohorts["development"]):
        raise ValueError("feature rows do not match selected cohorts")
    if train_h.ndim != 2 or dev_h.ndim != 2 or train_h.shape[1] != dev_h.shape[1]:
        raise ValueError("feature dimensions differ")
    if not (len(train_margin) == len(train_y) == train_h.shape[0] and len(dev_margin) == len(dev_y) == dev_h.shape[0]):
        raise ValueError("feature, margin, and label dimensions differ")
    if train_y.tolist() != [int(row["verifier_answer"] == "yes") for row in cohorts["training"]]:
        raise ValueError("training feature labels differ from selected source labels")
    if dev_y.tolist() != [int(row["verifier_answer"] == "yes") for row in cohorts["development"]]:
        raise ValueError("development feature labels differ from selected source labels")
    expected_mean = train_h.mean(axis=0)
    expected_scale = train_h.std(axis=0)
    expected_scale[expected_scale == 0] = 1.0
    if not np.allclose(mean, expected_mean, rtol=2e-5, atol=2e-5):
        raise ValueError("feature mean is not reconstructed from train rows")
    if not np.allclose(scale, expected_scale, rtol=2e-5, atol=2e-5):
        raise ValueError("feature scale is not reconstructed from train rows")
    if not np.allclose(train_x, (train_h - mean) / scale, rtol=2e-5, atol=2e-5):
        raise ValueError("stored training normalization differs")
    if not np.allclose(dev_x, (dev_h - mean) / scale, rtol=2e-5, atol=2e-5):
        raise ValueError("stored development normalization differs")
    token_ids = read_json(bundle / "feature_metadata.json")["action_token_ids"]
    if token_ids != spec["learner"]["action_token_ids"]:
        raise ValueError("action token ids differ from locked protocol")
    return train_x.astype(np.float32), train_margin.astype(np.float32), train_y, dev_x.astype(np.float32), dev_margin.astype(np.float32), dev_y


def replay_model_features(bundle: Path, spec: dict[str, Any], *, deadline: float):
    """Re-run tokenizer/model forwards for selected rows only and compare stored features."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    compute = spec["compute_limits"]
    torch.set_num_threads(int(compute["threads"]))
    model_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / spec["learner"]["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_ids = {}
    for label in ("Yes", "No"):
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"action {label} is not one token in the pinned tokenizer")
        action_ids[label.lower()] = int(tokens[0])
    if action_ids != spec["learner"]["action_token_ids"]:
        raise ValueError("independently loaded action token IDs differ from protocol")
    if read_json(bundle / "summary.json")["model_action_token_ids"] != action_ids:
        raise ValueError("summary action token IDs differ from independently loaded tokenizer")
    fixture = [{"role": "user", "content": "Tokenizer-only fixture. Is this a tokenization check?"}]
    prefix_ids = tokenizer.apply_chat_template(fixture, tokenize=True, add_generation_prompt=True)
    prefix_text = tokenizer.apply_chat_template(fixture, tokenize=False, add_generation_prompt=True)
    if tokenizer.encode(prefix_text, add_special_tokens=False) != prefix_ids:
        raise ValueError("pinned chat template prefix text differs from tokenized prefix")
    for label, token_id in (("Yes", action_ids["yes"]), ("No", action_ids["no"])):
        suffix = tokenizer.encode(label, add_special_tokens=False)
        if tokenizer.encode(prefix_text + label, add_special_tokens=False) != prefix_ids + suffix:
            raise ValueError(f"{label} action crosses the prefix token boundary")
        if not tokenizer.decode(prefix_ids + [token_id], skip_special_tokens=False).rstrip().endswith(label):
            raise ValueError(f"decoded action does not end in {label}")

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to("cpu")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cohorts = read_json(bundle / "cohorts.json")
    metadata = read_json(bundle / "feature_metadata.json")
    with np.load(bundle / "features.npz", allow_pickle=False) as arrays:
        expected = {
            "training": (arrays["train_hidden"], arrays["train_base_margin"]),
            "development": (arrays["development_hidden"], arrays["development_base_margin"]),
        }
    actual_tokens = {}
    max_abs_hidden_error = 0.0
    max_abs_margin_error = 0.0
    head_weight = model.lm_head.weight.index_select(0, torch.tensor([action_ids["yes"], action_ids["no"]]))
    head_bias = None
    if model.lm_head.bias is not None:
        head_bias = model.lm_head.bias.index_select(0, torch.tensor([action_ids["yes"], action_ids["no"]]))
    for name in ("training", "development"):
        rows = cohorts[name]
        hidden_chunks = []
        margins = []
        token_counts = []
        for offset in range(0, len(rows), int(compute["feature_batch_size"])):
            batch_rows = rows[offset:offset + int(compute["feature_batch_size"])]
            messages = [[{"role": "user", "content": row["user_message"]}] for row in batch_rows]
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, padding=True,
                return_tensors="pt", return_dict=True,
            )
            ids = encoded["input_ids"]
            attention = encoded["attention_mask"]
            lengths = attention.sum(dim=1)
            if torch.any(lengths > model.config.max_position_embeddings):
                raise ValueError("saved selected prompt exceeds context length")
            positions = lengths - 1
            with torch.inference_mode():
                hidden_all = model.model(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
                index = torch.arange(len(batch_rows), dtype=torch.long)
                hidden = hidden_all[index, positions].to(dtype=torch.float32).contiguous()
                action_logits = torch.nn.functional.linear(hidden, head_weight, head_bias)
                margins.append((action_logits[:, 0] - action_logits[:, 1]).cpu().numpy())
            hidden_chunks.append(hidden.cpu().numpy())
            token_counts.extend(lengths.tolist())
            if time.monotonic() >= deadline:
                raise TimeoutError("independent feature replay exceeded the frozen wall budget")
        actual_h = np.concatenate(hidden_chunks, axis=0)
        actual_m = np.concatenate(margins, axis=0)
        expected_h, expected_m = expected[name]
        if actual_h.shape != expected_h.shape or actual_m.shape != expected_m.shape:
            raise ValueError(f"independent model replay shape differs: {name}")
        if not np.allclose(actual_h, expected_h, rtol=2e-5, atol=2e-5):
            raise ValueError(f"independently replayed hidden features differ: {name}")
        if not np.allclose(actual_m, expected_m, rtol=2e-5, atol=2e-5):
            raise ValueError(f"independently replayed action margins differ: {name}")
        metadata_key = "train_prompt_token_counts" if name == "training" else "development_prompt_token_counts"
        if token_counts != metadata[metadata_key]:
            raise ValueError(f"independently replayed prompt token counts differ: {name}")
        max_abs_hidden_error = max(max_abs_hidden_error, float(np.max(np.abs(actual_h - expected_h))))
        max_abs_margin_error = max(max_abs_margin_error, float(np.max(np.abs(actual_m - expected_m))))
        actual_tokens[name] = len(token_counts)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    wall = time.monotonic() - started
    if peak_bytes > int(compute["max_peak_rss_bytes_campaign"]):
        raise MemoryError("independent feature replay exceeded the frozen RSS budget")
    return {
        "pass": True,
        "device": "cpu",
        "wall_seconds": wall,
        "peak_rss_bytes": peak_bytes,
        "rows_replayed": actual_tokens,
        "max_abs_hidden_error": max_abs_hidden_error,
        "max_abs_action_margin_error": max_abs_margin_error,
        "action_token_ids": action_ids,
    }


def verify_checkpoints(bundle: Path, spec: dict[str, Any], dev_x, dev_margin, dev_y, summary):
    checkpoints = read_json(bundle / "checkpoint_history.json")
    base_metrics = binary_metrics(dev_margin, dev_y, dev_margin)
    for key, expected in base_metrics.items():
        compare_metric(summary["base_metrics"].get(key), expected, key=f"base/{key}")
    max_kl = float(spec["learner"]["kl"]["maximum_selected_mean_nats_per_action"])
    selected = {}
    for arm, by_seed in checkpoints.items():
        selected[arm] = {}
        for seed, history in by_seed.items():
            if [int(item["update"]) for item in history] != spec["learner"]["optimizer"]["checkpoint_updates"]:
                raise ValueError(f"checkpoint grid differs: {arm}/{seed}")
            eligible = []
            for checkpoint in history:
                expected_actions = (
                    int(checkpoint["update"]) * int(spec["learner"]["optimizer"]["minibatch_size"])
                    * int(spec["learner"]["rloo"]["group_size"])
                ) if arm == "rloo_k4" else 0
                if int(checkpoint.get("optimizer_updates", -1)) != int(checkpoint["update"]):
                    raise ValueError(f"checkpoint optimizer-update counter differs: {arm}/{seed}/{checkpoint['update']}")
                if int(checkpoint.get("cumulative_rloo_sampled_actions", -1)) != expected_actions:
                    raise ValueError(f"checkpoint cumulative sampled-action counter differs: {arm}/{seed}/{checkpoint['update']}")
                weight = checkpoint["weight"]
                delta = 0.0 if weight is None else dev_x @ np.asarray(weight, dtype=np.float32)
                logits = dev_margin + delta + float(checkpoint["bias"])
                metrics = binary_metrics(logits, dev_y, dev_margin)
                for key, expected in metrics.items():
                    compare_metric(
                        checkpoint["metrics"].get(key), expected,
                        key=f"checkpoint/{arm}/{seed}/{checkpoint['update']}/{key}",
                    )
                is_eligible = metrics["mean_conditional_bernoulli_kl_nats_per_action"] <= max_kl
                if bool(checkpoint["eligible"]) != bool(is_eligible):
                    raise ValueError(f"checkpoint KL eligibility differs: {arm}/{seed}/{checkpoint['update']}")
                if is_eligible:
                    eligible.append((metrics["balanced_accuracy"], -int(checkpoint["update"]), checkpoint))
            if not eligible:
                raise ValueError(f"no eligible checkpoint in stored history: {arm}/{seed}")
            best = max(eligible, key=lambda item: (item[0], item[1]))[2]
            selected[arm][seed] = best
            if int(summary["per_seed"][arm][seed]["selected_updates"]) != int(best["update"]):
                raise ValueError(f"selected checkpoint does not follow locked dev rule: {arm}/{seed}")
            if summary["per_seed"][arm][seed]["selected_weight"] != best["weight"]:
                raise ValueError(f"selected adapter differs from history: {arm}/{seed}")
            close(summary["per_seed"][arm][seed]["selected_bias"], best["bias"])
            selected_weight = best["weight"]
            delta_norm_sq = float(best["bias"]) ** 2
            if selected_weight is not None:
                delta_norm_sq += float(np.square(np.asarray(selected_weight, dtype=np.float64)).sum())
            close(summary["per_seed"][arm][seed]["parameter_delta_l2_from_base"], math.sqrt(delta_norm_sq))
            expected_actions = (
                int(best["update"]) * int(spec["learner"]["optimizer"]["minibatch_size"])
                * int(spec["learner"]["rloo"]["group_size"])
            ) if arm == "rloo_k4" else 0
            if summary["per_seed"][arm][seed]["cumulative_rloo_sampled_actions_at_selected_update"] != expected_actions:
                raise ValueError(f"selected checkpoint action count differs: {arm}/{seed}")
            for key, value in best["metrics"].items():
                compare_metric(
                    summary["per_seed"][arm][seed]["selected_metrics"].get(key), value,
                    key=f"selected/{arm}/{seed}/{key}",
                )
    return base_metrics, selected


def verify_summary_aggregates(spec, summary, base_metrics, selected, rollouts):
    seeds = [str(value) for value in spec["learner"]["optimizer"]["seeds"]]
    arm_names = ("frozen_base", "scalar_calibration", "context_sft", "exact_expected_reward", "rloo_k4")
    means = {}
    for arm in arm_names:
        metrics = {seed: (base_metrics if arm == "frozen_base" else selected[arm][seed]["metrics"]) for seed in seeds}
        mean_row = {
            "mean_balanced_accuracy": statistics.fmean(row["balanced_accuracy"] for row in metrics.values()),
            "mean_exact_action_accuracy": statistics.fmean(row["exact_action_accuracy"] for row in metrics.values()),
            "mean_macro_f1": statistics.fmean(row["macro_f1"] for row in metrics.values()),
            "mean_action_nll": statistics.fmean(row["action_nll"] for row in metrics.values()),
            "mean_kl": statistics.fmean(row["mean_conditional_bernoulli_kl_nats_per_action"] for row in metrics.values()),
            "per_seed_balanced_accuracy": {seed: metrics[seed]["balanced_accuracy"] for seed in seeds},
            "per_seed_mean_kl": {seed: metrics[seed]["mean_conditional_bernoulli_kl_nats_per_action"] for seed in seeds},
            "per_seed_selected_updates": {seed: 0 if arm == "frozen_base" else int(selected[arm][seed]["update"]) for seed in seeds},
        }
        means[arm] = mean_row
        stored = summary["comparison"]["arms"][arm]
        for key, expected in mean_row.items():
            if isinstance(expected, dict):
                if set(stored[key]) != set(expected):
                    raise ValueError(f"per-seed aggregate keys differ: {arm}/{key}")
                for seed, value in expected.items():
                    close(stored[key][seed], value)
            else:
                close(stored[key], expected)
    rloo_diffs = {
        seed: selected["rloo_k4"][seed]["metrics"]["balanced_accuracy"] - base_metrics["balanced_accuracy"]
        for seed in seeds
    }
    if summary["comparison"]["rloo_seed_balanced_accuracy_differences_vs_base"].keys() != rloo_diffs.keys():
        raise ValueError("RLOO per-seed contrast keys differ")
    for seed, value in rloo_diffs.items():
        close(summary["comparison"]["rloo_seed_balanced_accuracy_differences_vs_base"][seed], value)
    if summary["comparison"]["rloo_seeds_not_worse_than_base"] != sum(value >= 0 for value in rloo_diffs.values()):
        raise ValueError("RLOO non-worse seed count differs")

    grouped = {seed: [row for row in rollouts.values() if str(row["seed"]) == seed] for seed in seeds}
    for seed, records in grouped.items():
        rewards = [float(value) for row in records for group in row["rewards"] for value in group]
        actions = [int(value) for row in records for group in row["actions"] for value in group]
        diagnostic = summary["rloo_rollout_diagnostics_by_seed"][seed]
        expected = {
            "updates": len(records),
            "prompt_groups": sum(len(row["actions"]) for row in records),
            "sampled_actions": len(actions),
            "sampled_yes_actions": sum(actions),
            "sampled_no_actions": len(actions) - sum(actions),
            "mean_verifier_reward": statistics.fmean(rewards),
            "verifier_reward_variance": statistics.pvariance(rewards),
            "mixed_reward_group_fraction": statistics.fmean(row["mixed_reward_group_rate"] for row in records),
            "zero_advantage_group_fraction": statistics.fmean(row["zero_advantage_group_rate"] for row in records),
        }
        for key, value in expected.items():
            if isinstance(value, int):
                if diagnostic[key] != value:
                    raise ValueError(f"RLOO diagnostic count differs: {seed}/{key}")
            else:
                close(diagnostic[key], value)
    return means


def verify_rollouts(bundle: Path, spec: dict[str, Any], train_y: np.ndarray, train_weights: dict[int, float], minibatches: dict[str, Any], summary: dict[str, Any]):
    rollouts = read_json(bundle / "rloo_rollouts.json")
    opt = spec["learner"]["optimizer"]
    k = int(spec["learner"]["rloo"]["group_size"])
    batch_size = int(opt["minibatch_size"])
    seeds = [str(seed) for seed in opt["seeds"]]
    expected_rollouts = len(seeds) * int(opt["maximum_updates"])
    if len(rollouts) != expected_rollouts:
        raise ValueError("rollout update count differs")
    if set(minibatches) != set(seeds):
        raise ValueError("minibatch seeds differ")
    for seed in seeds:
        schedule = minibatches[seed]
        if len(schedule) != int(opt["maximum_updates"]) or any(len(row) != batch_size for row in schedule):
            raise ValueError(f"minibatch schedule shape differs: {seed}")
        if any(int(index) < 0 or int(index) >= len(train_y) for batch in schedule for index in batch):
            raise ValueError(f"minibatch index out of range: {seed}")
        expected_rng = np.random.default_rng(int(seed) ^ 0x51A7C0DE)
        expected_schedule = expected_rng.integers(
            0, len(train_y), size=(int(opt["maximum_updates"]), batch_size), endpoint=False
        ).astype(int).tolist()
        if schedule != expected_schedule:
            raise ValueError(f"minibatch sequence does not reconstruct from locked seed: {seed}")
    records = {(str(row["seed"]), int(row["update"])): row for row in rollouts}
    if len(records) != expected_rollouts:
        raise ValueError("duplicate seed/update rollout record")
    for seed in seeds:
        for update in range(1, int(opt["maximum_updates"]) + 1):
            record = records.get((seed, update))
            if record is None:
                raise ValueError(f"missing rollout record: {seed}/{update}")
            indices = minibatches[seed][update - 1]
            if record["train_indices"] != indices:
                raise ValueError(f"rollout minibatch differs: {seed}/{update}")
            mixed_groups = 0
            zero_advantage_groups = 0
            for row_i, train_index in enumerate(indices):
                label = int(train_y[int(train_index)])
                class_weight = float(train_weights[label])
                actions = record["actions"][row_i]
                rewards = record["rewards"][row_i]
                advantages = record["leave_one_out_advantages"][row_i]
                probs = float(record["p_yes_before"][row_i])
                logps = record["sampled_action_log_probabilities"][row_i]
                if len(actions) != k or len(rewards) != k or len(advantages) != k or len(logps) != k:
                    raise ValueError(f"rollout group has wrong size: {seed}/{update}")
                if not 0 < probs < 1 or any(int(action) not in (0, 1) for action in actions):
                    raise ValueError(f"invalid action probability/value: {seed}/{update}")
                expected_rewards = [class_weight if int(action) == label else 0.0 for action in actions]
                if any(not math.isclose(float(a), float(b), abs_tol=1e-7) for a, b in zip(rewards, expected_rewards)):
                    raise ValueError(f"verifier reward differs from label/action: {seed}/{update}")
                expected_adv = [value - (sum(rewards) - value) / (k - 1) for value in rewards]
                if any(not math.isclose(float(a), float(b), abs_tol=1e-6) for a, b in zip(advantages, expected_adv)):
                    raise ValueError(f"leave-one-out advantage differs: {seed}/{update}")
                mixed_groups += int(min(rewards) != max(rewards))
                zero_advantage_groups += int(all(abs(float(value)) <= 1e-12 for value in advantages))
                for action, logp in zip(actions, logps):
                    expected_logp = math.log(probs if action == 1 else 1 - probs)
                    if not math.isclose(float(logp), expected_logp, rel_tol=2e-5, abs_tol=2e-5):
                        raise ValueError(f"sampled-action log-probability differs: {seed}/{update}")
            close(record["mixed_reward_group_rate"], mixed_groups / len(indices), tolerance=1e-7)
            close(record["zero_advantage_group_rate"], zero_advantage_groups / len(indices), tolerance=1e-7)
    return len(rollouts), records


def replay_optimizers(
    spec, train_x, train_margin, train_y, train_weights, minibatches, rollout_records,
    checkpoint_history, *, deadline: float,
):
    """Re-run all adapter updates from retained inputs and compare every frozen checkpoint."""
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(int(spec["compute_limits"]["threads"]))
    opt_spec = spec["learner"]["optimizer"]
    beta = float(spec["learner"]["kl"]["coefficient"])
    weights_for_rows = torch.as_tensor(
        [train_weights[int(label)] for label in train_y], dtype=torch.float32
    )
    hidden = torch.as_tensor(train_x, dtype=torch.float32)
    base = torch.as_tensor(train_margin, dtype=torch.float32)
    labels = torch.as_tensor(train_y, dtype=torch.float32)
    arms = ("scalar_calibration", "context_sft", "exact_expected_reward", "rloo_k4")
    for seed_value in opt_spec["seeds"]:
        seed = str(seed_value)
        for arm in arms:
            contextual = arm != "scalar_calibration"
            bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            weight = torch.nn.Parameter(torch.zeros(hidden.shape[1], dtype=torch.float32)) if contextual else None
            params = [bias] if weight is None else [weight, bias]
            optimizer = torch.optim.Adam(
                params,
                lr=float(opt_spec["learning_rate"]),
                weight_decay=float(opt_spec["weight_decay"]),
            )
            sample_rng = np.random.default_rng(int(seed) ^ 0xA17E6D4B)
            snapshots = {int(row["update"]): row for row in checkpoint_history[arm][seed]}
            for update in range(int(opt_spec["maximum_updates"]) + 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("independent optimizer replay exceeded the frozen wall budget")
                if update in snapshots:
                    saved = snapshots[update]
                    expected_weight = None if weight is None else weight.detach().cpu().numpy()
                    if expected_weight is None:
                        if saved["weight"] is not None:
                            raise ValueError(f"scalar checkpoint unexpectedly has a contextual adapter: {arm}/{seed}/{update}")
                    elif not np.allclose(np.asarray(saved["weight"], dtype=np.float32), expected_weight, rtol=1e-5, atol=1e-6):
                        raise ValueError(f"optimizer replay adapter mismatch: {arm}/{seed}/{update}")
                    close(saved["bias"], bias.detach().item(), tolerance=1e-5)
                if update == int(opt_spec["maximum_updates"]):
                    continue
                indices = torch.as_tensor(minibatches[seed][update], dtype=torch.long)
                prompt_labels = labels[indices]
                prompt_weights = weights_for_rows[indices]
                reference = base[indices]
                logits = reference + bias
                if weight is not None:
                    logits = logits + hidden[indices] @ weight
                log_p_yes = F.logsigmoid(logits)
                log_p_no = F.logsigmoid(-logits)
                p_yes = torch.sigmoid(logits)
                kl = p_yes * (log_p_yes - F.logsigmoid(reference)) + (1 - p_yes) * (log_p_no - F.logsigmoid(-reference))
                if arm in ("scalar_calibration", "context_sft"):
                    correct_logp = prompt_labels * log_p_yes + (1 - prompt_labels) * log_p_no
                    loss = (-prompt_weights * correct_logp + beta * kl).mean()
                elif arm == "exact_expected_reward":
                    p_correct = torch.where(prompt_labels == 1, p_yes, 1 - p_yes)
                    loss = (-prompt_weights * p_correct + beta * kl).mean()
                else:
                    record = rollout_records[(seed, update + 1)]
                    actions = torch.as_tensor(record["actions"], dtype=torch.float32)
                    sampled = sample_rng.binomial(
                        1,
                        p_yes.detach().cpu().numpy()[:, None],
                        size=(len(indices), int(spec["learner"]["rloo"]["group_size"])),
                    )
                    if not np.array_equal(sampled, actions.numpy().astype(np.int64)):
                        raise ValueError(f"RLOO actions do not replay from the frozen seed: {seed}/{update + 1}")
                    rewards = (actions == prompt_labels[:, None]).to(dtype=torch.float32) * prompt_weights[:, None]
                    group_size = actions.shape[1]
                    baseline = (rewards.sum(dim=1, keepdim=True) - rewards) / (group_size - 1)
                    advantages = (rewards - baseline).detach()
                    action_logp = actions * F.logsigmoid(logits[:, None]) + (1 - actions) * F.logsigmoid(-logits[:, None])
                    loss = -(advantages * action_logp).mean() + beta * kl.mean()
                    if not np.allclose(record["p_yes_before"], p_yes.detach().numpy(), rtol=2e-5, atol=2e-5):
                        raise ValueError(f"on-policy action probabilities differ from optimizer replay: {seed}/{update + 1}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(opt_spec["gradient_norm_clip"]))
                optimizer.step()
    return True


def audit(bundle: Path) -> dict[str, Any]:
    audit_started = time.monotonic()
    bundle = bundle.resolve()
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if (bundle / "failure.json").exists():
        raise ValueError("run bundle contains a failure record")
    manifest_count = verify_manifest(bundle)
    spec, protocol_hash = verify_protocol(bundle)
    audit_deadline = audit_started + float(spec["compute_limits"]["max_wall_seconds_campaign"])
    fingerprints = read_json(bundle / "input_fingerprints.json")
    summary = read_json(bundle / "summary.json")
    import datasets
    import safetensors
    import tokenizers
    import torch
    import transformers

    actual_runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": np.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
    }
    locked_runtime = {key: spec["runtime_lock"][key] for key in actual_runtime}
    if actual_runtime != locked_runtime or any(summary["runtime"].get(key) != value for key, value in locked_runtime.items()):
        raise ValueError("runtime versions differ from protocol or run summary")
    if summary["protocol_sha256"] != protocol_hash:
        raise ValueError("summary protocol digest differs")
    if int(summary["runtime"].get("torch_threads", -1)) != int(spec["compute_limits"]["threads"]):
        raise ValueError("runtime torch thread count differs from protocol")
    if summary["source_commit"] != fingerprints["source"]["commit"] or fingerprints["source"]["working_tree_clean"] is not True:
        raise ValueError("source commit/clean-tree records disagree")
    verify_source_snapshots(bundle, summary["source_commit"])
    cache = fingerprints["cached_dataset_and_model_sha256"]
    expected_cache = {
        "train_arrow": spec["data"]["dataset"]["train_arrow_sha256"],
        "validation_arrow": spec["data"]["dataset"]["validation_arrow_sha256"],
    }
    for name, digest in spec["learner"]["model"]["snapshot_file_sha256"].items():
        expected_cache[f"model_snapshot/{name}"] = digest
    if cache != expected_cache:
        raise ValueError("cached data/model hash record differs from lock")
    if fingerprints["dataset_revision"] != spec["data"]["dataset"]["revision"]:
        raise ValueError("dataset revision differs from lock")
    model_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / spec["learner"]["model"]["revision"]
    data_dir = Path.home() / ".cache/huggingface/datasets/google___boolq/default/0.0.0" / spec["data"]["dataset"]["revision"]
    actual_cache_paths = {
        "train_arrow": data_dir / "boolq-train.arrow",
        "validation_arrow": data_dir / "boolq-validation.arrow",
    }
    actual_cache_paths.update({f"model_snapshot/{name}": model_dir / name for name in spec["learner"]["model"]["snapshot_file_sha256"]})
    for name, path in actual_cache_paths.items():
        if not path.is_file() or sha256_file(path) != expected_cache[name]:
            raise ValueError(f"local cached source file does not match its frozen SHA-256: {name}")
    cohort_counts = verify_cohorts(bundle, spec, fingerprints)
    train_x, train_margin, train_y, dev_x, dev_margin, dev_y = verify_features(bundle, summary, spec)
    feature_replay = replay_model_features(bundle, spec, deadline=audit_deadline)
    cohort_weights = {cls: 0.5 / float((train_y == cls).mean()) for cls in (0, 1)}
    recorded_weights = read_json(bundle / "feature_metadata.json")["train_class_weights"]
    for cls, value in cohort_weights.items():
        close(recorded_weights[str(cls)], value)
    minibatches = read_json(bundle / "training_minibatches.json")
    rollout_count, rollouts = verify_rollouts(bundle, spec, train_y, cohort_weights, minibatches, summary)
    base_metrics, selected = verify_checkpoints(bundle, spec, dev_x, dev_margin, dev_y, summary)
    replay_optimizers(
        spec, train_x, train_margin, train_y, cohort_weights, minibatches, rollouts,
        read_json(bundle / "checkpoint_history.json"), deadline=audit_deadline,
    )
    verify_summary_aggregates(spec, summary, base_metrics, selected, rollouts)
    expected_rollout_n = len(spec["learner"]["optimizer"]["seeds"]) * int(spec["learner"]["optimizer"]["maximum_updates"])
    if rollout_count != expected_rollout_n:
        raise ValueError("RLOO rollout total differs from locked update budget")
    if summary["resource_measurements"]["rollout_action_count"] != (
        rollout_count * int(spec["learner"]["rloo"]["group_size"]) * int(spec["learner"]["optimizer"]["minibatch_size"])
    ):
        raise ValueError("sampled action count differs from raw rollouts")
    labels = dev_y.astype(int).tolist()
    seeds = [str(seed) for seed in spec["learner"]["optimizer"]["seeds"]]

    def per_prompt(arm):
        return [selected[arm][seed]["metrics"]["per_prompt_correct"] for seed in seeds]

    base_rows = [base_metrics["per_prompt_correct"] for _ in seeds]
    contrasts = {
        "primary_rloo_minus_base_balanced_accuracy": paired_bootstrap(
            per_prompt("rloo_k4"), base_rows, labels,
            int(spec["metrics"]["paired_bootstrap"]["resamples"]), int(spec["metrics"]["paired_bootstrap"]["seed"]),
        ),
        "rloo_minus_exact_expected_reward_balanced_accuracy": paired_bootstrap(
            per_prompt("rloo_k4"), per_prompt("exact_expected_reward"), labels,
            int(spec["metrics"]["paired_bootstrap"]["resamples"]), int(spec["metrics"]["paired_bootstrap"]["seed"]) + 1,
        ),
        "rloo_minus_context_sft_balanced_accuracy": paired_bootstrap(
            per_prompt("rloo_k4"), per_prompt("context_sft"), labels,
            int(spec["metrics"]["paired_bootstrap"]["resamples"]), int(spec["metrics"]["paired_bootstrap"]["seed"]) + 2,
        ),
    }
    stored = summary["comparison"]
    for name, values in contrasts.items():
        for key in values:
            close(stored[name][key], values[key])
    gate = spec["metrics"]["development_gate"]
    gradient_record = summary["objective_gradient_checks"]
    gradient_reconstruction = independently_check_rloo_gradient_identity()
    if (gradient_record.get("pass") is not True
            or int(gradient_record.get("enumerated_cases", -1)) != gradient_reconstruction["cases"]
            or int(gradient_record.get("group_size", -1)) != int(spec["learner"]["rloo"]["group_size"])):
        raise ValueError("runtime expected-gradient check record differs from independent enumeration")
    close(gradient_record["max_abs_gradient_error"], gradient_reconstruction["max_abs_error"], tolerance=1e-9)
    resources = summary["resource_measurements"]
    if int(resources.get("cpu_threads", -1)) != int(spec["compute_limits"]["threads"]):
        raise ValueError("recorded CPU thread ceiling differs from protocol")
    if int(resources.get("optimizer_updates_per_trained_arm_seed", -1)) != int(spec["learner"]["optimizer"]["maximum_updates"]):
        raise ValueError("recorded optimizer update budget differs from protocol")
    arm_walls = summary.get("arm_wall_seconds")
    expected_arms = {"scalar_calibration", "context_sft", "exact_expected_reward", "rloo_k4"}
    if not isinstance(arm_walls, dict) or set(arm_walls) != expected_arms:
        raise ValueError("per-arm wall-time record is missing or has unexpected arms")
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in arm_walls.values()):
        raise ValueError("per-arm wall-time record contains invalid values")
    resource_pass = (
        resources.get("device") == "cpu"
        and resources.get("network_offline") is True
        and float(resources["campaign_wall_seconds"]) <= float(spec["compute_limits"]["max_wall_seconds_campaign"])
        and int(resources["campaign_peak_rss_bytes"]) <= int(spec["compute_limits"]["max_peak_rss_bytes_campaign"])
    )
    rloo_ba = [float(selected["rloo_k4"][seed]["metrics"]["balanced_accuracy"]) for seed in seeds]
    base_ba = float(base_metrics["balanced_accuracy"])
    differences = [value - base_ba for value in rloo_ba]
    mean_kl = statistics.fmean(float(selected["rloo_k4"][s]["metrics"]["mean_conditional_bernoulli_kl_nats_per_action"]) for s in seeds)
    max_kl = max(float(selected["rloo_k4"][s]["metrics"]["mean_conditional_bernoulli_kl_nats_per_action"]) for s in seeds)
    checks = {
        "all_provenance_and_prompt_separation_checks_pass": True,
        "base_quality_floor_pass": base_ba >= gate["minimum_base_balanced_accuracy"] and min(base_metrics["yes_recall"], base_metrics["no_recall"]) >= gate["minimum_base_recall_per_class"],
        "rloo_mean_gain_threshold_pass": statistics.fmean(rloo_ba) - base_ba >= gate["rloo_minus_base_balanced_accuracy_at_least"],
        "rloo_paired_ci_pass": contrasts["primary_rloo_minus_base_balanced_accuracy"]["ci95_low"] > gate["rloo_minus_base_ci_lower_above"],
        "seed_consistency_pass": sum(diff >= 0 for diff in differences) >= gate["minimum_seeds_rloo_not_worse_than_base"],
        "mean_and_per_seed_kl_pass": mean_kl <= gate["maximum_selected_mean_kl_nats_per_action"] and max_kl <= gate["maximum_selected_per_seed_mean_kl_nats_per_action"],
        "expected_reward_gradient_checks_pass": True,
        "all_arms_within_resource_limits": resource_pass,
    }
    decision = summary["comparison"]["development_gate"]
    if checks != {key: value for key, value in decision.items() if key != "result"}:
        raise ValueError("development gate checks do not reconstruct")
    expected_result = "pass" if all(checks.values()) else "non-pass"
    if decision["result"] != expected_result:
        raise ValueError("development gate result does not reconstruct")
    limits = spec["compute_limits"]
    audit_wall = time.monotonic() - audit_started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    audit_peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    if audit_wall > float(limits["max_wall_seconds_campaign"]):
        raise TimeoutError("independent audit exceeded its frozen wall-time budget")
    if audit_peak_bytes > int(limits["max_peak_rss_bytes_campaign"]):
        raise MemoryError("independent audit exceeded its frozen peak-RSS budget")
    if resources["campaign_wall_seconds"] > limits["max_wall_seconds_campaign"]:
        raise ValueError("campaign exceeded frozen wall budget")
    if resources["campaign_peak_rss_bytes"] > limits["max_peak_rss_bytes_campaign"]:
        raise ValueError("campaign exceeded frozen memory budget")
    return {
        "audit": "pass",
        "protocol_id": spec["protocol_id"],
        "protocol_sha256": protocol_hash,
        "source_commit": summary["source_commit"],
        "manifest_files_verified": manifest_count,
        "cohort_rows_reconstructed": cohort_counts,
        "feature_normalization_reconstructed": True,
        "tokenizer_and_model_feature_replay": feature_replay,
        "rollout_updates_verified": rollout_count,
        "sampled_action_groups_verified": int(summary["resource_measurements"]["rollout_action_count"] // spec["learner"]["rloo"]["group_size"]),
        "checkpoint_metrics_selection_and_gate_reconstructed": True,
        "all_adapter_optimizer_updates_replayed": True,
        "audit_wall_seconds": audit_wall,
        "audit_peak_rss_bytes": audit_peak_bytes,
        "expected_reward_gradient_identity_independently_enumerated": gradient_reconstruction,
        "development_gate": decision,
        "audit_limits": [
            "Reconstructs prompts, labels, rank cohorts, tokenizer/model features, train-only normalization, adapter metrics, bootstrap intervals, optimizer updates, and RLOO action sampling.",
            "Uses the pinned model and the same local CPU framework stack; this is a same-host audit, not external reproduction.",
            "Reserved validation interval is used only for prompt-field ranking and prompt-hash disjointness; no reserved answer labels are read.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = audit(args.bundle)
    (args.bundle / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
