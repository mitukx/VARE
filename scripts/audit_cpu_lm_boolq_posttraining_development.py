#!/usr/bin/env python3
"""Independently audit a frozen train-only sequence-DPO development run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_cpu_lm_boolq_posttraining_development import (  # noqa: E402
    DATA_DIR, MODEL_DIR, canonical, generate_greedy, label_metrics, sha256_file, write_manifest,
)
from boolq_sequence_task import attach_base_rollout_rejections, make_sequence_rows  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def approx(left, right, tol=1e-10):
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def verify_manifest(output: Path):
    manifest = read_json(output / "manifest.json")
    expected = manifest["files"]
    actual_paths = {p.relative_to(output).as_posix() for p in output.rglob("*")
                    if p.is_file() and p.name not in ("manifest.json", "audit.json")}
    if set(expected) != actual_paths:
        raise ValueError("run manifest file inventory differs")
    for name, digest in expected.items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"run artifact hash mismatch: {name}")
    return sha256_file(output / "manifest.json")


def audit(output: Path, check_generation: bool = True):
    source_manifest_sha256 = verify_manifest(output)
    notice = (output / "DATASET-NOTICE.md").read_text(encoding="utf-8")
    if "BoolQ" not in notice or "CC BY-SA 3.0" not in notice or "Clark" not in notice:
        raise ValueError("BoolQ attribution/license notice is missing or incomplete")
    spec = read_json(output / "protocol.snapshot.json")
    lock = read_json(output / "protocol.lock.snapshot.json")
    lock_hash = lock.pop("sha256", None)
    protocol_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_hash != protocol_hash or canonical(lock) != canonical(spec):
        raise ValueError("run protocol does not match its lock")
    data = read_json(output / "development.json")
    result = data["summary"]
    if result["protocol_sha256"] != protocol_hash:
        raise ValueError("result protocol hash differs from the locked protocol")
    if result.get("training_method", "dpo") != spec["learner"].get("method", "dpo"):
        raise ValueError("result training method differs from the locked protocol")
    runner_snapshot = output / "run_cpu_lm_boolq_posttraining_development.snapshot.py"
    helper_snapshot = output / "boolq_sequence_task.snapshot.py"
    if sha256_file(runner_snapshot) != result["runner_sha256"]:
        raise ValueError("runner snapshot hash differs from result")
    if sha256_file(helper_snapshot) != result["task_helper_sha256"]:
        raise ValueError("task-helper snapshot hash differs from result")

    train = data["train_examples"]
    validation = data["validation_examples"]
    if len(train) != spec["dataset"]["development_training_examples"]:
        raise ValueError("unexpected training example count")
    if len(validation) != spec["dataset"]["development_validation_examples"]:
        raise ValueError("unexpected validation example count")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from datasets import load_dataset
    train_ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="train")
    validation_ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="validation")
    if len(train_ds) != spec["dataset"]["official_train_examples"] or \
            len(validation_ds) != spec["dataset"]["official_validation_examples"]:
        raise ValueError("cached BoolQ split sizes differ from the protocol")
    if train_ds._fingerprint != spec["dataset"]["cached_train_fingerprint"] or validation_ds._fingerprint != spec["dataset"]["cached_validation_fingerprint"]:
        raise ValueError("cached BoolQ split fingerprint differs from the lock")
    if sha256_file(DATA_DIR / "boolq-train.arrow") != spec["dataset"]["cached_train_arrow_sha256"] or \
            sha256_file(DATA_DIR / "boolq-validation.arrow") != spec["dataset"]["cached_validation_arrow_sha256"]:
        raise ValueError("cached BoolQ Arrow hash differs from the lock")
    train_range = spec["dataset"]["development_training_rank_range"]
    val_range = spec["dataset"]["development_validation_rank_range"]
    base_rollout_mode = spec["dataset"]["preference_construction"].startswith("base_rollout_")
    expected_train = make_sequence_rows(train_ds, "train", *train_range)
    expected_val = make_sequence_rows(validation_ds, "validation", *val_range)
    if base_rollout_mode:
        expected_train = attach_base_rollout_rejections(expected_train, result["base_training_generation"])
        expected_val = attach_base_rollout_rejections(expected_val, result["base_validation_generation"])
    for actual, expected in zip(train, expected_train):
        if actual != expected:
            raise ValueError("training rows differ from the locked hash ranks")
    for actual, expected in zip(validation, expected_val):
        if actual != expected:
            raise ValueError("validation rows differ from the locked hash ranks")
    train_hashes = {row["question_sha256"] for row in train}
    val_hashes = {row["question_sha256"] for row in validation}
    if len(train_hashes) != len(train) or len(val_hashes) != len(validation) or train_hashes & val_hashes:
        raise ValueError("development train/validation rows overlap or duplicate")

    tokenized = read_json(output / "tokenized_examples.json")
    for split, rows in (("train", train), ("validation", validation)):
        examples = tokenized[split]
        if len(examples) != len(rows):
            raise ValueError(f"tokenized {split} count differs")
        for row, example in zip(rows, examples):
            if row["dataset_index"] != example["dataset_index"]:
                raise ValueError(f"tokenized {split} dataset index differs")
            for kind in ("chosen", "rejected"):
                candidate = example[kind]
                ids, targets = candidate["input_ids"], candidate["response_ids"]
                if not targets or ids[-len(targets):] != targets:
                    raise ValueError(f"invalid {split} {kind} response token suffix")
        if len({row["question_sha256"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {split} question hash")

    seeds = spec["learner"]["seeds"]
    if [row["seed"] for row in result["per_seed"]] != seeds:
        raise ValueError("seed records differ from the protocol")
    computed = {}
    adapter_count = 0
    for epoch in spec["learner"]["checkpoints_epochs"]:
        records = []
        for seed_item in result["per_seed"]:
            checkpoint = seed_item["checkpoints"][str(epoch)]
            records.append(checkpoint["metrics"])
            ref = checkpoint["adapter_artifact"]
            artifact = output / ref["path"]
            if not artifact.is_file() or sha256_file(artifact) != ref["sha256"]:
                raise ValueError(f"adapter artifact hash mismatch: {ref['path']}")
            import numpy as np
            with np.load(artifact, allow_pickle=False) as arrays:
                if set(arrays.files) != {"A", "B"} or not all(np.isfinite(arrays[k]).all() for k in ("A", "B")):
                    raise ValueError(f"adapter artifact invalid: {ref['path']}")
            adapter_count += 1
            if len(checkpoint["metrics"]["relative_preference_margins"]) != len(validation):
                raise ValueError("validation preference-margin count differs")
        computed[str(epoch)] = {
            "mean_validation_dpo_preference_nll": statistics.fmean(m["mean_dpo_preference_nll"] for m in records),
            "mean_validation_preference_accuracy": statistics.fmean(m["preference_accuracy"] for m in records),
            "mean_validation_token_kl_to_base": statistics.fmean(m["mean_full_vocab_token_kl_to_base"] for m in records),
            "per_seed_nll": [m["mean_dpo_preference_nll"] for m in records],
        }
    for epoch, values in computed.items():
        for key, expected in values.items():
            actual = result["checkpoint_summary"][epoch][key]
            if isinstance(expected, list):
                if len(actual) != len(expected) or not all(approx(a, b) for a, b in zip(actual, expected)):
                    raise ValueError(f"checkpoint summary mismatch: {epoch}/{key}")
            elif not approx(actual, expected):
                raise ValueError(f"checkpoint summary mismatch: {epoch}/{key}")

    base_nll = result["base_validation_preference"]["mean_dpo_preference_nll"]
    eligible = [int(epoch) for epoch in computed if int(epoch) > 0 and
                computed[epoch]["mean_validation_dpo_preference_nll"] < base_nll and
                computed[epoch]["mean_validation_token_kl_to_base"] <= 0.5]
    selection_method = spec["metrics"].get("checkpoint_selection_method", "preference_nll")
    if selection_method in ("exact_match", "balanced_accuracy"):
        expected_generation_epochs = {str(epoch) for epoch in (eligible or [0])}
        checkpoint_records = result["checkpoint_generation_records"]
        if set(checkpoint_records) != expected_generation_epochs:
            raise ValueError("generated checkpoint set differs from NLL/KL-eligible epochs")
        recomputed_generation_summary = {}
        for epoch_text, records in checkpoint_records.items():
            if [item["seed"] for item in records] != seeds:
                raise ValueError("checkpoint generation seeds differ")
            accuracies, correct_counts, balanced_accuracies = [], [], []
            for item in records:
                generated = item["generated_validation"]
                if len(generated) != len(validation):
                    raise ValueError("checkpoint validation generation count differs")
                correct = sum(bool(row["exact_match"]) for row in generated)
                accuracy = statistics.fmean(float(row["exact_match"]) for row in generated)
                class_metrics = label_metrics(validation, generated)
                balanced_accuracy = class_metrics["balanced_accuracy"]
                if correct != item["correct"] or not approx(accuracy, item["accuracy"]):
                    raise ValueError("checkpoint generation metrics differ from retained outputs")
                if item.get("class_metrics") != class_metrics or not approx(item.get("balanced_accuracy"), balanced_accuracy):
                    raise ValueError("checkpoint class-stratified metrics differ from retained outputs")
                if [row["dataset_index"] for row in generated] != [row["dataset_index"] for row in validation]:
                    raise ValueError("checkpoint generation dataset order differs")
                accuracies.append(accuracy)
                correct_counts.append(correct)
                balanced_accuracies.append(balanced_accuracy)
            recomputed_generation_summary[epoch_text] = {
                "mean_exact_match_accuracy": statistics.fmean(accuracies),
                "per_seed_accuracy": accuracies,
                "per_seed_correct": correct_counts,
                "mean_balanced_accuracy": statistics.fmean(balanced_accuracies),
                "per_seed_balanced_accuracy": balanced_accuracies,
            }
        for epoch_text, values in recomputed_generation_summary.items():
            for key, expected in values.items():
                actual = result["checkpoint_generation_summary"][epoch_text][key]
                if isinstance(expected, list):
                    if len(actual) != len(expected) or not all(approx(a, b) for a, b in zip(actual, expected)):
                        raise ValueError("checkpoint generation summary mismatch")
                elif not approx(actual, expected):
                    raise ValueError("checkpoint generation summary mismatch")
        primary_key = ("mean_balanced_accuracy" if selection_method == "balanced_accuracy"
                       else "mean_exact_match_accuracy")
        selected = (min(eligible, key=lambda epoch: (
            -recomputed_generation_summary[str(epoch)][primary_key],
            -recomputed_generation_summary[str(epoch)]["mean_exact_match_accuracy"],
            computed[str(epoch)]["mean_validation_dpo_preference_nll"], epoch)) if eligible else 0)
    elif selection_method == "preference_nll":
        selected = min(eligible, key=lambda epoch: (computed[str(epoch)]["mean_validation_dpo_preference_nll"], epoch)) if eligible else 0
    else:
        raise ValueError("unknown checkpoint selection method")
    if selected != result["selected_epochs"]:
        raise ValueError("selected checkpoint differs from frozen selection rule")

    base_rows = result["base_validation_generation"]
    updated_rows = result["selected_adapters_and_generation"]
    if len(base_rows) != len(validation) or len(updated_rows) != len(seeds):
        raise ValueError("generation record counts differ")
    base_accuracy = statistics.fmean(float(r["exact_match"]) for r in base_rows)
    expected_base_classes = label_metrics(validation, base_rows)
    if result.get("base_validation_class_metrics") != expected_base_classes:
        raise ValueError("base class-stratified metrics differ from retained generations")
    updated_accuracies = [statistics.fmean(float(r["exact_match"]) for r in item["generated_validation"])
                          for item in updated_rows]
    if any(len(item["generated_validation"]) != len(validation) for item in updated_rows):
        raise ValueError("updated generation count differs")
    mean_updated = statistics.fmean(updated_accuracies)
    expected_selected_classes = [label_metrics(validation, item["generated_validation"]) for item in updated_rows]
    if result.get("selected_validation_class_metrics_by_seed") != expected_selected_classes:
        raise ValueError("selected class-stratified metrics differ from retained generations")
    at_least_two = sum(value >= base_accuracy for value in updated_accuracies) >= 2
    base_balanced = expected_base_classes["balanced_accuracy"]
    updated_balanced = statistics.fmean(item["balanced_accuracy"] for item in expected_selected_classes)
    at_least_two_balanced = sum(item["balanced_accuracy"] >= base_balanced
                                for item in expected_selected_classes) >= 2
    minimum_base_correct = spec["metrics"].get("minimum_base_exact_matches", 0)
    minimum_base_balanced = spec["metrics"].get("minimum_base_balanced_accuracy", 0.0)
    minimum_gain = spec["metrics"].get("minimum_balanced_accuracy_gain", 0.0)
    promoted = (selected > 0 and sum(bool(row["exact_match"]) for row in base_rows) >= minimum_base_correct and
                base_balanced >= minimum_base_balanced and updated_balanced >= base_balanced + minimum_gain and
                at_least_two_balanced)
    if not approx(result.get("selected_mean_balanced_accuracy"), updated_balanced) or \
            bool(result.get("at_least_two_seeds_balanced_no_worse")) != at_least_two_balanced:
        raise ValueError("selected balanced-accuracy gate metrics differ from retained outputs")
    expected_decision = "sequence_dpo_development_candidate" if promoted else "sequence_dpo_development_non_pass"
    if result["decision"] != expected_decision:
        raise ValueError("development decision differs from frozen selection rule")
    if base_rollout_mode:
        counts = {source: sum(row["rejected_source"] == source for row in train + validation)
                  for source in sorted({row["rejected_source"] for row in train + validation})}
        if result["rejection_source_counts"] != counts:
            raise ValueError("rejection source counts differ from retained pair provenance")

    parity_rows = []
    if check_generation:
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                                      torch_dtype=torch.float32).to("cpu").eval()
        parity_rows = generate_greedy(validation[:3], tokenizer, model, spec)
        for row, manual in zip(validation[:3], parity_rows):
            batch = tokenizer.apply_chat_template([[{"role": "user", "content": row["user_message"]}],
                                                   ], tokenize=True, add_generation_prompt=True, padding=True,
                                                  return_tensors="pt", return_dict=True)
            if not bool(batch["attention_mask"].all()):
                raise ValueError("single-prompt generation unexpectedly padded")
            with torch.inference_mode():
                generated = model.generate(**batch, max_new_tokens=spec["compute_limits"]["generation_max_new_tokens"],
                                           do_sample=False, pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=spec["generation"]["eos_token_ids"])
            text = tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if text != manual["generated_text"]:
                raise ValueError("manual greedy output differs from Hugging Face generation")
            manual["huggingface_parity"] = True

    audit_result = {
        "audit": "passed",
        "protocol_id": spec["protocol_id"],
        "protocol_sha256": protocol_hash,
        "source_manifest_sha256": source_manifest_sha256,
        "runner_snapshot_sha256": result["runner_sha256"],
        "train_examples": len(train),
        "validation_examples": len(validation),
        "nonzero_adapter_checkpoints_verified": adapter_count,
        "selected_epoch_recomputed": selected,
        "decision_recomputed": expected_decision,
        "base_exact_match": base_accuracy,
        "base_balanced_accuracy": base_balanced,
        "updated_exact_match_by_seed": updated_accuracies,
        "updated_balanced_accuracy_by_seed": [item["balanced_accuracy"] for item in expected_selected_classes],
        "generation_parity_examples": parity_rows,
    }
    (output / "audit.json").write_text(json.dumps(audit_result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    write_manifest(output)
    return audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--skip-generation-parity", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.run_directory.resolve(), not args.skip_generation_parity),
                     indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
