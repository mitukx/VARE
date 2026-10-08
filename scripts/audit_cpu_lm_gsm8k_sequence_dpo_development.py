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
from run_cpu_lm_gsm8k_sequence_dpo_development import (  # noqa: E402
    DATA_DIR, MODEL_DIR, canonical, generate_greedy, sha256_file, write_manifest,
)
from gsm8k_sequence_task import make_sequence_rows  # noqa: E402


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
    runner_snapshot = output / "run_cpu_lm_gsm8k_sequence_dpo_development.snapshot.py"
    helper_snapshot = output / "gsm8k_sequence_task.snapshot.py"
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
    dataset = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="train")
    if dataset._fingerprint != spec["dataset"]["cached_fingerprint"] or \
            sha256_file(DATA_DIR / "gsm8k-train.arrow") != spec["dataset"]["cached_train_arrow_sha256"]:
        raise ValueError("cached GSM8K train data differs from the lock")
    train_range = spec["dataset"].get("development_training_rank_range", [608, 672])
    val_range = spec["dataset"].get("development_validation_rank_range", [672, 736])
    expected_train = make_sequence_rows(dataset, "development_train", *train_range)
    expected_val = make_sequence_rows(dataset, "development_validation", *val_range)
    for actual, expected in zip(train, expected_train):
        if actual["dataset_index"] != expected["dataset_index"] or actual["question_sha256"] != expected["question_sha256"]:
            raise ValueError("training rows differ from the locked hash ranks")
    for actual, expected in zip(validation, expected_val):
        if actual["dataset_index"] != expected["dataset_index"] or actual["question_sha256"] != expected["question_sha256"]:
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
    selected = min(eligible, key=lambda epoch: (computed[str(epoch)]["mean_validation_dpo_preference_nll"], epoch)) if eligible else 0
    if selected != result["selected_epochs"]:
        raise ValueError("selected checkpoint differs from frozen selection rule")

    base_rows = result["base_validation_generation"]
    updated_rows = result["selected_adapters_and_generation"]
    if len(base_rows) != len(validation) or len(updated_rows) != len(seeds):
        raise ValueError("generation record counts differ")
    base_accuracy = statistics.fmean(float(r["exact_match"]) for r in base_rows)
    updated_accuracies = [statistics.fmean(float(r["exact_match"]) for r in item["generated_validation"])
                          for item in updated_rows]
    if any(len(item["generated_validation"]) != len(validation) for item in updated_rows):
        raise ValueError("updated generation count differs")
    mean_updated = statistics.fmean(updated_accuracies)
    at_least_two = sum(value >= base_accuracy for value in updated_accuracies) >= 2
    promoted = selected > 0 and mean_updated >= base_accuracy and at_least_two
    expected_decision = "sequence_dpo_development_candidate" if promoted else "sequence_dpo_development_non_pass"
    if result["decision"] != expected_decision:
        raise ValueError("development decision differs from frozen selection rule")

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
        "updated_exact_match_by_seed": updated_accuracies,
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
