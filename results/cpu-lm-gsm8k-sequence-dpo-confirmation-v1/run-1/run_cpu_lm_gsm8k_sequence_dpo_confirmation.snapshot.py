#!/usr/bin/env python3
"""Run the locked fixed-budget sequence-DPO confirmation on fresh GSM8K train rows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gsm8k_sequence_task import make_sequence_rows  # noqa: E402
from run_cpu_lm_gsm8k_sequence_dpo_development import (  # noqa: E402
    DATA_DIR, MODEL_DIR, encode_candidates, evaluate_pairs, generate_greedy,
    load_locked_spec, sha256_file, sigmoid_nll, train_to_checkpoints, write_json, write_manifest,
)

SPEC_PATH = ROOT / "protocols/cpu_lm_gsm8k_sequence_dpo_confirmation_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_lm_gsm8k_sequence_dpo_confirmation_v1.lock.json"


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def nll_deltas(margins_by_seed, beta):
    baseline = sigmoid_nll(0.0)
    return [statistics.fmean(sigmoid_nll(beta * margin) - baseline for margin in margins)
            for margins in margins_by_seed]


def paired_question_bootstrap(margins_by_seed, beta, replicates, seed):
    baseline = sigmoid_nll(0.0)
    count = len(margins_by_seed[0])
    if not count or any(len(values) != count for values in margins_by_seed):
        raise ValueError("per-seed held-out margins differ in length")
    per_question = [statistics.fmean(sigmoid_nll(beta * margins_by_seed[s][i]) - baseline
                                     for s in range(len(margins_by_seed))) for i in range(count)]
    rng = random.Random(seed)
    draws = sorted(statistics.fmean(per_question[rng.randrange(count)] for _ in range(count))
                   for _ in range(replicates))
    return {"mean_nll_change_updated_minus_base": statistics.fmean(per_question),
            "confidence_level": 0.95, "method": "paired_question_percentile_bootstrap",
            "replicates": replicates, "seed": seed,
            "ci_lower": draws[int(0.025 * replicates)],
            "ci_upper": draws[int(0.975 * replicates) - 1],
            "per_question_mean_deltas": per_question}


def run(output: Path, spec_path: Path = SPEC_PATH, lock_path: Path = LOCK_PATH):
    started = time.monotonic()
    spec, protocol_hash = load_locked_spec(spec_path, lock_path)
    if spec["phase"] != "independent_confirmation":
        raise ValueError("confirmation runner requires an independent_confirmation protocol")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(lock_path.read_bytes())
    sources = (Path(__file__), Path(__file__).with_name("gsm8k_sequence_task.py"),
               Path(__file__).with_name("run_cpu_lm_gsm8k_sequence_dpo_development.py"))
    for source in sources:
        (output / (source.stem + ".snapshot.py")).write_bytes(source.read_bytes())
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        os.environ[key] = "false" if key == "TOKENIZERS_PARALLELISM" else "1"
    try:
        import datasets
        import numpy as np
        import torch
        import transformers
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        runtime = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
                   "transformers": transformers.__version__, "numpy": np.__version__, "datasets": datasets.__version__}
        if runtime != spec["runtime"]:
            raise RuntimeError(f"runtime differs from frozen versions: {runtime}")
        torch.set_num_threads(spec["compute_limits"]["threads"])
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must not be initialized")
        if MODEL_DIR.name != spec["model"]["revision"]:
            raise ValueError("pinned model snapshot is not present")
        model_files = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
        model_hashes = {name: sha256_file(MODEL_DIR / name) for name in model_files}
        if model_hashes["generation_config.json"] != spec["model"]["generation_config_sha256"]:
            raise ValueError("pinned generation config hash differs")
        train_file_hash = sha256_file(DATA_DIR / "gsm8k-train.arrow")
        if train_file_hash != spec["dataset"]["cached_train_arrow_sha256"]:
            raise ValueError("cached training data hash differs from lock")
        ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="train")
        if ds._fingerprint != spec["dataset"]["cached_fingerprint"]:
            raise ValueError("cached train fingerprint differs from lock")

        # Reserved rows are first selected only after the committed lock passes verification.
        train_rows = make_sequence_rows(ds, "confirmation_update", 1504, 1632)
        heldout_rows = make_sequence_rows(ds, "confirmation_heldout", 1632, 2144)
        train_hashes = {row["question_sha256"] for row in train_rows}
        heldout_hashes = {row["question_sha256"] for row in heldout_rows}
        if len(train_hashes) != 128 or len(heldout_hashes) != 512 or train_hashes & heldout_hashes:
            raise ValueError("confirmation rows overlap or counts differ")

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                                      torch_dtype=torch.float32).to("cpu").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            if parameter.device.type != "cpu":
                raise RuntimeError("non-CPU model parameter detected")
        train_pairs = encode_candidates(train_rows, tokenizer, model, spec)
        heldout_pairs = encode_candidates(heldout_rows, tokenizer, model, spec)
        heldout_base = evaluate_pairs(heldout_pairs, model, None, None, spec["learner"]["beta"],
                                      spec["compute_limits"]["pair_microbatch_size"])
        train_reference = evaluate_pairs(train_pairs, model, None, None, spec["learner"]["beta"],
                                         spec["compute_limits"]["pair_microbatch_size"])
        base_generations = generate_greedy(heldout_rows, tokenizer, model, spec)
        base_accuracy = statistics.fmean(float(row["exact_match"]) for row in base_generations)

        seed_results = []
        fixed_epoch = spec["learner"]["fixed_update_epochs"]
        for seed in spec["learner"]["seeds"]:
            checkpoints, _ = train_to_checkpoints(train_pairs, heldout_pairs, model, spec, seed)
            checkpoint = checkpoints[str(fixed_epoch)]
            adapter_a = checkpoint["adapter"]["A"].detach()
            adapter_b = checkpoint["adapter"]["B"].detach()
            generated = generate_greedy(heldout_rows, tokenizer, model, spec, adapter_a, adapter_b)
            accuracy = statistics.fmean(float(row["exact_match"]) for row in generated)
            adapter_path = output / "adapters" / f"seed-{seed}-epoch-{fixed_epoch}.npz"
            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(adapter_path, A=adapter_a.numpy(), B=adapter_b.numpy())
            seed_results.append({"seed": seed, "update_epochs": fixed_epoch, "preference": checkpoint["metrics"],
                                 "generation_exact_match_accuracy": accuracy,
                                 "generation_exact_matches": sum(row["exact_match"] for row in generated),
                                 "generated_answers": generated,
                                 "adapter_artifact": {"path": adapter_path.relative_to(output).as_posix(),
                                                      "sha256": sha256_file(adapter_path)}})
            if rss_bytes() > spec["compute_limits"]["max_peak_rss_bytes"]:
                raise MemoryError("confirmation exceeded its peak RSS ceiling")
            if time.monotonic() - started > spec["compute_limits"]["max_wall_seconds"]:
                raise TimeoutError("confirmation exceeded its wall-time ceiling")

        margins = [row["preference"]["relative_preference_margins"] for row in seed_results]
        seed_deltas = nll_deltas(margins, spec["learner"]["beta"])
        boot = spec["metrics"]["bootstrap"]
        bootstrap = paired_question_bootstrap(margins, spec["learner"]["beta"], boot["replicates"], boot["seed"])
        mean_kl = statistics.fmean(row["preference"]["mean_full_vocab_token_kl_to_base"] for row in seed_results)
        mean_accuracy = statistics.fmean(row["generation_exact_match_accuracy"] for row in seed_results)
        two_seeds_improve = sum(value < 0 for value in seed_deltas) >= 2
        two_seeds_exact_not_worse = sum(row["generation_exact_match_accuracy"] >= base_accuracy
                                        for row in seed_results) >= 2
        passed = (bootstrap["ci_upper"] < 0 and two_seeds_improve and mean_kl <= 0.5 and
                  mean_accuracy >= base_accuracy and two_seeds_exact_not_worse)
        result = {
            "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_hash,
            "runner_sha256": sha256_file(Path(__file__)),
            "development_runner_sha256": sha256_file(Path(__file__).with_name("run_cpu_lm_gsm8k_sequence_dpo_development.py")),
            "task_helper_sha256": sha256_file(Path(__file__).with_name("gsm8k_sequence_task.py")),
            "model_id": spec["model"]["id"], "model_revision": MODEL_DIR.name, "model_file_sha256": model_hashes,
            "dataset_train_sha256": train_file_hash, "dataset_fingerprint": ds._fingerprint,
            "runtime": runtime, "platform": platform.platform(), "device": "cpu", "paid_compute": False,
            "network_disabled": True, "fixed_update_epochs": fixed_epoch,
            "base_train_preference": train_reference, "base_heldout_preference": heldout_base,
            "base_heldout_generation": base_generations, "base_exact_match_accuracy": base_accuracy,
            "seed_results": seed_results, "per_seed_nll_change": seed_deltas,
            "paired_question_bootstrap": bootstrap, "mean_updated_token_kl": mean_kl,
            "mean_updated_exact_match_accuracy": mean_accuracy,
            "at_least_two_seeds_improve_nll": two_seeds_improve,
            "at_least_two_seeds_exact_match_not_worse": two_seeds_exact_not_worse,
            "decision": "sequence_dpo_confirmation_pass" if passed else "sequence_dpo_confirmation_non_pass",
            "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
        }
        tokenized = {}
        for name, rows, pairs in (("update", train_rows, train_pairs), ("heldout", heldout_rows, heldout_pairs)):
            tokenized[name] = [{"dataset_index": row["dataset_index"],
                                "chosen": {"input_ids": pair[0]["input_ids"], "response_ids": pair[0]["response_ids"], "completion": pair[0]["completion"]},
                                "rejected": {"input_ids": pair[1]["input_ids"], "response_ids": pair[1]["response_ids"], "completion": pair[1]["completion"]}}
                               for row, pair in zip(rows, pairs)]
        write_json(output / "tokenized_examples.json", tokenized)
        write_json(output / "confirmation.json", {"summary": result, "update_examples": train_rows,
                                                    "heldout_examples": heldout_rows})
        write_manifest(output)
        return result
    except BaseException as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__, "message": str(exc),
                                              "elapsed_seconds": time.monotonic() - started,
                                              "peak_rss_bytes": rss_bytes()})
        raise
    finally:
        write_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cpu-lm-gsm8k-sequence-dpo-confirmation-v1/run-1")
    parser.add_argument("--protocol", type=Path, default=SPEC_PATH)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args()
    result = run(args.output.resolve(), args.protocol.resolve(), args.lock.resolve())
    keys = ("decision", "paired_question_bootstrap", "per_seed_nll_change", "mean_updated_token_kl",
            "base_exact_match_accuracy", "mean_updated_exact_match_accuracy", "elapsed_seconds", "peak_rss_bytes")
    print(json.dumps({key: result[key] for key in keys}, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
