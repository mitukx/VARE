"""TRAIN-only, one-GPU base feasibility gate; never performs an optimizer step."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.rewards import accuracy_reward

from scripts.math500_grader_v1 import exact_match, has_valid_final_box
from scripts.validate_math500_study_v1 import (
    EXPECTED_DEEPMATH_SHA256,
    deduplicate_train_rows,
    select_splits,
    sha256_file,
)


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_FILES_SHA256 = {
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
GATE_SEED = 20261012
NUM_COMPLETIONS = 4
MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKENS = 2048
MAX_RESERVED_VRAM_BYTES = 14 * 1024**3
PINNED_RUNTIME = {
    "torch": "2.11.0",
    "transformers": "5.5.4",
    "trl": "1.1.0",
    "accelerate": "1.13.0",
    "datasets": "4.8.4",
    "math-verify": "0.9.0",
}
TEMPERATURE = 0.7
TOP_P = 0.95
SYSTEM_PROMPT = "Solve the math problem. Show concise reasoning and put the final answer in \\boxed{...}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/math-grpo-cpu-first-2026-10-11/data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/math-grpo-cpu-first-2026-10-11/gpu-base-gate.json"),
    )
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    args = parser.parse_args()

    if platform.python_version() != "3.11.2":
        raise SystemExit(f"NO-GO: expected Python 3.11.2 from the pinned one-GPU reference run, found {platform.python_version()}")
    mismatched = {
        package: (importlib.metadata.version(package), expected)
        for package, expected in PINNED_RUNTIME.items()
        if Version(importlib.metadata.version(package)).base_version != expected
    }
    if mismatched:
        raise SystemExit(f"NO-GO: runtime does not match frozen versions: {mismatched}")
    if not torch.cuda.is_available():
        raise SystemExit("NO-GO: this gate requires one CUDA GPU; CPU/MPS fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"NO-GO: expected exactly one visible CUDA device, found {torch.cuda.device_count()}")
    for filename, expected_hash in MODEL_FILES_SHA256.items():
        model_file = args.model_dir / filename
        if not model_file.is_file():
            raise SystemExit(f"pinned model file is missing: {filename}")
        observed_hash = sha256_file(model_file)
        if observed_hash != expected_hash:
            raise SystemExit(f"pinned model file hash mismatch for {filename}: {observed_hash}")
    model_vram = torch.cuda.get_device_properties(0).total_memory
    if model_vram < 16 * 1024**3:
        raise SystemExit(f"NO-GO: requires a >=16 GiB CUDA device; found {model_vram} bytes")
    weight_hash = MODEL_FILES_SHA256["model.safetensors"]
    data_path = args.data_dir / "deepmath-train.parquet"
    if sha256_file(data_path) != EXPECTED_DEEPMATH_SHA256:
        raise SystemExit("pinned DeepMath TRAIN parquet hash mismatch")

    raw_rows = parquet.read_table(data_path, columns=["prompt", "solution"]).to_pylist()
    eligible_rows, dedupe_stats = deduplicate_train_rows(raw_rows)
    split = select_splits([], eligible_rows)
    wanted = set(split["base_gate_ids"])
    gate_rows = [
        row for row in eligible_rows
        if hashlib.sha256(row["prompt_key"].encode()).hexdigest() in wanted
    ]
    if len(gate_rows) != 32:
        raise SystemExit(f"expected 32 fixed TRAIN gate prompts; found {len(gate_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()

    torch.manual_seed(GATE_SEED)
    torch.cuda.manual_seed_all(GATE_SEED)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    outputs = []
    token_count = 0
    prompt_token_count = 0
    max_input_seen = 0
    group_reward_variance = []
    mixed_reward_groups = 0
    total_correct = 0
    total_parseable = 0

    for row_index, row in enumerate(gate_rows):
        if time.monotonic() - started > args.max_wall_seconds:
            raise SystemExit("NO-GO: TRAIN-only gate exceeded its wall-time cap")
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
        input_length = int(encoded["input_ids"].shape[1])
        if input_length > MAX_INPUT_TOKENS:
            raise SystemExit(f"NO-GO: gate prompt {row_index} exceeds {MAX_INPUT_TOKENS} input tokens")
        prompt_token_count += input_length
        max_input_seen = max(max_input_seen, input_length)
        inputs = {key: value.to("cuda:0") for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_return_sequences=NUM_COMPLETIONS,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = [
            tokenizer.decode(sequence[input_length:], skip_special_tokens=True)
            for sequence in generated
        ]
        rewards = accuracy_reward(
            completions=[[{"role": "assistant", "content": text}] for text in completions],
            solution=[row["solution"]] * len(completions),
        )
        if any(reward is None for reward in rewards):
            raise SystemExit("NO-GO: actual TRL reward function skipped a selected training example")
        group_reward_variance.append(float(torch.tensor(rewards, dtype=torch.float64).var(unbiased=False).item()))
        mixed_reward_groups += int(min(rewards) != max(rewards))
        for completion, reward in zip(completions, rewards):
            token_count += len(tokenizer.encode(completion, add_special_tokens=False))
            correct = exact_match(completion, row["solution"])
            parseable = has_valid_final_box(completion)
            total_correct += int(correct)
            total_parseable += int(parseable)
            outputs.append(
                {
                    "train_row_hash": hashlib.sha256(row["prompt_key"].encode()).hexdigest(),
                    "completion": completion,
                    "training_reward": reward,
                    "independent_task_success": correct,
                    "valid_final_box": parseable,
                }
            )
        del generated, inputs
        torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    success_rate = total_correct / len(outputs)
    parse_rate = total_parseable / len(outputs)
    mean_group_var = sum(group_reward_variance) / len(group_reward_variance)
    passed = (
        0.05 <= success_rate <= 0.90
        and parse_rate >= 0.90
        and mixed_reward_groups >= 6
        and mean_group_var > 0.0
        and peak_reserved <= MAX_RESERVED_VRAM_BYTES
        and peak_reserved <= 0.90 * total_vram
        and elapsed <= args.max_wall_seconds
    )
    result = {
        "phase": "train_only_base_feasibility",
        "decision": "pass" if passed else "stop",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": weight_hash},
        "data": {
            "repo": "trl-lib/DeepMath-103K",
            "revision": "066c50a88d4e14cefc056e31111db2dba17f6c68",
            "split": "train",
            "file_sha256": EXPECTED_DEEPMATH_SHA256,
            "deduplication": dedupe_stats,
            "gate_prompts": len(gate_rows),
            "prompt_split": "hash-ranked first 32 eligible training prompts; permanently excluded from policy updates",
        },
        "generation": {
            "seed": GATE_SEED,
            "completions_per_prompt": NUM_COMPLETIONS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "system_prompt": SYSTEM_PROMPT,
            "prompt_tokens": prompt_token_count,
            "max_input_tokens_observed": max_input_seen,
            "generated_tokens": token_count,
            "format_rate": parse_rate,
            "independent_task_success_rate": success_rate,
            "training_reward_group_mixed_count": mixed_reward_groups,
            "training_reward_group_variance_mean": mean_group_var,
            "training_reward_group_variances": group_reward_variance,
        },
        "resources": {
            "device": torch.cuda.get_device_name(0),
            "total_vram_bytes": total_vram,
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "elapsed_seconds": elapsed,
            "generated_tokens_per_second": token_count / elapsed if elapsed else None,
            "dtype": str(dtype),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "python": platform.python_version(),
        },
        "success_criteria": {
            "task_success_rate_inclusive_5_to_90_percent": 0.05 <= success_rate <= 0.90,
            "valid_final_box_at_least_90_percent": parse_rate >= 0.90,
            "at_least_6_mixed_reward_groups": mixed_reward_groups >= 6,
            "positive_mean_within_group_reward_variance": mean_group_var > 0.0,
            "peak_reserved_vram_at_most_14_gib": peak_reserved <= MAX_RESERVED_VRAM_BYTES,
            "peak_reserved_vram_below_90_percent": peak_reserved <= 0.90 * total_vram,
            "under_wall_time_cap": elapsed <= args.max_wall_seconds,
        },
        "optimizer_updates": 0,
        "math500_loaded": False,
        "completions": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("decision", "generation", "resources", "success_criteria", "optimizer_updates", "math500_loaded")}, indent=2))
    if not passed:
        raise SystemExit("NO-GO: one or more frozen base-gate criteria failed")


if __name__ == "__main__":
    main()
