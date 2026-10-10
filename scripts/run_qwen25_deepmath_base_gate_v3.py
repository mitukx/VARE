"""TRAIN-only base inference gate, with durable groups and identity-checked resume."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl.rewards import accuracy_reward

from scripts.gpu_gate_journal_v3 import GateJournal, GateWallTimeExceeded, run_prompt_groups, wall_time_limit
from scripts.math500_grader_v2 import exact_match, has_valid_final_box
from scripts.qwen25_deepmath_inputs_v2 import (
    GRPO_GENERATION_DEFAULTS,
    base_gate_generation_config,
    tokenize_generation_prompt,
)
from scripts.prompt_id_hash_v3 import verify_prompt_id_sha256
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
    "tokenizer_config.json": "5b5d4f65d0acd7b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
GATE_SEED = 20261012
NUM_COMPLETIONS = 4
MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKENS = 512
MAX_RESERVED_VRAM_BYTES = 14 * 1024**3
PINNED_RUNTIME = {
    "torch": "2.11.0",
    "transformers": "5.5.4",
    "trl": "1.1.0",
    "accelerate": "1.13.0",
    "datasets": "4.8.4",
    "math-verify": "0.9.0",
    "pyarrow": "22.0.0",
}
PROTOCOL_LOCK = Path("protocols/qwen25_deepmath_grpo_math500_v3.lock.json")
TEMPERATURE = GRPO_GENERATION_DEFAULTS["temperature"]
TOP_P = GRPO_GENERATION_DEFAULTS["top_p"]


def _check_runtime():
    python = Version(platform.python_version())
    if not (Version("3.11") <= python < Version("3.13")):
        raise SystemExit(f"NO-GO: Python 3.11 or 3.12 is required; found {python}")
    mismatched = {
        package: (importlib.metadata.version(package), expected)
        for package, expected in PINNED_RUNTIME.items()
        if Version(importlib.metadata.version(package)).base_version != expected
    }
    if mismatched:
        raise SystemExit(f"NO-GO: runtime does not match pinned versions: {mismatched}")
    if not torch.cuda.is_available():
        raise SystemExit("NO-GO: requires CUDA; CPU/MPS fallback is forbidden")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"NO-GO: expected one visible CUDA device, found {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    if device.total_memory < 16 * 1024**3:
        raise SystemExit(f"NO-GO: requires >=16 GiB VRAM; found {device.total_memory} bytes")
    return device


def _code_identity(protocol: dict) -> dict:
    repo_root = PROTOCOL_LOCK.resolve().parent.parent
    revision = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise SystemExit("NO-GO: execution checkout must be clean; persist run outputs outside the clone")
    expected = protocol["execution_code"]["files_sha256"]
    observed = {name: sha256_file(repo_root / name) for name in expected}
    if observed != expected:
        raise SystemExit(f"NO-GO: execution code hash differs from v3 lock: {observed}")
    return {"git_revision": revision, "clean_worktree": True, "files_sha256": observed}


def _verify_fixed_prompt_ids(protocol: dict, prompt_ids: list[str]) -> str:
    expected = protocol["training_data"]["sample"]["base_gate_id_list_sha256"]
    try:
        return verify_prompt_id_sha256(prompt_ids, expected, expected_count=32)
    except ValueError as error:
        raise SystemExit(f"NO-GO: fixed gate prompt ID cohort mismatch: {error}") from error


def _tensor_batch(encoded: dict) -> dict:
    batch = {}
    for key, value in encoded.items():
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        batch[key] = tensor
    return batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/math-grpo-cpu-first-2026-10-11/data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/math-grpo-cpu-first-2026-10-11/gpu-base-gate-v3.json"))
    parser.add_argument("--max-wall-seconds", type=int, default=7200)
    parser.add_argument("--max-generated-tokens", type=int, default=131072)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not (0 < args.max_wall_seconds <= 7200) or not (0 < args.max_generated_tokens <= 131072):
        raise SystemExit("time and generated-token limits must be positive and cannot exceed the frozen v3 ceilings")
    protocol = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    code = _code_identity(protocol)
    locked_budget = protocol["first_gpu_gate"]["budget"]
    if args.max_wall_seconds != locked_budget["max_wall_seconds"]:
        raise SystemExit("NO-GO: max_wall_seconds must equal the v3 locked experiment budget")
    if args.max_generated_tokens != locked_budget["max_generated_tokens"]:
        raise SystemExit("NO-GO: max_generated_tokens must equal the v3 locked experiment budget")
    # Validate data lineage and the expected prompt cohort before CUDA is queried
    # for inference resources or a model is loaded.
    for filename, expected_hash in MODEL_FILES_SHA256.items():
        model_file = args.model_dir / filename
        if not model_file.is_file():
            raise SystemExit(f"pinned model file is missing: {filename}")
        observed_hash = sha256_file(model_file)
        if observed_hash != expected_hash:
            raise SystemExit(f"pinned model file hash mismatch for {filename}: {observed_hash}")
    data_path = args.data_dir / "deepmath-train.parquet"
    if sha256_file(data_path) != EXPECTED_DEEPMATH_SHA256:
        raise SystemExit("pinned DeepMath TRAIN parquet hash mismatch")
    raw_rows = parquet.read_table(data_path, columns=["prompt", "solution"]).to_pylist()
    eligible_rows, dedupe_stats = deduplicate_train_rows(raw_rows)
    split = select_splits([], eligible_rows)
    wanted = set(split["base_gate_ids"])
    prompt_hash = _verify_fixed_prompt_ids(protocol, list(wanted))

    device = _check_runtime()
    gate_rows = [
        row for row in eligible_rows
        if hashlib.sha256(row["prompt_key"].encode()).hexdigest() in wanted
    ]
    if len(gate_rows) != 32:
        raise SystemExit(f"expected 32 fixed TRAIN gate prompts; found {len(gate_rows)}")

    identity = {
        "protocol_id": "qwen25_deepmath_grpo_math500_v3",
        "prompt_group_count": 32,
        "protocol_lock_sha256": sha256_file(PROTOCOL_LOCK),
        "experiment_budget": {"max_wall_seconds": args.max_wall_seconds,
                               "max_generated_tokens": args.max_generated_tokens},
        "execution_code": code,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_weights_sha256": MODEL_FILES_SHA256["model.safetensors"],
        "deepmath_train_sha256": EXPECTED_DEEPMATH_SHA256,
        "gate_prompt_ids_sha256": prompt_hash,
        "seed": GATE_SEED,
        "generation": {"temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
                       "completions": NUM_COMPLETIONS, "max_input_tokens": MAX_INPUT_TOKENS,
                       "system_prompt": None, "do_sample": True},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": importlib.metadata.version("transformers"),
                    "trl": importlib.metadata.version("trl"), "cuda_runtime": torch.version.cuda,
                    "gpu_name": device.name, "gpu_compute_capability": list(torch.cuda.get_device_capability(0))},
    }
    journal = GateJournal(args.output, identity, resume=args.resume)
    if journal.elapsed() >= args.max_wall_seconds:
        journal.stop("wall_time", "wall-time budget was exhausted before tokenizer/model loading")
        print(json.dumps({"decision": "wall_time", "state": str(journal.state_path), "partial": str(journal.partial_path)}))
        return
    outputs: list[dict] = []
    group_variances: list[float] = []
    mixed_groups = 0
    prompt_token_total = 0
    max_prompt_tokens = 0
    tokenized: list[dict] = []
    try:
        with wall_time_limit(args.max_wall_seconds - journal.elapsed()):
            tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
            for row_index, row in enumerate(gate_rows):
                encoded = tokenize_generation_prompt(tokenizer, row["prompt"])
                batch = _tensor_batch(encoded)
                input_ids = batch["input_ids"]
                prompt_length = int(input_ids.shape[-1])
                if prompt_length > MAX_INPUT_TOKENS:
                    raise SystemExit(f"gate prompt {row_index} exceeds {MAX_INPUT_TOKENS} input tokens")
                tokenized.append({"row": row, "batch": batch, "prompt_length": prompt_length})
                prompt_token_total += prompt_length
                max_prompt_tokens = max(max_prompt_tokens, prompt_length)

            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            torch.cuda.reset_peak_memory_stats(0)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True,
            ).to("cuda:0")
            model.eval()
            journal.record_peak_vram(torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0))
        def generate(prompt_index: int, _row: dict, per_completion_cap: int) -> dict:
            prompt = tokenized[prompt_index]
            inputs = {key: value.to("cuda:0") for key, value in prompt["batch"].items()}
            torch.manual_seed(GATE_SEED + prompt_index)
            torch.cuda.manual_seed_all(GATE_SEED + prompt_index)
            generation_config = GenerationConfig(
                **base_gate_generation_config(
                    max_new_tokens=per_completion_cap,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            )
            with torch.inference_mode():
                sequences = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    num_return_sequences=NUM_COMPLETIONS,
                )
            token_ids = [sequence[prompt["prompt_length"] :].tolist() for sequence in sequences]
            journal.record_peak_vram(torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0))
            completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
            rewards = accuracy_reward(
                completions=[[{"role": "assistant", "content": text}] for text in completions],
                solution=[prompt["row"]["solution"]] * len(completions),
            )
            if any(reward is None for reward in rewards):
                raise RuntimeError("TRL accuracy_reward skipped a selected training example")
            rewards = [float(reward) for reward in rewards]
            variance = float(torch.tensor(rewards, dtype=torch.float64).var(unbiased=False).item())
            group_variances.append(variance)
            nonlocal mixed_groups
            mixed_groups += int(min(rewards) != max(rewards))
            records = []
            for completion, reward in zip(completions, rewards, strict=True):
                correct = exact_match(completion, prompt["row"]["solution"])
                parseable = has_valid_final_box(completion)
                records.append({
                    "train_row_hash": hashlib.sha256(prompt["row"]["prompt_key"].encode()).hexdigest(),
                    "completion": completion,
                    "training_reward": reward,
                    "independent_task_success": correct,
                    "valid_final_box": parseable,
                })
            outputs.extend(records)
            del sequences, inputs
            torch.cuda.empty_cache()
            return {
                "train_row_hash": hashlib.sha256(prompt["row"]["prompt_key"].encode()).hexdigest(),
                "completion_token_ids": token_ids,
                "completions": records,
                "reward_variance": variance,
                "mixed_reward": min(rewards) != max(rewards),
                "prompt_tokens": prompt["prompt_length"],
            }

        status = run_prompt_groups(
            rows=gate_rows,
            journal=journal,
            generate=generate,
            completions_per_prompt=NUM_COMPLETIONS,
            max_new_tokens_per_completion=MAX_NEW_TOKENS,
            max_generated_tokens=args.max_generated_tokens,
            max_wall_seconds=args.max_wall_seconds,
            eos_token_id=tokenizer.eos_token_id,
        )
        if status != "complete":
            print(json.dumps({"decision": status, "state": str(journal.state_path), "partial": str(journal.partial_path)}))
            return
        groups = [journal.records[index] for index in sorted(journal.records)]
        outputs = [completion for group in groups for completion in group["completions"]]
        group_variances = [float(group["reward_variance"]) for group in groups]
        mixed_groups = sum(bool(group["mixed_reward"]) for group in groups)
        total_correct = sum(bool(output["independent_task_success"]) for output in outputs)
        total_parseable = sum(bool(output["valid_final_box"]) for output in outputs)
        actual_generated_tokens = sum(int(group["generated_tokens"]) for group in groups)
        elapsed = journal.elapsed()
        peak_allocated = max(journal.peak_allocated_vram_bytes, torch.cuda.max_memory_allocated(0))
        peak_reserved = max(journal.peak_reserved_vram_bytes, torch.cuda.max_memory_reserved(0))
        journal.record_peak_vram(peak_allocated, peak_reserved)
        total_vram = torch.cuda.get_device_properties(0).total_memory
        success_rate = total_correct / len(outputs)
        parse_rate = total_parseable / len(outputs)
        mean_group_var = sum(group_variances) / len(group_variances)
        passed = (
            0.05 <= success_rate <= 0.90 and parse_rate >= 0.90 and mixed_groups >= 6
            and mean_group_var > 0.0 and peak_reserved <= MAX_RESERVED_VRAM_BYTES
            and peak_reserved <= 0.90 * total_vram and elapsed <= args.max_wall_seconds
        )
        result = {
            "phase": "train_only_base_feasibility",
            "decision": "pass" if passed else "stop",
            "protocol_id": identity["protocol_id"],
            "identity": identity,
            "data": {"repo": "trl-lib/DeepMath-103K", "revision": "066c50a88d4e14cefc056e31111db2dba17f6c68",
                     "split": "train", "file_sha256": EXPECTED_DEEPMATH_SHA256,
                     "deduplication": dedupe_stats, "gate_prompts": len(gate_rows),
                     "gate_ids_sha256": identity["gate_prompt_ids_sha256"]},
            "generation": {"seed_rule": "20261012 + prompt_index; independent prompt groups support deterministic resume",
                           "completions_per_prompt": NUM_COMPLETIONS, "max_new_tokens_per_completion": MAX_NEW_TOKENS,
                           "temperature": TEMPERATURE, "top_p": TOP_P, "system_prompt": None,
                           "prompt_tokens": prompt_token_total, "max_input_tokens_observed": max_prompt_tokens,
                           "generated_tokens_from_generated_token_ids": actual_generated_tokens,
                           "valid_final_box_rate": parse_rate, "independent_task_success_rate": success_rate,
                           "training_reward_group_mixed_count": mixed_groups,
                           "training_reward_group_variance_mean": mean_group_var,
                           "training_reward_group_variances": group_variances},
            "resources": {"device": torch.cuda.get_device_name(0), "total_vram_bytes": total_vram,
                          "peak_allocated_vram_bytes": peak_allocated, "peak_reserved_vram_bytes": peak_reserved,
                          "elapsed_seconds": elapsed,
            "generated_tokens_per_second": actual_generated_tokens / elapsed if elapsed else None,
            "elapsed_scope": "cumulative wall time from initial journal creation through pauses and all resumed attempts",
                          "dtype": str(dtype), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                          "python": platform.python_version()},
            "optimizer_updates": 0,
            "math500_loaded": False,
            "success_criteria": {"task_success_5_to_90_percent": 0.05 <= success_rate <= 0.90,
                                 "format_at_least_90_percent": parse_rate >= 0.90,
                                 "at_least_6_mixed_groups": mixed_groups >= 6,
                                 "positive_reward_variance": mean_group_var > 0,
                                 "memory_limits_met": peak_reserved <= MAX_RESERVED_VRAM_BYTES and peak_reserved <= 0.9 * total_vram,
                                 "wall_time_met": elapsed <= args.max_wall_seconds},
        }
        journal.finalize(result)
        print(json.dumps({"decision": result["decision"], "generated_tokens": actual_generated_tokens,
                          "elapsed_seconds": elapsed, "output": str(args.output)}))
    except BaseException as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if torch.cuda.is_available():
            try:
                journal.record_peak_vram(torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0))
            except Exception:
                pass
        # Journal state is persisted by run_prompt_groups during generation. This
        # also covers failures while loading tokenizer/model before the first group.
        if journal.state_path.exists():
            decision = (
                "wall_time" if isinstance(error, GateWallTimeExceeded)
                else "oom" if isinstance(error, torch.cuda.OutOfMemoryError)
                else "interrupted" if isinstance(error, KeyboardInterrupt)
                else "failed"
            )
            journal.stop(decision, f"{type(error).__name__}: {error}")
        if isinstance(error, GateWallTimeExceeded):
            print(json.dumps({"decision": "wall_time", "state": str(journal.state_path), "partial": str(journal.partial_path)}))
            return
        raise


if __name__ == "__main__":
    main()
