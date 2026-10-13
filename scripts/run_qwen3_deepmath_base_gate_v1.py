"""Frozen TRAIN-only Qwen3/DeepMath signal gate; zero optimizer updates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from math_verify import parse as reward_parse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl.rewards import accuracy_reward

from scripts.gpu_gate_journal_v3 import GateJournal, run_prompt_groups
from scripts.math500_grader_v2 import exact_match, has_valid_final_box
from scripts.qwen3_deepmath_protocol_v1 import (
    MATH_INSTRUCTION,
    answer_target,
    cohort_rows,
    prompt_id,
    tokenize_generation_prompt,
)
from scripts.qwen25_gate_metrics_v4 import completion_shape, summarize_completions
from scripts.validate_math500_study_v1 import (
    EXPECTED_DEEPMATH_SHA256,
    deduplicate_train_rows,
    select_splits,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "protocols/qwen3_deepmath_grpo_math500_v1.lock.json"
PROTOCOL_ID = "qwen3_deepmath_grpo_math500_v1"
GATE_SEED = 20261013
NUM_COMPLETIONS = 4
MAX_INPUT_TOKENS = 512
MAX_SFT_SEQUENCE_TOKENS = 1024
MAX_NEW_TOKENS = 2048
MAX_WALL_SECONDS = 14_400
MAX_GENERATED_TOKENS = 262_144


def _git_identity(protocol: dict) -> dict:
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise SystemExit("NO-GO: execution checkout must be clean; keep outputs outside the clone")
    expected = protocol["execution_code"]["files_sha256"]
    observed = {name: sha256_file(ROOT / name) for name in expected}
    if observed != expected:
        raise SystemExit(f"NO-GO: execution source hash mismatch: {observed}")
    return {"git_revision": revision, "clean_worktree": True, "files_sha256": observed}


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def _device(protocol: dict) -> tuple[torch.device, int, dict]:
    expected = protocol["runtime"]["packages"]
    observed = {package: _version(package) for package in expected}
    if observed != expected:
        raise SystemExit(f"NO-GO: installed package versions differ from protocol: {observed}")
    if platform.python_version() != protocol["runtime"]["python"]:
        raise SystemExit(f"NO-GO: Python version differs from protocol: {platform.python_version()}")
    if torch.__version__ != protocol["runtime"]["torch"]:
        raise SystemExit(f"NO-GO: torch version differs from protocol: {torch.__version__}")
    if torch.version.cuda != protocol["runtime"]["cuda_runtime"]:
        raise SystemExit(f"NO-GO: CUDA build differs from protocol: {torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("NO-GO: exactly one visible CUDA GPU is required")
    props = torch.cuda.get_device_properties(0)
    gpu_rule = protocol["gate"]["device"]
    if props.name != gpu_rule["name"] or props.total_memory < gpu_rule["minimum_total_vram_bytes"]:
        raise SystemExit(f"NO-GO: GPU does not meet the frozen A100/VRAM gate: {props.name}, {props.total_memory}")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver,
        "gpu_name": props.name,
        "gpu_total_vram_bytes": props.total_memory,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "packages": observed,
    }
    max_reserved = int(props.total_memory * gpu_rule["maximum_reserved_fraction"])
    return torch.device("cuda:0"), max_reserved, runtime


def _training_target_is_valid(solution: str) -> bool:
    try:
        return bool(reward_parse(solution)) and exact_match(
            rf"\boxed{{{answer_target(solution)}}}", solution
        )
    except Exception:
        return False


def _verify_model(model_dir: Path, protocol: dict) -> dict:
    expected = protocol["model"]["files_sha256"]
    observed = {}
    for relative, digest in expected.items():
        path = model_dir / relative
        if not path.is_file():
            raise SystemExit(f"NO-GO: pinned model file is missing: {relative}")
        observed[relative] = sha256_file(path)
        if observed[relative] != digest:
            raise SystemExit(f"NO-GO: pinned model file hash mismatch: {relative}")
    return observed


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
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    protocol = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise SystemExit("NO-GO: protocol ID mismatch")
    if protocol["prompt_contract"]["instruction"] != MATH_INSTRUCTION:
        raise SystemExit("NO-GO: prompt instruction differs between helper and locked protocol")
    code = _git_identity(protocol)
    if sha256_file(args.data_dir / "deepmath-train.parquet") != EXPECTED_DEEPMATH_SHA256:
        raise SystemExit("NO-GO: pinned DeepMath TRAIN parquet hash mismatch")
    model_files = _verify_model(args.model_dir, protocol)

    raw_rows = parquet.read_table(
        args.data_dir / "deepmath-train.parquet", columns=["prompt", "solution"]
    ).to_pylist()
    eligible, dedupe = deduplicate_train_rows(raw_rows)
    retired = select_splits([], eligible)
    retired_ids = set(retired["base_gate_ids"]) | set(retired["train_ids"])
    gate_rows, update_rows, cohorts = cohort_rows(
        eligible, retired_ids, is_training_target_valid=_training_target_is_valid
    )
    frozen = protocol["training_data"]["cohorts"]
    if cohorts["gate_ids_sha256"] != frozen["gate_ids_sha256"]:
        raise SystemExit("NO-GO: fresh gate cohort fingerprint mismatch")
    if cohorts["update_ids_sha256"] != frozen["update_ids_sha256"]:
        raise SystemExit("NO-GO: update cohort fingerprint mismatch")
    if cohorts["retired_prompt_count"] != frozen["retired_prompt_count"]:
        raise SystemExit("NO-GO: retired legacy prompt set changed")
    if len(gate_rows) != 32 or len(update_rows) != 256:
        raise SystemExit("NO-GO: frozen train-only cohort sizes do not match")

    generation = protocol["gate"]["generation"]
    expected_generation = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_completions_per_prompt": NUM_COMPLETIONS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "repetition_penalty": 1.0,
        "prompt_instruction": protocol["prompt_contract"]["instruction"],
    }
    if generation != expected_generation:
        raise SystemExit("NO-GO: gate generation configuration differs from its locked protocol")
    budget = protocol["gate"]["budget"]
    if (budget["max_wall_seconds"] != MAX_WALL_SECONDS
            or budget["max_generated_tokens"] != MAX_GENERATED_TOKENS):
        raise SystemExit("NO-GO: gate budgets differ from the runner's frozen ceilings")

    device, max_reserved, runtime = _device(protocol)
    identity = {
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "execution_code": code,
        "model_id": protocol["model"]["repo_id"],
        "model_revision": protocol["model"]["revision"],
        "model_files_sha256": model_files,
        "deepmath_train_sha256": EXPECTED_DEEPMATH_SHA256,
        "gate_prompt_ids_sha256": cohorts["gate_ids_sha256"],
        "update_prompt_ids_sha256": cohorts["update_ids_sha256"],
        "gate_seed": GATE_SEED,
        "generation": protocol["gate"]["generation"],
        "max_wall_seconds": budget["max_wall_seconds"],
        "max_generated_tokens": budget["max_generated_tokens"],
        "runtime": runtime,
        "maximum_peak_reserved_vram_bytes": max_reserved,
    }
    journal = GateJournal(args.output, identity, resume=args.resume)
    if journal.elapsed() >= MAX_WALL_SECONDS:
        journal.stop("wall_time", "gate budget exhausted before tokenizer/model load")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenized = []
        total_prompt_tokens = 0
        max_update_prompt_tokens = 0
        max_sft_sequence_tokens = 0
        for index, row in enumerate(gate_rows):
            batch = _tensor_batch(tokenize_generation_prompt(tokenizer, row["prompt"]))
            length = int(batch["input_ids"].shape[-1])
            if length > MAX_INPUT_TOKENS:
                raise SystemExit(f"NO-GO: gate prompt {index} has {length} tokens (>512); no truncation allowed")
            total_prompt_tokens += length
            tokenized.append({"row": row, "batch": batch, "prompt_length": length})
        for index, row in enumerate(update_rows):
            problem = row["prompt"]
            prompt_batch = _tensor_batch(tokenize_generation_prompt(tokenizer, problem))
            max_update_prompt_tokens = max(max_update_prompt_tokens, int(prompt_batch["input_ids"].shape[-1]))
            target = rf"\boxed{{{answer_target(row['solution'])}}}"
            complete = tokenizer.apply_chat_template(
                [{"role": "system", "content": protocol["prompt_contract"]["instruction"]},
                 {"role": "user", "content": problem},
                 {"role": "assistant", "content": target}],
                tokenize=True,
                add_generation_prompt=False,
            )
            max_sft_sequence_tokens = max(max_sft_sequence_tokens, len(complete))
            if len(complete) > MAX_SFT_SEQUENCE_TOKENS:
                raise SystemExit(f"NO-GO: SFT example {index} exceeds the frozen 1024-token max_length")
        if max_update_prompt_tokens > MAX_INPUT_TOKENS:
            raise SystemExit("NO-GO: an update prompt exceeds 512 tokens; truncation is forbidden")

        dtype = torch.bfloat16
        torch.cuda.reset_peak_memory_stats(0)
        started = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, local_files_only=True, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device)
        model.eval()
        journal.record_peak_vram(torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0))
        if journal.peak_reserved_vram_bytes > max_reserved:
            journal.stop("vram_limit", "model-load peak reserved memory exceeded 90% of A100 VRAM")
            return
    except BaseException as error:
        if journal.state_path.exists():
            journal.stop("preflight_failed", f"{type(error).__name__}: {error}")
        raise

    def generate(prompt_index: int, _row: dict, per_completion_cap: int) -> dict:
        prompt = tokenized[prompt_index]
        inputs = {key: value.to(device) for key, value in prompt["batch"].items()}
        seed = GATE_SEED + prompt_index
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cfg = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            max_new_tokens=per_completion_cap,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.0,
        )
        with torch.inference_mode():
            sequences = model.generate(
                **inputs, generation_config=cfg, num_return_sequences=NUM_COMPLETIONS
            )
        token_ids = [sequence[prompt["prompt_length"] :].tolist() for sequence in sequences]
        journal.record_peak_vram(torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0))
        if journal.peak_reserved_vram_bytes > max_reserved:
            raise torch.cuda.OutOfMemoryError("reserved VRAM exceeded the frozen 90% cap")
        completions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in token_ids]
        rewards = accuracy_reward(
            completions=[[{"role": "assistant", "content": text}] for text in completions],
            solution=[prompt["row"]["solution"]] * len(completions),
        )
        if len(rewards) != NUM_COMPLETIONS or any(value is None for value in rewards):
            raise RuntimeError("training reward did not score every completion")
        rewards = [float(value) for value in rewards]
        variance = float(torch.tensor(rewards, dtype=torch.float64).var(unbiased=False).item())
        prompt_hash = prompt_id(prompt["row"]["prompt_key"])
        records = []
        for ids, text, reward in zip(token_ids, completions, rewards, strict=True):
            correct = exact_match(text, prompt["row"]["solution"])
            valid_box = has_valid_final_box(text)
            records.append({
                "prompt_id": prompt_hash,
                "completion_token_ids": ids,
                "completion": text,
                **completion_shape(ids, tokenizer.eos_token_id, per_completion_cap),
                "training_reward": reward,
                "independent_task_success": correct,
                "valid_final_box": valid_box,
                "training_reward_independent_checker_disagreement": bool(reward) != correct,
            })
        del sequences, inputs
        torch.cuda.empty_cache()
        return {
            "prompt_id": prompt_hash,
            "completion_token_ids": token_ids,
            "completions": records,
            "reward_variance": variance,
            "mixed_reward": min(rewards) != max(rewards),
            "prompt_tokens": prompt["prompt_length"],
        }

    try:
        status = run_prompt_groups(
            rows=gate_rows,
            journal=journal,
            generate=generate,
            completions_per_prompt=NUM_COMPLETIONS,
            max_new_tokens_per_completion=MAX_NEW_TOKENS,
            max_generated_tokens=MAX_GENERATED_TOKENS,
            max_wall_seconds=MAX_WALL_SECONDS,
            eos_token_id=tokenizer.eos_token_id,
        )
        if status != "complete":
            print(json.dumps({"decision": status, "state": str(journal.state_path), "partial": str(journal.partial_path)}))
            return
        groups = [journal.records[index] for index in sorted(journal.records)]
        outputs = [item for group in groups for item in group["completions"]]
        correct = sum(bool(item["independent_task_success"]) for item in outputs)
        valid = sum(bool(item["valid_final_box"]) for item in outputs)
        mixed = sum(bool(group["mixed_reward"]) for group in groups)
        variances = [float(group["reward_variance"]) for group in groups]
        success_rate = correct / len(outputs)
        format_rate = valid / len(outputs)
        agreement = not any(item["training_reward_independent_checker_disagreement"] for item in outputs)
        generated = sum(int(group["generated_tokens"]) for group in groups)
        peak_allocated = max(journal.peak_allocated_vram_bytes, torch.cuda.max_memory_allocated(0))
        peak_reserved = max(journal.peak_reserved_vram_bytes, torch.cuda.max_memory_reserved(0))
        journal.record_peak_vram(peak_allocated, peak_reserved)
        passed = (
            mixed >= 10
            and sum(variances) / len(variances) > 0
            and agreement
            and format_rate >= 0.95
            and 0.05 <= success_rate <= 0.90
            and generated <= MAX_GENERATED_TOKENS
            and journal.elapsed() <= MAX_WALL_SECONDS
            and peak_reserved < max_reserved
        )
        result = {
            "protocol_id": PROTOCOL_ID,
            "phase": "train_only_base_signal_gate",
            "decision": "pass" if passed else "stop",
            "gate_passed": passed,
            "identity": identity,
            "cohorts": cohorts,
            "deduplication": dedupe,
            "generation": {
                "prompt_tokens_total": total_prompt_tokens,
                "max_prompt_tokens": max(item["prompt_length"] for item in tokenized),
                "max_update_prompt_tokens": max_update_prompt_tokens,
                "max_sft_sequence_tokens": max_sft_sequence_tokens,
                "completion_count": len(outputs),
                "generated_token_ids": generated,
                "success_rate": success_rate,
                "valid_final_box_rate": format_rate,
                "mixed_reward_groups": mixed,
                "mean_within_group_reward_variance": sum(variances) / len(variances),
                "training_reward_checker_agreement": agreement,
                **summarize_completions(outputs),
            },
            "resources": {
                "device": torch.cuda.get_device_name(0),
                "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                "peak_allocated_vram_bytes": peak_allocated,
                "peak_reserved_vram_bytes": peak_reserved,
                "maximum_reserved_vram_bytes": max_reserved,
                "elapsed_seconds": journal.elapsed(),
                "model_load_and_run_seconds": time.time() - started,
                "dtype": str(dtype),
                "optimizer_updates": 0,
            },
            "evaluation_loaded": False,
            "success_criteria": {
                "mixed_groups_at_least_10": mixed >= 10,
                "positive_mean_reward_variance": sum(variances) / len(variances) > 0,
                "reward_checker_agreement_all": agreement,
                "format_at_least_95_percent": format_rate >= 0.95,
                "task_success_between_5_and_90_percent": 0.05 <= success_rate <= 0.90,
                "generated_token_budget_met": generated <= MAX_GENERATED_TOKENS,
                "wall_time_met": journal.elapsed() <= MAX_WALL_SECONDS,
                "reserved_vram_below_90_percent": peak_reserved < max_reserved,
            },
            "legacy_cohort_exclusion": {
                "legacy_protocol": "qwen25_deepmath_grpo_math500_v7",
                "legacy_gate_ids_sha256": "0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a",
                "legacy_update_ids_sha256": "80cfa5135aa56ebaf128d0928bd74ceddd9ef45cde42dcc5684663b2e822c5bd",
            },
        }
        journal.finalize(result)
        print(json.dumps({"decision": result["decision"], "output": str(args.output), "tokens": generated}))
    except BaseException as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        if journal.state_path.exists():
            decision = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            journal.stop(decision, f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
