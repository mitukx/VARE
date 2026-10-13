"""Run the locked one-update smoke and matched SFT/GRPO training arms."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pyarrow.parquet as parquet
import torch
from datasets import Dataset
from math_verify import parse as reward_parse
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from trl.rewards import accuracy_reward

from scripts.math500_grader_v2 import exact_match
from scripts.qwen3_deepmath_protocol_v1 import (
    MATH_INSTRUCTION,
    answer_target,
    cohort_rows,
    grpo_training_record,
    sft_record,
)
from scripts.validate_math500_study_v1 import (
    EXPECTED_DEEPMATH_SHA256,
    deduplicate_train_rows,
    select_splits,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "protocols/qwen3_deepmath_grpo_math500_v1.lock.json"
PROTOCOL_ID = "qwen3_deepmath_grpo_math500_v1"
SEEDS = [20261017, 20261019, 20261023, 20261027, 20261031]
MAX_PROMPT_TOKENS = 512
MAX_SFT_SEQUENCE_TOKENS = 1024


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _code_identity(protocol: dict) -> dict:
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
        raise SystemExit("NO-GO: execution checkout must be clean; keep all outputs outside the clone")
    expected = protocol["execution_code"]["files_sha256"]
    observed = {name: sha256_file(ROOT / name) for name in expected}
    if observed != expected:
        raise SystemExit(f"NO-GO: execution source hashes differ from the lock: {observed}")
    return {"git_revision": revision, "clean_worktree": True, "files_sha256": observed}


def _runtime(protocol: dict) -> dict:
    packages = protocol["runtime"]["packages"]
    observed = {name: importlib.metadata.version(name) for name in packages}
    if observed != packages:
        raise SystemExit(f"NO-GO: package versions differ from the lock: {observed}")
    if platform.python_version() != protocol["runtime"]["python"]:
        raise SystemExit(f"NO-GO: Python version differs from the lock: {platform.python_version()}")
    if torch.__version__ != protocol["runtime"]["torch"] or torch.version.cuda != protocol["runtime"]["cuda_runtime"]:
        raise SystemExit(f"NO-GO: PyTorch/CUDA version differs from the lock: {torch.__version__}/{torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("NO-GO: exactly one CUDA GPU is required")
    props = torch.cuda.get_device_properties(0)
    if props.name != protocol["gate"]["device"]["name"]:
        raise SystemExit(f"NO-GO: unexpected accelerator {props.name}")
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver,
        "gpu_name": props.name,
        "gpu_total_vram_bytes": props.total_memory,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "packages": observed,
    }


def _training_data(data_dir: Path, protocol: dict) -> tuple[list[dict], dict]:
    data_path = data_dir / "deepmath-train.parquet"
    if sha256_file(data_path) != EXPECTED_DEEPMATH_SHA256:
        raise SystemExit("NO-GO: DeepMath TRAIN file hash mismatch")
    raw = parquet.read_table(data_path, columns=["prompt", "solution"]).to_pylist()
    eligible, dedupe = deduplicate_train_rows(raw)
    old = select_splits([], eligible)
    retired = set(old["base_gate_ids"]) | set(old["train_ids"])

    def valid(solution: str) -> bool:
        try:
            return bool(reward_parse(solution)) and exact_match(
                rf"\boxed{{{answer_target(solution)}}}", solution
            )
        except Exception:
            return False

    gate, train, observed = cohort_rows(eligible, retired, is_training_target_valid=valid)
    expected = protocol["training_data"]["cohorts"]
    for key in ("retired_prompt_count", "gate_ids_sha256", "update_ids_sha256"):
        if observed[key] != expected[key]:
            raise SystemExit(f"NO-GO: train/gate cohort field mismatch: {key}")
    legacy = protocol["training_data"]["legacy_exclusion"]
    observed_legacy = {
        "legacy_gate_ids_sha256": _ids_sha256(retired & set(old["base_gate_ids"])),
        "legacy_update_ids_sha256": _ids_sha256(retired & set(old["train_ids"])),
    }
    if observed_legacy != legacy:
        raise SystemExit("NO-GO: legacy retired cohort fingerprint mismatch")
    if len(gate) != 32 or len(train) != 256:
        raise SystemExit("NO-GO: frozen DeepMath cohort sizes differ")
    if any(not valid(row["solution"]) for row in train):
        raise SystemExit("NO-GO: a selected SFT/GRPO training target failed independent self-grading")
    prepared = [grpo_training_record(row) for row in train]
    return prepared, {"deduplication": dedupe, "cohorts": observed, "legacy_exclusion": observed_legacy}


def _ids_sha256(values: set[str]) -> str:
    payload = "\n".join(sorted(values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_model(model_dir: Path, protocol: dict) -> dict:
    expected = protocol["model"]["files_sha256"]
    observed = {}
    for name, digest in expected.items():
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"NO-GO: missing pinned model file: {name}")
        observed[name] = sha256_file(path)
        if observed[name] != digest:
            raise SystemExit(f"NO-GO: model-file hash mismatch: {name}")
    return observed


def _peft_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )


def _load_base(model_dir: Path, *, train: bool):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.config.use_cache = not train
    return model


def _fingerprint_parameters(model, *, trainable: bool | None = None) -> str:
    digest = hashlib.sha256()
    items = sorted(model.named_parameters(), key=lambda item: item[0])
    seen = 0
    for name, parameter in items:
        if trainable is not None and parameter.requires_grad != trainable:
            continue
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        seen += 1
    if not seen:
        raise ValueError("parameter fingerprint did not include any tensors")
    return digest.hexdigest()


def _parameters_finite(model, *, trainable: bool) -> bool:
    selected = [parameter for parameter in model.parameters() if parameter.requires_grad == trainable]
    return bool(selected) and all(bool(torch.isfinite(parameter.detach()).all().item()) for parameter in selected)


def _state_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            _state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _make_optimizer(model, learning_rate: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("LoRA configuration produced no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


def _reward_func(completions, solution, prompt_id, completion_ids, log_extra, trainer_state, **_kwargs):
    """Keep the stock TRL accuracy reward while logging raw group provenance."""
    scores = accuracy_reward(completions=completions, solution=solution)
    if any(value is None for value in scores):
        return scores
    independent = []
    for completion, reference in zip(completions, solution, strict=True):
        if isinstance(completion, list):
            text = "".join(str(message.get("content", "")) for message in completion)
        else:
            text = str(completion)
        independent.append(exact_match(text, str(reference)))
    token_rows = []
    for ids in completion_ids:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        token_rows.append([int(token) for token in ids])
    log_extra("prompt_id", list(prompt_id))
    log_extra("completion_token_ids", token_rows)
    log_extra("independent_task_success", independent)
    log_extra("reward_score", [float(value) for value in scores])
    log_extra(
        "training_reward_checker_disagreement",
        [bool(float(score)) != correct for score, correct in zip(scores, independent, strict=True)],
    )
    log_extra("training_seed", [int(trainer_state.args.seed)] * len(scores))
    log_extra("trainer_global_step_at_rollout", [int(trainer_state.global_step)] * len(scores))
    return scores


def _optimizer_state_path(output_dir: Path) -> Path:
    return output_dir / "optimizer-state.pt"


def _audit_rollout_logs(
    trainer_output: Path,
    expected_prompt_ids: set[str] | None,
    expected_steps: int,
    expected_seed: int,
    eos_token_id: int | None,
) -> dict:
    import pyarrow.parquet as arrow_parquet

    files = sorted((trainer_output / "completions").glob("*.parquet"))
    if not files:
        raise RuntimeError(f"TRL completion parquet logs are missing under {trainer_output / 'completions'}")
    rows = []
    file_hashes = {path.name: sha256_file(path) for path in files}
    for path in files:
        rows.extend(arrow_parquet.read_table(path).to_pylist())
    expected_count = expected_steps * 16
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} logged completions, found {len(rows)}")
    required = {
        "prompt_id", "completion_token_ids", "independent_task_success",
        "training_reward_checker_disagreement", "trainer_global_step_at_rollout",
        "reward_score", "training_seed",
    }
    if not required.issubset(rows[0]):
        raise RuntimeError(f"TRL rollout log is missing provenance columns: {required - set(rows[0])}")
    counts: dict[str, int] = {}
    step_prompts: dict[int, set[str]] = {}
    prompt_steps: dict[str, set[int]] = {}
    group_scores: dict[str, list[float]] = {}
    successes = []
    observed_seeds = set()
    token_count = 0
    disagreement = 0
    for row in rows:
        group = str(row["prompt_id"])
        observed_seeds.add(int(row["training_seed"]))
        counts[group] = counts.get(group, 0) + 1
        step = int(row["trainer_global_step_at_rollout"])
        step_prompts.setdefault(step, set()).add(group)
        prompt_steps.setdefault(group, set()).add(step)
        ids = [int(value) for value in (row["completion_token_ids"] or [])]
        count = 0
        for token in ids:
            count += 1
            if eos_token_id is not None and token == eos_token_id:
                break
        token_count += count
        disagreement += int(bool(row["training_reward_checker_disagreement"]))
        group_scores.setdefault(group, []).append(float(row["reward_score"]))
        successes.append(float(bool(row["independent_task_success"])))
    if any(value != 4 for value in counts.values()):
        raise RuntimeError("a GRPO prompt group was logged with a partial completion count")
    if expected_prompt_ids is not None and set(counts) != expected_prompt_ids:
        raise RuntimeError("rollout provenance does not contain exactly the frozen update prompt IDs")
    if any(len(steps) != 1 for steps in prompt_steps.values()):
        raise RuntimeError("a training prompt was assigned to more than one rollout/update step")
    expected_group_count = expected_steps * 4
    if len(counts) != expected_group_count:
        raise RuntimeError(f"expected {expected_group_count} distinct prompt groups, found {len(counts)}")
    if set(step_prompts) != set(range(expected_steps)) or any(len(ids) != 4 for ids in step_prompts.values()):
        raise RuntimeError("each GRPO optimizer update must log exactly four distinct complete prompt groups")
    if disagreement:
        raise RuntimeError(f"training reward disagreed with the independent checker on {disagreement} rollouts")
    if observed_seeds != {expected_seed}:
        raise RuntimeError(f"rollout log seed mismatch: {observed_seeds}")
    return {
        "logged_completions": len(rows),
        "unique_prompt_groups": len(counts),
        "optimizer_updates_with_four_groups": len(step_prompts),
        "generated_token_ids": token_count,
        "mean_independent_task_success": sum(successes) / len(successes),
        "mixed_reward_groups": sum(min(values) != max(values) for values in group_scores.values()),
        "mean_within_group_reward_variance": sum(
            sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values)
            for values in group_scores.values()
        ) / len(group_scores),
        "training_reward_checker_disagreements": disagreement,
        "prompt_group_size_histogram": {"4": len(counts)},
        "completion_parquet_sha256": file_hashes,
    }


def _make_grpo_trainer(model_dir: Path, tokenizer, train_rows: list[dict], output_dir: Path, seed: int,
                       *, max_steps: int):
    model = _load_base(model_dir, train=True)
    model = get_peft_model(model, _peft_config())
    trainer_config = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        max_prompt_length=MAX_PROMPT_TOKENS,
        max_completion_length=2048,
        num_generations=4,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        beta=0.0,
        epsilon=0.2,
        loss_type="grpo",
        scale_rewards="group",
        num_iterations=1,
        generation_batch_size=None,
        steps_per_generation=None,
        remove_unused_columns=False,
        shuffle_dataset=True,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=16,
        save_total_limit=1,
        save_safetensors=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        log_completions=True,
        num_completions_to_print=4,
    )
    optimizer = _make_optimizer(model, learning_rate=1e-6)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=_reward_func,
        args=trainer_config,
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
        optimizers=(optimizer, None),
    )
    return trainer


def _make_sft_trainer(model_dir: Path, tokenizer, train_rows: list[dict], output_dir: Path, seed: int,
                      *, max_steps: int):
    records = [sft_record(row["prompt"][0]["content"], answer_target(row["solution"])) for row in train_rows]
    dataset = Dataset.from_list(records)
    model = _load_base(model_dir, train=True)
    model = get_peft_model(model, _peft_config())
    config = SFTConfig(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        max_length=MAX_SFT_SEQUENCE_TOKENS,
        completion_only_loss=True,
        packing=False,
        shuffle_dataset=True,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=16,
        save_total_limit=1,
        save_safetensors=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )
    optimizer = _make_optimizer(model, learning_rate=5e-5)
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None,
        optimizers=(optimizer, None),
    )
    return trainer


def _verify_adapter(output_dir: Path, protocol: dict) -> dict:
    manifest = protocol["adapter_manifest"]["required_files"]
    observed = {}
    for name in manifest:
        path = output_dir / name
        if not path.is_file():
            raise ValueError(f"missing adapter artifact: {name}")
        observed[name] = sha256_file(path)
    return observed


def _run_update_smoke(args, protocol: dict, code: dict, model_hashes: dict, runtime: dict, cohorts: dict) -> None:
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    train_rows, _ = _training_data(args.data_dir, protocol)
    seed = int(protocol["training"]["smoke_seed"])
    set_seed(seed)
    smoke_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.reset_peak_memory_stats(0)
    trainer = _make_grpo_trainer(
        args.model_dir, tokenizer, train_rows, output / "trainer", seed, max_steps=1
    )
    if trainer.args.generation_batch_size != 16 or trainer.args.steps_per_generation != 4:
        raise RuntimeError("TRL did not create the frozen 4-prompt / 4-generation accumulation window")
    base_before = _fingerprint_parameters(trainer.model, trainable=False)
    lora_before = _fingerprint_parameters(trainer.model, trainable=True)
    max_reserved = int(torch.cuda.get_device_properties(0).total_memory * 0.90)
    if torch.cuda.max_memory_reserved(0) >= max_reserved:
        raise RuntimeError("one-update smoke exceeded the frozen VRAM ceiling before training")
    started = time.monotonic()
    train_output = trainer.train()
    elapsed = time.monotonic() - started
    base_after = _fingerprint_parameters(trainer.model, trainable=False)
    lora_after = _fingerprint_parameters(trainer.model, trainable=True)
    if not _parameters_finite(trainer.model, trainable=True):
        raise RuntimeError("one-update smoke produced non-finite trainable parameters")
    if trainer.state.global_step != 1:
        raise RuntimeError(f"one-update smoke completed {trainer.state.global_step} optimizer steps")
    if base_before != base_after:
        raise RuntimeError("frozen base parameters changed during LoRA update")
    if lora_before == lora_after:
        raise RuntimeError("the one-step smoke did not change trainable parameters")
    smoke_log_history = list(trainer.state.log_history)
    smoke_metrics = dict(train_output.metrics)
    rollout_audit = _audit_rollout_logs(
        output / "trainer",
        expected_prompt_ids=None,
        expected_steps=1,
        expected_seed=seed,
        eos_token_id=tokenizer.eos_token_id,
    )

    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    optimizer_path = _optimizer_state_path(output)
    saved_optimizer = trainer.optimizer.state_dict()
    torch.save(saved_optimizer, optimizer_path)
    loaded_optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    trainer.optimizer.load_state_dict(loaded_optimizer)
    if not _state_equal(saved_optimizer, trainer.optimizer.state_dict()):
        raise RuntimeError("optimizer state failed same-instance serialization roundtrip")
    saved_state = _cpu_clone(saved_optimizer)

    # Remove the trained object, then reload the exact base+adapter and optimizer state.
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_base = _load_base(args.model_dir, train=False)
    reloaded_model = PeftModel.from_pretrained(reloaded_base, adapter_dir, is_trainable=True)
    reloaded_adapter = _fingerprint_parameters(reloaded_model, trainable=True)
    new_optimizer = _make_optimizer(reloaded_model, learning_rate=1e-6)
    new_optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu", weights_only=True))
    if reloaded_adapter != lora_after:
        raise RuntimeError("saved adapter parameters changed after reload")
    if not _state_equal(saved_state, new_optimizer.state_dict()):
        raise RuntimeError("optimizer state differs after reload into a fresh optimizer")
    reloaded_base_hash = _fingerprint_parameters(reloaded_model, trainable=False)
    if reloaded_base_hash != base_before:
        raise RuntimeError("base weights changed after save/reload")
    del reloaded_model, reloaded_base, new_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    total_wall = time.monotonic() - smoke_started
    passed = peak_reserved < max_reserved and total_wall <= protocol["training"]["smoke_max_wall_seconds"]
    adapter_files = _verify_adapter(adapter_dir, protocol)
    result = {
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "execution_code": code,
        "phase": "one_update_grpo_save_reload_smoke",
        "decision": "pass" if passed else "stop",
        "smoke_passed": passed,
        "optimizer_steps": 1,
        "training_wall_seconds": elapsed,
        "end_to_end_wall_seconds": total_wall,
        "train_rows_available": len(train_rows),
        "cohort_ids_sha256": cohorts["update_ids_sha256"],
        "model_file_sha256": model_hashes,
        "adapter_file_sha256": adapter_files,
        "optimizer_state_sha256": sha256_file(optimizer_path),
        "base_fingerprint_before": base_before,
        "base_fingerprint_after_update": base_after,
        "base_fingerprint_after_reload": reloaded_base_hash,
        "lora_fingerprint_before": lora_before,
        "lora_fingerprint_after_update": lora_after,
        "lora_fingerprint_after_reload": reloaded_adapter,
        "trainable_parameters_finite": True,
        "changed_trainable_weights": lora_before != lora_after,
        "base_weights_unchanged": base_before == base_after == reloaded_base_hash,
        "optimizer_state_roundtrip_exact": True,
        "rollout_audit": rollout_audit,
        "training_metrics": smoke_metrics,
        "trainer_log_history": smoke_log_history,
        "resources": {
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "maximum_reserved_vram_bytes": max_reserved,
            "wall_seconds": total_wall,
            "runtime": runtime,
        },
        "confirmation_loaded": False,
    }
    _write_json(output / "smoke.json", result)
    print(json.dumps({"decision": result["decision"], "output": str(output / "smoke.json")}))
    if not passed:
        raise SystemExit("NO-GO: one-update save/reload/resource smoke failed")


def _train_arm(model_dir: Path, tokenizer, train_rows: list[dict], output: Path,
               arm: str, seed: int, protocol: dict) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    final_adapter = output / "adapter"
    if (output / "train.json").exists():
        return json.loads((output / "train.json").read_text(encoding="utf-8"))
    expected_steps = int(protocol["training"]["optimizer_updates_per_seed"])
    set_seed(seed)
    if arm == "grpo":
        torch.cuda.reset_peak_memory_stats(0)
        trainer = _make_grpo_trainer(model_dir, tokenizer, train_rows, output / "trainer", seed, max_steps=expected_steps)
        if trainer.args.generation_batch_size != 16 or trainer.args.steps_per_generation != 4:
            raise RuntimeError("TRL accumulation/generation group contract differs from frozen protocol")
    elif arm == "sft":
        torch.cuda.reset_peak_memory_stats(0)
        trainer = _make_sft_trainer(model_dir, tokenizer, train_rows, output / "trainer", seed, max_steps=expected_steps)
    else:
        raise ValueError(f"unknown training arm: {arm}")
    limit = int(torch.cuda.get_device_properties(0).total_memory * 0.90)
    if torch.cuda.max_memory_reserved(0) >= limit:
        raise RuntimeError(f"{arm}/{seed} exceeded the frozen VRAM ceiling during model load")
    started = time.monotonic()
    from transformers.trainer_utils import get_last_checkpoint

    existing_entries = list((output / "trainer").glob("checkpoint-*")) if (output / "trainer").exists() else []
    checkpoint = get_last_checkpoint(str(output / "trainer")) if existing_entries else None
    if existing_entries and checkpoint is None:
        raise RuntimeError(f"trainer output contains checkpoint artifacts but no valid checkpoint: {output}")
    initial_base = _fingerprint_parameters(trainer.model, trainable=False)
    initial_trainable = None if checkpoint is not None else _fingerprint_parameters(trainer.model, trainable=True)
    trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.monotonic() - started
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(f"{arm}/{seed} finished {trainer.state.global_step} updates, expected {expected_steps}")
    final_trainable = _fingerprint_parameters(trainer.model, trainable=True)
    final_base = _fingerprint_parameters(trainer.model, trainable=False)
    if initial_base != final_base:
        raise RuntimeError(f"{arm}/{seed} changed frozen base parameters")
    if not _parameters_finite(trainer.model, trainable=True):
        raise RuntimeError(f"{arm}/{seed} produced non-finite trainable parameters")
    if checkpoint is None and initial_trainable == final_trainable:
        raise RuntimeError(f"{arm}/{seed} did not change its trainable parameters")
    rollout_audit = (
        _audit_rollout_logs(
            output / "trainer",
            expected_prompt_ids={str(row["prompt_id"]) for row in train_rows},
            expected_steps=expected_steps,
            expected_seed=seed,
            eos_token_id=tokenizer.eos_token_id,
        )
        if arm == "grpo"
        else None
    )
    trainer.model.save_pretrained(final_adapter, safe_serialization=True)
    optimizer_path = output / "optimizer-state.pt"
    torch.save(trainer.optimizer.state_dict(), optimizer_path)
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    adapter_files = _verify_adapter(final_adapter, protocol)
    record = {
        "arm": arm,
        "seed": seed,
        "optimizer_steps": trainer.state.global_step,
        "unique_training_prompt_count": len(train_rows),
        "cohort_ids_sha256": protocol["training_data"]["cohorts"]["update_ids_sha256"],
        "rollouts_per_optimizer_update": protocol["training"]["trl_semantics"]["completions_per_optimizer_update"] if arm == "grpo" else 0,
        "total_rollouts": protocol["training"]["trl_semantics"]["total_completions_per_seed"] if arm == "grpo" else 0,
        "trainable_parameter_fingerprint_before": initial_trainable,
        "trainable_parameter_fingerprint_after": final_trainable,
        "trainable_parameters_changed": None if initial_trainable is None else initial_trainable != final_trainable,
        "resumed_from_checkpoint": checkpoint,
        "frozen_base_fingerprint_before": initial_base,
        "frozen_base_fingerprint_after": final_base,
        "frozen_base_unchanged": initial_base == final_base,
        "adapter_file_sha256": adapter_files,
        "optimizer_state_sha256": sha256_file(optimizer_path),
        "trainer_log_history": trainer.state.log_history,
        "trainer_metrics": trainer.state.log_history[-1] if trainer.state.log_history else {},
        "rollout_audit": rollout_audit,
        "resources": {
            "gpu_name": torch.cuda.get_device_name(0),
            "peak_allocated_vram_bytes": peak_allocated,
            "peak_reserved_vram_bytes": peak_reserved,
            "maximum_reserved_vram_bytes": limit,
            "wall_seconds": elapsed,
        },
    }
    _write_json(output / "train.json", record)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    if peak_reserved >= limit:
        raise RuntimeError(f"{arm}/{seed} exceeded the frozen 90% VRAM limit")
    return record


def _train(args, protocol: dict, code: dict, model_hashes: dict, runtime: dict, cohort_audit: dict) -> None:
    gate_path = args.output / "base-gate.json"
    smoke_path = args.output / "smoke" / "smoke.json"
    if not gate_path.is_file() or not smoke_path.is_file():
        raise SystemExit("NO-GO: both the passing base gate and one-update smoke are required")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if gate.get("protocol_id") != PROTOCOL_ID or gate.get("decision") != "pass":
        raise SystemExit("NO-GO: frozen base gate did not pass")
    if smoke.get("protocol_id") != PROTOCOL_ID or smoke.get("decision") != "pass":
        raise SystemExit("NO-GO: frozen update/save/reload smoke did not pass")
    for upstream in (gate, smoke):
        lock_digest = upstream.get("protocol_lock_sha256")
        if lock_digest is None:
            lock_digest = upstream.get("identity", {}).get("protocol_lock_sha256")
        if lock_digest != sha256_file(LOCK_PATH):
            raise SystemExit("NO-GO: upstream evidence was produced under a different protocol lock")
        source_identity = upstream.get("execution_code")
        if source_identity is None:
            source_identity = upstream.get("identity", {}).get("execution_code")
        if source_identity != code:
            raise SystemExit("NO-GO: upstream evidence uses different execution source files")
        upstream_runtime = upstream.get("runtime")
        if upstream_runtime is None:
            upstream_runtime = upstream.get("resources", {}).get("runtime")
        if upstream_runtime is None:
            upstream_runtime = upstream.get("identity", {}).get("runtime")
        if upstream_runtime != runtime:
            raise SystemExit("NO-GO: upstream evidence uses a different runtime")
        upstream_model_hashes = upstream.get("model_file_sha256")
        if upstream_model_hashes is None:
            upstream_model_hashes = upstream.get("identity", {}).get("model_files_sha256")
        if upstream_model_hashes != model_hashes:
            raise SystemExit("NO-GO: upstream evidence was produced from different model files")
        upstream_cohort = upstream.get("cohort_ids_sha256")
        if upstream_cohort is None:
            upstream_cohort = upstream.get("identity", {}).get("update_prompt_ids_sha256")
        if upstream_cohort != protocol["training_data"]["cohorts"]["update_ids_sha256"]:
            raise SystemExit("NO-GO: upstream evidence names a different update cohort")
    if (args.output / "training-summary.json").exists() and not args.resume:
        raise FileExistsError(args.output / "training-summary.json")
    train_rows, data_audit = _training_data(args.data_dir, protocol)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    progress_path = args.output / "training-progress.json"
    if args.resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != {
            "protocol_id": PROTOCOL_ID,
            "protocol_lock_sha256": sha256_file(LOCK_PATH),
            "execution_code": code,
            "runtime": runtime,
            "model_hashes": model_hashes,
            "data_sha256": protocol["training_data"]["sha256"],
            "cohort_ids_sha256": protocol["training_data"]["cohorts"]["update_ids_sha256"],
            "seeds": SEEDS,
        }:
            raise SystemExit("NO-GO: training resume identity mismatch")
        records = progress["records"]
        phase_started = float(progress["phase_started_epoch"])
    else:
        identity = {
            "protocol_id": PROTOCOL_ID,
            "protocol_lock_sha256": sha256_file(LOCK_PATH),
            "execution_code": code,
            "runtime": runtime,
            "model_hashes": model_hashes,
            "data_sha256": protocol["training_data"]["sha256"],
            "cohort_ids_sha256": protocol["training_data"]["cohorts"]["update_ids_sha256"],
            "seeds": SEEDS,
        }
        if progress_path.exists():
            raise FileExistsError("training progress exists; pass --resume only for the identical lock and inputs")
        phase_started = time.time()
        records = {}
        progress = {"identity": identity, "phase_started_epoch": phase_started, "records": records}
        _write_json(progress_path, progress)

    max_wall = protocol["training"]["max_total_wall_seconds"]
    max_generated = protocol["training"]["max_total_generated_token_ids"]
    for seed in SEEDS:
        for arm in ("sft", "grpo"):
            key = f"{arm}-{seed}"
            if key in records:
                continue
            if time.time() - phase_started >= max_wall:
                _write_json(progress_path, progress)
                raise SystemExit("training phase exceeded its frozen cumulative wall-time budget")
            already_generated = sum(
                int(record.get("rollout_audit", {}).get("generated_token_ids", 0))
                for record in records.values()
            )
            if already_generated >= max_generated:
                raise SystemExit("training phase exhausted its frozen generated-token budget")
            arm_output = args.output / "candidates" / key
            record = _train_arm(args.model_dir, tokenizer, train_rows, arm_output, arm, seed, protocol)
            records[key] = record
            observed_generated = sum(
                int(item.get("rollout_audit", {}).get("generated_token_ids", 0))
                for item in records.values()
            )
            if observed_generated > max_generated:
                raise SystemExit("training phase exceeded its frozen generated-token budget")
            progress["records"] = records
            progress["elapsed_seconds"] = time.time() - phase_started
            _write_json(progress_path, progress)
    if len(records) != 10:
        raise RuntimeError("training did not produce all ten learned-arm records")
    required = protocol["adapter_manifest"]["required_files"]
    for seed in SEEDS:
        for arm in ("sft", "grpo"):
            adapter_dir = args.output / "candidates" / f"{arm}-{seed}" / "adapter"
            if _verify_adapter(adapter_dir, protocol) != records[f"{arm}-{seed}"]["adapter_file_sha256"]:
                raise RuntimeError(f"adapter manifest changed after training: {arm}/{seed}")
    summary = {
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "execution_code": code,
        "runtime": runtime,
        "status": "complete",
        "model_file_sha256": model_hashes,
        "cohort_audit": cohort_audit,
        "data_audit": data_audit,
        "training": records,
        "training_wall_seconds": time.time() - phase_started,
        "confirmation_opened": False,
        "all_ten_adapters_integrity_checked": True,
        "adapter_required_files": required,
    }
    _write_json(args.output / "training-summary.json", summary)
    print(json.dumps({"status": summary["status"], "candidate_count": len(records), "confirmation_opened": False}))


def _prepare(args):
    protocol = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol["prompt_contract"]["instruction"] != MATH_INSTRUCTION:
        raise SystemExit("NO-GO: protocol identity or prompt instruction mismatch")
    code = _code_identity(protocol)
    model_hashes = _verify_model(args.model_dir, protocol)
    runtime = _runtime(protocol)
    train_rows, data_audit = _training_data(args.data_dir, protocol)
    cohort_audit = data_audit["cohorts"]
    return protocol, code, model_hashes, runtime, cohort_audit


def _verify_model(model_dir: Path, protocol: dict) -> dict:
    output = {}
    for name, expected in protocol["model"]["files_sha256"].items():
        path = model_dir / name
        if not path.is_file():
            raise SystemExit(f"NO-GO: pinned model file missing: {name}")
        observed = sha256_file(path)
        if observed != expected:
            raise SystemExit(f"NO-GO: pinned model file changed: {name}")
        output[name] = observed
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "train"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol, code, model_hashes, runtime, cohort_audit = _prepare(args)
    if args.command == "smoke":
        if (args.output / "base-gate.json").is_file():
            gate = json.loads((args.output / "base-gate.json").read_text(encoding="utf-8"))
            if gate.get("decision") != "pass" or gate.get("protocol_id") != PROTOCOL_ID:
                raise SystemExit("NO-GO: base gate failed")
            if gate.get("identity", {}).get("protocol_lock_sha256") != sha256_file(LOCK_PATH):
                raise SystemExit("NO-GO: base gate used a different protocol lock")
            if gate.get("identity", {}).get("execution_code") != code:
                raise SystemExit("NO-GO: base gate used different execution source files")
            if gate.get("identity", {}).get("model_files_sha256") != model_hashes:
                raise SystemExit("NO-GO: base gate used a different model snapshot")
            if gate.get("cohorts", {}).get("update_ids_sha256") != cohort_audit["update_ids_sha256"]:
                raise SystemExit("NO-GO: base gate names a different update cohort")
        else:
            raise SystemExit("NO-GO: passing base-gate.json must be present")
        _run_update_smoke(args, protocol, code, model_hashes, runtime, cohort_audit)
    else:
        _train(args, protocol, code, model_hashes, runtime, cohort_audit)


if __name__ == "__main__":
    main()
