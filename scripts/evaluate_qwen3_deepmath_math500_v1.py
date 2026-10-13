"""Independently score frozen Qwen3 candidates on MATH-500 dev/confirmation."""

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

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from scripts.math500_grader_v2 import exact_match, has_valid_final_box
from scripts.qwen3_deepmath_protocol_v1 import tokenize_generation_prompt
from scripts.qwen3_deepmath_statistics_v1 import mean as _mean, paired_task_seed_bootstrap
from scripts.validate_math500_study_v1 import (
    EXPECTED_DEEPMATH_SHA256,
    EXPECTED_MATH500_SHA256,
    deduplicate_train_rows,
    select_splits,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "protocols/qwen3_deepmath_grpo_math500_v1.lock.json"
PROTOCOL_ID = "qwen3_deepmath_grpo_math500_v1"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _ids_sha256(values: list[str]) -> str:
    payload = "\n".join(sorted(values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _code_identity(protocol: dict) -> dict:
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise SystemExit("NO-GO: execution checkout must be clean; place run outputs outside this clone")
    expected = protocol["execution_code"]["files_sha256"]
    observed = {name: sha256_file(ROOT / name) for name in expected}
    if observed != expected:
        raise SystemExit("NO-GO: evaluation source hashes differ from the locked protocol")
    return {"git_revision": revision, "clean_worktree": True, "files_sha256": observed}


def _runtime(protocol: dict) -> dict:
    expected = protocol["runtime"]["packages"]
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise SystemExit(f"NO-GO: runtime package versions differ from lock: {observed}")
    if platform.python_version() != protocol["runtime"]["python"]:
        raise SystemExit(f"NO-GO: Python version differs from lock: {platform.python_version()}")
    if torch.__version__ != protocol["runtime"]["torch"] or torch.version.cuda != protocol["runtime"]["cuda_runtime"]:
        raise SystemExit(f"NO-GO: CUDA/PyTorch version differs from lock: {torch.__version__}/{torch.version.cuda}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("NO-GO: exactly one visible CUDA GPU is required")
    props = torch.cuda.get_device_properties(0)
    device = protocol["gate"]["device"]
    if props.name != device["name"] or props.total_memory < device["minimum_total_vram_bytes"]:
        raise SystemExit("NO-GO: the evaluation runtime does not match the frozen A100 requirement")
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


def _verify_model(model_dir: Path, protocol: dict) -> dict:
    observed = {}
    for relative, expected in protocol["model"]["files_sha256"].items():
        path = model_dir / relative
        if not path.is_file():
            raise SystemExit(f"NO-GO: pinned model file is missing: {relative}")
        observed[relative] = sha256_file(path)
        if observed[relative] != expected:
            raise SystemExit(f"NO-GO: pinned model file hash mismatch: {relative}")
    return observed


def _load_data(data_dir: Path, protocol: dict, phase: str) -> list[dict]:
    math_path = data_dir / "math500.jsonl"
    if sha256_file(data_dir / "deepmath-train.parquet") != EXPECTED_DEEPMATH_SHA256:
        raise SystemExit("NO-GO: DeepMath source hash mismatch")
    if sha256_file(math_path) != EXPECTED_MATH500_SHA256:
        raise SystemExit("NO-GO: MATH-500 source hash mismatch")
    math_rows = [json.loads(line) for line in math_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = __import__("pyarrow.parquet", fromlist=["read_table"]).read_table(
        data_dir / "deepmath-train.parquet", columns=["prompt", "solution"]
    ).to_pylist()
    eligible, _ = deduplicate_train_rows(train_rows)
    splits = select_splits(math_rows, eligible)
    expected_split = protocol["evaluation"]["splits"][phase]
    ids = splits["development_ids" if phase == "development" else "confirmation_ids"]
    if len(ids) != expected_split["count"] or _ids_sha256(ids) != expected_split["ids_sha256"]:
        raise SystemExit(f"NO-GO: frozen MATH-500 {phase} membership changed")
    by_id = {row["unique_id"]: row for row in math_rows}
    rows = [by_id[item_id] for item_id in sorted(ids)]
    return rows


def _prediction_identity(protocol: dict, phase: str, model_key: str, model_hashes: dict, adapter_hashes: dict | None) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "model_key": model_key,
        "model_revision": protocol["model"]["revision"],
        "model_files_sha256": model_hashes,
        "adapter_files_sha256": adapter_hashes,
        "prompt_instruction": protocol["prompt_contract"]["instruction"],
        "generation": protocol["evaluation"]["generation"],
    }


def _read_existing_predictions(path: Path, sidecar: Path, identity: dict, expected_ids: set[str]) -> set[str]:
    if not path.exists() and not sidecar.exists():
        return set()
    if not path.exists() or not sidecar.exists():
        raise RuntimeError(f"incomplete prediction artifact pair for {identity['model_key']}")
    stored_identity = json.loads(sidecar.read_text(encoding="utf-8"))
    if stored_identity.get("identity") != identity:
        raise RuntimeError(f"prediction resume identity mismatch for {identity['model_key']}")
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        item_id = item.get("unique_id")
        if item_id not in expected_ids or item_id in seen:
            raise RuntimeError(f"invalid/duplicate prediction ID on line {line_number}: {item_id}")
        seen.add(item_id)
    return seen


def _model_keys(protocol: dict) -> list[str]:
    return ["base"] + [f"{arm}-{seed}" for seed in protocol["training"]["seeds"] for arm in ("sft", "grpo")]


def _adapter_manifest(output_dir: Path, protocol: dict) -> dict[str, dict[str, str]]:
    output = {}
    expected_names = protocol["adapter_manifest"]["required_files"]
    for key in _model_keys(protocol):
        if key == "base":
            continue
        adapter_dir = output_dir / "candidates" / key / "adapter"
        manifest = {}
        for name in expected_names:
            path = adapter_dir / name
            if not path.is_file():
                raise SystemExit(f"NO-GO: missing {key} adapter artifact {name}")
            manifest[name] = sha256_file(path)
        expected = json.loads((output_dir / "candidates" / key / "train.json").read_text(encoding="utf-8"))
        if manifest != expected["adapter_file_sha256"]:
            raise SystemExit(f"NO-GO: {key} adapter differs from its completed training record")
        output[key] = manifest
    return output


def _append_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _evaluate_model(
    *,
    model_dir: Path,
    adapter_dir: Path | None,
    tokenizer,
    rows: list[dict],
    phase: str,
    model_key: str,
    output_dir: Path,
    identity: dict,
    protocol: dict,
) -> dict:
    expected_ids = {row["unique_id"] for row in rows}
    path = output_dir / "predictions" / f"{phase}-{model_key}.jsonl"
    sidecar = path.with_suffix(".identity.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = _read_existing_predictions(path, sidecar, identity, expected_ids)
    if not sidecar.exists():
        _write_json(sidecar, {"identity": identity})
    if complete == expected_ids:
        prior = list(_read_predictions(path).values())
        return {
            "path": str(path), "sha256": sha256_file(path), "prediction_count": len(complete),
            "generated_token_ids": sum(int(row["generated_token_count_through_first_eos"]) for row in prior),
            "resumed": True,
        }

    pending = [row for row in rows if row["unique_id"] not in complete]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.reset_peak_memory_stats(0)
    load_start = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda:0")
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=True, is_trainable=False)
    model.eval()
    reserved_limit = int(torch.cuda.get_device_properties(0).total_memory * 0.90)
    if torch.cuda.max_memory_reserved(0) >= reserved_limit:
        raise RuntimeError(f"{phase}/{model_key} exceeded the locked VRAM ceiling during model load")
    load_seconds = time.monotonic() - load_start
    started = time.monotonic()
    generation = protocol["evaluation"]["generation"]
    cfg = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=generation["max_new_tokens"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=generation["repetition_penalty"],
    )
    batch_size = generation["batch_size"]
    max_prompt = 0
    for offset in range(0, len(pending), batch_size):
        batch_rows = pending[offset : offset + batch_size]
        encodings = [tokenize_generation_prompt(tokenizer, row["problem"]) for row in batch_rows]
        prompt_lens = [len(item["input_ids"]) for item in encodings]
        if max(prompt_lens) > generation["max_input_tokens"]:
            raise RuntimeError(f"{phase}/{model_key} question exceeds the locked context limit")
        max_prompt = max(max_prompt, *prompt_lens)
        encoded = tokenizer.pad(encodings, padding=True, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            sequences = model.generate(**encoded, generation_config=cfg)
        completions = sequences[:, encoded["input_ids"].shape[1] :]
        batch_records = []
        for row, token_ids in zip(batch_rows, completions, strict=True):
            ids = [int(value) for value in token_ids.tolist()]
            count = 0
            for value in ids:
                count += 1
                if tokenizer.eos_token_id is not None and value == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(ids, skip_special_tokens=True)
            success = exact_match(text, row["answer"])
            valid = has_valid_final_box(text)
            batch_records.append({
                "unique_id": row["unique_id"],
                "level": int(row["level"]),
                "problem": row["problem"],
                "reference_answer": row["answer"],
                "completion": text,
                "completion_token_ids": ids,
                "generated_token_count_through_first_eos": count,
                "exact_match": bool(success),
                "valid_final_box": bool(valid),
                "hit_generation_cap": (
                    len(ids) >= generation["max_new_tokens"]
                    and (tokenizer.eos_token_id is None or tokenizer.eos_token_id not in ids)
                ),
            })
        _append_jsonl(path, batch_records)
        complete.update(row["unique_id"] for row in batch_records)
        if torch.cuda.max_memory_reserved(0) >= reserved_limit:
            raise RuntimeError(f"{phase}/{model_key} exceeded the locked VRAM ceiling during generation")
        del encoded, sequences, completions

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    peak_allocated = torch.cuda.max_memory_allocated(0)
    peak_reserved = torch.cuda.max_memory_reserved(0)
    if peak_reserved >= reserved_limit:
        raise RuntimeError(f"{phase}/{model_key} exceeded the locked 90% reserved VRAM ceiling")
    if len(complete) != len(expected_ids):
        raise RuntimeError(f"{phase}/{model_key} produced an incomplete prediction set")
    all_records = list(_read_predictions(path).values())
    result = {
        "path": str(path),
        "sha256": sha256_file(path),
        "prediction_count": len(complete),
        "generated_token_ids": sum(int(row["generated_token_count_through_first_eos"]) for row in all_records),
        "max_prompt_tokens": max_prompt,
        "model_load_seconds": load_seconds,
        "generation_seconds": elapsed,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "resumed": len(complete) > len(pending),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _read_predictions(path: Path) -> dict[str, dict]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            output[record["unique_id"]] = record
    return output


def _model_metrics(path: Path, expected_count: int) -> dict:
    rows = list(_read_predictions(path).values())
    if len(rows) != expected_count:
        raise RuntimeError(f"{path} has {len(rows)} scored rows; expected {expected_count}")
    by_level = {}
    for level in sorted({int(row["level"]) for row in rows}):
        selected = [row for row in rows if int(row["level"]) == level]
        by_level[str(level)] = {
            "count": len(selected),
            "exact_match": _mean([float(row["exact_match"]) for row in selected]),
            "valid_final_box": _mean([float(row["valid_final_box"]) for row in selected]),
        }
    return {
        "count": len(rows),
        "exact_match": _mean([float(row["exact_match"]) for row in rows]),
        "valid_final_box": _mean([float(row["valid_final_box"]) for row in rows]),
        "mean_generated_token_count": _mean([float(row["generated_token_count_through_first_eos"]) for row in rows]),
        "hit_generation_cap_count": sum(bool(row["hit_generation_cap"]) for row in rows),
        "by_level": by_level,
        "predictions_sha256": sha256_file(path),
    }


def _score_phase(args, protocol: dict, code: dict, runtime: dict, model_hashes: dict, phase: str) -> dict:
    gate_path = args.output / "base-gate.json"
    smoke_path = args.output / "smoke" / "smoke.json"
    training_path = args.output / "training-summary.json"
    if not gate_path.is_file() or not smoke_path.is_file() or not training_path.is_file():
        raise SystemExit("NO-GO: a passing gate, passing smoke, and complete training bundle are required")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if gate.get("decision") != "pass" or smoke.get("decision") != "pass" or training.get("status") != "complete":
        raise SystemExit("NO-GO: an upstream experiment phase did not pass its frozen integrity gate")
    for upstream in (gate, smoke, training):
        lock_digest = upstream.get("protocol_lock_sha256")
        if lock_digest is None:
            lock_digest = upstream.get("identity", {}).get("protocol_lock_sha256")
        if lock_digest != sha256_file(LOCK_PATH):
            raise SystemExit("NO-GO: an upstream phase used a different protocol lock")
        source_identity = upstream.get("execution_code")
        if source_identity is None:
            source_identity = upstream.get("identity", {}).get("execution_code")
        if source_identity != code:
            raise SystemExit("NO-GO: an upstream phase used different execution source files")
        upstream_runtime = upstream.get("runtime")
        if upstream_runtime is None:
            upstream_runtime = upstream.get("resources", {}).get("runtime")
        if upstream_runtime is None:
            upstream_runtime = upstream.get("identity", {}).get("runtime")
        if upstream_runtime != runtime:
            raise SystemExit("NO-GO: an upstream phase used a different runtime")
    if not training.get("all_ten_adapters_integrity_checked"):
        raise SystemExit("NO-GO: trained adapter integrity was not completed")
    if phase == "confirmation":
        dev_summary = args.output / "development-summary.json"
        if not dev_summary.is_file():
            raise SystemExit("NO-GO: confirmation is still sealed until the development gate passes")
        dev = json.loads(dev_summary.read_text(encoding="utf-8"))
        if dev.get("decision") != "advance_to_confirmation":
            raise SystemExit("NO-GO: frozen development advancement criteria did not pass")

    data_rows = _load_data(args.data_dir, protocol, phase)
    split_lock = protocol["evaluation"]["splits"][phase]
    if len(data_rows) != split_lock["count"]:
        raise SystemExit("NO-GO: evaluation split size mismatch")
    adapter_hashes = _adapter_manifest(args.output, protocol)
    identity_root = {
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_sha256": sha256_file(LOCK_PATH),
        "execution_code": code,
        "runtime": runtime,
        "model_files_sha256": model_hashes,
        "phase": phase,
        "split_ids_sha256": split_lock["ids_sha256"],
        "confirmation_opened": phase == "confirmation",
        "candidates_sha256": {
            key: hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
            for key, value in adapter_hashes.items()
        },
    }
    cap = protocol["evaluation"]["budgets"][phase]
    progress_path = args.output / f"{phase}-progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity_root:
            raise SystemExit(f"NO-GO: {phase} resume identity differs from original run")
        phase_started_epoch = float(progress["phase_started_epoch"])
        run_records = dict(progress.get("completed_models", {}))
    else:
        phase_started_epoch = time.time()
        run_records = {}
        _write_json(progress_path, {
            "identity": identity_root,
            "phase_started_epoch": phase_started_epoch,
            "completed_models": run_records,
            "generated_token_ids": 0,
        })
    all_generated_tokens = 0
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    generation = protocol["evaluation"]["generation"]
    max_eval_prompt_tokens = 0
    for row in data_rows:
        encoded_prompt = tokenize_generation_prompt(tokenizer, row["problem"])
        max_eval_prompt_tokens = max(max_eval_prompt_tokens, len(encoded_prompt["input_ids"]))
    if max_eval_prompt_tokens > generation["max_input_tokens"]:
        raise SystemExit(f"NO-GO: a {phase} prompt exceeds the locked input-token ceiling")
    keys = _model_keys(protocol)
    for key in keys:
        if time.time() - phase_started_epoch > cap["max_wall_seconds"]:
            raise SystemExit(f"NO-GO: {phase} exceeded its locked cumulative wall-time budget")
        adapter_dir = None if key == "base" else args.output / "candidates" / key / "adapter"
        identity = _prediction_identity(protocol, phase, key, model_hashes, adapter_hashes.get(key))
        record = _evaluate_model(
            model_dir=args.model_dir,
            adapter_dir=adapter_dir,
            tokenizer=tokenizer,
            rows=data_rows,
            phase=phase,
            model_key=key,
            output_dir=args.output,
            identity=identity,
            protocol=protocol,
        )
        run_records[key] = record
        all_generated_tokens = sum(int(item.get("generated_token_ids", 0)) for item in run_records.values())
        if all_generated_tokens > cap["max_generated_tokens"]:
            raise SystemExit(f"NO-GO: {phase} exceeded its locked generated-token ceiling")
        _write_json(progress_path, {
            "identity": identity_root,
            "phase_started_epoch": phase_started_epoch,
            "completed_models": run_records,
            "generated_token_ids": all_generated_tokens,
        })
    if time.time() - phase_started_epoch > cap["max_wall_seconds"]:
        raise SystemExit(f"NO-GO: {phase} exceeded its locked cumulative wall-time budget")

    metrics = {
        key: _model_metrics(args.output / "predictions" / f"{phase}-{key}.jsonl", split_lock["count"])
        for key in keys
    }
    seeds = [int(value) for value in protocol["training"]["seeds"]]
    sft_metrics = [metrics[f"sft-{seed}"] for seed in seeds]
    grpo_metrics = [metrics[f"grpo-{seed}"] for seed in seeds]
    if phase == "development":
        deltas = [grpo_metrics[i]["exact_match"] - sft_metrics[i]["exact_match"] for i in range(len(seeds))]
        mean_grpo = _mean([item["exact_match"] for item in grpo_metrics])
        mean_sft = _mean([item["exact_match"] for item in sft_metrics])
        format_by_seed = [item["valid_final_box"] for item in grpo_metrics]
        rules = {
            "mean_grpo_not_below_base": mean_grpo >= metrics["base"]["exact_match"],
            "mean_grpo_not_below_sft": mean_grpo >= mean_sft,
            "at_least_three_of_five_paired_seeds_nonnegative_vs_sft": sum(delta >= 0 for delta in deltas) >= 3,
            "every_grpo_seed_format_at_least_95_percent": all(value >= 0.95 for value in format_by_seed),
        }
        decision = "advance_to_confirmation" if all(rules.values()) else "stop_after_development"
        summary = {
            "protocol_id": PROTOCOL_ID,
            "phase": phase,
            "decision": decision,
            "frozen_advancement_rules": rules,
            "metrics": metrics,
            "paired_grpo_minus_sft_by_seed": dict(zip(seeds, deltas, strict=True)),
            "mean_grpo_exact_match": mean_grpo,
            "mean_sft_exact_match": mean_sft,
            "confirmation_opened": False,
            "run_records": run_records,
            "run_identity": identity_root,
            "elapsed_seconds": time.time() - phase_started_epoch,
        }
    else:
        grpo_rows = [_read_predictions(args.output / "predictions" / f"confirmation-grpo-{seed}.jsonl") for seed in seeds]
        sft_rows = [_read_predictions(args.output / "predictions" / f"confirmation-sft-{seed}.jsonl") for seed in seeds]
        ordered_ids = [row["unique_id"] for row in data_rows]
        grpo_matrix = [[float(grpo_rows[i][item_id]["exact_match"]) for item_id in ordered_ids] for i in range(len(seeds))]
        sft_matrix = [[float(sft_rows[i][item_id]["exact_match"]) for item_id in ordered_ids] for i in range(len(seeds))]
        uncertainty = paired_task_seed_bootstrap(
            grpo_matrix,
            sft_matrix,
            seed=protocol["evaluation"]["statistics"]["bootstrap_seed"],
            draws=protocol["evaluation"]["statistics"]["bootstrap_draws"],
        )
        per_level = {}
        for level in sorted({int(row["level"]) for row in data_rows}):
            level_ids = [row["unique_id"] for row in data_rows if int(row["level"]) == level]
            level_grpo = [[float(grpo_rows[i][item_id]["exact_match"]) for item_id in level_ids] for i in range(len(seeds))]
            level_sft = [[float(sft_rows[i][item_id]["exact_match"]) for item_id in level_ids] for i in range(len(seeds))]
            per_level[str(level)] = {
                "count": len(level_ids),
                "mean_grpo_minus_sft": _mean([
                    _mean(level_grpo[i]) - _mean(level_sft[i]) for i in range(len(seeds))
                ]),
            }
        rules = {
            "mean_grpo_minus_sft_at_least_2_percentage_points": uncertainty["mean_candidate_minus_control"] >= 0.02,
            "hierarchical_95_percent_interval_lower_above_zero": uncertainty["hierarchical_task_seed_95_percentile_interval"][0] > 0,
            "at_least_four_of_five_seeds_positive": sum(value > 0 for value in uncertainty["seed_deltas"]) >= 4,
            "no_level_regresses_more_than_5_percentage_points": all(value["mean_grpo_minus_sft"] >= -0.05 for value in per_level.values()),
            "every_grpo_seed_format_at_least_95_percent": all(item["valid_final_box"] >= 0.95 for item in grpo_metrics),
        }
        decision = "confirmation_rule_passed" if all(rules.values()) else "confirmation_rule_not_passed"
        summary = {
            "protocol_id": PROTOCOL_ID,
            "phase": phase,
            "decision": decision,
            "frozen_confirmation_rules": rules,
            "metrics": metrics,
            "paired_uncertainty": uncertainty,
            "per_level_grpo_minus_sft": per_level,
            "confirmation_opened": True,
            "run_records": run_records,
            "run_identity": identity_root,
            "elapsed_seconds": time.time() - phase_started_epoch,
        }
    output_path = args.output / f"{phase}-summary.json"
    _write_json(output_path, summary)
    print(json.dumps({"phase": phase, "decision": decision, "summary": str(output_path), "confirmation_opened": phase == "confirmation"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["development", "confirmation"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise SystemExit("NO-GO: protocol identity mismatch")
    code = _code_identity(protocol)
    runtime = _runtime(protocol)
    model_hashes = _verify_model(args.model_dir, protocol)
    _score_phase(args, protocol, code, runtime, model_hashes, args.phase)


if __name__ == "__main__":
    main()
