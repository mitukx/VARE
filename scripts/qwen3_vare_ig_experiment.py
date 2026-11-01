#!/usr/bin/env python3
"""Prospective VARE-IG pilot built on the existing Qwen3 MLX GRPO runner.

This entry point swaps only the task contract module and task-selection hook.
Model loading, exact-ratio sampling, trusted isolated grading, GRPO updates,
checkpointing, and recovery are delegated to
``scripts/qwen3_function_repair_grpo.py``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vare import grpo_mlx  # noqa: E402
from vare.config import PromotionConfig  # noqa: E402
from vare.learning_pipeline import informative_curriculum as curriculum  # noqa: E402
from vare.learning_pipeline import vare_ig_tasks as task_pack  # noqa: E402
from vare.learning_pipeline.verified_rollout_replay import (  # noqa: E402
    VerifiedGroupReplay,
    acknowledge_durable_decision,
)
from vare.mlx_prefix_reuse import build_shared_prefix, generate_profiled, CachedGroupGenerator  # noqa: E402
from vare.promotion import PromotionGate  # noqa: E402
from vare.types import EvaluationReport  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402


MASTER_PROTOCOL = ROOT / "protocols/vare_ig_qwen3_mlx_dev_v1.json"
TRAIN_TASK_HASHES = {task.task_id: task.sha256() for task in task_pack.TRAIN_TASKS}
EXPLORATION_TASK_HASHES = {task.task_id: task.sha256() for task in task_pack.EXPLORATION_TASKS}
DEVELOPMENT_TASK_HASHES = {task.task_id: task.sha256() for task in task_pack.DEVELOPMENT_TASKS}
TASK_HASHES_BY_SPLIT = {
    "train": TRAIN_TASK_HASHES,
    "exploration": EXPLORATION_TASK_HASHES,
    "development": DEVELOPMENT_TASK_HASHES,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _case_wall_time_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    """Sum measured verifier case times and count missing/invalid measurements."""
    measured_seconds = 0.0
    measured_count = 0
    unmeasured_count = 0
    for row in rows:
        cases = row.get("case_results", [])
        if not isinstance(cases, list):
            unmeasured_count += 1
            continue
        for case in cases:
            raw = case.get("wall_s") if isinstance(case, Mapping) else None
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                unmeasured_count += 1
                continue
            try:
                numeric = float(raw)
            except (OverflowError, TypeError, ValueError):
                unmeasured_count += 1
                continue
            if not math.isfinite(numeric) or numeric < 0:
                unmeasured_count += 1
                continue
            measured_seconds += numeric
            measured_count += 1
    return {
        "measured_seconds": measured_seconds,
        "measured_case_count": measured_count,
        "unmeasured_case_count": unmeasured_count,
    }


def source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "scripts/qwen3_vare_ig_experiment.py",
        ROOT / "scripts/freeze_vare_ig_protocol.py",
        ROOT / "scripts/audit_vare_ig_task_oracles.py",
        ROOT / "scripts/qwen3_function_repair_grpo.py",
        ROOT / "src/vare/learning_pipeline/vare_ig_tasks.py",
        ROOT / "src/vare/learning_pipeline/informative_curriculum.py",
        ROOT / "src/vare/learning_pipeline/verified_rollout_replay.py",
        ROOT / "src/vare/mlx_prefix_reuse.py",
        ROOT / "src/vare/grpo_mlx.py",
        ROOT / "src/vare/trajectory_store.py",
        ROOT / "src/vare/replay.py",
        ROOT / "src/vare/types.py",
        ROOT / "src/vare/config.py",
        ROOT / "src/vare/promotion.py",
        ROOT / "src/vare/function_repair_tasks.py",
        ROOT / "src/vare/isolated_candidate.py",
        ROOT / "src/vare/container_isolation.py",
    )
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def _concatenate_staged_rollouts(staged_paths: Sequence[Path], destination_path: Path) -> None:
    """Concatenate staged JSONL records byte-for-byte into a new file."""
    with destination_path.open("xb") as destination:
        for staged_path in staged_paths:
            with staged_path.open("rb") as source:
                shutil.copyfileobj(source, destination)
        destination.flush()
        os.fsync(destination.fileno())


def task_hashes() -> dict[str, dict[str, str]]:
    return {split: dict(hashes) for split, hashes in TASK_HASHES_BY_SPLIT.items()}


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def validate_protocol(path: Path, *, run_seed: int | None = None, require_arm: str | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen VARE-IG protocol is required before model execution: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("preregistered_before_model_inference"):
        raise RuntimeError("VARE-IG requires a protocol frozen before model inference")
    if payload.get("source_hashes") != runner.source_hashes():
        raise RuntimeError("existing runner source hashes differ from the frozen protocol")
    if payload.get("vare_ig_source_hashes") != source_hashes():
        raise RuntimeError("VARE-IG source hashes differ from the frozen protocol")
    if payload.get("task_contract_hashes") != task_hashes():
        raise RuntimeError("VARE-IG task hashes differ from the frozen protocol")
    if run_seed is not None and payload.get("sampling", {}).get("seed") != run_seed:
        raise RuntimeError("CLI seed differs from the frozen VARE-IG protocol")
    if require_arm is not None and payload.get("selection", {}).get("arm") != require_arm:
        raise RuntimeError("CLI scheduler arm differs from the frozen run protocol")
    if payload.get("model", {}).get("id") != runner.MODEL_ID or payload.get("model", {}).get("revision") != runner.MODEL_REVISION:
        raise RuntimeError("VARE-IG model identity differs from the pinned runner")
    if payload.get("learner", {}).get("config") != runner.GRPO_CONFIG.to_dict():
        raise RuntimeError("VARE-IG GRPO config differs from the existing runner")
    if payload.get("reward", {}).get("container_image") != runner.CONTAINER_IMAGE:
        raise RuntimeError("VARE-IG grader image differs from the pinned runner")
    return payload


def run_prefix_cache_benchmark(args: argparse.Namespace) -> int:
    protocol = validate_protocol(args.protocol, run_seed=args.seed)
    _install_existing_runner_contract(args.protocol)
    config = protocol["inference_benchmark"]
    if config.get("task_ids") != [task.task_id for task in task_pack.EXPLORATION_TASKS
                                  if task.task_id in set(config["task_ids"])]:
        raise RuntimeError("prefix benchmark tasks must be the locked Exploration subset in protocol order")
    by_id = {task.task_id: task for task in task_pack.EXPLORATION_TASKS}
    run = args.run_dir.expanduser().resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"refusing to overwrite prefix-cache evidence: {run}")
    run.mkdir(parents=True, exist_ok=True)
    preflight = runner.preflight(run, "vare-ig-prefix-cache")
    phase_started = time.monotonic()
    import mlx.core as mx

    mx.reset_peak_memory()
    load_started = time.perf_counter()
    model, tokenizer = runner.load_policy(seed=args.seed)
    model_load_seconds = time.perf_counter() - load_started
    policy_id = runner.make_policy_id(model, 0)
    policy_sha256 = grpo_mlx.trainable_digest(model)
    rows: list[dict[str, Any]] = []
    results_path = run / "prefix-cache-pairs.jsonl"
    for task_id in config["task_ids"]:
        task = by_id[task_id]
        messages = runner.task_prompt(task)
        tokenization_started = time.perf_counter()
        prompt_ids = [int(token) for token in tokenizer.apply_chat_template(
            list(messages), add_generation_prompt=True, tokenize=True,
        )]
        shared_tokenization_seconds = time.perf_counter() - tokenization_started
        prefix = build_shared_prefix(model, prompt_ids)
        independent_results = []
        cached_results = []
        task_rows = []
        for sample_index in range(int(config["samples_per_task"])):
            runner.resource_guard(run, phase_started)
            seed = runner.seed_for(args.seed, "vare-ig-prefix-cache-v1", task_id, sample_index)
            order = ("independent", "cached") if sample_index % 2 == 0 else ("cached", "independent")
            pair_results = {}
            for mode in order:
                if mode == "independent":
                    generated = generate_profiled(
                        model, tokenizer, messages=messages, seed=seed,
                        temperature=float(config["temperature"]),
                        max_tokens=int(config["max_new_tokens"]),
                    )
                    independent_results.append(generated)
                else:
                    generated = generate_profiled(
                        model, tokenizer, messages=messages, seed=seed,
                        temperature=float(config["temperature"]),
                        max_tokens=int(config["max_new_tokens"]),
                        prompt_ids=prompt_ids, shared_prefix=prefix,
                    )
                    cached_results.append(generated)
                pair_results[mode] = generated
            independent = pair_results["independent"]
            cached = pair_results["cached"]
            aligned = len(independent["sampler_logprobs"]) == len(cached["sampler_logprobs"])
            logprobs_equal = aligned and all(
                math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)
                for left, right in zip(independent["sampler_logprobs"], cached["sampler_logprobs"], strict=True)
            )
            outputs_equal = (
                independent["completion_ids"] == cached["completion_ids"]
                and independent["text"] == cached["text"]
                and independent["finish"] == cached["finish"]
                and logprobs_equal
            )
            task_rows.append({
                "task_id": task_id,
                "task_sha256": task.sha256(),
                "prompt_tokens": len(prompt_ids),
                "sample_index": sample_index,
                "seed": seed,
                "trial_order": list(order),
                "policy_checkpoint_id": policy_id,
                "policy_checkpoint_sha256": policy_sha256,
                "independent": independent,
                "cached": cached,
                "token_ids_equal": independent["completion_ids"] == cached["completion_ids"],
                "text_equal": independent["text"] == cached["text"],
                "sampler_logprobs_equal_within_tolerance": logprobs_equal,
                "outputs_and_probabilities_equal": outputs_equal,
                "independent_response_sha256": hashlib.sha256(independent["text"].encode()).hexdigest(),
                "cached_response_sha256": hashlib.sha256(cached["text"].encode()).hexdigest(),
            })
            runner.append_jsonl(results_path, task_rows[-1])
            rows.append(task_rows[-1])
            mx.clear_cache()
            runner.resource_guard(run, phase_started)

        independent_group_seconds = sum(
            float(item["tokenization_seconds"]) + float(item["generation_seconds"])
            for item in independent_results
        )
        cached_group_seconds = (
            shared_tokenization_seconds + float(prefix["prefill_seconds"])
            + sum(float(item["generation_seconds"]) + float(item["cache_clone_seconds"])
                  for item in cached_results)
        )
        for row in task_rows:
            row["group_independent_seconds"] = independent_group_seconds
            row["group_cached_seconds"] = cached_group_seconds
            row["group_speedup_independent_over_cached"] = independent_group_seconds / cached_group_seconds
            row["shared_prompt_tokenization_seconds"] = shared_tokenization_seconds
            row["shared_prefix_prefill_seconds"] = float(prefix["prefill_seconds"])
            row["shared_prefix_cache_bytes"] = int(prefix["cache_bytes"])
        # Rows are appended above for crash durability; rewrite a separate
        # summary because the per-group aggregate becomes known after all pairs.
        runner.atomic_json(run / f"prefix-cache-group-{task_id.replace('.', '_')}.json", {
            "task_id": task_id,
            "task_sha256": task.sha256(),
            "prompt_tokens": len(prompt_ids),
            "samples_per_mode": len(task_rows),
            "shared_prompt_tokenization_seconds": shared_tokenization_seconds,
            "shared_prefix_prefill_seconds": float(prefix["prefill_seconds"]),
            "shared_prefix_cache_bytes": int(prefix["cache_bytes"]),
            "independent_group_seconds": independent_group_seconds,
            "cached_group_seconds": cached_group_seconds,
            "group_speedup_independent_over_cached": independent_group_seconds / cached_group_seconds,
            "outputs_and_probabilities_equal": all(row["outputs_and_probabilities_equal"] for row in task_rows),
            "base_cache_cursor_unchanged": True,
            "completion_tokens": [len(item["completion_ids"]) for item in independent_results],
            "first_token_measurements": {
                "independent_prompt_prefill_and_first_sample_seconds": [
                    item["prompt_prefill_and_first_sample_seconds"] for item in independent_results
                ],
                "cached_tail_prefill_and_first_sample_seconds": [
                    item["prompt_prefill_and_first_sample_seconds"] for item in cached_results
                ],
                "independent_first_yield_seconds": [item["first_yield_seconds"] for item in independent_results],
                "cached_first_yield_seconds": [item["first_yield_seconds"] for item in cached_results],
            },
            "decode_tokens_per_second": {
                "independent": [item["generation_tokens_per_second"] for item in independent_results],
                "cached": [item["generation_tokens_per_second"] for item in cached_results],
            },
        })

    task_groups = [json.loads((run / f"prefix-cache-group-{task_id.replace('.', '_')}.json").read_text())
                   for task_id in config["task_ids"]]
    outputs_equal = all(row["outputs_and_probabilities_equal"] for row in rows)
    cursor_unchanged = all(group["base_cache_cursor_unchanged"] for group in task_groups)
    mean_speedup = sum(group["group_speedup_independent_over_cached"] for group in task_groups) / len(task_groups)
    minimum_group_speedup = float(config["training_enablement"]["minimum_mean_task_group_speedup"])
    summary = {
        "schema": "vare_ig_mlx_prefix_cache_benchmark_v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(args.protocol.expanduser().resolve()),
        "vare_ig_source_hashes": source_hashes(),
        "run_seed": args.seed,
        "model": {"id": runner.MODEL_ID, "revision": runner.MODEL_REVISION},
        "policy_checkpoint_id": policy_id,
        "policy_checkpoint_sha256": policy_sha256,
        "model_load_seconds": model_load_seconds,
        "preflight": preflight,
        "tasks": task_groups,
        "completion_pairs": len(rows),
        "independent_generations": len(rows),
        "cached_generations": len(rows),
        "outputs_and_probabilities_equal": outputs_equal,
        "cache_ownership_verified": cursor_unchanged,
        "mean_task_group_speedup": mean_speedup,
        "minimum_speedup_required_for_training_enablement": minimum_group_speedup,
        "training_enablement_criteria_passed": outputs_equal and cursor_unchanged
            and mean_speedup >= minimum_group_speedup,
        "profiled_completion_tokens_per_second": {
            "independent": [row["independent"]["generation_tokens_per_second"] for row in rows],
            "cached": [row["cached"]["generation_tokens_per_second"] for row in rows],
        },
        "resource_end": runner.resource_snapshot(run),
        "claim_boundary": "Two predeclared Exploration prompts; exact output and sampling-logprob check; task-group timing includes tokenization, shared prefill, cache cloning, and generation. This is a workload-specific measurement.",
    }
    runner.atomic_json(run / "prefix-cache-summary.json", summary)
    print(json.dumps({"prefix_cache_summary": str(run / "prefix-cache-summary.json"),
                      "pairs": len(rows), "outputs_equal": outputs_equal,
                      "mean_group_speedup": mean_speedup,
                      "training_enabled": summary["training_enablement_criteria_passed"]}, sort_keys=True), flush=True)
    del model, tokenizer
    return 0


def _install_existing_runner_contract(protocol_path: Path) -> None:
    """Use new tasks without changing the immutable old pilot module or lock."""
    lock_path = protocol_path.expanduser().resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("source_hashes") != runner.source_hashes() or \
            lock.get("vare_ig_source_hashes") != source_hashes() or \
            lock.get("task_contract_hashes") != task_hashes():
        raise RuntimeError("active VARE-IG lock no longer matches this task suite")
    runner.task_pack = task_pack
    # The legacy runner's protocol schema has train+development only. Keep its
    # validator intact and bind exploration hashes in this wrapper's lock.
    runner.task_hashes = lambda: {**TRAIN_TASK_HASHES, **DEVELOPMENT_TASK_HASHES}
    runner.DEFAULT_PROTOCOL = protocol_path
    if not getattr(runner, "_vare_ig_trajectory_ack_wrapped", False):
        original_atomic_json = runner.atomic_json

        def atomic_json_with_trajectory_ack(path: Path, value: Any) -> None:
            original_atomic_json(path, value)
            path = Path(path)
            update_step = None
            decision = None
            if path.name == "training-result.json" and value.get("update_applied") is False:
                update_step, decision = 0, value
            elif path.name == "learner-events.json":
                update_step = 1
                decision = {"update_applied": True, "effective_update_count": 1,
                            "event": "optimizer_update_committed"}
            elif path.name == "learner-events-reconciled.json":
                update_step = 1
                decision = {"update_applied": True, "effective_update_count": 1,
                            "event": value.get("event"),
                            "checkpoint_state_sha256": value.get("checkpoint_state_sha256")}
            elif path.name == "recovered-update-0002.json":
                update_step = 2
                decision = dict(value.get("update", {}))
            if update_step is not None and decision is not None:
                acknowledge_durable_decision(
                    path.parent, update_step=update_step,
                    acknowledgment_path=path, decision=decision,
                )

        runner.atomic_json = atomic_json_with_trajectory_ack
        runner._vare_ig_trajectory_ack_wrapped = True

    if not getattr(runner, "_vare_ig_restore_wrapped", False):
        original_restore = runner.restore_and_verify

        def restore_and_reconcile_ack(run: Path, *, seed: int, phase: str,
                                      protocol_path: Path = runner.DEFAULT_PROTOCOL):
            restored = original_restore(run, seed=seed, phase=phase, protocol_path=protocol_path)
            meta, _model, _optimizer, saved_state, _tokenizer = restored
            event_path = run / "learner-events.json"
            reconciled_path = run / "learner-events-reconciled.json"
            if not event_path.exists() and not reconciled_path.exists():
                checkpoint_path = runner.checkpoint_path(run, "update-0001")
                manifest_path = checkpoint_path / "checkpoint.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if int(manifest.get("step", -1)) == int(saved_state["update_step"]) == 1 and \
                        int(saved_state.get("optimizer_step", -1)) == 1:
                    runner.atomic_json(reconciled_path, {
                        "schema": "vare_ig_reconciled_learner_ack_v1",
                        "event": "optimizer_update_commit_ack_reconstructed_from_checkpoint",
                        "experiment_id": meta["experiment_id"],
                        "update_step": 1,
                        "checkpoint_manifest_sha256": saved_state["checkpoint_manifest_sha256"],
                        "checkpoint_state_sha256": sha256_file(run / "checkpoint-state.json"),
                        "parameter_sha256": saved_state["trainable_parameters_sha256"],
                        "optimizer_sha256": saved_state["optimizer_state_sha256"],
                        "source_boundary": "checkpoint durable; completion acknowledgment absent after SIGKILL",
                        "effective_update_count": 1,
                        "update_applied": True,
                        "duplicate_effective_update": False,
                        "restore_phase": phase,
                        "restore_checks": json.loads((run / f"recovery-restore-{phase}.json").read_text())["checks"],
                    })
            ack_path = event_path if event_path.exists() else reconciled_path
            if ack_path.exists():
                acknowledge_durable_decision(
                    run, update_step=1, acknowledgment_path=ack_path,
                    decision={"update_applied": True, "effective_update_count": 1,
                              "event": "optimizer_update_commit_ack_reconciled"},
                )
            return restored

        runner.restore_and_verify = restore_and_reconcile_ack
        runner._vare_ig_restore_wrapped = True


def _pilot_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=MASTER_PROTOCOL)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def _cache_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare independent MLX generation with shared-prefix reuse")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=MASTER_PROTOCOL)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def run_pilot(args: argparse.Namespace) -> int:
    protocol = validate_protocol(args.protocol, run_seed=args.seed)
    _install_existing_runner_contract(args.protocol)
    run = args.run_dir.expanduser().resolve()
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing pilot evidence: {run}")
    run.mkdir(parents=True, exist_ok=True)
    preflight = runner.preflight(run, "vare-ig-exploration")
    started = time.monotonic()
    model, tokenizer = runner.load_policy(seed=args.seed)
    policy_id = runner.make_policy_id(model, 0)
    runner.atomic_json(run / "pilot-meta.json", {
        "schema": "vare_ig_base_exploration_meta_v1",
        "experiment_id": protocol["planned_run_id"],
        "protocol_sha256": sha256_file(args.protocol.expanduser().resolve()),
        "starting_git_sha": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "vare_ig_source_hashes": source_hashes(),
        "task_hashes": task_hashes(),
        "seed": args.seed,
        "model": {"id": runner.MODEL_ID, "revision": runner.MODEL_REVISION},
        "policy_checkpoint_id": policy_id,
        "sampling": protocol["sampling"],
        "preflight": preflight,
    })
    rollouts_path = run / "exploration-rollouts.jsonl"
    if rollouts_path.exists():
        raise FileExistsError(rollouts_path)
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, list[float]]] = []
    for task in task_pack.EXPLORATION_TASKS:
        task_rewards = []
        for sample_index in range(int(protocol["sampling"]["samples_per_exploration_task"])):
            runner.resource_guard(run, started)
            sample_seed = runner.seed_for(args.seed, "vare-ig-exploration-v1", task.task_id, sample_index)
            generated = runner.generate(
                model, tokenizer, runner.task_prompt(task), seed=sample_seed,
                temperature=float(protocol["sampling"]["temperature"]),
                max_tokens=int(protocol["sampling"]["max_new_tokens"]),
            )
            score = runner.score_source(task, generated["text"])
            if generated["finish"] == "prompt_too_long":
                score = {"reward": 0.0, "grade_status": "prompt_too_long", "trusted_contract": True,
                         "passed": False, "cases": []}
            task_rewards.append(float(score["reward"]))
            row = {
                "schema": "vare_ig_exploration_rollout_v1",
                "task_id": task.task_id,
                "split": task.split,
                "task_family": task_pack.FAMILY_BY_EXPLORATION_TASK[task.task_id],
                "task_sha256": task.sha256(),
                "prompt_sha256": sha256_json(runner.task_prompt(task)),
                "policy_checkpoint_id": policy_id,
                "behavior_step": 0,
                "sample_index": sample_index,
                "seed": sample_seed,
                "temperature": protocol["sampling"]["temperature"],
                "top_p": protocol["sampling"]["top_p"],
                "top_k": protocol["sampling"]["top_k"],
                "sampling_distribution": "untruncated softmax(logits / temperature)",
                "prompt_tokens": len(generated["prompt_ids"]),
                "completion_tokens": len(generated["completion_ids"]),
                "generation_seconds": generated["seconds"],
                "finish": generated["finish"],
                "response_sha256": hashlib.sha256(generated["text"].encode()).hexdigest(),
                "candidate_source_sha256": score.get("candidate_source_sha256"),
                "reward": score["reward"],
                "grade_status": score["grade_status"],
                "trusted_contract": score["trusted_contract"],
                "verifier_id": runner.verifier_id(task),
                "case_results": score["cases"],
                "text": generated["text"],
            }
            with rollouts_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(row)
            runner.resource_guard(run, started)
        groups.append((task.task_id, task_rewards))

    family_observations: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    family_costs: dict[str, list[float]] = defaultdict(list)
    task_summaries = []
    for task in task_pack.EXPLORATION_TASKS:
        task_rows = [row for row in rows if row["task_id"] == task.task_id]
        rewards = [float(row["reward"]) for row in task_rows]
        family = task_pack.FAMILY_BY_EXPLORATION_TASK[task.task_id]
        family_observations[family].append((task.task_id, rewards))
        generation_s = sum(float(row["generation_seconds"]) for row in task_rows)
        verifier_time = _case_wall_time_summary(task_rows)
        verifier_s = float(verifier_time["measured_seconds"])
        group_cost_s = generation_s + verifier_s
        family_costs[family].append(group_cost_s)
        task_summaries.append({
            "task_id": task.task_id,
            "family": family,
            "task_sha256": task.sha256(),
            "n_completions": len(task_rows),
            "successes": sum(rewards),
            "pass_probability": sum(rewards) / len(rewards),
            "pass_probability_wilson_95": _wilson(int(sum(rewards)), len(rewards)),
            "rewards": rewards,
            "informative_group": len(set(rewards)) > 1,
            "unique_candidate_sources": len({row["candidate_source_sha256"] for row in task_rows
                                              if row["candidate_source_sha256"]}),
            "unique_responses": len({row["response_sha256"] for row in task_rows}),
            "completion_tokens": [row["completion_tokens"] for row in task_rows],
            "finish_reasons": dict(Counter(row["finish"] for row in task_rows)),
            "generation_seconds": generation_s,
            "verification_seconds": verifier_s,
            "measured_verifier_case_count": verifier_time["measured_case_count"],
            "unmeasured_verifier_case_count": verifier_time["unmeasured_case_count"],
            "generation_and_verification_seconds": group_cost_s,
            "failure_categories": dict(Counter(
                case["failure"] for row in task_rows for case in row["case_results"] if not case["passed"]
            )),
        })
    rates = curriculum.family_group_rates(groups, task_pack.FAMILY_BY_EXPLORATION_TASK)
    family_summaries = {}
    for family, rate in sorted(rates.items()):
        family_passes = sum(sum(outcomes) for _, outcomes in family_observations[family])
        family_completions = sum(len(outcomes) for _, outcomes in family_observations[family])
        pass_probability = family_passes / family_completions
        family_summaries[family] = {
            "exploration_task_ids": [task_id for task_id, _ in family_observations[family]],
            "observed_groups": rate.observed_groups,
            "informative_groups": rate.informative_groups,
            "observed_group_informativeness_rate": rate.informative_groups / rate.observed_groups,
            "beta_alpha": rate.alpha,
            "beta_beta": rate.beta,
            "scheduler_expected_informative_group_probability": rate.expected_informative_group_probability,
            "completions": family_completions,
            "successes": family_passes,
            "pass_probability": pass_probability,
            "pass_probability_wilson_95": _wilson(int(family_passes), family_completions),
            "iid_bernoulli_predicted_informative_probability":
                curriculum.independent_bernoulli_informative_probability(
                    pass_probability, int(protocol["sampling"]["group_size"])
                ),
            "mean_generation_and_verification_seconds_per_group":
                sum(family_costs[family]) / len(family_costs[family]),
            "mean_completion_tokens_per_group": sum(
                sum(row["completion_tokens"] for row in rows if row["task_id"] == task_id)
                for task_id, _ in family_observations[family]
            ) / len(family_observations[family]),
        }
    overall_verifier_time = _case_wall_time_summary(rows)
    result = {
        "schema": "vare_ig_exploration_summary_v1",
        "protocol_id": protocol["planned_run_id"],
        "protocol_sha256": sha256_file(args.protocol.expanduser().resolve()),
        "vare_ig_source_hashes": source_hashes(),
        "policy_checkpoint_id": policy_id,
        "model": {"id": runner.MODEL_ID, "revision": runner.MODEL_REVISION},
        "group_size": int(protocol["sampling"]["group_size"]),
        "task_ids": [task.task_id for task in task_pack.EXPLORATION_TASKS],
        "task_summaries": task_summaries,
        "family_summaries": family_summaries,
        "overall": {
            "groups": len(groups),
            "informative_groups": sum(len(set(rewards)) > 1 for _, rewards in groups),
            "completions": len(rows),
            "successes": sum(float(row["reward"]) for row in rows),
            "unique_candidate_sources": len({row["candidate_source_sha256"] for row in rows
                                              if row["candidate_source_sha256"]}),
            "unique_responses": len({row["response_sha256"] for row in rows}),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in rows),
            "generation_seconds": sum(float(row["generation_seconds"]) for row in rows),
            "verification_seconds": overall_verifier_time["measured_seconds"],
            "measured_verifier_case_count": overall_verifier_time["measured_case_count"],
            "unmeasured_verifier_case_count": overall_verifier_time["unmeasured_case_count"],
            "wall_seconds": round(time.monotonic() - started, 6),
            "finish_reasons": dict(Counter(row["finish"] for row in rows)),
            "failure_categories": dict(Counter(
                case["failure"] for row in rows for case in row["case_results"] if not case["passed"]
            )),
        },
        "claim_boundary": "Exploration-only base-policy estimates; no final evaluation or learning claim.",
        "resource_end": runner.resource_snapshot(run),
    }
    runner.atomic_json(run / "exploration-summary.json", result)
    print(json.dumps({"exploration_summary": str(run / "exploration-summary.json"),
                      "groups": result["overall"]["groups"],
                      "informative_groups": result["overall"]["informative_groups"],
                      "completions": result["overall"]["completions"]}, sort_keys=True), flush=True)
    del model, tokenizer
    return 0


def install_scheduled_training(*, arm: str, prior_path: Path, cache_benchmark_path: Path,
                               groups: int, run_seed: int,
                               protocol: Mapping[str, Any]) -> None:
    if arm not in {"uniform", "informative_per_cost"}:
        raise ValueError("arm must be uniform or informative_per_cost")
    prior_path = prior_path.expanduser().resolve()
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    prior_sha = sha256_file(prior_path)
    cache_benchmark_path = cache_benchmark_path.expanduser().resolve()
    cache_benchmark = json.loads(cache_benchmark_path.read_text(encoding="utf-8"))
    cache_benchmark_sha = sha256_file(cache_benchmark_path)
    selection = protocol.get("selection", {})
    if selection.get("arm") != arm or selection.get("group_budget") != groups:
        raise RuntimeError("selection arm or group budget differs from the frozen run protocol")
    if selection.get("exploration_summary_sha256") != prior_sha:
        raise RuntimeError("exploration summary differs from the frozen training prior")
    if selection.get("prefix_cache_benchmark_sha256") != cache_benchmark_sha:
        raise RuntimeError("prefix-cache benchmark differs from the frozen training protocol")
    cache_enabled = bool(cache_benchmark.get("training_enablement_criteria_passed"))
    if cache_enabled and not (
        cache_benchmark.get("outputs_and_probabilities_equal")
        and cache_benchmark.get("cache_ownership_verified")
    ):
        raise RuntimeError("prefix-cache training enablement lacks exact-output and cache-ownership evidence")
    if prior.get("task_ids") != [task.task_id for task in task_pack.EXPLORATION_TASKS]:
        raise RuntimeError("scheduler prior must contain exactly the locked exploration tasks")
    family_rates = {
        family: curriculum.GroupRate(
            informative_groups=int(summary["informative_groups"]),
            observed_groups=int(summary["observed_groups"]),
            successes=int(summary["successes"]),
            completions=int(summary["completions"]),
        )
        for family, summary in prior["family_summaries"].items()
    }
    family_costs = {
        family: float(summary["mean_generation_and_verification_seconds_per_group"])
        for family, summary in prior["family_summaries"].items()
    }
    if set(family_rates) != set(task_pack.FAMILY_BY_TASK.values()):
        raise RuntimeError("scheduler prior family set differs from Train task families")
    recovery_ids = protocol.get("recovery", {}).get("task_ids", [])
    if len(recovery_ids) != 3 or set(recovery_ids) - set(TRAIN_TASK_HASHES):
        raise RuntimeError("three locked Train tasks are required for the learner recovery batch")
    runner.RECOVERY_TASK_IDS = tuple(recovery_ids)

    original_prepare = runner.prepare_run

    def prepare_run_with_scheduler(run: Path, *, seed: int, protocol_path: Path = runner.DEFAULT_PROTOCOL):
        meta = original_prepare(run, seed=seed, protocol_path=protocol_path)
        meta["vare_ig"] = {
            "task_suite_source_sha256": sha256_file(ROOT / "src/vare/learning_pipeline/vare_ig_tasks.py"),
            "scheduler_source_sha256": sha256_file(ROOT / "src/vare/learning_pipeline/informative_curriculum.py"),
            "rollout_bridge_source_sha256": sha256_file(
                ROOT / "src/vare/learning_pipeline/verified_rollout_replay.py"
            ),
            "mlx_prefix_reuse_source_sha256": sha256_file(ROOT / "src/vare/mlx_prefix_reuse.py"),
            "arm": arm,
            "group_budget": groups,
            "exploration_summary_sha256": prior_sha,
            "prefix_cache_benchmark_sha256": cache_benchmark_sha,
            "prefix_cache_training_enabled": cache_enabled,
            "exploration_task_hashes": EXPLORATION_TASK_HASHES,
            "family_prior": prior["family_summaries"],
            "scheduler_version": curriculum.SCHEDULER_VERSION,
        }
        runner.atomic_json(run / "run_meta.json", meta)
        return meta

    runner.prepare_run = prepare_run_with_scheduler
    original_sample = runner.sample_tasks

    def enrich_rollout_rows(run: Path, phase: str, rows: Sequence[dict[str, Any]],
                            inference_profiles: Mapping[int, Mapping[str, Any]] | None = None
                            ) -> list[dict[str, Any]]:
        rollout_path = run / f"rollouts-{phase}.jsonl"
        raw_by_key = {}
        for line in rollout_path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            raw_by_key[(raw["task_id"], int(raw["sample_index"]))] = raw
        enriched = []
        for row in rows:
            raw = raw_by_key.get((row["instance_id"], int(row["sample_index"])))
            if raw is None:
                raise RuntimeError("rollout JSONL is missing a freshly generated sample")
            enriched.append({
                **row,
                "seed": int(raw["seed"]),
                "generation_seconds": float(raw["generation_seconds"]),
                "temperature": float(raw["temperature"]),
                "sampling_distribution": raw["sampling_distribution"],
                "finish": raw["finish"],
                **({"inference_profile": dict(inference_profiles[int(raw["seed"])])}
                   if inference_profiles and int(raw["seed"]) in inference_profiles else {}),
                **({"sample_generation_seconds": float(
                    inference_profiles[int(raw["seed"])].get("generation_seconds", raw["generation_seconds"])
                )} if inference_profiles and int(raw["seed"]) in inference_profiles else {}),
            })
        return enriched

    def sample_with_online_schedule(model: Any, tokenizer: Any, *, tasks: Sequence[Any], phase: str,
                                    update_step: int, policy_id: str, seed: int, max_tokens: int,
                                    run: Path, phase_started: float):
        if phase not in {"train-step-0", "train-step-1"}:
            return original_sample(model, tokenizer, tasks=tasks, phase=phase,
                                   update_step=update_step, policy_id=policy_id, seed=seed,
                                   max_tokens=max_tokens, run=run, phase_started=phase_started)

        scheduler = None
        scheduler_seed = None
        if phase == "train-step-0":
            scheduler_seed = runner.seed_for(run_seed, "vare-ig-scheduler-v1", "train-slate", 0)
            scheduler = curriculum.InformativeGroupScheduler(
                task_families={task.task_id: task_pack.FAMILY_BY_TASK[task.task_id]
                               for task in task_pack.TRAIN_TASKS},
                family_rates=family_rates,
                family_cost_seconds=family_costs,
                seed=scheduler_seed,
                mode=arm,
            )
            task_draws = None
            replay_seed = runner.seed_for(run_seed, "vare-ig-replay-v1", "train-slate", 0)
        else:
            task_draws = [task.task_id for task in tasks]
            if not task_draws or len(set(task_draws)) != len(task_draws) \
                    or set(task_draws) - set(recovery_ids):
                raise RuntimeError("step-1 rollout set must use unique predeclared recovery tasks")
            replay_seed = runner.seed_for(run_seed, "vare-ig-replay-v1", "recovery-slate", 0)

        by_id = {task.task_id: task for task in task_pack.TRAIN_TASKS}
        selected_samples = []
        selected_rows = []
        staged_paths = []
        trajectory_groups = []
        rollouts_path = run / f"rollouts-{phase}.jsonl"
        if rollouts_path.exists():
            raise FileExistsError(f"refusing to overwrite existing rollouts: {rollouts_path}")
        group_budget = groups if phase == "train-step-0" else len(task_draws)
        policy_checkpoint_sha256 = grpo_mlx.trainable_digest(model)
        bridge = VerifiedGroupReplay(
            run,
            run_id=f"vare-ig-{arm}-{run_seed}",
            group_budget=group_budget,
            group_size=runner.GROUP_SIZE,
            replay_seed=replay_seed,
        )
        for draw_index in range(group_budget):
            task_id = scheduler.next_task() if scheduler is not None else task_draws[draw_index]
            cached_generator = CachedGroupGenerator(max_prompt_tokens=runner.MAX_PROMPT_TOKENS)
            original_generate = runner.generate
            if cache_enabled:
                runner.generate = cached_generator
            try:
                samples, rows = original_sample(
                    model, tokenizer, tasks=[by_id[task_id]], phase=phase,
                    update_step=update_step, policy_id=policy_id, seed=seed,
                    max_tokens=max_tokens, run=run, phase_started=phase_started,
                )
            finally:
                runner.generate = original_generate
            profiles_by_seed = {
                int(record["seed"]): record["profile"] for record in cached_generator.records
            } if cache_enabled else None
            rows = enrich_rollout_rows(run, phase, rows, profiles_by_seed)
            if cache_enabled:
                for record in cached_generator.records:
                    runner.append_jsonl(run / "inference-profiles.jsonl", {
                        "phase": phase,
                        "task_id": task_id,
                        "group_index": draw_index,
                        "seed": int(record["seed"]),
                        "profile": record["profile"],
                    })
            if scheduler is not None:
                scheduler.observe(task_id, [sample.reward for sample in samples])
            group_id = f"vareig:{arm}:{run_seed}:{phase}:{draw_index}:{task_id}"
            trajectory_record = bridge.add_group(
                task=by_id[task_id],
                family=task_pack.FAMILY_BY_TASK[task_id],
                prompt_messages=runner.task_prompt(by_id[task_id]),
                samples=samples,
                rows=rows,
                policy_id=policy_id,
                policy_version=update_step,
                policy_checkpoint_sha256=policy_checkpoint_sha256,
                group_id=group_id,
            )
            trajectory_groups.append(trajectory_record)
            selected_samples.extend(samples)
            selected_rows.extend(rows)
            staged_dir = run / "scheduler-staged-rollouts"
            staged_dir.mkdir(exist_ok=True)
            staged_path = staged_dir / f"group-{draw_index:02d}-{task_id}.jsonl"
            os.replace(rollouts_path, staged_path)
            staged_paths.append(staged_path)

        selected_samples, selected_rows = bridge.sample_current(
            count=group_budget * runner.GROUP_SIZE,
            policy_id=policy_id,
            policy_version=update_step,
            policy_checkpoint_sha256=policy_checkpoint_sha256,
        )
        _concatenate_staged_rollouts(staged_paths, rollouts_path)
        if scheduler is not None:
            selection_record = {
                "schema": "vare_ig_online_selection_v1",
                "scheduler_version": curriculum.SCHEDULER_VERSION,
                "arm": arm,
                "policy_checkpoint_id": policy_id,
                "policy_checkpoint_sha256": policy_checkpoint_sha256,
                "behavior_step": update_step,
                "seed": run_seed,
                "scheduler_seed": scheduler_seed,
                "group_budget": groups,
                "completions_per_group": runner.GROUP_SIZE,
                "verification_cases_per_completion": 5,
                "prior_summary_sha256": prior_sha,
                "prior_summary_path": str(prior_path),
                "prefix_cache_benchmark_sha256": cache_benchmark_sha,
                "prefix_cache_enabled": cache_enabled,
                "eligible_train_task_ids": sorted(task.task_id for task in task_pack.TRAIN_TASKS),
                "development_and_heldout_eligible": False,
                "selection_history": scheduler.history,
                "family_rates_after_slate": {
                    family: vars(rate) for family, rate in scheduler.family_rates.items()
                },
                "rollouts_sha256": sha256_file(rollouts_path),
                "trajectory_groups": trajectory_groups,
                "trajectory_replay_sample_count": len(selected_samples),
                "freshness_policy_lag": 0,
                "group_provenance_homogeneous": True,
            }
            runner.atomic_json(run / "scheduler-selection.json", selection_record)
        else:
            recovery_record = {
                "schema": "vare_ig_recovery_trajectory_replay_v1",
                "policy_checkpoint_id": policy_id,
                "policy_checkpoint_sha256": policy_checkpoint_sha256,
                "prefix_cache_benchmark_sha256": cache_benchmark_sha,
                "prefix_cache_enabled": cache_enabled,
                "behavior_step": update_step,
                "trajectory_groups": trajectory_groups,
                "trajectory_replay_sample_count": len(selected_samples),
                "freshness_policy_lag": 0,
                "group_provenance_homogeneous": True,
            }
            runner.atomic_json(run / "recovery-trajectory-replay.json", recovery_record)
        return selected_samples, selected_rows

    runner.sample_tasks = sample_with_online_schedule


def record_development_promotion_gate(run: Path) -> dict[str, Any]:
    """Evaluate the configured fail-closed promotion gate on Development only."""
    final = json.loads((run / "final-summary.json").read_text(encoding="utf-8"))
    pre_rows = [json.loads(line) for line in (run / "development-pre.jsonl").read_text(encoding="utf-8").splitlines()]
    post_rows = [json.loads(line) for line in (run / "development-post.jsonl").read_text(encoding="utf-8").splitlines()]
    if not pre_rows or not post_rows:
        raise RuntimeError("promotion gate requires paired Development rows")
    base_policy_id = json.loads((run / "base-policy.json").read_text(encoding="utf-8"))["policy_checkpoint_id"]
    candidate_id = f"qwen3-4b-mlx:{runner.MODEL_REVISION[:12]}:adapter:{final['adapter_sha256'][:20]}"
    expected_ids = {task.task_id for task in task_pack.DEVELOPMENT_TASKS}
    if {row["task_id"] for row in pre_rows} != expected_ids or {row["task_id"] for row in post_rows} != expected_ids:
        raise RuntimeError("promotion gate task coverage differs from the frozen Development cohort")

    def report(policy_id: str, rows: Sequence[Mapping[str, Any]]) -> EvaluationReport:
        per_task = {str(row["task_id"]): float(row["reward"]) for row in rows}
        by_family: dict[str, list[float]] = defaultdict(list)
        for task_id, score in per_task.items():
            by_family[task_pack.FAMILY_BY_DEVELOPMENT_TASK[task_id]].append(score)
        slices = {family: sum(scores) / len(scores) for family, scores in by_family.items()}
        verifier_time = _case_wall_time_summary(rows)
        mean_cost = (
            sum(float(row.get("generation_seconds", 0.0)) for row in rows)
            + float(verifier_time["measured_seconds"])
        ) / len(rows)
        return EvaluationReport(
            policy_id=policy_id,
            primary=sum(per_task.values()) / len(per_task),
            slices=slices,
            cost=mean_cost,
            verifier_disagreement=0.0,
            n=len(per_task),
            metadata={
                "per_task_scores": per_task,
                "split": "development",
                "task_hashes": DEVELOPMENT_TASK_HASHES,
                "exposure": "development-only; not held-out evidence",
                "measured_verifier_case_count": verifier_time["measured_case_count"],
                "unmeasured_verifier_case_count": verifier_time["unmeasured_case_count"],
            },
        )

    config = PromotionConfig(
        min_primary_gain=0.01,
        min_eval_examples=64,
        paired_confidence_gate=True,
        min_paired_examples=64,
        paired_bootstrap_samples=2000,
        required_slice_names=tuple(sorted({task_pack.FAMILY_BY_DEVELOPMENT_TASK[task_id]
                                            for task_id in expected_ids})),
    )
    incumbent = report(base_policy_id, pre_rows)
    candidate = report(candidate_id, post_rows)
    decision = PromotionGate(config).decide(incumbent, candidate)
    result = {
        "schema": "vare_ig_development_promotion_gate_v1",
        "split": "development",
        "heldout": False,
        "candidate_adapter_sha256": final["adapter_sha256"],
        "candidate_checkpoint_manifest_sha256": sha256_file(
            run / "checkpoints" / (
                "recovered-update-0002" if json.loads(
                    (run / "recovered-update-0002.json").read_text(encoding="utf-8")
                )["update"]["update_applied"] else "update-0001"
            ) / "checkpoint.json"
        ),
        "config": {
            "min_eval_examples": config.min_eval_examples,
            "min_paired_examples": config.min_paired_examples,
            "paired_confidence_gate": config.paired_confidence_gate,
            "required_slice_names": list(config.required_slice_names),
        },
        "incumbent": {"policy_id": incumbent.policy_id, "primary": incumbent.primary,
                      "cost": incumbent.cost, "n": incumbent.n, "slices": incumbent.slices,
                      "measured_verifier_case_count": incumbent.metadata["measured_verifier_case_count"],
                      "unmeasured_verifier_case_count": incumbent.metadata["unmeasured_verifier_case_count"]},
        "candidate": {"policy_id": candidate.policy_id, "primary": candidate.primary,
                      "cost": candidate.cost, "n": candidate.n, "slices": candidate.slices,
                      "measured_verifier_case_count": candidate.metadata["measured_verifier_case_count"],
                      "unmeasured_verifier_case_count": candidate.metadata["unmeasured_verifier_case_count"]},
        "decision": {
            "accepted": decision.accepted,
            "reasons": list(decision.reasons),
            "primary_gain": decision.primary_gain,
            "worst_slice_regression": decision.worst_slice_regression,
            "paired_gain": decision.paired_gain,
            "paired_lcb": decision.paired_lcb,
            "paired_n": decision.paired_n,
        },
        "claim_boundary": "Development-only gate diagnostic; no promotion claim or held-out evidence.",
    }
    runner.atomic_json(run / "promotion-decision.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: qwen3_vare_ig_experiment.py pilot ... | --arm ARM --prior SUMMARY --groups N train ...")
    if argv[0] == "pilot":
        args = _pilot_parser().parse_args(argv[1:])
        try:
            return run_pilot(args)
        except runner.ResourceStop as exc:
            run = args.run_dir.expanduser().resolve()
            runner.atomic_json(run / "resource-stop-vare-ig-exploration.json", {
                "reason": str(exc), "resource": runner.resource_snapshot(run), "time": time.time()
            })
            print(json.dumps({"resource_stop": str(exc)}, sort_keys=True), flush=True)
            return 75

    if argv[0] == "cache-benchmark":
        args = _cache_benchmark_parser().parse_args(argv[1:])
        try:
            return run_prefix_cache_benchmark(args)
        except runner.ResourceStop as exc:
            run = args.run_dir.expanduser().resolve()
            runner.atomic_json(run / "resource-stop-prefix-cache.json", {
                "reason": str(exc), "resource": runner.resource_snapshot(run), "time": time.time()
            })
            print(json.dumps({"resource_stop": str(exc)}, sort_keys=True), flush=True)
            return 75

    wrapper_parser = argparse.ArgumentParser(add_help=False)
    wrapper_parser.add_argument("--arm", choices=("uniform", "informative_per_cost"))
    wrapper_parser.add_argument("--prior", type=Path)
    wrapper_parser.add_argument("--cache-benchmark", type=Path)
    wrapper_parser.add_argument("--groups", type=int, default=6)
    wrapper_parser.add_argument("--crash-after-checkpoint-before-ack", action="store_true")
    wrapper_args, runner_argv = wrapper_parser.parse_known_args(argv)
    if "--protocol" not in runner_argv:
        raise SystemExit("all runner phases require an explicit frozen --protocol")
    run_args = runner.parser().parse_args(runner_argv)
    _install_existing_runner_contract(run_args.protocol)
    run_protocol = validate_protocol(run_args.protocol, run_seed=run_args.seed)
    recovery_contract = run_protocol.get("recovery", {})
    expected_initial_sigkill = (
        run_args.phase == "train"
        and recovery_contract.get("initial_update_sigkill_boundary")
        == "after_checkpoint_before_completion_ack"
    )
    if run_args.phase == "train" and wrapper_args.crash_after_checkpoint_before_ack != expected_initial_sigkill:
        raise RuntimeError("initial-update SIGKILL flag does not match the frozen recovery boundary")
    expected_pending_sigkill = (
        run_args.phase == "prepare-recovery"
        and recovery_contract.get("second_boundary")
        == "after_fresh_on_policy_rollouts_rewards_and_behavior_logprobs_persisted_before_update_2"
    )
    actual_pending_sigkill = "--sigkill-after-pending" in runner_argv
    if run_args.phase == "prepare-recovery" and actual_pending_sigkill != expected_pending_sigkill:
        raise RuntimeError("pending-rollout SIGKILL flag does not match the frozen recovery boundary")
    if run_args.phase in {"train", "prepare-recovery"}:
        if wrapper_args.arm is None or wrapper_args.prior is None or wrapper_args.cache_benchmark is None:
            raise SystemExit("train and prepare-recovery phases require --arm, --prior, and --cache-benchmark")
        install_scheduled_training(
            arm=wrapper_args.arm, prior_path=wrapper_args.prior,
            cache_benchmark_path=wrapper_args.cache_benchmark, groups=wrapper_args.groups,
            run_seed=run_args.seed, protocol=run_protocol,
        )
        if wrapper_args.crash_after_checkpoint_before_ack and run_args.phase == "train":
            recovery = run_protocol.get("recovery", {})
            if recovery.get("initial_update_sigkill_boundary") != "after_checkpoint_before_completion_ack":
                raise RuntimeError("this run protocol does not authorize the requested SIGKILL boundary")
            original_atomic_json = runner.atomic_json

            def crash_before_ack(path: Path, value: Any) -> None:
                if Path(path).name == "learner-events.json":
                    print(json.dumps({
                        "sigkill_boundary": "after_checkpoint_before_completion_ack",
                        "checkpoint_state_sha256": sha256_file(run_args.run_dir / "checkpoint-state.json"),
                        "training_result_exists": (run_args.run_dir / "training-result.json").is_file(),
                    }, sort_keys=True), flush=True)
                    os.kill(os.getpid(), 9)
                original_atomic_json(path, value)

            runner.atomic_json = crash_before_ack
    elif wrapper_args.crash_after_checkpoint_before_ack:
        raise SystemExit("SIGKILL boundary option applies only to train phase")
    elif wrapper_args.arm is not None or wrapper_args.prior is not None or wrapper_args.cache_benchmark is not None:
        raise SystemExit("--arm, --prior, and --cache-benchmark apply only to train phases")
    status = runner.main(runner_argv)
    if status == 0 and run_args.phase == "resume":
        record_development_promotion_gate(run_args.run_dir.expanduser().resolve())
    return status


if __name__ == "__main__":
    raise SystemExit(main())
