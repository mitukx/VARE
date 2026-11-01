#!/usr/bin/env python3
"""Create the immutable VARE-IG master lock and post-pilot run locks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vare.learning_pipeline import vare_ig_tasks as tasks  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402


MASTER_PATH = ROOT / "protocols/vare_ig_qwen3_mlx_dev_v1.json"
RUN_PROTOCOL_DIR = ROOT / "protocols/vare_ig_qwen3_mlx_dev_v1"
PILOT_SEED = 20261103
RESEARCH_SEEDS = (20261104, 20261105)
RECOVERY_TASK_IDS = (
    "vare_ig.train.csv_record.v1",
    "vare_ig.train.stable_toposort.v1",
    "vare_ig.train.deep_merge_defaults.v1",
)
CACHE_BENCHMARK_TASK_IDS = (
    "vare_ig.explore.parse_csv_header.v1",
    "vare_ig.explore.graph_course_order.v1",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {path}")
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def live_starting_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def make_master() -> dict[str, object]:
    tasks.validate_task_pack()
    task_hashes = experiment.task_hashes()
    return {
        "schema": "vare_ig_qwen3_mlx_protocol_v1",
        "protocol_id": "vare-ig-qwen3-4b-mlx-development-v1",
        "planned_run_id": "vare-ig-qwen3-4b-mlx-exploration-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preregistered_before_model_inference": True,
        "starting_main_sha": "bc8803264e4a08721b0631ec04173f2785daa676",
        "implementation_start_sha": live_starting_head(),
        "source_hashes": runner.source_hashes(),
        "vare_ig_source_hashes": experiment.source_hashes(),
        "task_contract_hashes": task_hashes,
        "exposure": {
            "retired_task_ids_excluded": [
                "train.merge_closed_intervals.v1", "train.maximum_fixed_window.v1", "train.top_k_words.v1",
                "train.parse_env_lines.v1", "train.rotate_matrix_clockwise.v1", "train.valid_brackets.v1",
            ],
            "base_difficulty_pilot_split": "exploration",
            "scheduler_prior_may_use": "exploration only",
            "scheduler_eligible_split": "train only",
            "development_role": "descriptive greedy pre/post checks; not independent or held-out evaluation",
            "heldout_role": "none; no held-out task pool is used or claimed",
        },
        "research_question": (
            "Under matched group and verifier-call budgets, does an online family-level sampler based on observed trusted "
            "informative-group frequency per measured generation+verification second produce more informative groups than uniform sampling?"
        ),
        "prior_art_boundary": (
            "Difficulty-aware RL sampling and reward-variance curricula are established. This study evaluates a small, "
            "auditable implementation in a distinct verifier-bound coding workload; it makes no algorithmic novelty claim."
        ),
        "model": {"id": runner.MODEL_ID, "revision": runner.MODEL_REVISION, "local_only": True},
        "runtime_expected": {
            "python": "3.12.12",
            "mlx": "0.32.3",
            "mlx_lm": "0.32.0",
            "inference": "MLX/Metal",
            "host": "Apple Silicon M1 Pro, 32 GiB unified memory",
        },
        "sampling": {
            "seed": PILOT_SEED,
            "research_seeds": list(RESEARCH_SEEDS),
            "matched_seed_across_arms": True,
            "group_size": runner.GROUP_SIZE,
            "samples_per_exploration_task": 4,
            "max_new_tokens": runner.TRAIN_MAX_NEW_TOKENS,
            "max_prompt_tokens": runner.MAX_PROMPT_TOKENS,
            "temperature": runner.GRPO_CONFIG.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "distribution": "untruncated temperature-softmax; exact behavior log-probabilities",
            "generation_seed_schedule": "sha256(base_seed:phase:task_id:sample_index)[:8]",
        },
        "inference_benchmark": {
            "task_ids": list(CACHE_BENCHMARK_TASK_IDS),
            "split": "exploration",
            "samples_per_task": 4,
            "seed": PILOT_SEED,
            "temperature": runner.GRPO_CONFIG.temperature,
            "max_new_tokens": runner.TRAIN_MAX_NEW_TOKENS,
            "paired_seed_schedule": "sha256(base_seed:vare-ig-prefix-cache-v1:task_id:sample_index)[:8]",
            "trial_order": "independent-then-cached for even sample index; cached-then-independent for odd index",
            "comparison": "same model parameters, prompt token IDs, temperature-softmax sampler, seed, and token cap",
            "exactness": {"completion_token_ids": True, "decoded_text": True,
                          "sampler_logprob_abs_tol": 1e-6, "sampler_logprob_rel_tol": 1e-6,
                          "shared_cache_cursor_unchanged": True},
            "training_enablement": {
                "minimum_mean_task_group_speedup": 1.02,
                "require_exact_outputs_and_probabilities": True,
                "require_shared_cache_ownership_check": True,
                "applies_equally_to_both_scheduler_arms": True,
            },
        },
        "selection": {
            "arms": ["uniform", "informative_per_cost"],
            "scheduler_version": "vare_informative_group_cost_scheduler_v1",
            "pilot_task_count": len(tasks.EXPLORATION_TASKS),
            "pilot_groups": len(tasks.EXPLORATION_TASKS),
            "pilot_completions": len(tasks.EXPLORATION_TASKS) * runner.GROUP_SIZE,
            "training_groups_per_run": 6,
            "training_completions_per_run": 6 * runner.GROUP_SIZE,
            "training_runs_per_arm": len(RESEARCH_SEEDS),
            "selection_without_replacement": True,
            "uniform_probability": "1 / remaining eligible Train tasks at each draw",
            "adaptive_score": "Beta(1,1)-smoothed observed mixed-reward group rate for the task family divided by mean pilot generation+verification seconds per group",
            "online_update": "after each selected task group, update family mixed-group and reward counts before the next draw",
            "family_level_prior_only": True,
            "within_task_completion_independence_assumption": False,
            "group_cost_definition": "sum of actual completion generation seconds and parent-reported isolated case wall seconds for one four-completion task group",
            "inference_optimization": "shared-prefix reuse is used in training only if the predeclared base-model benchmark passes exact-output, log-probability, cache-ownership, and 1.02x mean group-speedup criteria; otherwise independent generation remains in use",
            "fixed_budget": {
                "groups_per_run": 6,
                "candidate_generations_per_run": 24,
                "verifier_cases_per_candidate": 5,
                "verifier_calls_per_run": 120,
                "candidate_max_new_tokens": runner.TRAIN_MAX_NEW_TOKENS,
                "interpretation": "fixed groups, candidate count, token cap, and verifier-call count; realized tokens and seconds are measured and may vary",
            },
        },
        "learner": {
            "objective": "existing VARE MLX clipped-ratio GRPO",
            "config": runner.GRPO_CONFIG.to_dict(),
            "lora": runner.LORA_CONFIG,
            "optimizer": runner.OPTIMIZER_CONFIG,
            "zero_variance": "skip all-equal reward group; never create an update from it",
            "update_count": "at most one initial applied update per run; later on-policy recovery update only if its new groups are informative",
            "sampling_consistency": "full-vocabulary temperature-softmax, top_p=1, top_k=0, min_p=0",
            "completion_only_loss": True,
            "policy_lag": 0,
        },
        "reward": {
            "schema": "vare_parent_owned_function_oracle_v1",
            "container_image": runner.CONTAINER_IMAGE,
            "timeout_seconds": runner.EVAL_TIMEOUT_SECONDS,
            "memory_mb": runner.EVAL_MEMORY_MB,
            "all_five_parent_owned_cases_required": True,
            "candidate_execution": "existing DockerIsolationRunner, network disabled, digest-pinned Python image",
            "expected_values": "trusted parent module only; never included in candidate prompt or candidate workspace",
            "untrusted_repository_pytest_authority": False,
        },
        "recovery": {
            "task_ids": list(RECOVERY_TASK_IDS),
            "initial_update_sigkill_boundary": "after_checkpoint_before_completion_ack",
            "initial_update_sigkill_arm": "informative_per_cost",
            "initial_update_sigkill_seed": RESEARCH_SEEDS[0],
            "second_boundary": "after_fresh_on_policy_rollouts_rewards_and_behavior_logprobs_persisted_before_update_2",
            "replay_control": "same retained checkpoint and pending group data replayed from the same checkpoint; compare parameters and optimizer byte digests",
            "update_boundary_only_if_real_update_exists": True,
        },
        "development": {
            "tasks": list(experiment.DEVELOPMENT_TASK_HASHES),
            "evaluation": "greedy temperature=0, one completion per task per available checkpoint",
            "interpretation": "small descriptive Development-only check, no significance or independent generalization claim",
        },
        "resources": {
            "one_4b_model_process_at_a_time": True,
            "runner_min_free_disk_gib": runner.DISK_MIN_GIB,
            "runner_min_available_memory_gib": runner.AVAILABLE_MEMORY_MIN_GIB,
            "runner_min_memory_free_percent": runner.MEMORY_FREE_MIN_PCT,
            "runner_start_swap_fraction_max_exclusive": runner.START_SWAP_MAX,
            "runner_stop_swap_fraction": runner.STOP_SWAP_MAX,
            "runner_max_phase_seconds": runner.MAX_RUN_SECONDS,
            "offline_model_cache_required": True,
            "resource_snapshot_before_and_during_each_phase": True,
        },
        "analysis": {
            "primary_outcome": "empirical informative task groups per fixed candidate/verifier budget",
            "secondary_outcomes": [
                "informative groups per 1000 generated tokens", "informative groups per minute",
                "actual applied optimizer updates", "effective completion tokens trained",
                "gradient norm and finite-gradient checks", "parameter and optimizer state digests",
                "per-family and per-task pass rate", "unique candidate/source hashes",
                "generation and verification seconds", "descriptive Development pre/post rewards",
            ],
            "inference_cost_claim": "descriptive on the locked workload; no speedup claim without matched realized work",
            "uncertainty": "report task/family counts and Wilson intervals descriptively; no inferential or significance claim from this small pilot",
        },
        "claim_boundary": "Prospective synthetic coding task and scheduler study; no held-out model capability claim unless separately certified.",
    }


def make_run_protocol(master: dict[str, object], prior_path: Path, cache_path: Path,
                      arm: str, seed: int) -> dict[str, object]:
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    expected_master_hash = sha256_file(MASTER_PATH)
    if prior.get("protocol_sha256") != expected_master_hash:
        raise RuntimeError("pilot summary was not generated under the current master protocol")
    if prior.get("task_ids") != [task.task_id for task in tasks.EXPLORATION_TASKS]:
        raise RuntimeError("pilot summary does not match the locked exploration pool")
    expected_source_hashes = experiment.source_hashes()
    if prior.get("vare_ig_source_hashes") != expected_source_hashes:
        raise RuntimeError("pilot summary source hashes differ from the current implementation")
    payload = dict(master)
    payload["protocol_id"] = f"vare-ig-{arm}-seed-{seed}-v1"
    payload["planned_run_id"] = f"vare-ig-qwen3-{arm}-seed-{seed}-v1"
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["preregistered_before_model_inference"] = True  # for this seed/arm run; the exploration pilot is its separate prior phase
    payload["sampling"] = dict(master["sampling"])
    payload["sampling"]["seed"] = seed
    payload["selection"] = dict(master["selection"])
    payload["selection"]["arm"] = arm
    payload["selection"]["group_budget"] = 6
    payload["selection"]["exploration_summary_sha256"] = sha256_file(prior_path)
    cache_summary = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache_summary.get("protocol_sha256") != expected_master_hash:
        raise RuntimeError("cache benchmark was not generated under the current master protocol")
    if cache_summary.get("vare_ig_source_hashes") != expected_source_hashes:
        raise RuntimeError("cache benchmark source hashes differ from the current implementation")
    payload["selection"]["prefix_cache_benchmark_sha256"] = sha256_file(cache_path)
    payload["task_contracts"] = {
        "train": {"task_hashes": experiment.TRAIN_TASK_HASHES},
        "development": {"task_hashes": experiment.DEVELOPMENT_TASK_HASHES},
    }
    payload["recovery"] = dict(master["recovery"])
    payload["recovery"]["initial_update_sigkill_boundary"] = (
        "after_checkpoint_before_completion_ack"
        if arm == "informative_per_cost" and seed == RESEARCH_SEEDS[0]
        else None
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    master = subs.add_parser("master", help="freeze protocol before base-model exploration")
    master.add_argument("--output", type=Path, default=MASTER_PATH)
    runs = subs.add_parser("runs", help="freeze arm/seed run protocols after the exploration prior is retained")
    runs.add_argument("--pilot-summary", type=Path, required=True)
    runs.add_argument("--cache-summary", type=Path, required=True)
    runs.add_argument("--output-dir", type=Path, default=RUN_PROTOCOL_DIR)
    args = parser.parse_args()
    if args.command == "master":
        payload = make_master()
        write_once(args.output, payload)
        print(f"frozen {args.output} sha256={sha256_file(args.output)}")
        return 0
    master = json.loads(MASTER_PATH.read_text(encoding="utf-8"))
    if master.get("source_hashes") != runner.source_hashes() or \
            master.get("vare_ig_source_hashes") != experiment.source_hashes() or \
            master.get("task_contract_hashes") != experiment.task_hashes():
        raise RuntimeError("cannot create train locks from a changed master protocol")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for arm in ("uniform", "informative_per_cost"):
        for seed in RESEARCH_SEEDS:
            path = args.output_dir / f"{arm}-seed-{seed}.json"
            payload = make_run_protocol(
                master, args.pilot_summary.expanduser().resolve(),
                args.cache_summary.expanduser().resolve(), arm, seed,
            )
            write_once(path, payload)
            print(f"frozen {path.relative_to(ROOT)} sha256={sha256_file(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
