#!/usr/bin/env python3
"""Freeze the Sprint XIII online-actor protocol from local pinned inputs.

This command reads the cached tokenizer only. It does not load model weights,
perform a forward pass, generate text, or start training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys


MODEL_SNAPSHOT = Path(
    "/Users/user/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
    "/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
)
RVL_ROOT = Path("/tmp/vare-sprint-xi-rvl")


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_messages(task: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Produce a route as a short sequence of labels. Follow the stopping rule.",
        },
        {
            "role": "user",
            "content": (
                f"{task['instruction']} The desired consecutive pair is {task['target_pair']}. "
                f"Avoid the consecutive pair {task['forbidden_pair']}. Use only A, B, C, or D, "
                "with at least two and at most five route labels."
            ),
        },
    ]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def sha_map(root: Path, paths: list[Path]) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): file_sha(path) for path in paths}


def validate_protocol_task_contract(root: dict) -> None:
    """Require run and task identities before model work can start."""
    if not isinstance(root.get("run_id"), str) or not root["run_id"].strip():
        raise ValueError("protocol requires a non-empty run_id")
    tasks = root.get("tasks")
    manifest_tasks = root.get("task_manifest", {}).get("tasks")
    if not isinstance(tasks, list) or tasks != manifest_tasks:
        raise ValueError("protocol tasks must exactly match task_manifest.tasks")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("protocols/sprint_xiii_online_actor_v3.lock.json"),
    )
    args = parser.parse_args()
    repository_root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    output = args.output if args.output.is_absolute() else repository_root / args.output
    sidecar = Path(str(output) + ".sha256")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"protocol freeze is append-only: {output}")
    if git(repository_root, "status", "--porcelain"):
        raise RuntimeError("protocol source checkout must be clean before freezing")
    if not MODEL_SNAPSHOT.is_dir() or not RVL_ROOT.is_dir():
        raise FileNotFoundError("the frozen local model or RVL source checkout is absent")

    # Loading this tokenizer is a deterministic identity check, not model inference.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SNAPSHOT), local_files_only=True)
    action_labels = ["A", "B", "C", "D"]
    action_token_ids = [
        tokenizer.encode(label, add_special_tokens=False) for label in action_labels
    ]
    model_config = json.loads((MODEL_SNAPSHOT / "config.json").read_text())
    if any(len(tokens) != 1 for tokens in action_token_ids):
        raise RuntimeError("the four frozen route actions must each be one token")
    if tokenizer.eos_token_id in [tokens[0] for tokens in action_token_ids]:
        raise RuntimeError("EOS token collides with a route action")

    raw_tasks = [
        {
            "task_id": "train-route-v3-01",
            "split": "train",
            "instruction": "Plan a feasible route for synthetic cohort v3 training case train-route-v3-01.",
            "target_pair": "AC",
            "forbidden_pair": "BD",
        },
        {
            "task_id": "train-route-v3-02",
            "split": "train",
            "instruction": "Plan a feasible route for synthetic cohort v3 training case train-route-v3-02.",
            "target_pair": "DB",
            "forbidden_pair": "AC",
        },
        {
            "task_id": "eval-route-v3-01",
            "split": "evaluation",
            "instruction": "Plan a feasible route for synthetic cohort v3 evaluation case eval-route-v3-01.",
            "target_pair": "BA",
            "forbidden_pair": "CD",
        },
        {
            "task_id": "eval-route-v3-02",
            "split": "evaluation",
            "instruction": "Plan a feasible route for synthetic cohort v3 evaluation case eval-route-v3-02.",
            "target_pair": "CB",
            "forbidden_pair": "DA",
        },
        {
            "task_id": "eval-route-v3-03",
            "split": "evaluation",
            "instruction": "Plan a feasible route for synthetic cohort v3 evaluation case eval-route-v3-03.",
            "target_pair": "CD",
            "forbidden_pair": "AB",
        },
        {
            "task_id": "eval-route-v3-04",
            "split": "evaluation",
            "instruction": "Plan a feasible route for synthetic cohort v3 evaluation case eval-route-v3-04.",
            "target_pair": "AD",
            "forbidden_pair": "BC",
        },
    ]
    tasks = []
    for task in raw_tasks:
        task = dict(task)
        task["task_sha256"] = hashlib.sha256(canonical(task)).hexdigest()
        text = tokenizer.apply_chat_template(
            prompt_messages(task), tokenize=False, add_generation_prompt=True
        )
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        task["prompt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        task["prompt_token_ids"] = token_ids
        task["prompt_token_count"] = len(token_ids)
        tasks.append(task)

    current_commit = git(repository_root, "rev-parse", "HEAD")
    if len(current_commit) != 40:
        raise RuntimeError("could not resolve the exact source commit")
    source_files = [
        repository_root / "scripts/sprint_xiii/online_actor.py",
        repository_root / "scripts/sprint_xiii/audit_online_actor.py",
        repository_root / "scripts/sprint_xiii/freeze_online_actor_protocol.py",
        repository_root / "tests/test_sprint_xiii_online_actor_contract.py",
        *sorted((repository_root / "src/vare").rglob("*.py")),
    ]
    snapshot_files = sorted(path for path in MODEL_SNAPSHOT.iterdir() if path.is_file())
    rvl_files = sorted((RVL_ROOT / "src/rvl_systems").rglob("*.py"))
    if not source_files or not snapshot_files or not rvl_files:
        raise RuntimeError("empty source or model identity set")

    import torch

    seeds = [404, 505, 606]
    execution_order = [
        [404, "A"], [404, "B"], [404, "C"], [404, "D"], [404, "E"],
        [505, "C"], [505, "E"], [505, "B"], [505, "D"], [505, "A"],
        [606, "E"], [606, "D"], [606, "A"], [606, "C"], [606, "B"],
    ]
    required_faults = [
        "rollout_write_before_ack",
        "verifier_write_before_replay_admission",
        "optimizer_apply_before_commit",
        "checkpoint_before_publication",
        "publication_before_actor_refresh",
    ]
    root = {
        "schema_version": 1,
        "protocol_id": "vare-sprint-xiii-online-actor-imperfect-verifier-v3",
        "run_id": "sprint-xiii-20261029-online-actor-v3",
        "status": "PREREGISTERED_NO_MODEL_FORWARD_OR_GENERATION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "On matched synthetic trajectories, accepting stale v0 reward changes the real "
            "GRPO update relative to current-v1 regrading; use-time rejection plus a fresh "
            "replacement preserves verifier freshness, and durable regrading with process "
            "fault recovery reproduces the uninterrupted D reference state."
        ),
        "cheapest_falsification": (
            "Contract tests and exhaustive action-space verifier calibration; the first "
            "constant-reward group, absent gradient divergence on mismatched rewards, or "
            "any D/E state mismatch falsifies its corresponding mechanism claim."
        ),
        "primary_outcomes": [
            "Per-seed B-versus-D token-gradient divergence on the exact same stale rollout",
            "C stale-group rejection followed by a fresh p1/v1 replacement update",
            "Exact D/E final adapter, AdamW state, RNG, checkpoint and evaluation replay",
            "Post-update policy distribution shift on a repeated prompt at a fixed token budget",
        ],
        "secondary_outcomes": [
            "Independent synthetic trusted reward per evaluation task",
            "Verifier false-positive/false-negative/disagreement rates over the enumerated route space",
            "Generated/prompt token counts, child RSS, host available memory, swap growth, wall time",
            "Queue-inclusive rollout-to-commit latency, verifier call count and crash recovery latency",
        ],
        "claim_boundary": (
            "Synthetic systems and online-optimizer feasibility only. This protocol contains no "
            "external repository repair task, Coding RL capability claim, held-out claim, or "
            "generalization claim. Model pretraining exposure to route wording is unknown."
        ),
        "decision_rule": {
            "abort": [
                "Any source, model, protocol, task, prompt, RVL revision, or runtime hash mismatch",
                "Any constant reward group; preserve artifacts and do not replace or retune it",
                "Any unplanned process failure, missing fault injection, child/resource limit violation, or audit failure",
            ],
            "systems_go": (
                "All 15 cases complete; every effective update is independently replayed; "
                "all five SIGKILL cutpoints are observed; D/E adapter, optimizer, RNG and "
                "evaluation states match exactly; each fresh actor probe exceeds the frozen TV floor."
            ),
            "learning_claim_go": (
                "Only a task-scoped synthetic reward outcome with all protocol gates passing; "
                "never label this coding capability or held-out improvement."
            ),
            "coding_rl": "NO_GO: no fresh eligible coding Development families are assigned here.",
            "heldout_training": "NO_GO: held-out source/grader certification remains below 15/15.",
        },
        "source_commit": current_commit,
        "repository_root": str(repository_root),
        "source_hashes": sha_map(repository_root, source_files),
        "rvl_source_root": str(RVL_ROOT),
        "rvl_commit": git(RVL_ROOT, "rev-parse", "HEAD"),
        "rvl_source_hashes": sha_map(RVL_ROOT, rvl_files),
        "model": {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "snapshot_path": str(MODEL_SNAPSHOT),
            "snapshot_files": sha_map(MODEL_SNAPSHOT, snapshot_files),
            "weight_file": str(MODEL_SNAPSHOT / "model.safetensors"),
            "weight_sha256": file_sha(MODEL_SNAPSHOT / "model.safetensors"),
            "architecture": "Frozen pretrained base plus one trainable fp32 vocabulary-logit-bias vector",
            "trainable_parameter_count": int(model_config["vocab_size"]),
            "trainable_scope": "Only the added output bias; base checkpoint weights stay frozen.",
            "adapter_identity": "zero-initialized vocabulary-sized fp32 bias; per-case checkpointed",
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "platform": platform.platform(),
            "device": "cpu",
            "precision": "fp32",
            "torch_threads": 1,
            "network_during_campaign": "offline; cached model only",
            "paid_compute": False,
            "tokenizer_only_before_protocol_freeze": True,
            "model_forward_or_generation_before_protocol_freeze": False,
            "mps_used": False,
            "cuda_used": bool(torch.cuda.is_available()),
        },
        "action_labels": action_labels,
        "action_token_ids": [row[0] for row in action_token_ids],
        "task_manifest": {
            "task_units": "synthetic route-pair prompts; not repository tasks",
            "tasks": tasks,
            "training_task_ids": [task["task_id"] for task in tasks if task["split"] == "train"],
            "evaluation_task_ids": [task["task_id"] for task in tasks if task["split"] == "evaluation"],
            "candidate_visible": ["task instructions and target/forbidden pair values"],
            "trusted_evaluator_location": "scripts/sprint_xiii/audit_online_actor.py only",
            "trusted_labels_available_to_learner": False,
        },
        # Runtime and auditor consume this canonical task array directly.
        "tasks": tasks,
        "generation": {
            "minimum_actions": 2,
            "maximum_actions": 5,
            "group_size": 16,
            "temperature": 2.0,
            "termination_logit_offset": 2.5,
            "seed_rule": "fixed integer formulas in the checked-in runner; no reseeding or resampling after outcomes",
            "response_token_count_includes_eos": True,
        },
        "training_tasks": ["train-route-01", "train-route-02"],
        "evaluation": {
            "task_ids": [task["task_id"] for task in tasks if task["split"] == "evaluation"],
            "samples_per_task": 8,
            "seed_rule": "same seed and task-derived offset in each baseline and final arm",
            "primary_summary": "unweighted mean of independently recomputed per-task binary trusted rewards",
            "task_unit": "one synthetic evaluation prompt; report four task rows and seed-level results",
        },
        "seeds": seeds,
        "execution_order": execution_order,
        "arms": {
            "A": "Synchronous rollout and update using current v0 then current v1 verifier.",
            "B": "Prefetched p0/v0 group arrives after transition; explicit naive staleness bypass admits stale v0 reward.",
            "C": "VARE use-time freshness rejects the stale p0/v0 group; a fresh p1/v1 replacement supplies update 2.",
            "D": "Prefetched p0/v0 group is scored and regraded by a separate current-v1 scorer process, then VARE admits it. v1 is not the trusted evaluator.",
            "E": "Same training/data as D plus real SIGKILL cutpoints; effective state must exactly match D.",
        },
        "fairness": {
            "same_model_seed_tasks_temperature_token_cap_update_count_and_eval": True,
            "same_initial_checkpoint_per_seed_and_arm": True,
            "all_arms_target_two_effective_updates": True,
            "C_has_extra_replacement_rollout_and_its_added_cost_is_reported": True,
            "B_is_a_deliberate_stale-reward negative control": True,
            "A_is_synchronous_current-verifier_reference": True,
            "seed_is_replication_unit; attempts and route strings are not independent units": True,
            "execution_order_is_fixed_and_counterbalanced_across_seeds": True,
        },
        "verifiers": {
            "0": {
                "version": 0,
                "rule": "min(1, count(target first label in first 3 route positions) / 2); ignores pair order and forbidden pair",
                "role": "partial stale proxy",
            },
            "1": {
                "version": 1,
                "rule": "zero if forbidden pair appears in first 3 route positions; otherwise 0.75*target pair anywhere + 0.25*min(2,target first-label count across route)/2",
                "role": "improved current proxy; still imperfect outside inspected forbidden prefix",
            },
            "trusted": {
                "rule": "binary 1 iff target consecutive pair appears anywhere and forbidden pair appears nowhere",
                "implementation": "independent audit module, no runner import",
                "visible_to_training_scorer_or_learner": False,
            },
        },
        "fault_plan": {
            "seed-404-E": {"rollout_write_before_ack": True},
            "seed-505-E": {
                "verifier_write_before_admission": True,
                "optimizer_apply_before_commit": True,
            },
            "seed-606-E": {
                "checkpoint_before_publication": True,
                "publication_before_actor_refresh": True,
            },
            "required_faults": required_faults,
            "signal": "SIGKILL; restarted child must have a distinct PID",
            "effective_update_rule": "uncommitted optimizer work may be recomputed; active transaction id prevents duplicate published updates",
        },
        "training": {
            "trainer": "RVL HFCausalLMGRPOTrainer from the exact pinned source checkout",
            "optimizer": "AdamW",
            "learning_rate": 0.01,
            "clip_epsilon": 0.2,
            "max_grad_norm": 1.0,
            "advantage_epsilon": 1e-6,
            "clip_advantage": 5.0,
            "objective_backend": "torch",
            "dropout": False,
            "updates_per_case": 2,
        },
        "audit": {
            "audit_version": "online-actor-independent-audit-v1",
            "behavior_logprob_abs_tolerance": 5e-4,
            "gradient_abs_tolerance": 2e-5,
            "state_abs_tolerance": 1e-6,
            "state_relative_tolerance": 1e-5,
            "minimum_refresh_tv": 1e-6,
            "required_assertions": [
                "prompt/token/checkpoint/policy/verifier provenance",
                "behavior token logprobs and independent token-level clipped-GRPO/AdamW replay",
                "VARE group-aware replay and independent policy/verifier freshness decisions",
                "trusted evaluation recomputation on disjoint synthetic prompts",
                "journal/hash/ack/commit/active-checkpoint identity and all actual SIGKILLs",
                "D/E exact adapter, optimizer, RNG and independent evaluation identity",
                "three historical audit-v2 mutant classes: old pointer, wrong gradient origin, incorrect evaluation mean",
            ],
        },
        "limits": {
            "max_child_rss_bytes": 16 * 1024**3,
            "min_available_bytes": 4 * 1024**3,
            "hard_min_available_bytes": 2 * 1024**3,
            "min_free_swap_bytes": 512 * 1024**2,
            "max_swap_growth_bytes": 512 * 1024**2,
            "min_disk_bytes": 10 * 1024**3,
            "wall_seconds": 1200,
            "audit_wall_seconds": 1200,
            "max_total_generated_tokens": 6000,
            "max_total_model_tokens": 150_000,
            "max_worker_concurrency": 1,
            "max_learner_concurrency": 1,
            "max_concurrent_children": 2,
            "torch_threads": 1,
            "gpu_allocation": 0,
        },
        "stopping_rule": (
            "Append every output. Stop whole run at any constant reward group, source/provenance "
            "mismatch, unplanned child exit, missing fault, audit mismatch or frozen resource cap. "
            "No retry or replacement cohort after model generation begins."
        ),
        "reproduction": {
            "freeze": f"{sys.executable} scripts/sprint_xiii/freeze_online_actor_protocol.py --output protocols/sprint_xiii_online_actor_v3.lock.json",
            "campaign": f"{sys.executable} scripts/sprint_xiii/online_actor.py campaign --protocol protocols/sprint_xiii_online_actor_v3.lock.json --output results/sprint_xiii_20261029/run-002",
            "audit": f"{sys.executable} scripts/sprint_xiii/audit_online_actor.py --protocol protocols/sprint_xiii_online_actor_v3.lock.json --output results/sprint_xiii_20261029/run-002 --receipt results/sprint_xiii_20261029/run-002/independent-audit.json",
            "environment": "Use the recorded offline Python environment and preserve the cached snapshot; do not overwrite an existing run directory.",
        },
    }
    validate_protocol_task_contract(root)
    payload = json.dumps(root, sort_keys=True, indent=2, ensure_ascii=False).encode() + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    sidecar.write_text(
        json.dumps({"sha256": hashlib.sha256(payload).hexdigest()}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "protocol": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_commit": current_commit,
                "tokenizer_only": True,
                "model_forward_or_generation": False,
                "task_count": len(tasks),
                "evaluation_task_count": sum(task["split"] == "evaluation" for task in tasks),
                "action_token_ids": [row[0] for row in action_token_ids],
                "eos_token_id": tokenizer.eos_token_id,
                "execution_cases": len(execution_order),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
