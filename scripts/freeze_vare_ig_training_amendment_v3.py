#!/usr/bin/env python3
"""Freeze fresh VARE-IG training runs with incremental sample durability."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402


NEW_SEEDS = (20261116, 20261117)
ARMS = ("uniform", "informative_per_cost")
LAUNCHER = ROOT / "scripts/run_vare_ig_durable_training.py"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {path}")
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-master", type=Path,
                        default=ROOT / "protocols/vare_ig_qwen3_mlx_dev_v2.json")
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--resource-stop", type=Path, required=True)
    parser.add_argument("--incomplete-run-dir", type=Path, required=True)
    parser.add_argument("--implementation-start-sha", required=True)
    parser.add_argument("--output-master", type=Path,
                        default=ROOT / "protocols/vare_ig_qwen3_mlx_dev_v3.json")
    parser.add_argument("--output-run-dir", type=Path,
                        default=ROOT / "protocols/vare_ig_qwen3_mlx_dev_v3")
    args = parser.parse_args()

    parent_path = args.parent_master.expanduser().resolve()
    prior_path = args.pilot_summary.expanduser().resolve()
    cache_path = args.cache_summary.expanduser().resolve()
    stop_path = args.resource_stop.expanduser().resolve()
    incomplete_run = args.incomplete_run_dir.expanduser().resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_sha = sha256_file(parent_path)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    stopped = json.loads(stop_path.read_text(encoding="utf-8"))
    first_update = json.loads((incomplete_run / "training-result.json").read_text(encoding="utf-8"))
    state = json.loads((incomplete_run / "checkpoint-state.json").read_text(encoding="utf-8"))

    if parent.get("schema") != "vare_ig_qwen3_mlx_protocol_v2":
        raise RuntimeError("V3 must amend the retained V2 master lock")
    if prior.get("protocol_sha256") != parent.get("protocol_amendment", {}).get("parent_master_sha256"):
        raise RuntimeError("pilot summary is not bound to the V2 parent amendment")
    if cache.get("protocol_sha256") != prior.get("protocol_sha256"):
        raise RuntimeError("prefix-cache summary does not share the pilot's frozen parent")
    if parent.get("task_contract_hashes") != experiment.task_hashes():
        raise RuntimeError("task contracts changed; this is not an execution-only amendment")
    if parent.get("source_hashes") != runner.source_hashes() \
            or parent.get("vare_ig_source_hashes") != experiment.source_hashes():
        raise RuntimeError("the V2 implementation source hashes no longer match")
    if set(NEW_SEEDS) & set(parent["sampling"].get("research_seeds", [])):
        raise RuntimeError("V3 must use a fresh matched seed pair")
    if not (first_update.get("update_applied") is True and state.get("update_step") == 1
            and state.get("optimizer_step") == 1):
        raise RuntimeError("incomplete run does not contain a durable, effective update 1")
    if stopped.get("phase") != "prepare-recovery" or stopped.get("resource", {}).get(
            "available_memory_gib", 99) >= runner.AVAILABLE_MEMORY_MIN_GIB:
        raise RuntimeError("referenced V2 diagnostic is not the locked-floor recovery stop")
    rollout_path = incomplete_run / "rollouts-train-step-1.jsonl"
    rows = [json.loads(line) for line in rollout_path.read_text(encoding="utf-8").splitlines()]
    exact_sample_path = incomplete_run / "training-samples-streamed-train-step-1.jsonl"
    pending_path = incomplete_run / "pending-update-0002.json"
    if len(rows) != 12 or exact_sample_path.exists() or pending_path.exists():
        raise RuntimeError("referenced stop must have 12 rollouts but lack a persisted exact learner batch")

    start_sha = subprocess.run(
        ["git", "rev-parse", "--verify", f"{args.implementation_start_sha}^{{commit}}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tool_sha = sha256_file(Path(__file__).resolve())
    launcher_sha = sha256_file(LAUNCHER)
    amendment = {
        "schema": "vare_ig_training_protocol_amendment_v1",
        "parent_master_path": parent_path.relative_to(ROOT).as_posix(),
        "parent_master_sha256": parent_sha,
        "pilot_summary_path": prior_path.relative_to(ROOT).as_posix(),
        "pilot_summary_sha256": sha256_file(prior_path),
        "prefix_cache_summary_path": cache_path.relative_to(ROOT).as_posix(),
        "prefix_cache_summary_sha256": sha256_file(cache_path),
        "resource_stop_path": stop_path.relative_to(ROOT).as_posix(),
        "resource_stop_sha256": sha256_file(stop_path),
        "incomplete_run_directory": incomplete_run.relative_to(ROOT).as_posix(),
        "incomplete_rollout_sha256": sha256_file(rollout_path),
        "incomplete_batch_disposition": (
            "the 12 step-1 rollout texts are retained, but the runner did not persist exact token IDs before the "
            "behavior-logprob pass; this batch is excluded from training and will not be replayed"
        ),
        "incomplete_outcomes_used_for_selection_or_tuning": False,
        "fresh_matched_research_seeds": list(NEW_SEEDS),
        "seed_change_reason": "avoid reusing all V2 sampling seeds after a resource-stopped recovery attempt",
        "execution_change": (
            "a hash-locked launcher clears reusable MLX buffers before the unchanged resource guard and fsyncs "
            "each exact sample, including prompt/completion token IDs and verifier provenance, immediately after "
            "its graded rollout is appended and before any behavior-logprob pass"
        ),
        "launcher_path": LAUNCHER.relative_to(ROOT).as_posix(),
        "launcher_sha256": launcher_sha,
        "unchanged_contracts": [
            "model and revision", "task contracts and splits", "trusted parent-owned reward oracle",
            "scheduler algorithm and family-level prior", "sampling distribution and temperatures",
            "group sizes and candidate/verifier budgets", "learner objective and optimizer",
            "resource thresholds", "Development evaluation contract",
        ],
        "protocol_amendment_tool_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "protocol_amendment_tool_sha256": tool_sha,
        "implementation_start_sha": start_sha,
        "frozen_at_utc": timestamp,
    }

    master = deepcopy(parent)
    master["schema"] = "vare_ig_qwen3_mlx_protocol_v3"
    master["protocol_id"] = "vare-ig-qwen3-4b-mlx-training-amendment-v3"
    master["planned_run_id"] = "vare-ig-qwen3-4b-mlx-training-v3"
    master["frozen_at_utc"] = timestamp
    master["implementation_start_sha"] = start_sha
    master["sampling"]["research_seeds"] = list(NEW_SEEDS)
    master["selection"]["exploration_summary_sha256"] = sha256_file(prior_path)
    master["selection"]["prefix_cache_benchmark_sha256"] = sha256_file(cache_path)
    master["recovery"]["initial_update_sigkill_seed"] = NEW_SEEDS[0]
    master["protocol_stage"] = "post-V2 resource-stop amendment frozen before V3 model runs"
    master["protocol_preregistration_scope"] = (
        "the four fresh V3 training runs; V1 Exploration/cache evidence and V2 seed-20261114 update-1 evidence "
        "remain preserved under their existing immutable locks"
    )
    master["execution"] = {
        "durable_training_launcher_sha256": launcher_sha,
        "cache_release_before_resource_guard": True,
        "persist_token_ids_before_behavior_logprobs": True,
        "sample_artifact": "training-samples-streamed-{phase}.jsonl; one fsynced exact learner sample per graded rollout",
    }
    master["protocol_stage"] = "post-V2 resource-stop amendment frozen before V3 model runs"
    master["protocol_amendment"] = amendment
    write_once(args.output_master.expanduser().resolve(), master)
    master_sha = sha256_file(args.output_master.expanduser().resolve())

    output_dir = args.output_run_dir.expanduser().resolve()
    for arm in ARMS:
        for seed in NEW_SEEDS:
            payload = deepcopy(master)
            payload["protocol_id"] = f"vare-ig-{arm}-seed-{seed}-v3"
            payload["planned_run_id"] = f"vare-ig-qwen3-{arm}-seed-{seed}-v3"
            payload["frozen_at_utc"] = timestamp
            payload["preregistered_before_model_inference"] = True
            payload["sampling"]["seed"] = seed
            payload["selection"]["arm"] = arm
            payload["selection"]["group_budget"] = 6
            payload["recovery"]["initial_update_sigkill_boundary"] = (
                "after_checkpoint_before_completion_ack"
                if arm == "informative_per_cost" and seed == NEW_SEEDS[0]
                else None
            )
            payload["recovery"]["initial_update_sigkill_seed"] = NEW_SEEDS[0]
            payload["base_master_protocol_sha256"] = master_sha
            payload["protocol_amendment"] = amendment
            path = output_dir / f"{arm}-seed-{seed}.json"
            write_once(path, payload)
            print(f"frozen {path.relative_to(ROOT)} sha256={sha256_file(path)}")
    print(f"frozen {args.output_master.expanduser().resolve()} sha256={master_sha}")
    print(f"launcher sha256={launcher_sha}")
    print(f"amendment tool sha256={tool_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
