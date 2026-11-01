#!/usr/bin/env python3
"""Freeze a source-corrected VARE-IG training amendment without rewriting v1 locks."""
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

from vare.learning_pipeline import vare_ig_tasks as tasks  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402


NEW_SEEDS = (20261114, 20261115)
ARMS = ("uniform", "informative_per_cost")
ABORTED_SOURCE_KEY = "scripts/qwen3_vare_ig_experiment.py"


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
                        default=ROOT / "protocols/vare_ig_qwen3_mlx_dev_v1.json")
    parser.add_argument("--pilot-summary", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--aborted-attempt", type=Path, required=True)
    parser.add_argument("--output-master", type=Path,
                        default=ROOT / "protocols/vare_ig_qwen3_mlx_dev_v2.json")
    parser.add_argument("--output-run-dir", type=Path,
                        default=ROOT / "protocols/vare_ig_qwen3_mlx_dev_v2")
    parser.add_argument("--implementation-start-sha", required=True)
    args = parser.parse_args()

    parent_path = args.parent_master.expanduser().resolve()
    prior_path = args.pilot_summary.expanduser().resolve()
    cache_path = args.cache_summary.expanduser().resolve()
    aborted_path = args.aborted_attempt.expanduser().resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_sha = sha256_file(parent_path)
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    aborted = json.loads(aborted_path.read_text(encoding="utf-8"))

    if prior.get("protocol_sha256") != parent_sha or cache.get("protocol_sha256") != parent_sha:
        raise RuntimeError("pilot and prefix-cache evidence must remain bound to the parent master lock")
    if prior.get("vare_ig_source_hashes") != parent.get("vare_ig_source_hashes"):
        raise RuntimeError("pilot source hashes differ from the parent master lock")
    if cache.get("vare_ig_source_hashes") != parent.get("vare_ig_source_hashes"):
        raise RuntimeError("cache benchmark source hashes differ from the parent master lock")
    if prior.get("task_ids") != [task.task_id for task in tasks.EXPLORATION_TASKS]:
        raise RuntimeError("pilot task set differs from the frozen Exploration pool")
    if parent.get("task_contract_hashes") != experiment.task_hashes():
        raise RuntimeError("task contracts changed; this is not an implementation-only amendment")
    if aborted.get("optimizer_update_applied") is not False or aborted.get("optimizer_phase_reached") is not False:
        raise RuntimeError("the referenced attempt is not a pre-optimizer aborted run")

    old_sources = parent["vare_ig_source_hashes"]
    current_sources = experiment.source_hashes()
    if set(old_sources) != set(current_sources):
        raise RuntimeError("source hash inventory changed; amendment needs manual review")
    deltas = {key: {"parent": old_sources[key], "amended": current_sources[key]}
              for key in old_sources if old_sources[key] != current_sources[key]}
    if set(deltas) != {ABORTED_SOURCE_KEY}:
        raise RuntimeError(f"unexpected VARE-IG source changes in amendment: {sorted(deltas)}")
    if parent.get("source_hashes") != runner.source_hashes():
        raise RuntimeError("the existing GRPO runner changed; a wider protocol amendment is required")
    if set(NEW_SEEDS) & set(parent["sampling"].get("research_seeds", [])):
        raise RuntimeError("amended seed pair must not reuse the parent seed pair")

    start_sha = subprocess.run(
        ["git", "rev-parse", "--verify", f"{args.implementation_start_sha}^{{commit}}"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tool_sha = sha256_file(Path(__file__).resolve())
    amendment = {
        "schema": "vare_ig_training_protocol_amendment_v1",
        "parent_master_path": str(parent_path),
        "parent_master_sha256": parent_sha,
        "pilot_summary_path": str(prior_path),
        "pilot_summary_sha256": sha256_file(prior_path),
        "prefix_cache_summary_path": str(cache_path),
        "prefix_cache_summary_sha256": sha256_file(cache_path),
        "aborted_attempt_diagnostic_path": str(aborted_path),
        "aborted_attempt_diagnostic_sha256": sha256_file(aborted_path),
        "aborted_attempt_disposition": (
            "uniform seed 20261104 generated and staged six groups, then failed in text-to-binary rollout "
            "consolidation before the optimizer phase; its rollout and reward outcomes are excluded"
        ),
        "aborted_attempt_outcomes_used_for_selection_or_tuning": False,
        "fresh_matched_research_seeds": list(NEW_SEEDS),
        "seed_change_reason": "avoid reusing the seed from the aborted pre-update rollout attempt",
        "implementation_change": (
            "make staged rollout consolidation byte-to-byte and validate the explicitly supplied active lock"
        ),
        "changed_vare_ig_sources": deltas,
        "unchanged_contracts": [
            "model and revision", "task contracts and splits", "trusted parent-owned reward oracle",
            "scheduler algorithm and family-level prior", "sampling distribution and temperatures",
            "group sizes and candidate/verifier budgets", "learner objective and optimizer",
            "recovery boundaries", "Development evaluation contract",
        ],
        "protocol_amendment_tool_path": str(Path(__file__).resolve()),
        "protocol_amendment_tool_sha256": tool_sha,
        "implementation_start_sha": start_sha,
        "frozen_at_utc": timestamp,
    }

    master = deepcopy(parent)
    master["schema"] = "vare_ig_qwen3_mlx_protocol_v2"
    master["protocol_id"] = "vare-ig-qwen3-4b-mlx-training-amendment-v2"
    master["planned_run_id"] = "vare-ig-qwen3-4b-mlx-training-v2"
    master["frozen_at_utc"] = timestamp
    master["implementation_start_sha"] = start_sha
    master["source_hashes"] = runner.source_hashes()
    master["vare_ig_source_hashes"] = current_sources
    master["sampling"]["research_seeds"] = list(NEW_SEEDS)
    master["recovery"]["initial_update_sigkill_seed"] = NEW_SEEDS[0]
    master["protocol_stage"] = "post-pilot amendment frozen before corrected training arms"
    master["protocol_preregistration_scope"] = (
        "the four corrected training runs; the Exploration pilot and prefix-cache benchmark remain governed by the parent v1 lock"
    )
    master["protocol_amendment"] = amendment
    write_once(args.output_master.expanduser().resolve(), master)
    master_sha = sha256_file(args.output_master.expanduser().resolve())

    output_dir = args.output_run_dir.expanduser().resolve()
    for arm in ARMS:
        for seed in NEW_SEEDS:
            payload = deepcopy(master)
            payload["protocol_id"] = f"vare-ig-{arm}-seed-{seed}-v2"
            payload["planned_run_id"] = f"vare-ig-qwen3-{arm}-seed-{seed}-v2"
            payload["frozen_at_utc"] = timestamp
            payload["preregistered_before_model_inference"] = True
            payload["sampling"]["seed"] = seed
            payload["selection"]["arm"] = arm
            payload["selection"]["group_budget"] = 6
            payload["selection"]["exploration_summary_sha256"] = sha256_file(prior_path)
            payload["selection"]["prefix_cache_benchmark_sha256"] = sha256_file(cache_path)
            payload["task_contracts"] = {
                "train": {"task_hashes": experiment.TRAIN_TASK_HASHES},
                "development": {"task_hashes": experiment.DEVELOPMENT_TASK_HASHES},
            }
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
    print(f"amendment tool sha256={tool_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
