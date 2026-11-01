#!/usr/bin/env python3
"""Regrade a frozen VARE-IG base-policy pilot without resampling the model.

Use this only when the original rollout JSONL is complete but a recorded
grader-runtime failure invalidated its reward fields. Completion text, seeds,
policy identity, tasks, and the master protocol are copied and hash-bound.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from vare.learning_pipeline import informative_curriculum as curriculum  # noqa: E402
from vare.learning_pipeline import vare_ig_tasks as tasks  # noqa: E402
from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / den
    return max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--docker-host", required=True)
    args = parser.parse_args()

    if os.environ.get("DOCKER_HOST") != args.docker_host:
        raise SystemExit("DOCKER_HOST must exactly match --docker-host")
    docker = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}} {{.OSType}} {{.Architecture}}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = experiment.validate_protocol(protocol_path, run_seed=20261103)
    experiment._install_existing_runner_contract(protocol_path)

    source_run = args.source_run.expanduser().resolve()
    output_run = args.output_run.expanduser().resolve()
    if output_run.exists() and any(output_run.iterdir()):
        raise FileExistsError(f"refusing to overwrite regraded evidence: {output_run}")
    source_rollouts = source_run / "exploration-rollouts.jsonl"
    source_meta_path = source_run / "pilot-meta.json"
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    raw_lines = source_rollouts.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines]
    expected_tasks = {task.task_id: task for task in tasks.EXPLORATION_TASKS}
    expected_count = len(tasks.EXPLORATION_TASKS) * int(protocol["sampling"]["samples_per_exploration_task"])
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} saved completions, got {len(rows)}")
    expected_keys = {
        (task_id, sample_index)
        for task_id in expected_tasks
        for sample_index in range(int(protocol["sampling"]["samples_per_exploration_task"]))
    }
    actual_keys = {(row["task_id"], int(row["sample_index"])) for row in rows}
    if actual_keys != expected_keys:
        raise RuntimeError("saved rollout set is incomplete, duplicated, or outside the frozen Exploration pool")
    if source_meta.get("protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("saved rollouts were not generated under the requested master protocol")
    if source_meta.get("vare_ig_source_hashes") != experiment.source_hashes():
        raise RuntimeError("saved rollout source hashes differ from the locked implementation")
    for row in rows:
        task = expected_tasks.get(row["task_id"])
        if task is None or row.get("task_sha256") != task.sha256():
            raise RuntimeError(f"saved rollout task contract mismatch: {row.get('task_id')}")
        if row.get("policy_checkpoint_id") != source_meta.get("policy_checkpoint_id"):
            raise RuntimeError("policy checkpoint identity changed within the saved rollout set")

    output_run.mkdir(parents=True, exist_ok=True)
    input_sha = sha256_file(source_rollouts)
    helper_sha = sha256_file(Path(__file__).resolve())
    started = time.monotonic()
    per_task: dict[str, list[float]] = defaultdict(list)
    grade_path = output_run / "exploration-rollouts-regraded.jsonl"
    if grade_path.exists():
        raise FileExistsError(grade_path)
    with grade_path.open("x", encoding="utf-8") as out:
        for index, (line, row) in enumerate(zip(raw_lines, rows, strict=True)):
            task = expected_tasks[row["task_id"]]
            score = runner.score_source(task, row["text"])
            if not score.get("trusted_contract"):
                raise RuntimeError(f"untrusted grader result for {task.task_id}/{row['sample_index']}")
            infra_failures = {
                "protocol_mismatch", "container_start_failure", "container_start_timeout",
                "container_unavailable", "daemon_unavailable",
            }
            observed_failures = {case.get("failure") for case in score.get("cases", [])}
            if observed_failures & infra_failures:
                raise RuntimeError(
                    f"grader infrastructure failure for {task.task_id}/{row['sample_index']}: "
                    f"{sorted(observed_failures & infra_failures)}"
                )
            graded = {
                **row,
                "source_rollout_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                "regrade_index": index,
                "supersedes_invalid_grade_status": row.get("grade_status"),
                "candidate_source_sha256": score.get("candidate_source_sha256"),
                "reward": float(score["reward"]),
                "grade_status": score["grade_status"],
                "trusted_contract": bool(score["trusted_contract"]),
                "verifier_id": runner.verifier_id(task),
                "case_results": score["cases"],
            }
            per_task[task.task_id].append(float(graded["reward"]))
            out.write(json.dumps(graded, sort_keys=True, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())

    family_observations: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    family_costs: dict[str, list[float]] = defaultdict(list)
    task_summaries = []
    groups = []
    graded_rows = [json.loads(line) for line in grade_path.read_text(encoding="utf-8").splitlines()]
    for task in tasks.EXPLORATION_TASKS:
        task_rows = [row for row in graded_rows if row["task_id"] == task.task_id]
        rewards = [float(row["reward"]) for row in task_rows]
        family = tasks.FAMILY_BY_EXPLORATION_TASK[task.task_id]
        family_observations[family].append((task.task_id, rewards))
        groups.append((task.task_id, rewards))
        generation_s = sum(float(row["generation_seconds"]) for row in task_rows)
        verifier_s = sum(float(case["wall_s"]) for row in task_rows for case in row["case_results"]
                         if case.get("wall_s") is not None)
        group_cost_s = generation_s + verifier_s
        family_costs[family].append(group_cost_s)
        task_summaries.append({
            "task_id": task.task_id,
            "family": family,
            "task_sha256": task.sha256(),
            "n_completions": len(task_rows),
            "successes": sum(rewards),
            "pass_probability": sum(rewards) / len(rewards),
            "pass_probability_wilson_95": wilson(int(sum(rewards)), len(rewards)),
            "rewards": rewards,
            "informative_group": len(set(rewards)) > 1,
            "unique_candidate_sources": len({row["candidate_source_sha256"] for row in task_rows
                                              if row["candidate_source_sha256"]}),
            "unique_responses": len({row["response_sha256"] for row in task_rows}),
            "completion_tokens": [row["completion_tokens"] for row in task_rows],
            "finish_reasons": dict(Counter(row["finish"] for row in task_rows)),
            "generation_seconds": generation_s,
            "verification_seconds": verifier_s,
            "unmeasured_case_calls": sum(
                case.get("wall_s") is None for row in task_rows for case in row["case_results"]
            ),
            "generation_and_verification_seconds": group_cost_s,
            "failure_categories": dict(Counter(
                case["failure"] for row in task_rows for case in row["case_results"] if not case["passed"]
            )),
        })
    rates = curriculum.family_group_rates(groups, tasks.FAMILY_BY_EXPLORATION_TASK)
    family_summaries = {}
    for family, rate in sorted(rates.items()):
        family_passes = sum(sum(outcomes) for _, outcomes in family_observations[family])
        family_completions = sum(len(outcomes) for _, outcomes in family_observations[family])
        probability = family_passes / family_completions
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
            "pass_probability": probability,
            "pass_probability_wilson_95": wilson(int(family_passes), family_completions),
            "iid_bernoulli_predicted_informative_probability":
                curriculum.independent_bernoulli_informative_probability(probability, int(protocol["sampling"]["group_size"])),
            "mean_generation_and_verification_seconds_per_group":
                sum(family_costs[family]) / len(family_costs[family]),
            "mean_completion_tokens_per_group": sum(
                sum(row["completion_tokens"] for row in graded_rows if row["task_id"] == task_id)
                for task_id, _ in family_observations[family]
            ) / len(family_observations[family]),
        }
    generation_total = sum(float(row["generation_seconds"]) for row in graded_rows)
    verification_total = sum(float(case["wall_s"]) for row in graded_rows for case in row["case_results"]
                             if case.get("wall_s") is not None)
    result = {
        "schema": "vare_ig_exploration_summary_v1",
        "protocol_id": protocol["planned_run_id"],
        "protocol_sha256": sha256_file(protocol_path),
        "vare_ig_source_hashes": experiment.source_hashes(),
        "policy_checkpoint_id": source_meta["policy_checkpoint_id"],
        "model": source_meta["model"],
        "group_size": int(protocol["sampling"]["group_size"]),
        "task_ids": [task.task_id for task in tasks.EXPLORATION_TASKS],
        "task_summaries": task_summaries,
        "family_summaries": family_summaries,
        "overall": {
            "groups": len(groups),
            "informative_groups": sum(len(set(rewards)) > 1 for _, rewards in groups),
            "completions": len(graded_rows),
            "successes": sum(float(row["reward"]) for row in graded_rows),
            "unique_candidate_sources": len({row["candidate_source_sha256"] for row in graded_rows
                                              if row["candidate_source_sha256"]}),
            "unique_responses": len({row["response_sha256"] for row in graded_rows}),
            "completion_tokens": sum(int(row["completion_tokens"]) for row in graded_rows),
            "generation_seconds": generation_total,
            "verification_seconds": verification_total,
            "unmeasured_case_calls": sum(
                case.get("wall_s") is None for row in graded_rows for case in row["case_results"]
            ),
            "wall_seconds": round(time.monotonic() - started, 6),
            "finish_reasons": dict(Counter(row["finish"] for row in graded_rows)),
            "failure_categories": dict(Counter(
                case["failure"] for row in graded_rows for case in row["case_results"] if not case["passed"]
            )),
        },
        "claim_boundary": "Exploration-only base-policy estimates regraded from saved model outputs; no final evaluation or learning claim.",
        "resource_end": runner.resource_snapshot(output_run),
        "regrade": {
            "source_run": str(source_run),
            "source_rollouts_sha256": input_sha,
            "docker_host": args.docker_host,
            "docker_server": docker,
            "helper_sha256": helper_sha,
            "original_grade_fields_discarded": True,
            "completion_text_and_seed_schedule_unchanged": True,
            "wall_seconds_measures_regrading_only": True,
        },
    }
    result_path = output_run / "exploration-summary.json"
    atomic_json(result_path, result)
    manifest = {
        "schema": "vare_ig_exploration_regrade_manifest_v1",
        "source_run": str(source_run),
        "source_rollouts_sha256": input_sha,
        "source_pilot_meta_sha256": sha256_file(source_meta_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "regraded_rollouts_sha256": sha256_file(grade_path),
        "summary_sha256": sha256_file(result_path),
        "helper_path": str(Path(__file__).resolve()),
        "helper_sha256": helper_sha,
        "docker_host": args.docker_host,
        "docker_server": docker,
        "policy_checkpoint_id": source_meta["policy_checkpoint_id"],
        "model": source_meta["model"],
        "completed_completions": len(graded_rows),
        "regrade_wall_seconds": result["overall"]["wall_seconds"],
    }
    atomic_json(output_run / "regrade-manifest.json", manifest)
    out_meta = {
        **source_meta,
        "schema": "vare_ig_base_exploration_regraded_meta_v1",
        "source_rollouts_sha256": input_sha,
        "regrade_manifest_sha256": sha256_file(output_run / "regrade-manifest.json"),
        "docker_host": args.docker_host,
        "docker_server": docker,
        "generation_reused_without_resampling": True,
    }
    atomic_json(output_run / "pilot-meta.json", out_meta)
    print(json.dumps({
        "summary": str(result_path),
        "completions": len(graded_rows),
        "informative_groups": result["overall"]["informative_groups"],
        "successes": result["overall"]["successes"],
        "summary_sha256": sha256_file(result_path),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
