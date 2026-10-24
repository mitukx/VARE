#!/usr/bin/env python3
"""Train and evaluate a small policy controller against #1912 verifier rewards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare import mlxlm_1912_repo_task as task  # noqa: E402

PROTOCOL_PATH = ROOT / "protocols/verifier_integrity_policy_gradient_v1.json"
LOCK_PATH = ROOT / "protocols/verifier_integrity_policy_gradient_v1.lock.json"
PROTOCOL_ID = "verifier-integrity-policy-gradient-v1-mlxlm-1912"
RUNNER_PATH = "scripts/run_verifier_integrity_policy_gradient_v1.py"
INVALID_ACTIONS = {"delegate-installed-source", "delegate-installed-bytecode"}
CONDITIONS = (
    "contaminated-feedback",
    "sanitized-feedback",
    "reward-shuffle-control",
    "frozen-policy-control",
)
T_CRIT_95_DF7 = 2.364624251


class ExperimentStop(RuntimeError):
    """Fail-closed stop that preserves the partial run directory."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ExperimentStop("unexpected protocol identity")
    if lock.get("protocol_sha256") != sha256_bytes(protocol_bytes):
        raise ExperimentStop("protocol bytes differ from the frozen lock")
    expected_implementation = lock.get("implementation_sha256")
    actual_implementation = {
        relative: sha256_file(ROOT / relative)
        for relative in protocol["implementation_paths"]
    }
    if actual_implementation != expected_implementation:
        raise ExperimentStop("locked implementation source hash mismatch")
    expected_candidates = lock.get("candidate_sha256")
    actual_candidates = {
        item["id"]: sha256_file(ROOT / item["source"])
        for item in protocol["candidate_actions"]
    }
    declared_candidates = {
        item["id"]: item["sha256"] for item in protocol["candidate_actions"]
    }
    if (
        actual_candidates != expected_candidates
        or actual_candidates != declared_candidates
    ):
        raise ExperimentStop("frozen candidate bytes changed")
    historical = protocol["task"]["historical_source_evidence"]
    if sha256_file(ROOT / historical) != lock.get("historical_source_evidence_sha256"):
        raise ExperimentStop("historical pilot evidence manifest changed")
    return protocol, lock


def resource_snapshot() -> dict[str, float]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    process = psutil.Process()
    return {
        "host_available_memory_mib": memory.available / 2**20,
        "host_available_memory_percent": memory.available / memory.total * 100,
        "host_swap_used_mib": swap.used / 2**20,
        "process_rss_mib": process.memory_info().rss / 2**20,
    }


def enforce_resources(protocol: dict[str, Any], started: float) -> dict[str, float]:
    snapshot = resource_snapshot()
    limits = protocol["budgets"]
    if snapshot["host_swap_used_mib"] > limits["abort_if_host_swap_used_mib_exceeds"]:
        raise ExperimentStop("host swap limit exceeded")
    if (
        snapshot["host_available_memory_percent"]
        < limits["abort_if_host_available_memory_percent_below"]
    ):
        raise ExperimentStop("host available-memory floor breached")
    if time.monotonic() - started > limits["maximum_wall_seconds"]:
        raise ExperimentStop("total wall-time budget exceeded")
    return snapshot


def verify_local_images(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import subprocess

    result: dict[str, dict[str, Any]] = {}
    for name, image in protocol["images"].items():
        completed = subprocess.run(
            ["docker", "--context", "colima", "image", "inspect", image],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "DOCKER_CONTEXT": "colima"},
        )
        row = json.loads(completed.stdout)[0]
        digest = image.split("@", 1)[1]
        if not any(item.endswith("@" + digest) for item in row.get("RepoDigests", [])):
            raise ExperimentStop(f"local image lacks the pinned repo digest: {name}")
        result[name] = {
            "image": image,
            "image_id": row["Id"],
            "size_bytes": row["Size"],
            "architecture": row.get("Architecture"),
            "os": row.get("Os"),
            "repo_digests": row.get("RepoDigests", []),
        }
    return result


def check_optimizer_semantics() -> dict[str, Any]:
    """Synthetic preflight: a rewarded action must gain probability under Adam."""
    logits = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = torch.optim.Adam([logits], lr=0.08)
    actions = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    advantages = rewards - rewards.mean()
    before = torch.softmax(logits.detach(), dim=0)
    log_prob = torch.log_softmax(logits, dim=0).index_select(0, actions)
    loss = -(log_prob * advantages).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = torch.softmax(logits.detach(), dim=0)
    if not after[0] > before[0]:
        raise ExperimentStop(
            "optimizer semantic preflight did not increase rewarded action probability"
        )
    return {
        "status": "pass",
        "optimizer": "torch.optim.Adam",
        "torch_version": torch.__version__,
        "before": before.tolist(),
        "after": after.tolist(),
        "optimizer_step_calls": 1,
        "task_grader_calls": 0,
    }


def candidate_workspaces(protocol: dict[str, Any], output: Path) -> dict[str, Path]:
    workspaces: dict[str, Path] = {}
    for action in protocol["candidate_actions"]:
        workspace = output / "candidate-workspaces" / action["id"]
        target = workspace / "mlx_lm" / "sample_utils.py"
        target.parent.mkdir(parents=True, exist_ok=False)
        source = ROOT / action["source"]
        shutil.copyfile(source, target)
        if sha256_file(target) != action["sha256"]:
            raise ExperimentStop(f"candidate snapshot hash mismatch: {action['id']}")
        workspaces[action["id"]] = workspace
    return workspaces


def grade_action(
    action_id: str,
    workspace: Path,
    image: str,
    sequence: int,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    original_verifier = task.verify_protocol_v4

    def verify_for_image() -> dict[str, Any]:
        frozen = copy.deepcopy(original_verifier())
        frozen["runtime"]["image"] = image
        return frozen

    task.verify_protocol_v4 = verify_for_image
    try:
        report = task.grade_workspace(
            workspace,
            image=image,
            timeout_s=protocol["budgets"]["candidate_timeout_seconds"],
            memory_mb=protocol["budgets"]["candidate_memory_limit_mib"],
        )
    finally:
        task.verify_protocol_v4 = original_verifier
    expected_cases = protocol["task"]["trusted_case_count"]
    if len(report.get("cases", [])) != expected_cases:
        raise ExperimentStop(
            f"grader returned {len(report.get('cases', []))} cases; expected {expected_cases}"
        )
    return {
        "sequence": sequence,
        "action_id": action_id,
        "image": image,
        "task_passed": bool(report["passed"]),
        "case_count": len(report["cases"]),
        "case_results": report["cases"],
        "candidate_source_sha256": report["candidate_source_sha256"],
        "grader_source_sha256": report["grader_source_sha256"],
        "grader_adapter_sha256": report["grader_adapter_sha256"],
    }


def action_probabilities(logits: torch.Tensor) -> list[float]:
    return torch.softmax(logits.detach(), dim=0).tolist()


def sample_action(
    logits: torch.Tensor,
    generator: torch.Generator,
    action_ids: list[str],
) -> str:
    probs = torch.softmax(logits.detach(), dim=0)
    index = int(torch.multinomial(probs, 1, generator=generator).item())
    return action_ids[index]


def train_condition(
    condition: str,
    seed: int,
    protocol: dict[str, Any],
    workspaces: dict[str, Path],
    output: Path,
    sequence: int,
    run_started: float,
) -> tuple[dict[str, Any], int]:
    action_ids = [item["id"] for item in protocol["candidate_actions"]]
    condition_spec = protocol["conditions"][condition]
    image = protocol["images"][condition_spec["training_image"]]
    controller = protocol["controller"]
    logits = torch.nn.Parameter(
        torch.tensor(controller["initial_logits"], dtype=torch.float64)
    )
    learning_rate = (
        0.0 if condition == "frozen-policy-control" else controller["learning_rate"]
    )
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    action_generator = torch.Generator(device="cpu").manual_seed(seed * 100_003 + 17)
    shuffle_generator = torch.Generator(device="cpu").manual_seed(seed * 100_003 + 29)
    batch_size = controller["batch_size"]
    action_count = controller["training_actions_per_seed_condition"]
    event_path = output / "training-events.jsonl"
    optimizer_event_path = output / "optimizer-events.jsonl"
    started = time.monotonic()
    policy_curve = [
        {
            "optimizer_step": 0,
            "invalid_action_probability": sum(
                action_probabilities(logits)[action_ids.index(action)]
                for action in INVALID_ACTIONS
            ),
            "probabilities": action_probabilities(logits),
        }
    ]
    false_rewards = 0
    with event_path.open("a", encoding="utf-8") as stream:
        for batch_index in range(action_count // batch_size):
            batch: list[dict[str, Any]] = []
            for within_batch in range(batch_size):
                enforce_resources(protocol, run_started)
                action_id = sample_action(logits, action_generator, action_ids)
                grade_started = time.monotonic()
                call_sequence = sequence
                sequence += 1
                grade = grade_action(
                    action_id,
                    workspaces[action_id],
                    image,
                    call_sequence,
                    protocol,
                )
                raw_reward = int(grade["task_passed"])
                if (
                    condition == "contaminated-feedback"
                    and action_id in INVALID_ACTIONS
                    and raw_reward
                ):
                    false_rewards += 1
                event = {
                    "condition": condition,
                    "seed": seed,
                    "phase": "train",
                    "batch_index": batch_index,
                    "within_batch": within_batch,
                    "optimizer_step_before_batch": batch_index,
                    "action_id": action_id,
                    "image_role": condition_spec["training_image"],
                    "raw_verifier_reward": raw_reward,
                    "grade_wall_seconds": time.monotonic() - grade_started,
                    "grade": grade,
                }
                batch.append(event)
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
            raw_rewards = [event["raw_verifier_reward"] for event in batch]
            if condition == "reward-shuffle-control":
                permutation = torch.randperm(
                    batch_size, generator=shuffle_generator
                ).tolist()
                training_rewards = [raw_rewards[index] for index in permutation]
            else:
                permutation = list(range(batch_size))
                training_rewards = raw_rewards
            for index, event in enumerate(batch):
                event["training_reward"] = training_rewards[index]
                event["training_reward_source_slot"] = permutation[index]
                event["reward_shuffle_permutation"] = permutation

            action_tensor = torch.tensor(
                [action_ids.index(event["action_id"]) for event in batch],
                dtype=torch.long,
            )
            reward_tensor = torch.tensor(training_rewards, dtype=torch.float64)
            advantages = reward_tensor - reward_tensor.mean()
            log_probabilities = torch.log_softmax(logits, dim=0).index_select(
                0, action_tensor
            )
            loss = -(log_probabilities * advantages).mean()
            before = logits.detach().clone()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            changed = not torch.equal(before, logits.detach())
            probabilities = action_probabilities(logits)
            with optimizer_event_path.open("a", encoding="utf-8") as optimizer_stream:
                optimizer_stream.write(
                    json.dumps(
                        {
                            "condition": condition,
                            "seed": seed,
                            "batch_index": batch_index,
                            "action_ids": [event["action_id"] for event in batch],
                            "raw_batch_rewards": raw_rewards,
                            "training_batch_rewards": training_rewards,
                            "reward_shuffle_permutation": permutation,
                            "loss": float(loss.detach().item()),
                            "parameters_before": before.tolist(),
                            "parameters_after": logits.detach().tolist(),
                            "parameters_changed": changed,
                            "optimizer_step": batch_index + 1,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                optimizer_stream.flush()
            policy_curve.append(
                {
                    "optimizer_step": batch_index + 1,
                    "loss": float(loss.detach().item()),
                    "raw_batch_rewards": raw_rewards,
                    "training_batch_rewards": training_rewards,
                    "reward_shuffle_permutation": permutation,
                    "parameters_changed": changed,
                    "invalid_action_probability": sum(
                        probabilities[action_ids.index(action)]
                        for action in INVALID_ACTIONS
                    ),
                    "probabilities": probabilities,
                }
            )
    state = {
        "protocol_id": PROTOCOL_ID,
        "condition": condition,
        "seed": seed,
        "action_ids": action_ids,
        "logits": logits.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_class": type(optimizer).__name__,
        "learning_rate": learning_rate,
        "torch_version": torch.__version__,
        "training_action_count": action_count,
        "optimizer_step_calls": action_count // batch_size,
        "parameter_updates_committed": sum(
            1 for row in policy_curve[1:] if row["parameters_changed"]
        ),
    }
    checkpoint_path = output / "checkpoints" / f"{condition}-seed-{seed}.pt"
    torch.save(state, checkpoint_path)
    record = {
        "condition": condition,
        "seed": seed,
        "training_image": condition_spec["training_image"],
        "training_actions": action_count,
        "optimizer_step_calls": action_count // batch_size,
        "parameter_updates_committed": state["parameter_updates_committed"],
        "training_false_rewards_on_known_invalid_actions": false_rewards,
        "final_logits": logits.detach().tolist(),
        "final_action_probabilities": action_probabilities(logits),
        "final_invalid_action_probability": policy_curve[-1][
            "invalid_action_probability"
        ],
        "policy_learning_curve": policy_curve,
        "checkpoint_path": str(checkpoint_path.relative_to(output)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    return record, sequence


def evaluate_condition(
    condition: str,
    seed: int,
    checkpoint: Path | None,
    protocol: dict[str, Any],
    workspaces: dict[str, Path],
    output: Path,
    sequence: int,
    run_started: float,
) -> tuple[dict[str, Any], int]:
    action_ids = [item["id"] for item in protocol["candidate_actions"]]
    if checkpoint is None:
        logits = torch.tensor(
            protocol["controller"]["initial_logits"], dtype=torch.float64
        )
        checkpoint_sha = None
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        logits = saved["logits"]
        checkpoint_sha = sha256_file(checkpoint)
    generator = torch.Generator(device="cpu").manual_seed(seed * 100_003 + 701)
    image = protocol["images"][protocol["evaluation"]["image"]]
    evaluations: list[dict[str, Any]] = []
    event_path = output / "evaluation-events.jsonl"
    started = time.monotonic()
    with event_path.open("a", encoding="utf-8") as stream:
        for index in range(
            protocol["controller"]["evaluation_actions_per_seed_condition"]
        ):
            enforce_resources(protocol, run_started)
            action_id = sample_action(logits, generator, action_ids)
            grade_started = time.monotonic()
            call_sequence = sequence
            sequence += 1
            grade = grade_action(
                action_id, workspaces[action_id], image, call_sequence, protocol
            )
            event = {
                "condition": condition,
                "seed": seed,
                "phase": "trusted-evaluation",
                "evaluation_index": index,
                "action_id": action_id,
                "is_delegate_action": action_id in INVALID_ACTIONS,
                "is_invalid_delegate_action": action_id in INVALID_ACTIONS
                and not grade["task_passed"],
                "clean_false_reward_acceptance": action_id in INVALID_ACTIONS
                and grade["task_passed"],
                "image_role": "sanitized",
                "task_passed": grade["task_passed"],
                "case_count": grade["case_count"],
                "grade_wall_seconds": time.monotonic() - grade_started,
                "grade": grade,
            }
            evaluations.append(event)
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
    total = len(evaluations)
    invalid = sum(event["is_invalid_delegate_action"] for event in evaluations)
    delegates = sum(event["is_delegate_action"] for event in evaluations)
    false_acceptances = sum(
        event["clean_false_reward_acceptance"] for event in evaluations
    )
    successes = sum(event["task_passed"] for event in evaluations)
    return {
        "condition": condition,
        "seed": seed,
        "evaluation_actions": total,
        "invalid_delegate_selections": invalid,
        "invalid_action_share": invalid / total,
        "delegate_action_share": delegates / total,
        "clean_false_reward_acceptances": false_acceptances,
        "trusted_task_successes": successes,
        "trusted_task_success_share": successes / total,
        "evaluation_image": image,
        "checkpoint_sha256": checkpoint_sha,
        "elapsed_wall_seconds": time.monotonic() - started,
    }, sequence


def paired_t_interval(values: list[float]) -> list[float]:
    mean = statistics.mean(values)
    if len(values) != 8:
        raise ExperimentStop("frozen analysis requires all eight complete seeds")
    margin = T_CRIT_95_DF7 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - margin, mean + margin]


def sign_flip_pvalue(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(
            statistics.mean(
                value * sign for value, sign in zip(values, signs, strict=True)
            )
        )
        if statistic >= observed - 1e-15:
            extreme += 1
        total += 1
    return extreme / total


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, pvalue) in enumerate(ordered):
        corrected = min(1.0, (count - rank) * pvalue)
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted


def analyze(
    training: list[dict[str, Any]], evaluation: list[dict[str, Any]]
) -> dict[str, Any]:
    by_condition = {
        condition: sorted(
            [row for row in evaluation if row["condition"] == condition],
            key=lambda row: row["seed"],
        )
        for condition in CONDITIONS
    }
    means: dict[str, Any] = {}
    for condition, rows in by_condition.items():
        values = [row["invalid_action_share"] for row in rows]
        means[condition] = {
            "mean_invalid_action_share": statistics.mean(values),
            "seed_sd_invalid_action_share": statistics.stdev(values),
            "seed_t95_interval": paired_t_interval(values),
            "mean_trusted_task_success_share": statistics.mean(
                row["trusted_task_success_share"] for row in rows
            ),
            "per_seed": rows,
        }
    primary_name = "contaminated-minus-sanitized"
    contrasts = {
        primary_name: [
            dirty["invalid_action_share"] - clean["invalid_action_share"]
            for dirty, clean in zip(
                by_condition["contaminated-feedback"],
                by_condition["sanitized-feedback"],
                strict=True,
            )
        ],
        "contaminated-minus-reward-shuffle": [
            dirty["invalid_action_share"] - shuffled["invalid_action_share"]
            for dirty, shuffled in zip(
                by_condition["contaminated-feedback"],
                by_condition["reward-shuffle-control"],
                strict=True,
            )
        ],
        "contaminated-minus-frozen-policy": [
            dirty["invalid_action_share"] - frozen["invalid_action_share"]
            for dirty, frozen in zip(
                by_condition["contaminated-feedback"],
                by_condition["frozen-policy-control"],
                strict=True,
            )
        ],
    }
    raw_p = {name: sign_flip_pvalue(values) for name, values in contrasts.items()}
    adjusted_p = holm_adjust(raw_p)
    contrast_summary = {}
    for name, values in contrasts.items():
        contrast_summary[name] = {
            "paired_seed_differences": values,
            "mean_difference": statistics.mean(values),
            "paired_seed_t95_interval": paired_t_interval(values),
            "exact_two_sided_sign_flip_p": raw_p[name],
            "holm_adjusted_p": adjusted_p[name],
        }
    false_rewards = sum(
        int(row.get("training_false_rewards_on_known_invalid_actions", 0))
        for row in training
    )
    committed_updates = sum(row["parameter_updates_committed"] for row in training)
    return {
        "seed_count": 8,
        "task_count": 1,
        "primary_metric": "clean-oracle invalid-delegate action share",
        "conditions": means,
        "planned_paired_contrasts": contrast_summary,
        "training_false_reward_count": false_rewards,
        "optimizer_step_calls": sum(row["optimizer_step_calls"] for row in training),
        "committed_parameter_updates": committed_updates,
        "training_curve_by_condition_seed_step": [
            {"condition": row["condition"], "seed": row["seed"], **point}
            for row in training
            for point in row["policy_learning_curve"]
        ],
    }


def snapshot_inputs(output: Path, protocol: dict[str, Any]) -> None:
    snapshots = output / "snapshots"
    snapshots.mkdir()
    shutil.copyfile(PROTOCOL_PATH, snapshots / "protocol.json")
    shutil.copyfile(LOCK_PATH, snapshots / "protocol.lock.json")
    shutil.copyfile(ROOT / RUNNER_PATH, snapshots / "runner.py")
    for action in protocol["candidate_actions"]:
        dest = snapshots / "candidates" / f"{action['id']}.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / action["source"], dest)


def write_evidence(
    output: Path, protocol: dict[str, Any], lock: dict[str, Any]
) -> None:
    artifacts = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {
            "evidence-manifest.json",
            "evidence-lock.json",
        }:
            continue
        artifacts.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest_path = output / "evidence-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "vare.verifier-integrity-policy-gradient-evidence.v1",
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_lock = {
        "schema": "vare.verifier-integrity-policy-gradient-lock.v1",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(ROOT / RUNNER_PATH),
        "manifest_sha256": sha256_file(manifest_path),
        "implementation_sha256": lock["implementation_sha256"],
        "candidate_sha256": lock["candidate_sha256"],
    }
    (output / "evidence-lock.json").write_text(
        json.dumps(evidence_lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def preflight(protocol: dict[str, Any]) -> dict[str, Any]:
    mps_backend = getattr(torch.backends, "mps", None)
    accelerator_available = torch.cuda.is_available() or (
        mps_backend is not None and mps_backend.is_available()
    )
    image_records = verify_local_images(protocol)
    task.verify_protocol_v4()
    resources = resource_snapshot()
    limits = protocol["budgets"]
    if resources["host_swap_used_mib"] > limits["abort_if_host_swap_used_mib_exceeds"]:
        raise ExperimentStop("resource preflight exceeds host swap ceiling")
    if (
        resources["host_available_memory_percent"]
        < limits["abort_if_host_available_memory_percent_below"]
    ):
        raise ExperimentStop("resource preflight is below available-memory floor")
    return {
        "status": "pass",
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_device": "cpu",
        "accelerator_available_but_unused": accelerator_available,
        "accelerator_used": False,
        "images": image_records,
        "host_resources": resources,
        "optimizer_semantics": check_optimizer_semantics(),
        "grader_protocol": "mlxlm-1912-isolated-task-v4",
        "grader_case_count": protocol["task"]["trusted_case_count"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="verify frozen inputs/resources without grading",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        protocol, lock = load_frozen_inputs()
        preflight_result = preflight(protocol)
        if args.preflight:
            print(json.dumps(preflight_result, indent=2, sort_keys=True))
            return 0
        if args.output is None:
            parser.error("--output is required for an experiment run")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "checkpoints").mkdir()
        snapshot_inputs(output, protocol)
        (output / "preflight.json").write_text(
            json.dumps(preflight_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        images = verify_local_images(protocol)
        (output / "images.json").write_text(
            json.dumps(images, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        workspaces = candidate_workspaces(protocol, output)
        started = time.monotonic()
        sequence = 0
        training_rows: list[dict[str, Any]] = []
        evaluation_summaries: list[dict[str, Any]] = []
        run_state = output / "run-state.json"
        try:
            for condition in CONDITIONS:
                for seed in protocol["controller"]["seeds"]:
                    training, sequence = train_condition(
                        condition,
                        seed,
                        protocol,
                        workspaces,
                        output,
                        sequence,
                        started,
                    )
                    training_rows.append(training)
                    checkpoint = output / training["checkpoint_path"]
                    evaluation, sequence = evaluate_condition(
                        condition,
                        seed,
                        checkpoint,
                        protocol,
                        workspaces,
                        output,
                        sequence,
                        started,
                    )
                    evaluation_summaries.append(evaluation)
                    (run_state).write_text(
                        json.dumps(
                            {
                                "status": "running",
                                "completed_condition_seed_pairs": len(training_rows),
                                "expected_condition_seed_pairs": len(CONDITIONS)
                                * len(protocol["controller"]["seeds"]),
                                "completed_grade_calls": sequence,
                                "planned_grade_calls": protocol["budgets"][
                                    "maximum_grader_invocations"
                                ],
                                "wall_seconds": time.monotonic() - started,
                                "host_resources": resource_snapshot(),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print(
                        json.dumps(
                            {
                                "condition": condition,
                                "seed": seed,
                                "training_actions": training["training_actions"],
                                "evaluation_actions": evaluation["evaluation_actions"],
                                "invalid_share": evaluation["invalid_action_share"],
                                "completed_grade_calls": sequence,
                                "wall_seconds": round(time.monotonic() - started, 1),
                            }
                        ),
                        flush=True,
                    )
            summary = {
                "protocol_id": PROTOCOL_ID,
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "implementation_commit": os.environ.get(
                    "VARE_IMPLEMENTATION_COMMIT", "unrecorded"
                ),
                "task_id": protocol["task"]["task_id"],
                "candidate_action_count": len(protocol["candidate_actions"]),
                "condition_count": len(CONDITIONS),
                "seed_count": len(protocol["controller"]["seeds"]),
                "training_grade_calls": len(training_rows)
                * protocol["controller"]["training_actions_per_seed_condition"],
                "evaluation_grade_calls": len(evaluation_summaries)
                * protocol["controller"]["evaluation_actions_per_seed_condition"],
                "actual_grader_invocations": sequence,
                "actual_candidate_case_executions": sequence
                * protocol["task"]["trusted_case_count"],
                "optimizer_step_calls": sum(
                    row["optimizer_step_calls"] for row in training_rows
                ),
                "committed_parameter_updates": sum(
                    row["parameter_updates_committed"] for row in training_rows
                ),
                "model_calls": 0,
                "accelerator_compute": False,
                "wall_seconds": time.monotonic() - started,
                "initial_host_resources": preflight_result["host_resources"],
                "final_host_resources": resource_snapshot(),
                "analysis": analyze(training_rows, evaluation_summaries),
                "training_seed_records": training_rows,
                "evaluation_seed_records": evaluation_summaries,
                "evaluation_events_file": "evaluation-events.jsonl",
                "training_events_file": "training-events.jsonl",
                "claim_boundary": protocol["claim_boundary"],
            }
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            run_state.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "completed_grade_calls": sequence,
                        "wall_seconds": time.monotonic() - started,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "status": "aborted",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "completed_grade_calls": sequence,
                        "completed_training_seed_records": len(training_rows),
                        "wall_seconds": time.monotonic() - started,
                        "host_resources": resource_snapshot(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            write_evidence(output, protocol, lock)
            raise
        write_evidence(output, protocol, lock)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(output),
                    "grade_calls": sequence,
                    "optimizer_step_calls": summary["optimizer_step_calls"],
                    "committed_parameter_updates": summary[
                        "committed_parameter_updates"
                    ],
                    "wall_seconds": summary["wall_seconds"],
                },
                indent=2,
            )
        )
        return 0
    except ExperimentStop as exc:
        parser.exit(2, f"STOP: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
