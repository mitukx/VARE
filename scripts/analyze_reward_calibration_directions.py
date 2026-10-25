"""Recompute descriptive reward, advantage, and GRPO-gradient drift metrics.

This is a post-run analysis of the frozen four-group calibration. It does not
sample trajectories, call an optimizer step, or change the calibration gate.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALIBRATION = ROOT / "results/verifier_consistent_async_grpo_v7/reward-calibration-v1/pilot-1"
PROTOCOL = ROOT / "benchmarks/verifier_consistent_async_grpo_v7/reward-calibration-v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_create_json(path: Path, value: dict[str, Any]) -> str:
    """Create once, fsync, and verify bytes; never replace existing evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    written = path.read_bytes()
    if written != payload:
        raise RuntimeError("analysis artifact read-back differs from written bytes")
    return hashlib.sha256(payload).hexdigest()


def strict_reversals(left: list[float], right: list[float]) -> tuple[int, int]:
    reversals = 0
    one_sided_ties = 0
    for i, j in itertools.combinations(range(len(left)), 2):
        delta_left = left[i] - left[j]
        delta_right = right[i] - right[j]
        if delta_left * delta_right < 0:
            reversals += 1
        elif (delta_left == 0) != (delta_right == 0):
            one_sided_ties += 1
    return reversals, one_sided_ties


def reward_metrics(groups: list[dict[str, Any]], compute_advantages) -> dict[str, Any]:
    members = [member for group in groups for member in group["members"]]
    shifts = [abs(float(row["v0"]) - float(row["v1"])) for row in members]
    group_results = []
    total_reversals = 0
    total_tie_breaks = 0
    for group in groups:
        rewards = {
            version: [float(row[version]) for row in group["members"]]
            for version in ("v0", "v1")
        }
        advantages = {
            version: compute_advantages(rewards[version])
            for version in ("v0", "v1")
        }
        reversals, tie_breaks = strict_reversals(rewards["v0"], rewards["v1"])
        total_reversals += reversals
        total_tie_breaks += tie_breaks
        centered_delta = [b - a for a, b in zip(advantages["v0"], advantages["v1"])]
        sign_switches = sum(
            (a < 0 < b) or (b < 0 < a)
            for a, b in zip(advantages["v0"], advantages["v1"])
        )
        group_results.append({
            "task_id": group["task_id"],
            "reward_counts": {
                version: {
                    "positive": sum(x > 0 for x in rewards[version]),
                    "negative": sum(x < 0 for x in rewards[version]),
                    "zero": sum(x == 0 for x in rewards[version]),
                }
                for version in ("v0", "v1")
            },
            "rewards": rewards,
            "centered_advantages": advantages,
            "advantage_change": {
                "mean_absolute": sum(abs(x) for x in centered_delta) / len(centered_delta),
                "max_absolute": max(abs(x) for x in centered_delta),
                "negative_positive_sign_switches": sign_switches,
                "strict_preference_reversals": reversals,
                "one_sided_tie_breaks": tie_breaks,
            },
        })
    return {
        "members": len(members),
        "strictly_changed_rewards": sum(float(row["v0"]) != float(row["v1"]) for row in members),
        "fraction_strictly_changed": sum(float(row["v0"]) != float(row["v1"]) for row in members) / len(members),
        "mean_absolute_reward_disagreement": sum(shifts) / len(shifts),
        "fraction_absolute_shift_at_least_0_25": sum(x >= 0.25 for x in shifts) / len(shifts),
        "strict_pairwise_preference_reversals": total_reversals,
        "one_sided_tie_breaks": total_tie_breaks,
        "pairwise_comparisons": sum(len(g["members"]) * (len(g["members"]) - 1) // 2 for g in groups),
        "groups": group_results,
    }


def gradient_for_group(model, trainer, helper, compute_group_advantages,
                       rows: list[dict[str, Any]], version: str):
    import torch

    samples = []
    for row in rows:
        generation = helper.Generation(
            prompt_id="calibration-group",
            prompt=row["prompt"],
            response=row["output_text"],
            logprob=float(row["behavior_logprob"]),
            token_count=int(row["token_count"]),
            latency_s=float(row["generation_latency_s"]),
            metadata=row["generation_metadata"],
        )
        samples.append(helper.VerifiedGeneration(
            generation=generation,
            reward=float(row[version]),
            verifier_latency_s=0.0,
            verifier_version=0 if version == "v0" else 1,
        ))
    records = compute_group_advantages(samples, eps=1e-6, clip=5.0)
    model.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    max_abs_log_ratio = 0.0
    for sample, record in zip(samples, records):
        objective, _, ratio, _ = trainer._sample_objective(sample, record.advantage)
        (-objective / len(samples)).backward()
        max_abs_log_ratio = max(max_abs_log_ratio, float(ratio.detach().cpu()))
    named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    vector = torch.cat([
        (torch.zeros_like(param) if param.grad is None else param.grad).detach().reshape(-1).to(torch.float64)
        for _, param in named_parameters
    ])
    return vector, len(named_parameters), sum(param.numel() for _, param in named_parameters), max_abs_log_ratio


def analyze_gradients(groups: list[dict[str, Any]], model_dir: Path, rvl_source: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    sys.path.insert(0, str(rvl_source.resolve() / "src"))
    from rvl_systems import types as helper
    from rvl_systems.grpo import compute_group_advantages
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir.resolve(), local_files_only=True, dtype=torch.float32
    )
    model.config.use_cache = False
    model.eval()
    trainer = HFCausalLMGRPOTrainer(
        model,
        config=HFTTrainerConfig(
            learning_rate=1e-6,
            clip_eps=0.2,
            max_grad_norm=1.0,
            advantage_eps=1e-6,
            clip_advantage=5.0,
            disable_dropout=True,
            objective_backend="torch",
        ),
    )
    state_before = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    results = []
    for group in groups:
        v0, parameter_tensors, parameter_count, v0_ratio = gradient_for_group(
            model, trainer, helper, compute_group_advantages, group["members"], "v0"
        )
        v1, _, _, v1_ratio = gradient_for_group(
            model, trainer, helper, compute_group_advantages, group["members"], "v1"
        )
        norm0 = float(torch.linalg.vector_norm(v0))
        norm1 = float(torch.linalg.vector_norm(v1))
        diff = v1 - v0
        diff_norm = float(torch.linalg.vector_norm(diff))
        denominator = norm0 * norm1
        cosine = float(torch.dot(v0, v1) / denominator) if denominator else None
        both_nonzero = (v0 != 0) & (v1 != 0)
        sign_reversals = int(((v0 * v1) < 0).sum())
        results.append({
            "task_id": group["task_id"],
            "trainable_parameter_tensors": parameter_tensors,
            "trainable_parameter_count": parameter_count,
            "v0_gradient_l2": norm0,
            "v1_gradient_l2": norm1,
            "gradient_difference_l2": diff_norm,
            "relative_gradient_difference": diff_norm / max(norm0, 1e-30),
            "gradient_cosine_similarity": cosine,
            "coordinate_sign_reversals": sign_reversals,
            "coordinates_nonzero_in_both_versions": int(both_nonzero.sum()),
            "max_recorded_behavior_logratio_v0": v0_ratio,
            "max_recorded_behavior_logratio_v1": v1_ratio,
        })
        del v0, v1, diff, both_nonzero
    state_unchanged = all(
        torch.equal(state_before[name], value.detach().cpu())
        for name, value in model.state_dict().items()
    )
    if not state_unchanged:
        raise RuntimeError("gradient analysis modified model weights")
    return {
        "model_state_unchanged": state_unchanged,
        "optimizer_steps_executed": 0,
        "objective": "Pinned RVL token-level clipped GRPO surrogate; exact recorded outputs, token ids, behavior logprobs, reward vectors, and initial checkpoint; no optimizer.step().",
        "model_gradient_groups": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT.parent / "tinystories-1m-pinned")
    parser.add_argument("--rvl-source", type=Path, default=ROOT.parent / "rvl-pinned")
    args = parser.parse_args()

    start = time.monotonic()
    from scripts.audit_verifier_reward_calibration import audit
    from scripts.audit_verifier_reward_calibration import independent_reward
    sys.path.insert(0, str(args.rvl_source.resolve() / "src"))
    from rvl_systems import types as helper
    from rvl_systems.grpo import compute_group_advantages

    audit_result = audit(CALIBRATION)
    if audit_result["status"] != "PASS_INDEPENDENT_RECONSTRUCTION":
        raise RuntimeError("frozen calibration independent reward audit did not pass")
    protocol = json.loads(PROTOCOL.read_text())
    model_files = protocol["model"]["files"]
    model_dir = args.model_dir.resolve()
    model_file_checks = {
        name: sha256_file(model_dir / name) == expected
        for name, expected in model_files.items()
    }
    if not all(model_file_checks.values()):
        raise RuntimeError("local model checkpoint does not match the frozen calibration")
    groups = [
        json.loads((CALIBRATION / "groups" / f"{task['id']}.json").read_text())
        for task in protocol["sampling"]["tasks"]
    ]
    for task, group in zip(protocol["sampling"]["tasks"], groups):
        for row in group["members"]:
            category, reward0, reward1 = independent_reward(
                row["output_text"], task["target_class"], protocol["reward_definition"]
            )
            if (category != row["surface_class"] or reward0 != float(row["v0"])
                    or reward1 != float(row["v1"])):
                raise RuntimeError("raw calibration rewards failed independent reconstruction")

    def advantages(rewards: list[float]) -> list[float]:
        samples = [
            helper.VerifiedGeneration(
                generation=helper.Generation(
                    prompt_id="group", prompt="prompt", response=str(index), logprob=0.0,
                    token_count=1, latency_s=0.0,
                ), reward=value, verifier_latency_s=0.0, verifier_version=0,
            )
            for index, value in enumerate(rewards)
        ]
        return [record.advantage for record in compute_group_advantages(samples, eps=1e-6, clip=5.0)]

    metrics = reward_metrics(groups, advantages)
    gradient_results = analyze_gradients(groups, model_dir, args.rvl_source)
    source_hashes = {
        "analysis_script": sha256_file(Path(__file__).resolve()),
        "calibration_protocol": sha256_file(PROTOCOL),
        "independent_reward_auditor": sha256_file(ROOT / "scripts/audit_verifier_reward_calibration.py"),
        "rvl_grpo": sha256_file(args.rvl_source / "src/rvl_systems/grpo.py"),
        "rvl_hf_trainer": sha256_file(args.rvl_source / "src/rvl_systems/hf_trainer.py"),
        "rvl_objective": sha256_file(args.rvl_source / "src/rvl_systems/triton_grpo.py"),
    }
    input_hashes = {
        str(path.relative_to(CALIBRATION)): sha256_file(path)
        for path in sorted(CALIBRATION.rglob("*")) if path.is_file()
    }
    result = {
        "schema": "vare.reward-calibration-postrun-analysis.v1",
        "status": "DESCRIPTIVE_NO_GO_PRESERVED",
        "calibration_status": audit_result["calibration_gate"],
        "independent_reward_audit": audit_result,
        "reward_and_advantage_metrics": metrics,
        "exact_grpo_gradient_analysis": gradient_results,
        "identities": {
            "model_revision": protocol["model"]["revision"],
            "model_files_match_frozen_manifest": model_file_checks,
            "rvl_revision": protocol["runtime"]["rvl_revision"],
            "rvl_source_directory": str(args.rvl_source.resolve()),
            "source_sha256": source_hashes,
            "calibration_input_sha256": input_hashes,
        },
        "analysis_runtime": {
            "device": "cpu",
            "precision": "fp32",
            "torch_threads": 1,
            "elapsed_seconds": round(time.monotonic() - start, 6),
        },
        "interpretation": {
            "training_direction": "Gradient direction is computed from the exact pinned GRPO objective for each saved group and reward version at the identical starting checkpoint. No optimizer step or model update is applied.",
            "inference_limit": "Four fixed synthetic calibration prompts are descriptive only; these results do not establish coding capability, causal policy regression, asynchronous behavior, or general verifier-drift prevalence.",
        },
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    digest = atomic_create_json(output, result)
    print(json.dumps({"output": str(output), "sha256": digest, "status": result["status"]}, indent=2))


if __name__ == "__main__":
    main()
