#!/usr/bin/env python3
"""Frozen CPU parity run for VARE trajectory targets and pinned RVL GRPO."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RVL_REVISION = "c7e646b043cb56e5ea3c2623bb8a61e065451f72"
LOCK_PATH = ROOT / "protocols/rvl_trajectory_gradient_parity_v5.lock.json"
OUTPUT_PATH = ROOT / "outputs/rvl_trajectory_gradient_parity_v5.json"
RVL_FILES = (
    "src/rvl_systems/hf_trainer.py",
    "src/rvl_systems/triton_grpo.py",
    "src/rvl_systems/grpo.py",
    "src/rvl_systems/types.py",
    "src/rvl_systems/lab/token_replay.py",
    "src/rvl_systems/lab/contracts.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_lock() -> dict[str, Any]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    require(lock.get("protocol_id") == "vare-rvl-trajectory-gradient-parity-v5", "unexpected protocol ID")
    require(lock.get("status") == "preregistered-before-run", "protocol is not preregistered")
    require(lock.get("rvl_revision") == RVL_REVISION, "protocol RVL revision mismatch")
    require(lock.get("runner_sha256") == sha256_file(Path(__file__)), "runner hash differs from protocol")
    require(set(lock["rvl_source_files_sha256"]) == set(RVL_FILES), "locked RVL files differ from runner")
    require({"seed", "runtime", "trainer", "fixture", "numeric_tolerances"} <= lock.keys(),
            "protocol is missing required sections")
    require({"prompt_id", "prompt", "prompt_token_ids", "temperature", "groups", "total_samples"} <= lock["fixture"].keys(),
            "fixture protocol is missing required fields")
    require({"clip_eps", "max_grad_norm", "advantage_eps", "advantage_clip"} <= lock["trainer"].keys(),
            "trainer protocol is missing required fields")
    require({"loss_abs_tolerance", "gradient_abs_tolerance", "negative_control_min_gradient_delta"}
            <= lock["numeric_tolerances"].keys(), "numeric tolerance protocol is missing required fields")
    for relpath, expected in lock["vare_source_files_sha256"].items():
        require(sha256_file(ROOT / relpath) == expected, f"VARE source hash mismatch: {relpath}")
    return lock


def extract_pinned_rvl(git_path: Path, destination: Path, expected_hashes: dict[str, str]) -> dict[str, str]:
    resolved = subprocess.run(
        ["git", "-C", str(git_path), "rev-parse", "--verify", f"{RVL_REVISION}^{{commit}}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    require(resolved == RVL_REVISION, "RVL revision resolved to a different commit")
    archive = subprocess.run(
        ["git", "-C", str(git_path), "archive", "--format=tar", RVL_REVISION],
        check=True, capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(destination, filter="data")
    source_hashes = {name: sha256_file(destination / name) for name in RVL_FILES}
    require(source_hashes == expected_hashes, "one or more pinned RVL source hashes differ")
    sys.path.insert(0, str(destination / "src"))
    return source_hashes


def clone_state_dict(model: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def state_dict_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.keys() == right.keys() and all(
        left[key].dtype == right[key].dtype
        and left[key].shape == right[key].shape
        and bool((left[key] == right[key]).all())
        for key in left
    )


def nested_equal(left: Any, right: Any, torch: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k], torch) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            nested_equal(a, b, torch) for a, b in zip(left, right, strict=True)
        )
    return left == right


def parameter_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def module_modes(model: Any) -> dict[str, bool]:
    return {name: bool(module.training) for name, module in model.named_modules()}


def current_token_logprobs(torch: Any, model: Any, prompt_ids: list[int], response_ids: list[int], temperature: float) -> list[float]:
    sequence = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=sequence).logits[0]
        start = len(prompt_ids) - 1
        response_logits = logits[start : start + len(response_ids)].float() / temperature
        targets = torch.tensor(response_ids, dtype=torch.long)
        values = torch.log_softmax(response_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return [float(value) for value in values.tolist()]


def independent_materialized_loss_and_grads(
    *, torch: Any, model: Any, samples: list[Any], experiences: list[Any],
    target_groups: dict[str, tuple[Any, ...]], advantages: list[float], clip_eps: float,
) -> tuple[float, dict[str, Any]]:
    """Independent autograd oracle; does not call RVL objective helpers."""
    by_group: dict[str, list[int]] = {}
    for index, experience in enumerate(experiences):
        by_group.setdefault(experience.attempt.metadata["vare_rollout_group"], []).append(index)
    total_samples = len(samples)
    loss = torch.zeros((), dtype=torch.float32)
    for group_id, indices in by_group.items():
        group_targets = target_groups[group_id]
        group_size = len(indices)
        for index in indices:
            experience = experiences[index]
            sample = samples[index]
            generation_index = experience.attempt.metadata["rvl_generation_index"]
            execution_id = f"{group_id}:member-{generation_index}"
            targets = [target for target in group_targets if target.execution_id == execution_id]
            require(bool(targets), f"missing trajectory targets for {execution_id}")
            prompt_ids = [int(value) for value in sample.generation.metadata["prompt_token_ids"]]
            response_ids = [int(value) for value in sample.generation.metadata["response_token_ids"]]
            sequence = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
            logits = model(input_ids=sequence).logits[0]
            start = len(prompt_ids) - 1
            temperature = float(sample.generation.metadata["sampling_temperature"])
            log_probs = torch.log_softmax(
                logits[start : start + len(response_ids)].float() / temperature, dim=-1
            )
            advantage = torch.tensor(float(advantages[index]), dtype=torch.float32)
            weighted_objective = torch.zeros((), dtype=torch.float32)
            for target in targets:
                token_index = int(target.token_index)
                current = log_probs[token_index, int(target.token_id)]
                log_ratio = torch.clamp(
                    current - float(target.behavior_logprob), min=-20.0, max=20.0
                )
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                token_objective = torch.minimum(ratio * advantage, clipped_ratio * advantage)
                weighted_objective = weighted_objective + float(target.weight) * token_objective
            # Each group's weights sum to one. RVL averages the full batch,
            # so scale each group objective by group_size / total_samples.
            loss = loss - weighted_objective * (group_size / total_samples)
    loss.backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return float(loss.detach()), grads


def run(git_path: Path) -> dict[str, Any]:
    lock = load_lock()
    fixture = lock["fixture"]
    import torch
    torch.set_num_threads(int(lock["runtime"]["threads"]))
    require(not torch.cuda.is_available(), "protocol requires CPU; CUDA must be unavailable/unused")
    random.seed(int(lock["seed"]))
    torch.manual_seed(int(lock["seed"]))

    with tempfile.TemporaryDirectory(prefix="vare-rvl-parity-") as temp_dir:
        temp_root = Path(temp_dir)
        source_dir = temp_root / "rvl"
        source_dir.mkdir()
        source_hashes = extract_pinned_rvl(git_path, source_dir, lock["rvl_source_files_sha256"])

        from rvl_systems.lab.token_replay import TokenReplay
        from rvl_systems.types import Generation, VerifiedGeneration
        from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
        from vare.integrations.rvl import RVLTokenReplayReader
        from vare.integrations.rvl_grpo import RVLGRPOHooks
        from vare.integrations.rvl_trajectory import rvl_experiences_to_trajectory_group

        class TinyCausalLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.logits_by_context = torch.nn.Embedding(32, 32)
                torch.nn.init.normal_(self.logits_by_context.weight, mean=0.0, std=0.15)

            def forward(self, *, input_ids):
                from types import SimpleNamespace
                return SimpleNamespace(logits=self.logits_by_context(input_ids))

        model = TinyCausalLM()
        model.train()
        trainer = HFCausalLMGRPOTrainer(
            model,
            config=HFTTrainerConfig(
                learning_rate=1e-3,
                clip_eps=float(lock["trainer"]["clip_eps"]),
                max_grad_norm=float(lock["trainer"]["max_grad_norm"]),
                advantage_eps=float(lock["trainer"]["advantage_eps"]),
                clip_advantage=float(lock["trainer"]["advantage_clip"]),
                disable_dropout=True,
                objective_backend="torch",
            ),
        )
        database_path = temp_root / "token-replay.sqlite"
        replay = TokenReplay(database_path, capacity=8)
        replay.bind_verification({"verifier": "synthetic-fixture-v1", "seed": lock["seed"]})

        prompt_ids = list(fixture["prompt_token_ids"])
        prompt_id, prompt, temperature = fixture["prompt_id"], fixture["prompt"], float(fixture["temperature"])
        rewards_by_group = fixture["groups"]
        inserted_rows: list[dict[str, Any]] = []
        for group_id, group in rewards_by_group.items():
            verified: list[Any] = []
            for member_index, item in enumerate(group):
                response_ids = list(item["response_token_ids"])
                behavior_logps = current_token_logprobs(torch, model, prompt_ids, response_ids, temperature)
                require(len(behavior_logps) == len(response_ids), "logprob/token cardinality mismatch")
                generation = Generation(
                    prompt_id=prompt_id,
                    prompt=prompt,
                    response=item["response"],
                    logprob=math.fsum(behavior_logps),
                    token_count=len(response_ids),
                    latency_s=0.001 * (member_index + 1),
                    metadata={
                        "family": "synthetic-parity",
                        "backend": "tiny-causal-lm",
                        "model": "vare-parity/tiny-causal-lm-v1",
                        "precision": "fp32",
                        "policy_id": "policy-7",
                        "sampling_temperature": temperature,
                        "logprob_distribution": "neutral_temperature_scaled_policy",
                        "prompt_token_ids": prompt_ids,
                        "response_token_ids": response_ids,
                        "response_token_logprobs": behavior_logps,
                    },
                )
                verified.append(VerifiedGeneration(
                    generation=generation,
                    reward=float(item["reward"]),
                    verifier_latency_s=0.001,
                    verifier_version=3,
                    metadata={
                        "verifier_name": "synthetic-fixture-v1",
                        "trusted": True,
                        "confidence": 1.0,
                        "synthetic_fixture": True,
                    },
                ))
            require(replay.put(group_id, 7, verified), f"TokenReplay rejected group {group_id}")
            inserted_rows.append({
                "group_id": group_id,
                "rewards": [float(row.reward) for row in verified],
                "response_lengths": [int(row.generation.token_count) for row in verified],
                "response_token_ids": [list(row.generation.metadata["response_token_ids"]) for row in verified],
                "behavior_token_logprobs": [list(row.generation.metadata["response_token_logprobs"]) for row in verified],
            })
        replay.db.close()

        experiences = RVLTokenReplayReader(database_path).ready_experiences(
            current_policy_version=7, current_verifier_version=3,
        )
        require(len(experiences) == int(fixture["total_samples"]), "unexpected replay sample count")
        hooks = RVLGRPOHooks(
            backend=object(), trainer=trainer, eval_tasks=[], score_fn=lambda _task, _response: 0.0,
            generation_factory=Generation, verified_generation_factory=VerifiedGeneration,
        )
        samples = [hooks._to_verified_generation(item) for item in experiences]
        advantages = RVLGRPOHooks._group_relative_advantages(
            experiences, samples,
            eps=float(trainer.config.advantage_eps), clip=float(trainer.config.clip_advantage),
        )
        require(advantages is not None and len(advantages) == len(samples), "group advantages missing")
        groups: dict[str, list[Any]] = {}
        for experience in experiences:
            groups.setdefault(experience.attempt.metadata["vare_rollout_group"], []).append(experience)
        target_groups = {
            group_id: rvl_experiences_to_trajectory_group(group_experiences).materialize()
            for group_id, group_experiences in groups.items()
        }
        target_sums = {
            group_id: math.fsum(float(target.weight) for target in targets)
            for group_id, targets in target_groups.items()
        }
        require(all(math.isclose(value, 1.0, abs_tol=1e-12) for value in target_sums.values()),
                "materialized group target weights must sum to one")

        rewards = [float(sample.reward) for sample in samples]
        mean_reward = math.fsum(rewards) / len(rewards)
        variance = math.fsum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        scale = math.sqrt(variance + float(lock["trainer"]["advantage_eps"]))
        merged_advantages = [
            max(-float(lock["trainer"]["advantage_clip"]), min(float(lock["trainer"]["advantage_clip"]),
                (reward - mean_reward) / scale))
            for reward in rewards
        ]

        initial_model_state = clone_state_dict(model)
        initial_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
        initial_torch_rng = torch.get_rng_state().clone()
        initial_python_rng = random.getstate()
        initial_modes = module_modes(model)
        initial_fingerprint = parameter_fingerprint(model)

        model.zero_grad(set_to_none=True)
        reference_loss, reference_grads = independent_materialized_loss_and_grads(
            torch=torch, model=model, samples=samples, experiences=experiences,
            target_groups=target_groups, advantages=advantages, clip_eps=float(trainer.config.clip_eps),
        )
        reference_grad_norm = math.sqrt(math.fsum(
            float(torch.sum(gradient.float() ** 2)) for gradient in reference_grads.values()
        ))
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(initial_torch_rng)
        random.setstate(initial_python_rng)

        intercepted_calls = 0
        original_step = trainer.optimizer.step

        def counted_no_op_step(*_args: Any, **_kwargs: Any) -> None:
            nonlocal intercepted_calls
            intercepted_calls += 1
            return None

        trainer.optimizer.step = counted_no_op_step
        try:
            trainer_result = trainer.train_step(samples, advantages=advantages)
        finally:
            trainer.optimizer.step = original_step
        native_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        require(intercepted_calls == 1, "expected exactly one intercepted optimizer step")
        require(native_grads.keys() == reference_grads.keys(), "gradient parameter sets differ")
        grad_diffs = {
            name: float((native_grads[name] - reference_grads[name]).abs().max())
            for name in native_grads
        }
        max_grad_diff = max(grad_diffs.values(), default=0.0)
        native_loss = float(trainer_result["loss"])
        loss_diff = abs(native_loss - reference_loss)
        require(math.isfinite(native_loss) and math.isfinite(reference_loss), "loss must be finite")
        require(loss_diff <= float(lock["numeric_tolerances"]["loss_abs_tolerance"]), "scalar loss mismatch")
        require(max_grad_diff <= float(lock["numeric_tolerances"]["gradient_abs_tolerance"]), "gradient mismatch")
        require(float(trainer_result["grad_norm"]) < float(lock["trainer"]["max_grad_norm"]), "unexpected grad clipping")
        require(state_dict_equal(initial_model_state, clone_state_dict(model)), "model state changed")
        require(nested_equal(initial_optimizer_state, trainer.optimizer.state_dict(), torch), "optimizer state changed")
        require(torch.equal(initial_torch_rng, torch.get_rng_state()), "torch CPU RNG changed")
        require(initial_python_rng == random.getstate(), "Python RNG changed")
        require(initial_fingerprint == parameter_fingerprint(model), "parameter fingerprint changed")

        for name, module in model.named_modules():
            module.training = initial_modes[name]
        require(initial_modes == module_modes(model), "module modes were not restored")
        model.zero_grad(set_to_none=True)

        model.zero_grad(set_to_none=True)
        merged_loss, merged_grads = independent_materialized_loss_and_grads(
            torch=torch, model=model, samples=samples, experiences=experiences,
            target_groups=target_groups, advantages=merged_advantages,
            clip_eps=float(trainer.config.clip_eps),
        )
        merged_grad_delta = max(
            float((merged_grads[name] - reference_grads[name]).abs().max())
            for name in reference_grads
        )
        model.zero_grad(set_to_none=True)
        require(merged_grad_delta > float(lock["numeric_tolerances"]["negative_control_min_gradient_delta"]),
                "prompt-merged negative control did not diverge")
        require(state_dict_equal(initial_model_state, clone_state_dict(model)),
                "negative control changed model state")

        batch_scaled_weight = math.fsum(
            math.fsum(
                float(target.weight) * len(groups[group_id]) / len(samples)
                for target in targets
            )
            for group_id, targets in target_groups.items()
        )
        require(math.isclose(batch_scaled_weight, 1.0, rel_tol=0.0, abs_tol=1e-12),
                "batch-scaled trajectory weights do not sum to one")
        require(any(bool(torch.count_nonzero(gradient)) for gradient in native_grads.values()),
                "fixture gradients are vacuously zero")
        require(all(bool(torch.isfinite(gradient).all()) for gradient in native_grads.values()),
                "nonfinite native gradients")

        return {
            "protocol_id": lock["protocol_id"],
            "protocol_sha256": sha256_file(LOCK_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "rvl_revision": RVL_REVISION,
            "rvl_source_files_sha256": source_hashes,
            "vare_source_files_sha256": lock["vare_source_files_sha256"],
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
                "device": "cpu",
                "threads": torch.get_num_threads(),
                "paid_compute": False,
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            },
            "fixture": {
                "prompt_id": prompt_id,
                "groups": inserted_rows,
                "sample_count": len(samples),
                "temperature": temperature,
                "groupwise_advantages": [float(value) for value in advantages],
                "prompt_merged_control_advantages": merged_advantages,
                "per_group_materialized_weight_sums": target_sums,
                "batch_scaled_target_weight_sum": batch_scaled_weight,
            },
            "result": {
                "status": "completed",
                "optimizer_step_calls_intercepted": intercepted_calls,
                "parameter_update_performed": False,
                "native_trainer_loss": native_loss,
                "materialized_reference_loss": reference_loss,
                "absolute_loss_difference": loss_diff,
                "reference_gradient_global_norm": reference_grad_norm,
                "native_gradient_norm": float(trainer_result["grad_norm"]),
                "max_absolute_gradient_difference": max_grad_diff,
                "per_parameter_max_absolute_gradient_difference": grad_diffs,
                "prompt_merged_control_loss": merged_loss,
                "prompt_merged_control_max_gradient_difference": merged_grad_delta,
                "model_parameters_and_buffers_unchanged": True,
                "optimizer_state_unchanged": True,
                "torch_cpu_rng_unchanged": True,
                "python_rng_unchanged": True,
                "module_modes_restored": True,
                "native_gradients_finite": True,
                "native_gradients_nonzero": True,
                "claim_scope": "CPU loss/gradient interoperability only; no capability gain, real-model training, or RSI claim",
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvl-git", type=Path, required=True, help="local Git object database containing pinned RVL")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        result = run(args.rvl_git.resolve())
    except Exception as exc:
        failure_path = args.output.with_name(args.output.stem + ".failure.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(json.dumps({
            "status": "failed",
            "protocol_path": str(LOCK_PATH),
            "protocol_sha256": sha256_file(LOCK_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "note": "The runner always replaces optimizer.step with a counted no-op before the pinned train_step; inspect the traceback to identify whether backward was reached.",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    result["result"]["wall_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["result"]["status"],
        "output": str(args.output),
        "loss_abs_diff": result["result"]["absolute_loss_difference"],
        "max_grad_abs_diff": result["result"]["max_absolute_gradient_difference"],
        "merged_control_grad_delta": result["result"]["prompt_merged_control_max_gradient_difference"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
