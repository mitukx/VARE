#!/usr/bin/env python3
"""Frozen CPU integration audit for VARE rollout groups and pinned RVL GRPO."""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/rvl_grpo_group_boundary_real_trainer_v1.lock.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class ResourceLimit(RuntimeError):
    pass


def _timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("frozen wall-time limit reached")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tensor_fingerprint(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_difference(reference: dict[str, Any], candidate: dict[str, Any], torch: Any) -> tuple[int, float]:
    changed = 0
    maximum = 0.0
    for name, before in reference.items():
        after = candidate[name].detach().to(device="cpu")
        if not torch.equal(before, after):
            changed += 1
            maximum = max(maximum, float((before - after).abs().max().item()))
    return changed, maximum


def _normalized(rewards: list[float], eps: float, clip: float) -> list[float]:
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    scale = math.sqrt(variance + eps)
    return [max(-clip, min(clip, (reward - mean) / scale)) for reward in rewards]


def _nested_equal(left: Any, right: Any, torch: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_nested_equal(left[key], right[key], torch) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_nested_equal(a, b, torch) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _validate_inputs(lock: dict[str, Any], model_path: Path, rvl_source: Path) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("protocol is frozen for macOS arm64")
    if not model_path.is_dir() or lock["model"]["revision"] not in str(model_path):
        raise FileNotFoundError("model path must be the exact frozen snapshot")
    if not (rvl_source / "src/rvl_systems/hf_backend.py").is_file():
        raise FileNotFoundError("--rvl-source must contain src/rvl_systems/hf_backend.py")
    if _sha(Path(__file__)) != lock["runner_sha256"]:
        raise RuntimeError("runner hash differs from the frozen protocol")
    auditor = ROOT / "scripts/audit_rvl_grpo_group_boundary_real_trainer_v1.py"
    if _sha(auditor) != lock["auditor_sha256"]:
        raise RuntimeError("auditor hash differs from the frozen protocol")
    adapter = ROOT / "src/vare/integrations/rvl_grpo.py"
    if _sha(adapter) != lock["vare_adapter_sha256"]:
        raise RuntimeError("VARE adapter source differs from the frozen protocol")
    actual_rvl = {
        path.relative_to(rvl_source).as_posix(): _sha(path)
        for path in sorted((rvl_source / "src/rvl_systems").rglob("*.py"))
    }
    if actual_rvl != lock["rvl_source_files_sha256"]:
        raise RuntimeError("pinned RVL source hashes do not match")
    for name, expected in lock["model"]["files_sha256"].items():
        if _sha(model_path / name) != expected:
            raise RuntimeError(f"pinned model file hash does not match: {name}")


def run(model_path: Path, rvl_source: Path, output_dir: Path) -> dict[str, Any]:
    lock = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _validate_inputs(lock, model_path, rvl_source)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retained output: {output_dir}")
    output_dir.mkdir(parents=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(lock["runtime"]["threads"])
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(rvl_source / "src"))

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    expected_runtime = lock["runtime"]
    if sys.version.split()[0] != expected_runtime["python"]:
        raise RuntimeError(f"Python mismatch: expected {expected_runtime['python']}")
    if torch.__version__ != expected_runtime["torch"]:
        raise RuntimeError(f"PyTorch mismatch: expected {expected_runtime['torch']}")
    if transformers.__version__ != expected_runtime["transformers"]:
        raise RuntimeError(f"Transformers mismatch: expected {expected_runtime['transformers']}")
    torch.set_num_threads(int(expected_runtime["threads"]))
    torch.manual_seed(int(lock["sampling"]["seed"]))

    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from rvl_systems.types import VerifiedGeneration
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    from vare.types import Attempt, Experience, Task, Verification

    prompt_spec = lock["sampling"]
    prompt = prompt_spec["rendered_prompt"]
    state: dict[str, Any] = {
        "protocol_id": lock["protocol_id"],
        "protocol_sha256": _sha(PROTOCOL_PATH),
        "runner_sha256": _sha(Path(__file__)),
        "vare_adapter_sha256": _sha(ROOT / "src/vare/integrations/rvl_grpo.py"),
        "rvl_revision": lock["rvl_revision"],
        "source_revision": lock["source_revision"],
        "model_id": MODEL_ID,
        "model_revision": lock["model"]["revision"],
        "status": "running",
        "device": "cpu",
        "responses": [],
        "arms": {},
        "wall_started_unix": time.time(),
    }
    _write_json(output_dir / "progress.json", state)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(int(expected_runtime["max_wall_seconds"]))
    max_rss = int(expected_runtime["max_peak_rss_bytes"])
    backend = None
    trainer = None
    try:
        if torch.cuda.is_available():
            raise RuntimeError("CUDA unexpectedly available; protocol forbids accelerator use")
        state["mps_available_but_disabled"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        backend = HFLocalBackend(
            model_name=str(model_path),
            max_new_tokens=int(prompt_spec["max_new_tokens"]),
            device="cpu",
            precision="fp32",
        )
        model = backend.model
        if next(model.parameters()).device.type != "cpu":
            raise RuntimeError("loaded model is not on CPU")
        state["load_seconds"] = time.monotonic() - started
        state["load_peak_rss_bytes"] = _peak_rss_bytes()
        if state["load_peak_rss_bytes"] > max_rss:
            raise ResourceLimit("peak RSS cap exceeded after model load")

        generations = asyncio.run(
            backend.generate(
                prompt_spec["prompt_id"],
                prompt,
                n=int(prompt_spec["samples"]),
                temperature=float(prompt_spec["temperature"]),
                seed=int(prompt_spec["seed"]),
            )
        )
        if len(generations) != int(prompt_spec["samples"]):
            raise RuntimeError("backend returned an unexpected generation count")
        rewards = [value for group in prompt_spec["group_rewards"] for value in group]
        if len(rewards) != len(generations):
            raise RuntimeError("frozen reward vector does not match generated samples")
        state["responses"] = [
            {
                "index": index,
                "text": generation.response,
                "reward": rewards[index],
                "token_count": generation.token_count,
                "logprob": generation.logprob,
                "response_token_ids": generation.metadata.get("response_token_ids", []),
                "response_token_logprobs": generation.metadata.get(
                    "response_token_logprobs", []
                ),
                "prompt_token_ids": generation.metadata.get("prompt_token_ids", []),
            }
            for index, generation in enumerate(generations)
        ]
        distinct_gate = (
            bool(state["responses"][1]["response_token_ids"])
            and all(
                state["responses"][1]["response_token_ids"]
                != state["responses"][index]["response_token_ids"]
                for index in (0, 2, 3)
            )
        )
        state["response_diversity_gate"] = distinct_gate
        _write_json(output_dir / "progress.json", state)
        if not distinct_gate:
            raise RuntimeError("frozen first-group response-diversity gate failed; no resampling")

        trainer = HFCausalLMGRPOTrainer(
            model,
            config=HFTTrainerConfig(
                learning_rate=float(lock["update"]["learning_rate"]),
                clip_eps=float(lock["update"]["clip_eps"]),
                advantage_eps=float(lock["update"]["advantage_epsilon"]),
                clip_advantage=float(lock["update"]["advantage_clip"]),
                objective_backend="torch",
            ),
        )
        import rvl_systems.hf_trainer as hf_trainer_module

        advantage_captures: list[list[float]] = []
        original_compute_advantages = hf_trainer_module.compute_group_advantages

        def capture_compute_advantages(samples, *, eps=1e-6, clip=5.0):
            records = original_compute_advantages(samples, eps=eps, clip=clip)
            advantage_captures.append([record.advantage for record in records])
            return records

        hf_trainer_module.compute_group_advantages = capture_compute_advantages
        generation_type = type(generations[0])
        task = Task(id=prompt_spec["prompt_id"], prompt=prompt, family="synthetic-group-boundary")
        verified = []
        experiences = []
        for index, (generation, reward) in enumerate(zip(generations, rewards, strict=True)):
            verification = Verification(
                score=float(reward),
                passed=bool(reward),
                confidence=1.0,
                verifier_version=0,
                verifier_name="frozen-synthetic-group-boundary-witness",
                trusted=True,
            )
            verified.append(
                VerifiedGeneration(
                    generation=generation,
                    reward=float(reward),
                    verifier_latency_s=0.0,
                    verifier_version=0,
                    metadata={"vare_trusted": True},
                )
            )
            experiences.append(
                Experience(
                    attempt=Attempt(
                        task=task,
                        output=generation.response,
                        policy_id="policy-0",
                        policy_version=0,
                        created_step=0,
                        logprob=generation.logprob,
                        latency_ms=1000.0 * generation.latency_s,
                        metadata={"rvl_generation": RVLGRPOHooks._generation_fields(generation)},
                    ),
                    verification=verification,
                    policy_lag=0,
                    verifier_lag=0,
                    shift_score=0.0,
                )
            )

        class RecordingTrainer(HFCausalLMGRPOTrainer):
            def __init__(self, inner: HFCausalLMGRPOTrainer) -> None:
                self.__dict__ = inner.__dict__
                self.calls = 0
                self.optimizer_step_calls = 0
                self.arm: dict[str, Any] | None = None
                self.initial_state: dict[str, Any] | None = None
                original_optimizer_step = self.optimizer.step

                def count_optimizer_step(*args, **kwargs):
                    self.optimizer_step_calls += 1
                    return original_optimizer_step(*args, **kwargs)

                self.optimizer.step = count_optimizer_step

            def train_step(self, samples, *, advantages=None):
                self.calls += 1
                if self.calls != 1:
                    raise RuntimeError("more than one optimizer step in an arm")
                advantage_captures.clear()
                metrics = super().train_step(samples, advantages=advantages)
                if advantages is None:
                    if len(advantage_captures) != 1:
                        raise RuntimeError("could not capture native trainer advantages")
                    consumed = advantage_captures[0]
                    capture_method = "intercepted pinned trainer compute_group_advantages output"
                else:
                    consumed = [float(value) for value in advantages]
                    capture_method = "explicit VARE adapter argument received by train_step"
                gradients = [parameter.grad for parameter in self.model.parameters()
                             if parameter.grad is not None]
                finite = sum(int(bool(torch.isfinite(gradient).all().item())) for gradient in gradients)
                nonzero = sum(int(bool(torch.count_nonzero(gradient).item())) for gradient in gradients)
                if self.initial_state is None:
                    raise RuntimeError("missing arm incumbent snapshot")
                changed = 0
                max_delta = 0.0
                for key, value in self.initial_state["model"].items():
                    current = self.model.state_dict()[key].detach().cpu()
                    if not torch.equal(value, current):
                        changed += 1
                        max_delta = max(max_delta, float((value - current).abs().max().item()))
                self.arm = {
                    "consumed_advantages": consumed,
                    "advantage_capture_method": capture_method,
                    "metrics": metrics,
                    "train_step_calls": self.calls,
                    "optimizer_step_calls": self.optimizer_step_calls,
                    "gradient_tensor_count": len(gradients),
                    "finite_gradient_tensor_count": finite,
                    "nonzero_gradient_tensor_count": nonzero,
                    "changed_parameter_tensor_count": changed,
                    "max_abs_parameter_delta": max_delta,
                    "post_step_parameter_fingerprint": _tensor_fingerprint(
                        self.model.state_dict()
                    ),
                }
                return metrics

        trainer = RecordingTrainer(trainer)
        initial_state = trainer.snapshot_training_state()
        initial_fingerprint = _tensor_fingerprint(initial_state["model"])
        initial_mode = bool(trainer.model.training)
        initial_rng = torch.get_rng_state().clone()
        native_model_state: dict[str, Any] | None = None
        state["incumbent_parameter_fingerprint"] = initial_fingerprint
        state["arms"] = {}

        async def run_adapter_arm(group_ids: list[str], group_sizes: list[int]) -> dict[str, Any]:
            trainer.restore_training_state(initial_state)
            torch.set_rng_state(initial_rng)
            initial_state_matched = (
                _tensor_fingerprint(trainer.model.state_dict()) == initial_fingerprint
                and _nested_equal(trainer.optimizer.state_dict(), initial_state["optimizer"], torch)
                and torch.equal(torch.get_rng_state(), initial_rng)
                and bool(trainer.model.training) == initial_mode
            )
            if not initial_state_matched:
                raise RuntimeError("failed to restore exact incumbent before adapter arm")
            trainer.calls = 0
            trainer.optimizer_step_calls = 0
            trainer.arm = None
            trainer.initial_state = initial_state
            for experience, group_id, group_size in zip(
                experiences, group_ids, group_sizes, strict=True
            ):
                experience.attempt.metadata["vare_rollout_group"] = group_id
                experience.attempt.metadata["vare_rollout_group_size"] = group_size
            hooks = RVLGRPOHooks(
                backend=backend,
                trainer=trainer,
                eval_tasks=[],
                score_fn=lambda _task, _response: 0.0,
                config=RVLGRPOConfig(train_temperature=float(prompt_spec["temperature"])),
                generation_factory=generation_type,
                verified_generation_factory=VerifiedGeneration,
            )
            arm_started = time.monotonic()
            candidate_id = await hooks.train_candidate("policy-0", experiences)
            arm_result = dict(trainer.arm or {})
            arm_result["initial_parameter_fingerprint"] = initial_fingerprint
            arm_result["initial_state_matched"] = initial_state_matched
            candidate_state = hooks._states[candidate_id]
            if native_model_state is None:
                raise RuntimeError("native update state is missing")
            arm_result["max_abs_parameter_difference_vs_native"] = _parameter_difference(
                native_model_state, candidate_state["model"], torch
            )[1]
            arm_result["post_step_parameter_fingerprint"] = _tensor_fingerprint(
                candidate_state["model"]
            )
            arm_result["elapsed_seconds"] = time.monotonic() - arm_started
            arm_result["incumbent_restored_exact"] = (
                _tensor_fingerprint(trainer.model.state_dict()) == initial_fingerprint
                and _nested_equal(
                    trainer.optimizer.state_dict(), initial_state["optimizer"], torch
                )
                and torch.equal(torch.get_rng_state(), initial_rng)
                and bool(trainer.model.training) == initial_mode
            )
            hooks._states.clear()
            return arm_result

        trainer.initial_state = initial_state
        baseline_started = time.monotonic()
        native_initial_state_matched = (
            _tensor_fingerprint(trainer.model.state_dict()) == initial_fingerprint
            and _nested_equal(trainer.optimizer.state_dict(), initial_state["optimizer"], torch)
            and torch.equal(torch.get_rng_state(), initial_rng)
            and bool(trainer.model.training) == initial_mode
        )
        if not native_initial_state_matched:
            raise RuntimeError("failed to verify native-arm incumbent state")
        trainer.calls = 0
        trainer.optimizer_step_calls = 0
        advantage_captures.clear()
        trainer.train_step(verified)
        native_arm = dict(trainer.arm or {})
        native_arm["initial_parameter_fingerprint"] = initial_fingerprint
        native_arm["initial_state_matched"] = native_initial_state_matched
        native_arm["elapsed_seconds"] = time.monotonic() - baseline_started
        native_arm["incumbent_restored_exact"] = False
        native_fingerprint = native_arm.get("post_step_parameter_fingerprint")
        native_model_state = {
            name: value.detach().to(device="cpu").clone()
            for name, value in trainer.model.state_dict().items()
        }
        native_arm["max_abs_parameter_difference_vs_native"] = 0.0
        trainer.restore_training_state(initial_state)
        native_arm["incumbent_restored_exact"] = (
            _tensor_fingerprint(trainer.model.state_dict()) == initial_fingerprint
            and _nested_equal(trainer.optimizer.state_dict(), initial_state["optimizer"], torch)
            and torch.equal(torch.get_rng_state(), initial_rng)
            and bool(trainer.model.training) == initial_mode
        )
        state["arms"]["native_prompt_grouping"] = native_arm
        _write_json(output_dir / "progress.json", state)

        control_arm = asyncio.run(
            run_adapter_arm(["vare-single-group"] * 8, [8] * 8)
        )
        state["arms"]["adapter_single_group_control"] = control_arm
        _write_json(output_dir / "progress.json", state)
        treatment_arm = asyncio.run(
            run_adapter_arm(
                ["vare-group-a"] * 4 + ["vare-group-b"] * 4,
                [4] * 8,
            )
        )
        state["arms"]["adapter_two_group_treatment"] = treatment_arm
        _write_json(output_dir / "progress.json", state)

        def reject_malformed_batch(batch: list[Experience]) -> dict[str, Any]:
            trainer.restore_training_state(initial_state)
            torch.set_rng_state(initial_rng)
            before = _tensor_fingerprint(trainer.model.state_dict())
            trainer.calls = 0
            trainer.optimizer_step_calls = 0
            invalid_hooks = RVLGRPOHooks(
                backend=backend,
                trainer=trainer,
                eval_tasks=[],
                score_fn=lambda _task, _response: 0.0,
                config=RVLGRPOConfig(train_temperature=float(prompt_spec["temperature"])),
                generation_factory=generation_type,
                verified_generation_factory=VerifiedGeneration,
            )
            rejected = False
            error = None
            try:
                asyncio.run(invalid_hooks.train_candidate("policy-0", batch))
            except ValueError as exc:
                rejected = True
                error = str(exc)
            result = {
                "rejected": rejected,
                "error": error,
                "train_step_calls": trainer.calls,
                "optimizer_step_calls": trainer.optimizer_step_calls,
                "incumbent_unchanged": (
                    _tensor_fingerprint(trainer.model.state_dict()) == before
                    and _nested_equal(trainer.optimizer.state_dict(), initial_state["optimizer"], torch)
                    and torch.equal(torch.get_rng_state(), initial_rng)
                    and bool(trainer.model.training) == initial_mode
                ),
            }
            invalid_hooks._states.clear()
            return result

        incomplete_batch = copy.deepcopy(experiences[:3])
        for experience in incomplete_batch:
            experience.attempt.metadata["vare_rollout_group"] = "invalid-incomplete"
            experience.attempt.metadata["vare_rollout_group_size"] = 4
        mixed_batch = copy.deepcopy(experiences[:2])
        mixed_batch[0].attempt.metadata["vare_rollout_group"] = "invalid-mixed"
        mixed_batch[0].attempt.metadata["vare_rollout_group_size"] = 2
        mixed_batch[1].attempt.metadata.pop("vare_rollout_group", None)
        mixed_batch[1].attempt.metadata.pop("vare_rollout_group_size", None)
        state["negative_controls"] = {
            "incomplete_group": reject_malformed_batch(incomplete_batch),
            "mixed_group_metadata": reject_malformed_batch(mixed_batch),
        }

        state["same_group_update_matches_native"] = (
            control_arm.get("post_step_parameter_fingerprint") == native_fingerprint
        )
        state["two_group_update_differs_from_native"] = (
            treatment_arm.get("post_step_parameter_fingerprint") != native_fingerprint
        )
        state["max_abs_parameter_difference_two_group_vs_native"] = treatment_arm[
            "max_abs_parameter_difference_vs_native"
        ]
        state["wall_seconds"] = time.monotonic() - started
        state["peak_rss_bytes"] = _peak_rss_bytes()
        frozen_rewards = [value for group in prompt_spec["group_rewards"] for value in group]
        expected_native = _normalized(
            frozen_rewards,
            float(lock["update"]["advantage_epsilon"]),
            float(lock["update"]["advantage_clip"]),
        )
        expected_groupwise = _normalized(
            frozen_rewards[:4],
            float(lock["update"]["advantage_epsilon"]),
            float(lock["update"]["advantage_clip"]),
        ) + _normalized(
            frozen_rewards[4:],
            float(lock["update"]["advantage_epsilon"]),
            float(lock["update"]["advantage_clip"]),
        )
        state["frozen_gate_checks"] = {
            "native_vector_matches_frozen_prompt_normalization": all(
                abs(actual - expected) <= 1e-10
                for actual, expected in zip(native_arm["consumed_advantages"], expected_native, strict=True)
            ),
            "same_group_control_matches_native_vector_and_update": (
                control_arm["consumed_advantages"] == native_arm["consumed_advantages"]
                and control_arm["post_step_parameter_fingerprint"] == native_fingerprint
                and control_arm["max_abs_parameter_difference_vs_native"] == 0.0
            ),
            "two_group_vector_matches_frozen_group_normalization": all(
                abs(actual - expected) <= 1e-10
                for actual, expected in zip(treatment_arm["consumed_advantages"], expected_groupwise, strict=True)
            ),
            "two_group_treatment_update_differs_from_native": state["two_group_update_differs_from_native"]
            and treatment_arm["max_abs_parameter_difference_vs_native"] > 0.0,
            "one_actual_optimizer_step_per_arm": all(
                arm["train_step_calls"] == 1 and arm["optimizer_step_calls"] == 1
                for arm in state["arms"].values()
            ),
            "all_arms_have_finite_nonzero_gradients": all(
                arm["gradient_tensor_count"] > 0
                and arm["finite_gradient_tensor_count"] == arm["gradient_tensor_count"]
                and arm["nonzero_gradient_tensor_count"] > 0
                for arm in state["arms"].values()
            ),
            "all_arms_change_parameters": all(
                arm["changed_parameter_tensor_count"] > 0
                and arm["max_abs_parameter_delta"] > 0.0
                for arm in state["arms"].values()
            ),
            "all_arms_restore_incumbent_exactly": all(
                arm["incumbent_restored_exact"] for arm in state["arms"].values()
            ) and _tensor_fingerprint(trainer.model.state_dict()) == initial_fingerprint,
            "all_arms_start_from_exact_same_incumbent": all(
                arm["initial_state_matched"]
                and arm["initial_parameter_fingerprint"] == initial_fingerprint
                for arm in state["arms"].values()
            ),
            "malformed_groups_fail_before_any_step": all(
                result["rejected"]
                and result["train_step_calls"] == 0
                and result["optimizer_step_calls"] == 0
                and result["incumbent_unchanged"]
                for result in state["negative_controls"].values()
            ),
            "resource_caps_met": state["wall_seconds"] <= expected_runtime["max_wall_seconds"]
            and state["peak_rss_bytes"] <= max_rss,
        }
        all_pass = all(state["frozen_gate_checks"].values())
        state["status"] = "completed" if all_pass else "failed"
        state["decision"] = "pass" if all_pass else "non-pass; one or more frozen gates failed"
        _write_json(output_dir / "summary.json", state)
        return state
    except BaseException as exc:
        state["status"] = "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc()
        state["wall_seconds"] = time.monotonic() - started
        state["peak_rss_bytes"] = _peak_rss_bytes()
        _write_json(output_dir / "summary.json", state)
        raise
    finally:
        signal.alarm(0)
        if trainer is not None:
            del trainer
        if backend is not None:
            del backend
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.model_path.resolve(), args.rvl_source.resolve(), args.output_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
