#!/usr/bin/env python3
"""One-step CPU feasibility smoke for the pinned VARE -> RVL real-model path.

This records execution and round-trip feasibility only. It does not evaluate
policy quality, task performance, or capability improvement.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/rvl_cpu_real_model_update_path_v3.lock.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DEFAULT_MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots"
    / MODEL_REVISION
)
VARE_SOURCE = ROOT / "src"


class ResourceLimit(RuntimeError):
    pass


def _timeout(_signum: int, _frame: Any) -> None:
    raise TimeoutError("frozen 1200-second wall limit reached")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_fingerprint(state: dict[str, Any]) -> str:
    """Hash ordered tensor names, dtype, shape, and exact CPU bytes."""
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    # On macOS ru_maxrss is reported in bytes (unlike Linux, where it is KiB).
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_prompt(system: str, question: str) -> str:
    return (
        "<|im_start|>system\n"
        + system
        + "<|im_end|>\n<|im_start|>user\n"
        + question
        + "\nReply with exactly one uppercase option letter (A, B, C, or D)."
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def _choice_reward(response: str, answer: str) -> float:
    match = re.match(r"^\s*([A-D])(?=$|[\s).,:;])", response)
    return float(bool(match and match.group(1) == answer))


def _validate_lock(lock: dict[str, Any], model_path: Path, rvl_source: Path) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("protocol is frozen for macOS arm64")
    if not model_path.is_dir() or MODEL_REVISION not in str(model_path):
        raise FileNotFoundError(f"expected cached model snapshot {MODEL_REVISION}: {model_path}")
    if not (rvl_source / "src/rvl_systems/hf_backend.py").is_file():
        raise FileNotFoundError("--rvl-source must contain src/rvl_systems/hf_backend.py")
    if lock.get("protocol_id") != "vare-rvl-cpu-real-model-update-path-v3":
        raise ValueError("unexpected protocol ID")
    if _sha256_file(Path(__file__)) != lock.get("runner_sha256"):
        raise ValueError("runner source hash does not match frozen protocol")
    vare_adapter = VARE_SOURCE / "vare/integrations/rvl_grpo.py"
    if _sha256_file(vare_adapter) != lock.get("vare_adapter_sha256"):
        raise ValueError("VARE adapter source hash does not match frozen protocol")
    expected_rvl = lock.get("rvl_source_files_sha256", {})
    actual_rvl_files = {
        path.relative_to(rvl_source).as_posix(): _sha256_file(path)
        for path in sorted((rvl_source / "src/rvl_systems").rglob("*.py"))
    }
    if not expected_rvl or actual_rvl_files != expected_rvl:
        mismatches = sorted(
            key
            for key in set(actual_rvl_files) | set(expected_rvl)
            if actual_rvl_files.get(key) != expected_rvl.get(key)
        )
        raise ValueError(f"pinned RVL source file hash mismatch: {mismatches[:8]}")
    for name, expected in lock["model"]["files_sha256"].items():
        actual = _sha256_file(model_path / name)
        if actual != expected:
            raise ValueError(f"cached model hash mismatch for {name}: {actual}")


def run(model_path: Path, rvl_source: Path, output_dir: Path) -> dict[str, Any]:
    lock = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    _validate_lock(lock, model_path, rvl_source)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite retained output: {output_dir}")
    output_dir.mkdir(parents=True)

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"
    sys.path.insert(0, str(VARE_SOURCE))
    sys.path.insert(0, str(rvl_source / "src"))

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if sys.version_info[:3] != (3, 12, 12):
        raise RuntimeError(f"Python mismatch: expected 3.12.12, got {sys.version.split()[0]}")
    if torch.__version__ != "2.9.1":
        raise RuntimeError(f"PyTorch mismatch: expected 2.9.1, got {torch.__version__}")
    if transformers.__version__ != "4.57.3":
        raise RuntimeError(
            f"Transformers mismatch: expected 4.57.3, got {transformers.__version__}"
        )
    torch.set_num_threads(4)
    torch.manual_seed(20261009)

    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    from vare.types import Attempt, Experience, Task, Verification

    prompt_specs = lock["task"]["groups"]
    state: dict[str, Any] = {
        "protocol_id": lock["protocol_id"],
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "runner_sha256": _sha256_file(Path(__file__)),
        "vare_adapter_sha256": _sha256_file(VARE_SOURCE / "vare/integrations/rvl_grpo.py"),
        "source_revision": lock["source_revision"],
        "rvl_revision": lock["rvl_revision"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "status": "running",
        "device": "cpu",
        "responses": [],
        "wall_started_unix": time.time(),
    }
    _write_json(output_dir / "progress.json", state)
    started = time.monotonic()
    max_rss = int(lock["runtime"]["max_peak_rss_bytes"])
    selected: tuple[dict[str, Any], list[Any], list[float]] | None = None
    backend = None
    trainer = None
    hooks = None
    candidate_id = None
    tokenizer = None
    model = None
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # The device is explicitly CPU below; this records that MPS existed
            # but was not selected or moved onto during this run.
            state["mps_available_but_disabled"] = True
        else:
            state["mps_available_but_disabled"] = False
        if torch.cuda.is_available():
            raise RuntimeError("CUDA unexpectedly available; protocol forbids accelerator use")

        backend = HFLocalBackend(
            model_name=str(model_path),
            max_new_tokens=8,
            device="cpu",
            precision="fp32",
        )
        model = backend.model
        tokenizer = backend.tokenizer
        if next(model.parameters()).device.type != "cpu":
            raise RuntimeError("loaded model is not on CPU")
        state["load_peak_rss_bytes"] = _peak_rss_bytes()
        state["load_seconds"] = time.monotonic() - started
        if state["load_peak_rss_bytes"] > max_rss:
            raise ResourceLimit("peak RSS cap exceeded after model load")
        _write_json(output_dir / "progress.json", state)

        system = "Solve the calculation accurately."
        sample_config = lock["task"]["sampling"]
        for group_index, spec in enumerate(prompt_specs):
            prompt = _render_prompt(system, spec["question"])
            generations = __import__("asyncio").run(
                backend.generate(
                    spec["prompt_id"],
                    prompt,
                    n=int(sample_config["samples_per_prompt"]),
                    temperature=float(sample_config["temperature"]),
                    seed=int(sample_config["seed"]) + group_index,
                )
            )
            rewards = [_choice_reward(item.response, spec["answer"]) for item in generations]
            group_record = {
                "prompt_id": spec["prompt_id"],
                "question": spec["question"],
                "answer_key": spec["answer"],
                "rewards": rewards,
                "responses": [
                    {
                        "text": item.response,
                        "reward": reward,
                        "token_count": item.token_count,
                        "logprob": item.logprob,
                        "response_token_ids": item.metadata.get("response_token_ids", []),
                        "response_token_logprobs": item.metadata.get(
                            "response_token_logprobs", []
                        ),
                    }
                    for item, reward in zip(generations, rewards, strict=True)
                ],
            }
            state["responses"].append(group_record)
            _write_json(output_dir / "progress.json", state)
            if len(set(rewards)) > 1:
                selected = (spec, generations, rewards)
                break

        if selected is None:
            state["status"] = "no_mixed_reward_group"
            state["decision"] = "fail; no optimizer step attempted"
            return state

        spec, generations, rewards = selected
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

        class RecordingTrainer(HFCausalLMGRPOTrainer):
            def __init__(self, inner: HFCausalLMGRPOTrainer) -> None:
                # Reuse exactly the already-loaded model and optimizer; only
                # intercepts the one actual trainer call for its evidence.
                self.__dict__ = inner.__dict__
                self.calls = 0
                self.metrics: dict[str, float] | None = None
                self.received_advantages: list[float] | None = None
                self.finite_gradient_tensor_count = 0
                self.nonzero_gradient_tensor_count = 0

            def train_step(self, samples, *, advantages=None):
                self.calls += 1
                self.received_advantages = advantages
                self.metrics = super().train_step(samples, advantages=advantages)
                gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
                self.finite_gradient_tensor_count = sum(
                    int(bool(torch.isfinite(gradient).all().item())) for gradient in gradients
                )
                self.nonzero_gradient_tensor_count = sum(
                    int(bool(torch.count_nonzero(gradient).item())) for gradient in gradients
                )
                if (
                    not gradients
                    or self.finite_gradient_tensor_count != len(gradients)
                    or self.nonzero_gradient_tensor_count == 0
                ):
                    raise RuntimeError("optimizer step lacked finite, nonzero gradients")
                return self.metrics

        trainer = RecordingTrainer(trainer)
        task = Task(
            id=spec["prompt_id"],
            prompt=_render_prompt(system, spec["question"]),
            family="update-path-smoke",
        )
        group_id = f"cpu-update-smoke:{spec['prompt_id']}"
        experiences = []
        for index, (generation, reward) in enumerate(zip(generations, rewards, strict=True)):
            raw_generation = RVLGRPOHooks._generation_fields(generation)
            attempt = Attempt(
                task=task,
                output=generation.response,
                policy_id="policy-0",
                policy_version=0,
                created_step=0,
                logprob=generation.logprob,
                latency_ms=generation.latency_s * 1000.0,
                metadata={
                    "rvl_generation": raw_generation,
                    "vare_rollout_group": group_id,
                    "vare_rollout_group_size": len(generations),
                    "sampling_temperature": float(sample_config["temperature"]),
                    "smoke_member": index,
                },
            )
            verification = Verification(
                score=reward,
                passed=bool(reward),
                confidence=1.0,
                verifier_version=0,
                verifier_name="frozen-exact-choice-smoke",
                trusted=True,
            )
            experiences.append(
                Experience(
                    attempt=attempt,
                    verification=verification,
                    policy_lag=0,
                    verifier_lag=0,
                    shift_score=0.0,
                )
            )

        hooks = RVLGRPOHooks(
            backend=backend,
            trainer=trainer,
            eval_tasks=[],
            score_fn=lambda _task, _response: 0.0,
            config=RVLGRPOConfig(train_temperature=float(sample_config["temperature"])),
            generation_factory=type(generations[0]),
            verified_generation_factory=__import__("rvl_systems.types", fromlist=["VerifiedGeneration"]).VerifiedGeneration,
        )
        incumbent_fingerprint = _tensor_fingerprint(trainer.model.state_dict())
        incumbent_rng = torch.get_rng_state().clone()
        incumbent_training_mode = bool(trainer.model.training)
        candidate_id = __import__("asyncio").run(
            hooks.train_candidate("policy-0", experiences)
        )
        restored_fingerprint = _tensor_fingerprint(trainer.model.state_dict())
        incumbent_optimizer = hooks._states["policy-0"]["optimizer"]
        incumbent_restored_exact = (
            restored_fingerprint == incumbent_fingerprint
            and trainer.optimizer.state_dict() == incumbent_optimizer
            and len(trainer.optimizer.state) == 0
            and torch.equal(torch.get_rng_state(), incumbent_rng)
            and torch.equal(torch.get_rng_state(), hooks._states["policy-0"]["cpu_rng"])
            and bool(trainer.model.training) == incumbent_training_mode
        )
        if not incumbent_restored_exact:
            raise RuntimeError("train_candidate did not restore exact incumbent model/optimizer")
        state["incumbent_restored_exact"] = True
        state["incumbent_parameter_fingerprint"] = incumbent_fingerprint
        state["incumbent_rng_restored_exact"] = True
        state["incumbent_training_mode_restored_exact"] = True
        if trainer.calls != 1:
            raise RuntimeError(f"expected one optimizer call, observed {trainer.calls}")
        if not trainer.metrics or any(not math.isfinite(float(x)) for x in trainer.metrics.values()):
            raise RuntimeError("trainer metrics missing or non-finite")
        if trainer.received_advantages is None or len(trainer.received_advantages) != 4:
            raise RuntimeError("VARE did not provide one group-relative advantage per rollout")
        if len(set(trainer.received_advantages)) == 1:
            raise RuntimeError("selected nonconstant rewards produced constant advantages")
        state["selected_prompt_id"] = spec["prompt_id"]
        state["selected_rewards"] = rewards
        state["received_advantages"] = trainer.received_advantages
        state["trainer_metrics"] = trainer.metrics
        state["optimizer_step_calls"] = trainer.calls
        state["finite_gradient_tensor_count"] = trainer.finite_gradient_tensor_count
        state["nonzero_gradient_tensor_count"] = trainer.nonzero_gradient_tensor_count
        state["update_peak_rss_bytes"] = _peak_rss_bytes()
        state["update_seconds"] = time.monotonic() - started
        if state["update_peak_rss_bytes"] > max_rss:
            raise ResourceLimit("peak RSS cap exceeded during optimizer update")

        base_state = hooks._states["policy-0"]["model"]
        candidate_state = hooks._states[candidate_id]
        candidate_model_state = candidate_state["model"]
        parameter_delta_count = 0
        max_abs_parameter_delta = 0.0
        for key, base_tensor in base_state.items():
            candidate_tensor = candidate_model_state[key]
            if not torch.equal(base_tensor, candidate_tensor):
                parameter_delta_count += 1
                delta = float((base_tensor - candidate_tensor).abs().max().item())
                max_abs_parameter_delta = max(max_abs_parameter_delta, delta)
        if parameter_delta_count == 0 or max_abs_parameter_delta <= 0.0:
            raise RuntimeError("candidate weights did not change")
        state["changed_parameter_tensor_count"] = parameter_delta_count
        state["max_abs_parameter_delta"] = max_abs_parameter_delta

        # Load the candidate inside the trainer, save locally, then release all
        # optimizer/snapshot copies before loading the saved policy again.
        trainer.restore_training_state(candidate_state)
        expected_fingerprint = _tensor_fingerprint(trainer.model.state_dict())
        import tempfile

        with tempfile.TemporaryDirectory(prefix="vare-rvl-cpu-update-") as temp_name:
            checkpoint_dir = Path(temp_name) / "candidate"
            trainer.model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_hashes = {
                path.name: _sha256_file(path)
                for path in sorted(checkpoint_dir.iterdir())
                if path.is_file()
            }
            state["checkpoint_file_sha256"] = checkpoint_hashes

            # Drop duplicate model and Adam states before reload. Keep only the
            # expected digest, so round-trip verification does not retain a
            # second full parameter copy in memory.
            hooks._states.clear()
            del candidate_state, candidate_model_state, base_state
            del hooks, trainer, model, backend, generations, experiences
            gc.collect()
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir, local_files_only=True
            )
            tokenizer_prompt_matches = all(
                tokenizer(prompt, add_special_tokens=False)["input_ids"]
                == loaded_tokenizer(prompt, add_special_tokens=False)["input_ids"]
                for prompt in (
                    _render_prompt(system, row["question"]) for row in prompt_specs
                )
            )
            if not tokenizer_prompt_matches:
                raise RuntimeError("candidate tokenizer changed tokenization on frozen prompts")
            state["tokenizer_prompt_roundtrip_exact"] = True
            del tokenizer, loaded_tokenizer
            reloaded_model = AutoModelForCausalLM.from_pretrained(
                checkpoint_dir, local_files_only=True, dtype=torch.float32
            ).to("cpu")
            observed_fingerprint = _tensor_fingerprint(reloaded_model.state_dict())
            if observed_fingerprint != expected_fingerprint:
                raise RuntimeError("candidate save/reload parameter fingerprint mismatch")
            state["reloaded_parameter_fingerprint"] = observed_fingerprint
            state["roundtrip_exact"] = True
            state["reloaded_model_device"] = str(next(reloaded_model.parameters()).device)
            del loaded_tokenizer, reloaded_model
            gc.collect()

        state["peak_rss_bytes"] = _peak_rss_bytes()
        state["wall_seconds"] = time.monotonic() - started
        if state["peak_rss_bytes"] > max_rss:
            raise ResourceLimit("peak RSS cap exceeded")
        if state["wall_seconds"] > float(lock["runtime"]["max_wall_seconds"]):
            raise ResourceLimit("wall-time cap exceeded")
        state["status"] = "passed"
        state["decision"] = "CPU one-step update path feasible under this smoke; task efficacy remains untested"
        return state
    except BaseException as exc:
        state["status"] = "failed"
        state["error_type"] = type(exc).__name__
        state["error"] = str(exc)
        state["traceback"] = traceback.format_exc()
        state["peak_rss_bytes"] = _peak_rss_bytes()
        state["wall_seconds"] = time.monotonic() - started
        return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/rvl-cpu-real-model-update-path-v3/run-1",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, _timeout)
    output_dir = args.output_dir.resolve()
    output_preexisted = output_dir.exists()
    try:
        if output_preexisted:
            raise FileExistsError(f"refusing to overwrite retained output: {output_dir}")
        lock = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        signal.alarm(int(lock["runtime"]["max_wall_seconds"]))
        result = run(args.model_path.resolve(), args.rvl_source.resolve(), output_dir)
    except BaseException as exc:
        result = {
            "protocol_id": "vare-rvl-cpu-real-model-update-path-v3",
            "protocol_sha256": _sha256_file(PROTOCOL_PATH) if PROTOCOL_PATH.is_file() else None,
            "runner_sha256": _sha256_file(Path(__file__)),
            "status": "failed_preflight",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "python": sys.version,
            "platform": platform.platform(),
            "wall_time_unix": time.time(),
        }
        if not output_preexisted and not output_dir.exists():
            output_dir.mkdir(parents=True)
    if not output_preexisted:
        _write_json(output_dir / "progress.json", result)
        _write_json(output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
