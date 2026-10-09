#!/usr/bin/env python3
"""CPU-only integration validation for rollback of tensor and module-mode state."""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib
import json
import os
import platform
import sys
import time
import types
from pathlib import Path

for _key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
    os.environ[_key] = "1"


UPSTREAM_COMMIT = "c7e646b043cb56e5ea3c2623bb8a61e065451f72"
UPSTREAM_SOURCE_SHA256 = {
    "backends.py": "26abeffd24ecef37093a179dc33506f0853018331c0002bd01010d016b31faca",
    "grpo.py": "a2ef6217385a206f5e6c0f43fb40e900ae3bf1b2066a1c8d1ab1878fd25d724d",
    "hf_trainer.py": "375985c1d0cca49a9d98d8b00ed41cefe000b91461e9a0783e94312466fd0210",
    "triton_grpo.py": "ed7ee082db036e1631c1f8eaacfb8691b102401c8342dcb9ed4dba5d26a22cf9",
    "types.py": "a3e9f06cc70972a7b5fd50b757ffff992d3a6156d1ffab5133e410471dcc5d7b",
}


class Generation:
    def __init__(self, prompt_id, prompt, response, logprob, token_count, latency_s, metadata=None):
        self.prompt_id = prompt_id
        self.prompt = prompt
        self.response = response
        self.logprob = logprob
        self.token_count = token_count
        self.latency_s = latency_s
        self.metadata = metadata or {}


class VerifiedGeneration:
    def __init__(self, generation, reward, verifier_latency_s, verifier_version, metadata=None):
        self.generation = generation
        self.reward = reward
        self.verifier_latency_s = verifier_latency_s
        self.verifier_version = verifier_version
        self.metadata = metadata or {}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def equal_nested(left, right, torch) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal_nested(left[key], right[key], torch) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            equal_nested(a, b, torch) for a, b in zip(left, right)
        )
    return left == right


class ModelStateBackend:
    def __init__(self, trainer, initial_model_state, torch):
        self.trainer = trainer
        self.initial_model_state = initial_model_state
        self.torch = torch

    async def generate(self, prompt_id, prompt, *, n, temperature, seed):
        current = {key: value.detach().cpu() for key, value in self.trainer.model.state_dict().items()}
        unchanged = equal_nested(self.initial_model_state, current, self.torch)
        response = "incumbent" if unchanged else "candidate"
        return [Generation(
            prompt_id, prompt, response, -0.1, 1, 0.001,
            {"sampling_temperature": max(temperature, 0.1), "prompt_token_ids": [1],
             "response_token_ids": [3], "response_token_logprobs": [-0.1]},
        )]


def make_experience(task, response_token: int, reward: float, logprob: float, vare_types):
    Attempt, Experience, Verification = vare_types
    response = "A" if response_token == 3 else "B"
    raw = {
        "prompt_id": "same-prompt", "prompt": task.prompt, "response": response,
        "logprob": logprob, "token_count": 1, "latency_s": 0.001,
        "metadata": {"sampling_temperature": 0.8, "prompt_token_ids": [1],
                     "response_token_ids": [response_token], "response_token_logprobs": [logprob]},
    }
    attempt = Attempt(task, response, "policy-0", 0, 0, logprob=logprob,
                      metadata={"rvl_generation": raw})
    verification = Verification(float(reward), bool(reward), 0.0, 1, "tiny-oracle", trusted=True)
    return Experience(attempt, verification, 0, 0, 0.0)


def import_pinned_trainer(source: Path):
    for name, expected in UPSTREAM_SOURCE_SHA256.items():
        path = source / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError("pinned RVL source mismatch: %s" % name)
    # Import the pinned submodules without executing RVL's broad package
    # initializer; only the trainer and its direct Python dependencies are used.
    package = types.ModuleType("rvl_systems")
    package.__path__ = [str(source)]
    package.__package__ = "rvl_systems"
    sys.modules["rvl_systems"] = package
    module = importlib.import_module("rvl_systems.hf_trainer")
    return module.HFCausalLMGRPOTrainer, module.HFTTrainerConfig


def run(source: Path) -> dict:
    started = time.monotonic()
    source = source.resolve()
    HFCausalLMGRPOTrainer, HFTTrainerConfig = import_pinned_trainer(source)
    import torch
    import transformers
    from transformers import GPT2Config, GPT2LMHeadModel

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    from vare.types import Attempt, Experience, Task, Verification

    runtime = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
               "transformers": transformers.__version__}
    if runtime != {"python": "3.12.12", "torch": "2.9.1", "transformers": "4.57.3"}:
        raise RuntimeError("runtime differs from frozen smoke protocol: %r" % runtime)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("CPU-only smoke cannot use or initialize CUDA")
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=16, n_positions=8, n_ctx=8, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )).to("cpu")
    model.eval()
    trainer = HFCausalLMGRPOTrainer(
        model, config=HFTTrainerConfig(learning_rate=1e-2, objective_backend="torch")
    )
    before = copy.deepcopy(trainer.snapshot_training_state())
    before_module_modes = {name: module.training for name, module in model.named_modules()}
    task = Task(id="tiny-random-lm", prompt="one token", family="synthetic-smoke")
    vare_types = (Attempt, Experience, Verification)
    experiences = []
    for token, reward in ((3, 0.0), (4, 1.0)):
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([[1, token]])).logits[0, 0]
            logprob = float(torch.log_softmax(logits / 0.8, dim=-1)[token])
        experiences.append(make_experience(task, token, reward, logprob, vare_types))

    backend = ModelStateBackend(trainer, before["model"], torch)
    hooks = RVLGRPOHooks(
        backend=backend, trainer=trainer, eval_tasks=[task],
        score_fn=lambda _task, response: float(response == "candidate"),
        config=RVLGRPOConfig(seed=1), generation_factory=Generation,
        verified_generation_factory=VerifiedGeneration,
    )

    actual_optimizer_step = trainer.optimizer.step
    checks: dict[str, bool] = {}

    def optimizer_step_then_fail(*args, **kwargs):
        actual_optimizer_step(*args, **kwargs)
        after_update = trainer.snapshot_training_state()
        checks["model_changed_before_failure"] = not equal_nested(
            before["model"], after_update["model"], torch
        )
        checks["optimizer_changed_before_failure"] = bool(trainer.optimizer.state)
        if not checks["model_changed_before_failure"] or not checks["optimizer_changed_before_failure"]:
            raise AssertionError("real GRPO step did not mutate both model and optimizer state")
        checks["failure_injected_inside_train_step"] = True
        raise RuntimeError("injected failure immediately after optimizer mutation")

    trainer.optimizer.step = optimizer_step_then_fail
    try:
        asyncio.run(hooks.train_candidate("policy-0", experiences))
    except RuntimeError as exc:
        checks["failure_propagated"] = "injected failure immediately after optimizer mutation" in str(exc)
    else:
        raise AssertionError("candidate failure did not propagate")

    after = trainer.snapshot_training_state()
    checks["model_restored"] = equal_nested(before["model"], after["model"], torch)
    checks["optimizer_restored"] = equal_nested(before["optimizer"], after["optimizer"], torch)
    checks["rng_restored"] = torch.equal(before["cpu_rng"], after["cpu_rng"])
    checks["active_policy_unchanged"] = hooks.active_policy() == ("policy-0", 0)
    checks["failed_candidate_removed"] = set(hooks._states) == {"policy-0"}
    checks["counter_unchanged"] = hooks._counter == 1
    attempt = asyncio.run(hooks.rollout(task, "policy-0", 0, 1))
    checks["next_rollout_uses_incumbent"] = attempt.output == "incumbent"
    checks["module_modes_restored"] = before_module_modes == {
        name: module.training for name, module in model.named_modules()
    }
    checks["root_mode_restored"] = model.training is False
    checks["cuda_uninitialized"] = not torch.cuda.is_initialized()

    wall_seconds = round(time.monotonic() - started, 3)
    if wall_seconds > 120:
        raise RuntimeError("smoke exceeded the frozen 120-second wall limit")
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "upstream_repository": "https://github.com/mitukx/Recursive-Verification-Lag",
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
        "runtime": {**runtime, "device": "cpu", "torch_threads": 1},
        "model": {"architecture": "GPT2LMHeadModel", "initialization": "random",
                  "parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "vocab_size": 16, "layers": 1, "hidden_size": 16, "heads": 2},
        "procedure": "real pinned HFCausalLMGRPOTrainer.train_step with optimizer.step wrapped to mutate model/optimizer and then raise before train_step returns",
        "checks": checks,
        "wall_seconds": wall_seconds,
        "claim_boundary": "Tiny random-model integration smoke only; no pretrained-model capability, training-quality, or external-reproduction claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rvl-source", type=Path, required=True,
                        help="src/rvl_systems directory from the pinned upstream commit")
    args = parser.parse_args()
    result = run(args.rvl_source)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
