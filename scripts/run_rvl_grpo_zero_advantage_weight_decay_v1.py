#!/usr/bin/env python3
"""CPU reproducer for AdamW drift under zero GRPO advantage in pinned RVL."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/rvl_grpo_zero_advantage_weight_decay_v1.lock.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canon_sha(data: dict[str, Any]) -> str:
    body = {key: value for key, value in data.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fixture(model, rewards: list[float]):
    import torch
    from src.rvl_systems.types import Generation, VerifiedGeneration
    model.eval()
    samples = []
    for index, reward in enumerate(rewards):
        prompt_ids = [1, 2]
        response_ids = [3 + index, 7 + index]
        with torch.no_grad():
            logits = model(torch.tensor([prompt_ids + response_ids])).logits[0, 1:3]
            logps = torch.log_softmax(logits.float(), dim=-1).gather(
                1, torch.tensor(response_ids)[:, None]
            ).squeeze(1).tolist()
        generation = Generation(
            prompt_id="same-prompt", prompt="offline fixture", response=f"trace-{index}",
            logprob=sum(logps), token_count=len(response_ids), latency_s=0.0,
            metadata={"prompt_token_ids": prompt_ids, "response_token_ids": response_ids,
                      "response_token_logprobs": logps, "sampling_temperature": 1.0},
        )
        samples.append(VerifiedGeneration(generation, reward, 0.0, 0))
    return samples


def _model(seed: int):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    ))


def _one_step(trainer_type, config_type, seed: int, rewards: list[float], *, lr: float, explicit_wd: float | None = None) -> dict[str, Any]:
    import torch
    model = _model(seed)
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("fixture model must remain on CPU")
    kwargs = {"learning_rate": lr}
    if explicit_wd is not None:
        kwargs["weight_decay"] = explicit_wd
    trainer = trainer_type(model, config=config_type(**kwargs))
    effective_wd = float(trainer.optimizer.param_groups[0]["weight_decay"])
    samples = _fixture(model, rewards)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.eval()
    with torch.no_grad():
        logits_before = model(torch.tensor([[1, 2]])).logits[0, -1].float()
        p_before = torch.softmax(logits_before, dim=-1)
    metrics = trainer.train_step(samples)
    grads = [parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None]
    after = model.state_dict()
    changed = [key for key, value in before.items() if not torch.equal(value, after[key])]
    parameters = dict(model.named_parameters())
    expected_decay = {key: value * (1.0 - lr * effective_wd) for key, value in before.items() if key in parameters}
    max_formula_error = max(float((after[key] - expected_decay[key]).abs().max()) for key in expected_decay)
    max_delta = max(float((after[key] - before[key]).abs().max()) for key in expected_decay)
    model.eval()
    with torch.no_grad():
        logits_after = model(torch.tensor([[1, 2]])).logits[0, -1].float()
        p_after = torch.softmax(logits_after, dim=-1)
        kl = float(torch.sum(p_before * (torch.log(p_before) - torch.log(p_after))))
    # Recompute group advantages from the declared population-variance estimator.
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    scale = math.sqrt(variance + 1e-6)
    advantages = [(reward - mean) / scale for reward in rewards]
    return {
        "seed": seed, "rewards": rewards, "recomputed_advantages": advantages,
        "effective_weight_decay": effective_wd, "metrics": metrics,
        "gradient_tensor_count": len(grads),
        "nonzero_gradient_tensor_count": sum(int(torch.count_nonzero(grad) > 0) for grad in grads),
        "changed_parameter_tensor_count": len(changed), "max_abs_parameter_delta": max_delta,
        "max_abs_error_vs_adamw_decay_formula": max_formula_error,
        "next_token_policy_kl_before_to_after": kl,
    }


def run(rvl_source: Path, arm: str, output: Path) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if canon_sha(lock) != lock.get("sha256"):
        raise RuntimeError("frozen protocol digest mismatch")
    if sha(Path(__file__)) != lock["runner_sha256"]:
        raise RuntimeError("runner source hash mismatch")
    trainer_sha = sha(rvl_source / "src/rvl_systems/hf_trainer.py")
    expected_trainer_sha = (lock["sources"]["trainer_sha256"] if arm == "base"
                            else lock["candidate"]["trainer_sha256"])
    if trainer_sha != expected_trainer_sha:
        raise RuntimeError(f"trainer source hash mismatch for {arm} arm")
    if arm == "candidate" and sha(rvl_source / lock["candidate"]["regression_test_file"]) != lock["candidate"]["regression_test_sha256"]:
        raise RuntimeError("candidate regression test source hash mismatch")
    sys.path.insert(0, str(rvl_source.resolve()))
    import torch
    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    if torch.__version__ != lock["runtime"]["torch"]:
        raise RuntimeError("PyTorch runtime mismatch")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must remain unused")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    base_rows = []
    for seed in lock["fixture"]["seeds"]:
        base_rows.append(_one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed,
                                   [1.0] * 4, lr=lock["fixture"]["learning_rate"]))
    result: dict[str, Any] = {
        "protocol_id": lock["protocol_id"], "protocol_sha256": sha(LOCK),
        "runner_sha256": sha(Path(__file__)), "arm": arm, "rvl_revision": lock["sources"]["rvl_revision"],
        "trainer_source_sha256": trainer_sha,
        "base_constant_reward": base_rows, "status": "complete",
    }
    if arm == "candidate":
        candidate_default = []
        candidate_explicit = []
        mixed = []
        for seed in lock["fixture"]["seeds"]:
            candidate_default.append(_one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed,
                [1.0] * 4, lr=lock["fixture"]["learning_rate"]))
            candidate_explicit.append(_one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed,
                [1.0] * 4, lr=lock["fixture"]["learning_rate"], explicit_wd=0.01))
            mixed.append(_one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed,
                lock["fixture"]["mixed_rewards"], lr=lock["fixture"]["learning_rate"]))
        result["candidate_default_constant_reward"] = candidate_default
        result["candidate_explicit_weight_decay"] = candidate_explicit
        result["candidate_mixed_reward_control"] = mixed
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "candidate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.rvl_source.resolve(), args.arm, args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
