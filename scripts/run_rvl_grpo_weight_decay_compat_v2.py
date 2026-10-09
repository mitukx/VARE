#!/usr/bin/env python3
"""Frozen CPU comparison for RVL's explicit, compatibility-preserving decay config."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/rvl_grpo_weight_decay_compat_v2.lock.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(data: dict[str, Any]) -> str:
    body = {key: value for key, value in data.items() if key != "sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_model(seed: int):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    ))
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("fixture must remain on CPU")
    return model


def make_samples(model, rewards: list[float]):
    import torch
    from src.rvl_systems.types import Generation, VerifiedGeneration

    model.eval()
    rows = []
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
            logprob=sum(logps), token_count=2, latency_s=0.0,
            metadata={"prompt_token_ids": prompt_ids, "response_token_ids": response_ids,
                      "response_token_logprobs": logps, "sampling_temperature": 1.0},
        )
        rows.append(VerifiedGeneration(generation, reward, 0.0, 0))
    return rows


def one_step(trainer_type, config_type, seed: int, rewards: list[float], *, lr: float,
             weight_decay: float | None = None) -> dict[str, Any]:
    import torch

    model = make_model(seed)
    kwargs = {"learning_rate": lr}
    if weight_decay is not None:
        kwargs["weight_decay"] = weight_decay
    trainer = trainer_type(model, config=config_type(**kwargs))
    effective_wd = float(trainer.optimizer.param_groups[0]["weight_decay"])
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    metrics = trainer.train_step(make_samples(model, rewards))
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    params = dict(model.named_parameters())
    changed = [name for name, p in params.items() if not torch.equal(p, before[name])]
    formula_error = max(float((params[name] - before[name] * (1.0 - lr * effective_wd)).abs().max())
                        for name in params)
    delta = max(float((params[name] - before[name]).abs().max()) for name in params)
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    scale = math.sqrt(variance + 1e-6)
    advantages = [(reward - mean) / scale for reward in rewards]
    return {
        "seed": seed, "rewards": rewards, "advantages": advantages,
        "effective_weight_decay": effective_wd, "metrics": metrics,
        "gradient_tensor_count": len(grads),
        "nonzero_gradient_tensor_count": sum(int(torch.count_nonzero(g) > 0) for g in grads),
        "changed_parameter_tensor_count": len(changed), "max_abs_parameter_delta": delta,
        "max_abs_error_vs_decay_only_formula": formula_error,
    }


def run(source: Path, arm: str, output: Path) -> dict[str, Any]:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if canonical_sha(lock) != lock.get("sha256"):
        raise RuntimeError("protocol digest mismatch")
    if sha(Path(__file__)) != lock["runner_sha256"]:
        raise RuntimeError("runner hash mismatch")
    hashes = lock["base_source_hashes"] if arm == "base" else lock["candidate_source_hashes"]
    for relative, expected in hashes.items():
        if sha(source / relative) != expected:
            raise RuntimeError(f"source hash mismatch: {relative}")
    sys.path.insert(0, str(source.resolve()))
    import torch
    from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig

    if torch.__version__ != lock["runtime"]["torch"] or torch.cuda.is_available():
        raise RuntimeError("runtime mismatch or CUDA available")
    if output.exists():
        raise FileExistsError(output)
    seeds = lock["fixture"]["seeds"]
    lr = lock["fixture"]["learning_rate"]
    result: dict[str, Any] = {
        "protocol_id": lock["protocol_id"], "protocol_raw_sha256": sha(LOCK),
        "runner_sha256": sha(Path(__file__)), "rvl_revision": lock["rvl_revision"],
        "candidate_revision": lock["candidate_revision"], "torch_version": torch.__version__,
        "arm": arm,
    }
    if arm == "base":
        result["base_implicit_default"] = [
            one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed, [1.0] * 4, lr=lr)
            for seed in seeds
        ]
    else:
        result["candidate_default"] = [
            one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed, [1.0] * 4, lr=lr)
            for seed in seeds
        ]
        result["candidate_explicit_zero"] = [
            one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed, [1.0] * 4, lr=lr,
                     weight_decay=0.0) for seed in seeds
        ]
        result["candidate_explicit_legacy"] = [
            one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed, [1.0] * 4, lr=lr,
                     weight_decay=0.01) for seed in seeds
        ]
        result["candidate_mixed_reward_control"] = [
            one_step(HFCausalLMGRPOTrainer, HFTTrainerConfig, seed,
                     lock["fixture"]["mixed_rewards"], lr=lr) for seed in seeds
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rvl-source", required=True, type=Path)
    parser.add_argument("--arm", choices=("base", "candidate"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.rvl_source.resolve(), args.arm, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
