"""Real RVL/HF inference probe; random tiny GPT-2, CPU, no model downloads."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--vare-source", type=Path, required=True)
parser.add_argument("--rvl-source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.vare_source / "src"), str(args.rvl_source)]

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
from vare.types import Task
from src.rvl_systems.hf_backend import HFLocalBackend
from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer
from src.rvl_systems.types import Generation, VerifiedGeneration

torch.set_num_threads(1)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 4

    def __call__(self, prompt, *, return_tensors):
        return {"input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.long)}

    def decode(self, ids, *, skip_special_tokens):
        return " ".join(str(x) for x in ids)


def setup(backend_class=HFLocalBackend):
    torch.manual_seed(321)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=5, n_positions=16, n_embd=8, n_layer=1, n_head=1,
        bos_token_id=1, eos_token_id=4, pad_token_id=0,
    ))
    trainer = HFCausalLMGRPOTrainer(model)
    # Populate actual AdamW moments through the actual trainer, before probing.
    with torch.no_grad():
        old_logp = torch.log_softmax(model(torch.tensor([[1, 2, 3]])).logits[0, 1], -1)[3].item()
    sample = VerifiedGeneration(Generation(
        prompt_id="warmup", prompt="fixture", response="3", logprob=old_logp,
        token_count=1, latency_s=0.0, metadata={"prompt_token_ids": [1, 2],
        "response_token_ids": [3], "response_token_logprobs": [old_logp],
        "sampling_temperature": 1.0}), reward=1.0, verifier_latency_s=0.0,
        verifier_version=0)
    trainer.train_step([sample], advantages=[1.0])
    model.transformer.drop.eval()
    backend = backend_class("random-tiny-gpt2", device="cpu", max_new_tokens=2)
    backend._torch, backend._tokenizer, backend._model = torch, Tokenizer(), model
    backend._resolved_precision = "fp32"
    task = Task("eval", "fixture prompt")
    hooks = RVLGRPOHooks(backend=backend, trainer=trainer, eval_tasks=[task],
                        score_fn=lambda *_: 1.0, config=RVLGRPOConfig(seed=7))
    return model, trainer, backend, task, hooks


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def state(trainer):
    return copy.deepcopy(trainer.snapshot_training_state())


def compare(before, trainer):
    after = state(trainer)
    return {key + "_preserved": equal(value, after[key]) for key, value in before.items()}


async def standard_cases():
    out = {}
    for case in ("evaluate_success", "evaluate_score_failure", "rollout_success"):
        _, trainer, _, task, hooks = setup()
        before = state(trainer)
        if case == "evaluate_score_failure":
            def fail(*_):
                raise RuntimeError("score fixture failure after real generation")
            hooks.score_fn = fail
        error = None
        try:
            if case == "rollout_success":
                await hooks.rollout(task, "policy-0", 0, 0)
            else:
                await hooks.evaluate("policy-0")
        except RuntimeError as exc:
            error = str(exc)
        out[case] = {"error": error, "state": compare(before, trainer),
                     "loaded_id": hooks._loaded_id}
    return out


async def cancellation_case():
    model, trainer, _, _, hooks = setup()
    before = state(trainer)
    original = model.generate
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    finished = threading.Event()
    calls = []

    def controlled(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        entered[index].set()
        try:
            if not release[index].wait(10):
                raise TimeoutError("fixture release timed out")
            return original(*args, **kwargs)
        finally:
            if index == 0:
                finished.set()

    model.generate = controlled
    first = asyncio.create_task(hooks.evaluate("policy-0"))
    second = None
    try:
        assert await asyncio.to_thread(entered[0].wait, 5)
        first.cancel()
        # Bounded observation while the worker is deliberately blocked.
        await asyncio.wait({first}, timeout=0.05)
        first.cancel()
        lock_held_after_cancel = hooks._model_lock.locked()
        attempted = asyncio.Event()

        async def follow():
            attempted.set()
            return await hooks.rollout(Task("next", "next prompt"), "policy-0", 0, 1)

        second = asyncio.create_task(follow())
        await attempted.wait()
        overlapped = await asyncio.to_thread(entered[1].wait, 0.1)
        result = {"cancelled_call_pending_until_worker_finishes": not first.done(),
                  "lock_held_after_cancel": lock_held_after_cancel,
                  "no_next_generation_overlap": not overlapped,
                  "first_worker_live_at_observation": not finished.is_set()}
        release[0].set()
        try:
            await first
        except asyncio.CancelledError:
            result["cancellation_propagated"] = True
        assert await asyncio.to_thread(entered[1].wait, 5)
        release[1].set()
        await second
        result["final_state"] = compare(before, trainer)
        return result
    finally:
        for event in release:
            event.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


async def main():
    result = {"standard": await standard_cases(), "cancellation": await cancellation_case()}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


asyncio.run(main())
