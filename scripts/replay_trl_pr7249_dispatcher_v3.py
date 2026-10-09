#!/usr/bin/env python3
"""Exercise pinned AsyncGRPO normalization through Accelerate dispatch and Trainer (v1)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset


COMMITS = {
    "base": "a98fa6a4428f9aae58dfb26d729d7437f662f27a",
    "candidate": "5234eb7c70f4ca7eb92fe8e01611a33eaff17f40",
}
SOURCE_HASHES = {
    "base": "b84c4544a4ac2aac5b08681a93fb3d49d1f24ff2f2f9fe53a6c35ab5c8f0ea40",
    "candidate": "5c4fc857c563db0d9f937a2edf89dd74c4f218a31f3a05bf33bffbf5285749dc",
}
EXPECTED_GLOBAL = [0, 5, 7]


def sample(tokens: int, sid: int) -> dict:
    mask = [0, 0] + [1] * (tokens // 2) + [0] + [1] * (tokens - tokens // 2)
    return {
        "input_ids": [1] * len(mask), "completion_mask": mask,
        "old_log_probs": [-math.log(2)] * len(mask), "advantage": -1.0 - sid / 10,
        "group_id": sid, "prompt_id": sid, "metrics": {},
    }


def microbatches() -> list[list[list[dict]]]:
    counts = [[0, 0], [0, 5], [4, 3]]
    return [[[sample(row[rank], 100 + 10 * step + rank)] for rank in range(2)]
            for step, row in enumerate(counts)]


def build_model():
    from types import SimpleNamespace
    from transformers import PretrainedConfig

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(2, 2)
            torch.nn.init.ones_(self.embed.weight)

        def forward(self, input_ids, position_ids, use_cache=False):
            return SimpleNamespace(last_hidden_state=self.embed(input_ids))

    class TokenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig()
            self.base_model = Backbone()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.zeros_(self.lm_head.weight)

        def get_output_embeddings(self):
            return self.lm_head

        def forward(self, input_ids, position_ids, labels=None, completion_mask=None, use_cache=False):
            hidden = self.base_model(input_ids, position_ids).last_hidden_state
            log_probs = self.lm_head(hidden[:, :-1]).log_softmax(-1)
            return {
                "log_probs": log_probs.gather(-1, input_ids[:, 1:, None]).squeeze(-1),
                "entropy": -(log_probs.exp() * log_probs).sum(-1),
                "aux_loss": self.lm_head.weight.sum() + 2.0,
            }

    return TokenModel()


def source_check(repo: Path, arm: str) -> str:
    import subprocess
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()
    source = repo / "trl/experimental/async_grpo/async_grpo_trainer.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if commit != COMMITS[arm] or dirty or digest != SOURCE_HASHES[arm]:
        raise RuntimeError(f"source mismatch: commit={commit}, dirty={bool(dirty)}, sha256={digest}")
    return commit


def loss_oracle(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    output = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                   labels=batch["input_ids"], completion_mask=batch["completion_mask"], use_cache=False)
    mask = batch["completion_mask"][:, 1:].float()
    old = batch["old_log_probs"][:, 1:]
    adv = batch["advantages"][:, 1:].float()
    ratio = (output["log_probs"] - old).exp()
    clipped = ratio.clamp(0.8, 1.2)
    per_token = -torch.minimum(ratio * adv, clipped * adv)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max())


def digest(state: dict) -> str:
    return hashlib.sha256(b"".join(k.encode() + state[k].detach().cpu().contiguous().numpy().tobytes()
                                    for k in sorted(state))).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--arm", choices=("base", "candidate"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    commit = source_check(repo, args.arm)
    torch.set_num_threads(1)
    if not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    if dist.get_world_size() != 2 or dist.get_backend() != "gloo":
        raise RuntimeError("expected 2-process CPU Gloo")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    from accelerate.data_loader import DataLoaderDispatcher
    from transformers import Trainer, TrainingArguments
    from trl.experimental.async_grpo import AsyncGRPOTrainer
    from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout
    from trl.models.utils import _ForwardRedirection

    torch.manual_seed(12345)
    model = build_model()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 if step == 0 else 0.5)
    train_args = TrainingArguments(
        output_dir=str(args.output / f"trainer-{rank}"), use_cpu=True, report_to="none",
        logging_strategy="no", save_strategy="no", disable_tqdm=True, max_steps=2,
        gradient_accumulation_steps=2, per_device_train_batch_size=1, max_grad_norm=0.0,
        remove_unused_columns=False, ddp_find_unused_parameters=False, seed=12345,
        accelerator_config={"split_batches": True, "dispatch_batches": True, "even_batches": False},
    )
    trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
    Trainer.__init__(trainer, model=model, args=train_args,
                     compute_loss_func="non-None value to disable Trainer auto-scaling",
                     optimizers=(optimizer, scheduler))
    trainer.model_accepts_loss_kwargs = False
    trainer.epsilon_low = trainer.epsilon_high = 0.2
    trainer.aux_loss_enabled = False
    trainer.router_aux_loss_coef = 0.1
    trainer._metrics = {"train": defaultdict(list)}
    trainer._step_forward_tokens = trainer._step_trained_tokens = 0
    trainer._step_seq_len_weighted = trainer._step_samples = trainer._step_forward_s = 0
    trainer._step_microbatches = trainer._current_train_step_time = 0
    trainer.rollout_worker = None
    trainer._forward_redirection = _ForwardRedirection()
    trainer.args.beta = 0.0

    rows = microbatches()
    collator = DataCollatorForRollout(pad_token_id=0, num_processes=2, metrics=trainer._metrics["train"])

    class LockedStream(IterableDataset):
        def __iter__(self):
            yield from rows

    def get_loader():
        loader = DataLoader(LockedStream(), batch_size=1, collate_fn=collator, num_workers=0)
        prepared = trainer.accelerator.prepare(loader)
        if not isinstance(prepared, DataLoaderDispatcher):
            raise RuntimeError(f"expected production DataLoaderDispatcher, got {type(prepared).__name__}")
        return prepared

    trainer._inner_training_loop = Trainer._inner_training_loop.__get__(trainer, AsyncGRPOTrainer)
    trainer.log = Trainer.log.__get__(trainer, AsyncGRPOTrainer)
    trainer.get_train_dataloader = get_loader
    original_step = trainer.training_step
    consumed = []
    step_losses = []
    per_update_grads = []

    def capture_step(model_arg, batch, num_items_in_batch):
        local = int(batch["completion_mask"].sum().item())
        global_count = int(batch["global_n_tokens"][0].item())
        consumed.append((local, global_count))
        loss = original_step(model_arg, batch, num_items_in_batch)
        step_losses.append(float(loss.detach()))
        if len(step_losses) in (2, 3):
            unwrapped = trainer.accelerator.unwrap_model(model_arg)
            per_update_grads.append({k: p.grad.detach().clone() for k, p in unwrapped.named_parameters()
                                     if p.grad is not None})
        return loss

    trainer.training_step = capture_step
    train_result = trainer.train()
    expected_local = [0, 5, 3] if rank == 1 else [0, 0, 4]
    if consumed != list(zip(expected_local, EXPECTED_GLOBAL)):
        raise RuntimeError(f"dispatcher counts/layout mismatch: {consumed}")
    actual_model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
    actual_state = {k: v.detach().clone() for k, v in actual_model.state_dict().items()}

    # Two independent pooled-token reference updates: a two-microbatch window, then a short final window.
    oracle = build_model()
    oracle.load_state_dict(initial)
    reference_updates = []
    for indices, lr in (([0, 1], 0.1), ([2], 0.05)):
        pooled_rows = [[example for i in indices for example in rows[i][r]] for r in range(2)]
        pooled = collator([pooled_rows])
        oracle.zero_grad(set_to_none=True)
        reference_loss = loss_oracle(oracle, pooled)
        reference_loss.backward()
        grads = {k: p.grad.detach().clone() for k, p in oracle.named_parameters() if p.grad is not None}
        with torch.no_grad():
            for p in oracle.parameters():
                p.add_(p.grad, alpha=-lr)
        reference_updates.append({"loss": float(reference_loss.detach()), "grads": grads,
                                  "state": {k: v.detach().clone() for k, v in oracle.state_dict().items()}})

    # Trainer accumulates its loss returns per optimizer window. Average rank-local diagnostics as in the DDP objective.
    local_window_losses = [sum(step_losses[:2]), step_losses[2]]
    loss_tensor = torch.tensor(local_window_losses, dtype=torch.float64)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    loss_tensor /= dist.get_world_size()
    grad_errors, update_errors, loss_errors = [], [], []
    for idx, ref in enumerate(reference_updates):
        if idx >= len(per_update_grads):
            raise RuntimeError("missing update-boundary gradient snapshot")
        grad_errors.append(max(err(per_update_grads[idx][k], ref["grads"][k]) for k in ref["grads"]))
        update_errors.append(max(err(actual_state[k], ref["state"][k]) for k in ref["state"]) if idx == 1 else 0.0)
        loss_errors.append(abs(float(loss_tensor[idx]) - ref["loss"]))
    states = [None] * 2
    dist.all_gather_object(states, digest(actual_state))
    result = {
        "rank": rank, "global_step": train_result.global_step,
        "observed_dispatcher_type": "DataLoaderDispatcher",
        "observed_local_tokens_and_replicated_window_counts": consumed,
        "rank_mean_window_losses": loss_tensor.tolist(),
        "oracle_window_losses": [item["loss"] for item in reference_updates],
        "loss_abs_errors": loss_errors, "gradient_abs_errors": grad_errors,
        "final_parameter_abs_error": update_errors[-1], "post_step_state_sha256": digest(actual_state),
        "elapsed_seconds": time.monotonic() - started,
    }
    gathered = [None] * 2
    dist.all_gather_object(gathered, result)
    if rank == 0:
        summary = {"protocol_id": "trl-pr7249-dispatcher-v3", "arm": args.arm, "source_commit": commit,
                   "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                               "transformers": __import__("transformers").__version__,
                               "accelerate": __import__("accelerate").__version__},
                   "rank_results": gathered, "rank_state_equal": len(set(states)) == 1}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
