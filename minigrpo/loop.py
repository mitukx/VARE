"""Resumable GRPO loop over an intact group.

A trainer supplies text, token log probabilities, and one optimizer update.
This module grades integer answers, writes JSONL metrics, and refuses to drop
a group member. Confirm tasks are not accepted here.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from minigrpo.expr import parse_final_integer
from minigrpo.grpo import TokenRow, group_advantages, reference_kl, reward_value, sequence_loss
from minigrpo.splits import SplitTask, dev_tasks, derive_seeds, train_tasks

MAX_STEPS = 10_000
DEV_SEED_OFFSET = 50_000_000


@dataclass(frozen=True, slots=True)
class RunConfig:
    backend: str
    model: str
    revision: str
    steps: int
    group_size: int
    max_new_tokens: int
    learning_rate: float
    kl_coef: float
    clip_eps: float
    dr_grpo: bool
    epochs: int
    lora_rank: int
    lora_scale: float
    lora_dropout: float
    num_layers: int
    seed: int
    temperature: float
    output: str
    max_seq_length: int = 512
    reward_mode: str = "correctness"
    init_adapter: str = ""
    eval_dev: bool = True

    def identity(self) -> dict[str, object]:
        data = {
            "backend": self.backend,
            "clip_eps": self.clip_eps,
            "dr_grpo": self.dr_grpo,
            "epochs": self.epochs,
            "eval_dev": self.eval_dev,
            "group_size": self.group_size,
            "init_adapter": self.init_adapter,
            "kl_coef": self.kl_coef,
            "learning_rate": self.learning_rate,
            "lora_dropout": self.lora_dropout,
            "lora_rank": self.lora_rank,
            "lora_scale": self.lora_scale,
            "max_new_tokens": self.max_new_tokens,
            "max_seq_length": self.max_seq_length,
            "model": self.model,
            "num_layers": self.num_layers,
            "reward_mode": self.reward_mode,
            "revision": self.revision,
            "seed": self.seed,
            "steps": self.steps,
            "temperature": self.temperature,
        }
        return data


@dataclass(frozen=True, slots=True)
class Rollout:
    text: str
    old_logp: tuple[float, ...]
    mask: tuple[int, ...]
    entropy: tuple[float, ...]
    ref_logp: tuple[float, ...] | None = None
    token_ids: tuple[int, ...] = ()
    prompt_length: int = 0


class Trainer(Protocol):
    def prepare(self, resume: Path | None) -> None: ...

    def generate(self, prompt: str, seed: int) -> str: ...

    def encode(self, prompt: str, completion: str) -> Rollout: ...

    def update(self, rollouts: list[Rollout], advantages: list[float], *, epochs: int) -> float: ...

    def save(self, directory: Path) -> None: ...


def rollout_seed(base: int, step: int, index: int, group_size: int) -> int:
    if not 1 <= step <= MAX_STEPS or not 0 <= index < group_size or group_size >= 1000:
        raise ValueError("rollout seed is outside the supported grid")
    return base + step * 1_000 + index


def dev_sample_seed(base: int, task_index: int) -> int:
    if task_index < 0:
        raise ValueError("task index must be non-negative")
    return base + DEV_SEED_OFFSET + task_index


def check_payload() -> dict[str, object]:
    train = train_tasks()
    dev = dev_tasks()
    confirm = set(derive_seeds("confirm"))
    used = {task.seed for task in train} | {task.seed for task in dev}
    if used & confirm:
        raise RuntimeError("confirm seed leaked into train or dev")
    if {task.split for task in train} != {"train"} or {task.split for task in dev} != {"dev"}:
        raise RuntimeError("split labels are wrong")
    return {
        "confirm_tasks_used": 0,
        "dev_tasks": len(dev),
        "status": "checked",
        "train_tasks": len(train),
    }


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise SystemExit(f"{path} contains a non-object row")
            rows.append(payload)
    return rows


def checkpoint_dir(output: Path, step: int) -> Path:
    return output / "checkpoints" / f"step-{step:04d}"


def _validate(config: RunConfig, tasks: list[SplitTask] | tuple[SplitTask, ...]) -> None:
    if config.steps < 1 or config.steps > MAX_STEPS:
        raise ValueError("steps must lie between 1 and 10000")
    if config.group_size < 2 or config.group_size >= 1000:
        raise ValueError("group_size must lie between 2 and 999")
    if not tasks:
        raise ValueError("training tasks are empty")
    if any(task.split == "confirm" for task in tasks):
        raise ValueError("confirm tasks cannot be trained")
    if config.reward_mode not in {"correctness", "format"}:
        raise ValueError("reward_mode must be correctness or format")


def _completed_steps(rows: list[dict[str, object]]) -> list[int]:
    steps = [int(row["step"]) for row in rows if row.get("kind") == "train"]
    if steps != list(range(1, len(steps) + 1)):
        raise SystemExit("metrics.jsonl train steps are missing or duplicated")
    return steps


def _masked_mean(values: tuple[float, ...], mask: tuple[int, ...]) -> float:
    chosen = [value for value, bit in zip(values, mask, strict=True) if bit]
    if not chosen:
        raise RuntimeError("completion mask is empty")
    return sum(chosen) / len(chosen)


def _kl_mean(rollouts: list[Rollout]) -> float:
    total = 0.0
    count = 0
    for rollout in rollouts:
        if rollout.ref_logp is None:
            continue
        for logp, ref, bit in zip(rollout.old_logp, rollout.ref_logp, rollout.mask, strict=True):
            if bit:
                total += reference_kl(logp, ref)
                count += 1
    return 0.0 if count == 0 else total / count


def _load_identity(output: Path, config: RunConfig) -> None:
    path = output / "run.json"
    identity = config.identity()
    if not path.exists():
        path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved != identity:
        raise SystemExit("output directory was created with a different run configuration")


def execute(
    trainer: Trainer,
    config: RunConfig,
    tasks: list[SplitTask] | tuple[SplitTask, ...],
    dev: list[SplitTask] | tuple[SplitTask, ...],
) -> None:
    _validate(config, tasks)
    if any(task.split == "confirm" for task in dev):
        raise ValueError("confirm tasks cannot be evaluated by this loop")
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    _load_identity(output, config)
    metrics = output / "metrics.jsonl"
    rows = read_jsonl(metrics)
    done = _completed_steps(rows)
    if done and done[-1] > config.steps:
        raise SystemExit("metrics.jsonl has more steps than this run")
    resume = checkpoint_dir(output, done[-1]) if done else None
    if resume is not None and not resume.exists():
        raise SystemExit(f"missing checkpoint for step {done[-1]}")
    need_train = not done or done[-1] < config.steps
    if need_train:
        trainer.prepare(resume if done else None)
    task_list = list(tasks)
    for step in range((done[-1] + 1) if done else 1, config.steps + 1):
        task = task_list[(step - 1) % len(task_list)]
        rollouts: list[Rollout] = []
        rewards: list[float] = []
        for index in range(config.group_size):
            seed = rollout_seed(config.seed, step, index, config.group_size)
            text = trainer.generate(task.prompt, seed)
            rollout = trainer.encode(task.prompt, text)
            if sum(rollout.mask) < 1:
                raise RuntimeError("empty completion; refusing to drop a group member")
            rollouts.append(rollout)
            rewards.append(reward_value(config.reward_mode, task.expression, text))
        advantages = group_advantages(rewards, config.group_size, dr_grpo=config.dr_grpo)
        if len(advantages) != len(rollouts):
            raise RuntimeError("group was truncated")
        loss = trainer.update(rollouts, advantages, epochs=config.epochs)
        if not isinstance(loss, float) or not math.isfinite(loss):
            raise RuntimeError("non-finite loss")
        sequence_loss(
            [
                TokenRow(
                    logp=rollout.old_logp,
                    logp_old=rollout.old_logp,
                    mask=rollout.mask,
                    advantage=advantage,
                    logp_ref=rollout.ref_logp,
                    entropy=rollout.entropy,
                )
                for rollout, advantage in zip(rollouts, advantages, strict=True)
            ],
            clip_eps=config.clip_eps,
            kl_coef=config.kl_coef,
            dr_grpo=config.dr_grpo,
        )
        directory = checkpoint_dir(output, step)
        trainer.save(directory)
        lengths = [float(sum(rollout.mask)) for rollout in rollouts]
        entropies = [_masked_mean(rollout.entropy, rollout.mask) for rollout in rollouts]
        formats = [1.0 if parse_final_integer(rollout.text) is not None else 0.0 for rollout in rollouts]
        append_jsonl(
            metrics,
            {
                "dr_grpo": config.dr_grpo,
                "entropy_mean": sum(entropies) / len(entropies),
                "format_compliance_rate": sum(formats) / len(formats),
                "group_size": config.group_size,
                "kind": "train",
                "kl_coef": config.kl_coef,
                "kl_mean": _kl_mean(rollouts),
                "loss": loss,
                "response_length_mean": sum(lengths) / len(lengths),
                "reward_mean": sum(rewards) / len(rewards),
                "reward_mode": config.reward_mode,
                "split": task.split,
                "step": step,
                "task_seed": task.seed,
            },
        )
    if not config.eval_dev:
        return
    rows = read_jsonl(metrics)
    if any(row.get("kind") == "dev_summary" and int(row.get("step", -1)) == config.steps for row in rows):
        return
    if not need_train:
        trainer.prepare(checkpoint_dir(output, config.steps))
    correct = 0
    compliant = 0
    for index, task in enumerate(dev):
        text = trainer.generate(task.prompt, dev_sample_seed(config.seed, index))
        hit = reward_value("correctness", task.expression, text) == 1.0
        formatted = parse_final_integer(text) is not None
        correct += int(hit)
        compliant += int(formatted)
        append_jsonl(
            metrics,
            {
                "correct": hit,
                "format": formatted,
                "kind": "dev",
                "split": "dev",
                "step": config.steps,
                "task_seed": task.seed,
            },
        )
    if not dev:
        return
    append_jsonl(
        metrics,
        {
            "accuracy": correct / len(dev),
            "format_compliance_rate": compliant / len(dev),
            "kind": "dev_summary",
            "n": len(dev),
            "split": "dev",
            "step": config.steps,
        },
    )
