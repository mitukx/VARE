#!/usr/bin/env python3
"""Train a small GRPO policy on the integer seed-split train rows.

Dev is scored after the last step. The confirm split is not loaded.
``--check`` does not import MLX or PyTorch and does not write metrics.
"""

from __future__ import annotations

import argparse
import json

from minigrpo.loop import RunConfig, check_payload, execute
from minigrpo.splits import dev_tasks, train_tasks


def _config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        backend=args.backend,
        model=args.model,
        revision=args.revision,
        steps=args.steps,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        clip_eps=args.clip_eps,
        dr_grpo=args.dr_grpo,
        epochs=args.epochs,
        lora_rank=args.lora_rank,
        lora_scale=args.lora_scale,
        lora_dropout=0.0,
        num_layers=args.num_layers,
        seed=args.seed,
        temperature=args.temperature,
        output=args.output,
        reward_mode="correctness",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Integer-expression GRPO")
    parser.add_argument("--backend", choices=("mlx", "torch"), required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dr-grpo", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(check_payload(), sort_keys=True))
        return 0
    if not args.model or not args.output:
        raise SystemExit("--model and --output are required")
    config = _config(args)
    if args.backend == "mlx":
        from minigrpo.mlx_train import MlxTrainer

        trainer = MlxTrainer(config)
    else:
        from minigrpo.torch_train import TorchTrainer

        trainer = TorchTrainer(config)
    execute(trainer, config, train_tasks(), dev_tasks())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
