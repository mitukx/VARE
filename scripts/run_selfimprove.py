#!/usr/bin/env python3
"""Propose integer problems, train with minigrpo, and roll back through PromotionGate.

Dev is the held-out split. Confirm is not scored. ``--reward format`` is the
hackable reward: it pays for a strict final line and ignores the value.
``--check`` does not import MLX or PyTorch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minigrpo.loop import RunConfig
from selfimprove.loop import run_rounds
from selfimprove.model import ModelSolver, build_trainer
from selfimprove.propose import propose
from selfimprove.reports import DEV_COUNT, dev_gate


def check_payload() -> dict[str, object]:
    problems = propose(1, 2, 4)
    if any(problem.value is None for problem in problems):
        raise RuntimeError("check kept an empty problem")
    return {
        "confirm_tasks_used": 0,
        "dev_tasks": DEV_COUNT,
        "min_eval_examples": dev_gate().config.min_eval_examples,
        "proposed_example": problems[0].expression,
        "status": "checked",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Integer proposer and promotion loop")
    parser.add_argument("--backend", choices=("mlx", "torch"), required=True)
    parser.add_argument("--model", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps-per-round", type=int, default=8)
    parser.add_argument("--problems", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reward", choices=("correctness", "format"), default="correctness")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        print(json.dumps(check_payload(), sort_keys=True))
        return 0
    if not args.model or not args.output:
        raise SystemExit("--model and --output are required")
    config = RunConfig(
        backend=args.backend,
        model=args.model,
        revision=args.revision,
        steps=args.steps_per_round,
        group_size=args.group_size,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        kl_coef=0.0,
        clip_eps=0.2,
        dr_grpo=False,
        epochs=1,
        lora_rank=8,
        lora_scale=20.0,
        lora_dropout=0.0,
        num_layers=args.num_layers,
        seed=args.seed,
        temperature=args.temperature,
        output=args.output,
        reward_mode=args.reward,
        eval_dev=False,
    )
    solver = ModelSolver(build_trainer(args.backend, config), config, Path(args.output))
    run_rounds(
        solver,
        output=Path(args.output),
        rounds=args.rounds,
        problems_per_round=args.problems,
        steps_per_round=args.steps_per_round,
        reward_mode=args.reward,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
