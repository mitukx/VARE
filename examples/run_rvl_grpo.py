"""Real-model VARE loop using Recursive-Verification-Lag's HF/GRPO substrate.

Example:
  PYTHONPATH=/path/to/Recursive-Verification-Lag:/path/to/VARE/src \
    python examples/run_rvl_grpo.py --dataset gsm8k --model Qwen/Qwen2.5-0.5B-Instruct

This script intentionally does not install heavyweight GPU dependencies.
Use the pinned environment from the RVL repository.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from vare.config import EngineConfig, PromotionConfig
from vare.engine import CapabilityLoop
from vare.evidence import HashChainLedger
from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
from vare.telemetry import EventLog
from vare.types import Task
from vare.verifiers import FunctionalAttemptVerifier, VerifierEnsemble, VerifierMember


def _load_jsonl(path: str):
    train, evaluation = [], []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        raw = json.loads(line)
        task = Task(
            id=str(raw.get("id", f"task-{index}")),
            prompt=str(raw["prompt"]),
            family=str(raw.get("family", "default")),
            metadata={"answer": str(raw["answer"]), "split": str(raw.get("split", "train"))},
        )
        (train if task.metadata["split"] == "train" else evaluation).append(task)
    if not train or not evaluation:
        raise ValueError("JSONL must contain both train and non-train examples")
    return train, evaluation


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", choices=["gsm8k", "jsonl"], default="gsm8k")
    parser.add_argument("--tasks-jsonl")
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--prompts-per-round", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--train-temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="artifacts/rvl-grpo")
    args = parser.parse_args()

    try:
        from src.rvl_systems.hf_backend import HFLocalBackend
        from src.rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
        from src.rvl_systems.rlvr_benchmark import load_gsm8k_tasks, response_reward
    except ImportError as exc:
        raise RuntimeError(
            "Recursive-Verification-Lag must be importable; add its repository root to PYTHONPATH"
        ) from exc

    if args.dataset == "gsm8k":
        train_raw, eval_raw = load_gsm8k_tasks(
            train_limit=args.train_limit,
            eval_limit=args.eval_limit,
            seed=args.seed,
        )
        train_tasks = [
            Task(x.task_id, x.prompt, family="gsm8k", metadata={"answer": x.answer, "split": "train"})
            for x in train_raw
        ]
        eval_tasks = [
            Task(x.task_id, x.prompt, family="gsm8k", metadata={"answer": x.answer, "split": "eval"})
            for x in eval_raw
        ]
    else:
        if not args.tasks_jsonl:
            parser.error("--tasks-jsonl is required for --dataset jsonl")
        train_tasks, eval_tasks = _load_jsonl(args.tasks_jsonl)
        train_tasks = train_tasks[: args.train_limit]
        eval_tasks = eval_tasks[: args.eval_limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = HFLocalBackend(
        args.model,
        max_new_tokens=args.max_new_tokens,
        precision=args.precision,
    )
    trainer = HFCausalLMGRPOTrainer(
        backend.model,
        config=HFTTrainerConfig(learning_rate=args.learning_rate),
    )
    score_fn = lambda task, response: response_reward(response, str(task.metadata["answer"]))
    hooks = RVLGRPOHooks(
        backend=backend,
        trainer=trainer,
        eval_tasks=eval_tasks,
        score_fn=score_fn,
        config=RVLGRPOConfig(train_temperature=args.train_temperature, seed=args.seed),
    )
    verifier = VerifierEnsemble([
        VerifierMember(
            FunctionalAttemptVerifier(
                lambda attempt: score_fn(attempt.task, attempt.output),
                name="exact-task-reward",
                version=1,
                trusted=True,
            )
        )
    ])
    loop = CapabilityLoop(
        hooks=hooks,
        verifier=verifier,
        config=EngineConfig(
            rollout_concurrency=1,
            replay_batch_size=args.prompts_per_round * args.samples_per_prompt,
            samples_per_task=args.samples_per_prompt,
            preserve_rollout_groups=True,
            promotion=PromotionConfig(
                min_primary_gain=0.0,
                max_slice_regression=0.03,
                min_eval_examples=len(eval_tasks),
                paired_confidence_gate=True,
                min_paired_examples=len(eval_tasks),
            ),
        ),
        event_log=EventLog(output_dir / "events.jsonl"),
        decision_ledger=HashChainLedger(output_dir / "promotion-ledger.jsonl"),
        seed=args.seed,
    )

    results = []
    for round_index in range(args.rounds):
        result = await loop.run_round(
            train_tasks,
            round_index=round_index,
            rollout_count=args.prompts_per_round * args.samples_per_prompt,
        )
        results.append(asdict(result))
        print(
            f"round={round_index} accepted={result.decision.accepted} "
            f"gain={result.decision.primary_gain:+.4f} lcb={result.decision.paired_lcb}"
        )
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main())
