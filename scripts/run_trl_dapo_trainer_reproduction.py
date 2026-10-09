#!/usr/bin/env python3
"""Run one offline, CPU-only DAPO normalization arm through real TRL GRPOTrainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "protocols/trl_dapo_trainer_reproduction_v1.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_and_model(source_root: Path, model_root: Path, revision_arm: str) -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    source = lock["source"]
    model = lock["model"]
    expected_source_hash = source["baseline_sha256"] if revision_arm == "baseline" else source["fixed_sha256"]
    source_file = source_root / source["trainer_path"]
    if sha256(source_file) != expected_source_hash:
        raise ValueError(f"TRL trainer source hash mismatch: {source_file}")
    found_model_hashes = {}
    for filename, expected_hash in model["files_sha256"].items():
        path = model_root / filename
        found_model_hashes[filename] = sha256(path)
        if found_model_hashes[filename] != expected_hash:
            raise ValueError(f"model file hash mismatch: {path}")
    return {
        "trl_trainer_sha256": expected_source_hash,
        "model_revision": model["revision"],
        "model_files_sha256": found_model_hashes,
    }


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    source_root = args.source_root.resolve()
    model_root = args.model_root.resolve()
    provenance = verify_source_and_model(source_root, model_root, args.revision_arm)

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    trl_path = Path(sys.modules["trl"].__file__).resolve()
    if source_root not in trl_path.parents:
        raise ValueError(f"loaded TRL from {trl_path}, expected source under {source_root}")

    set_seed(20261009)
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(model_root), local_files_only=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    dataset = Dataset.from_list([{"prompt": "The capital of France is"} for _ in range(32)])

    def reward_by_character_length(completions, **kwargs):
        return [float(len(completion)) for completion in completions]

    output_dir = Path(tempfile.mkdtemp(prefix="vare-trl-dapo-trainer-"))
    config = GRPOConfig(
        output_dir=str(output_dir),
        loss_type="dapo",
        learning_rate=1e-7,
        per_device_train_batch_size=3,
        num_generations=3,
        max_completion_length=8,
        steps_per_generation=args.steps_per_generation,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_iterations=1,
        beta=0.0,
        max_steps=args.max_steps,
        seed=20261009,
        data_seed=20261009,
        report_to="none",
        use_cpu=True,
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_by_character_length,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    captured = []
    original_compute_loss = trainer._compute_loss

    def capture_compute_loss(wrapped_model, inputs):
        loss = original_compute_loss(wrapped_model, inputs)
        captured.append(
            {
                "actual_loss": float(loss.detach().cpu()),
                "num_items_in_batch": float(inputs["num_items_in_batch"].detach().cpu()),
                "completion_token_counts": inputs["completion_mask"].sum(-1).detach().cpu().tolist(),
                "advantages": inputs["advantages"].detach().cpu().tolist(),
            }
        )
        return loss

    trainer._compute_loss = capture_compute_loss
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()
    }
    start = time.perf_counter()
    trainer.train()
    wall_seconds = time.perf_counter() - start

    expected_ratio = (
        args.gradient_accumulation_steps / args.steps_per_generation
        if args.revision_arm == "baseline"
        else 1.0
    )
    records = []
    for index, row in enumerate(captured):
        advantages = torch.tensor(row["advantages"], dtype=torch.float64)
        token_counts = torch.tensor(row["completion_token_counts"], dtype=torch.float64)
        expected_loss = float(
            (-advantages * token_counts).sum()
            / (row["num_items_in_batch"] * args.gradient_accumulation_steps / args.steps_per_generation)
        )
        observed_ratio = row["actual_loss"] / expected_loss if abs(expected_loss) > 1e-12 else None
        records.append(
            {
                "microbatch_index": index,
                "actual_loss": row["actual_loss"],
                "expected_window_loss": expected_loss,
                "actual_over_expected": observed_ratio,
                "predicted_ratio": expected_ratio,
                "num_items_in_batch": row["num_items_in_batch"],
                "completion_token_counts": row["completion_token_counts"],
                "reward_signal_nonzero": bool(float(advantages.abs().sum()) > 0),
            }
        )

    changed_parameter_tensors = 0
    total_absolute_parameter_delta = 0.0
    max_absolute_parameter_delta = 0.0
    for name, parameter in trainer.model.named_parameters():
        before = initial_parameters[name]
        delta = (parameter.detach().cpu() - before.cpu()).abs()
        if bool(torch.any(delta != 0)):
            changed_parameter_tensors += 1
        total_absolute_parameter_delta += float(delta.sum())
        max_absolute_parameter_delta = max(max_absolute_parameter_delta, float(delta.max()))

    acceptance_errors = []
    expected_count = args.max_steps * args.gradient_accumulation_steps
    if len(captured) != expected_count:
        acceptance_errors.append(
            f"captured {len(captured)} microbatches; expected {expected_count}"
        )
    if not all(row["reward_signal_nonzero"] for row in records):
        acceptance_errors.append("a captured batch had no nonzero relative-reward signal")
    if any(row["actual_over_expected"] is None for row in records):
        acceptance_errors.append("a captured expected loss was zero")
    for row in records:
        if row["actual_over_expected"] is not None and not math.isclose(
            row["actual_over_expected"], expected_ratio, rel_tol=1e-4, abs_tol=1e-6
        ):
            acceptance_errors.append(
                f"microbatch {row['microbatch_index']} ratio "
                f"{row['actual_over_expected']} differs from expected {expected_ratio}"
            )
    if changed_parameter_tensors == 0:
        acceptance_errors.append("optimizer completed without changing any model parameter")

    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        max_rss *= 1024
    shutil.rmtree(output_dir, ignore_errors=True)
    return {
        "protocol_id": "trl_dapo_trainer_reproduction_v1",
        "status": "pass" if not acceptance_errors else "nonpass",
        "acceptance_errors": acceptance_errors,
        "revision_arm": args.revision_arm,
        "configuration": {
            "steps_per_generation": args.steps_per_generation,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "trl_source": str(trl_path),
            "device": str(trainer.args.device),
            "world_size": trainer.accelerator.num_processes,
        },
        "provenance": provenance,
        "captured_microbatch_count": len(captured),
        "expected_observed_ratio": expected_ratio,
        "records": records,
        "actual_optimizer_update": {
            "changed_parameter_tensor_count": changed_parameter_tensors,
            "total_absolute_parameter_delta": total_absolute_parameter_delta,
            "max_absolute_parameter_delta": max_absolute_parameter_delta,
        },
        "resources": {
            "wall_seconds": wall_seconds,
            "peak_rss_gib": max_rss / 1024**3,
        },
        "train_metrics": trainer.state.log_history,
        "disclaimer": "Real pinned GRPOTrainer execution with a synthetic character-length reward; no capability evaluation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-arm", choices=("baseline", "fixed"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--steps-per-generation", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
