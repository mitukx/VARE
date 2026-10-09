#!/usr/bin/env python3
"""Independently reconstruct a frozen GSM8K DPO bundle's adapter-to-margin path."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/cpu_lm_gsm8k_dpo_margin_reconstruction_v4.lock.json"
DEFAULT_BUNDLE = ROOT / "results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1"
MODEL_ID = "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rss_bytes() -> int:
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def summarize(margins: list[float], signs: list[int]) -> dict[str, float]:
    losses = [max(-sign * margin, 0.0) + math.log1p(math.exp(-abs(sign * margin)))
              for margin, sign in zip(margins, signs, strict=True)]
    correct = [1.0 if sign * margin > 0 else 0.5 if margin == 0 else 0.0
               for margin, sign in zip(margins, signs, strict=True)]
    return {"nll": statistics.fmean(losses), "accuracy": statistics.fmean(correct)}


def bernoulli_kl(base: list[float], updated: list[float]) -> float:
    total = 0.0
    for before, after in zip(base, updated, strict=True):
        p, q = sigmoid(before), sigmoid(after)
        total += q * math.log(q / p) + (1.0 - q) * math.log((1.0 - q) / (1.0 - p))
    return total / len(base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--model-dir", type=Path, default=Path.home() / ".cache/huggingface/hub" / MODEL_ID)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/cpu-lm-gsm8k-dpo-margin-reconstruction-v4/run-1")
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"output must be new: {args.output}")
    try:
        args.output.mkdir(parents=True)
        return _run(args, started)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = {
            "protocol_id": "vare-cpu-lm-gsm8k-dpo-margin-reconstruction-v4",
            "status": "non_pass_or_incomplete",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": rss_bytes(),
            "partial_files": sorted(path.relative_to(args.output).as_posix()
                                     for path in args.output.rglob("*") if path.is_file()),
        }
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, started: float) -> int:

    lock = load(PROTOCOL)
    lock_hash = lock.pop("sha256", None)
    if lock_hash != hashlib.sha256(canonical(lock)).hexdigest():
        raise ValueError("protocol lock hash mismatch")
    if digest(Path(__file__)) != lock["reproducer"]["sha256"]:
        raise ValueError("reproducer source hash mismatch")
    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    bundle = args.bundle.resolve()
    model_dir = args.model_dir.resolve()
    confirmation_lock = load(ROOT / lock["source"]["confirmation_protocol"])
    confirmation_lock_hash = confirmation_lock.pop("sha256", None)
    if confirmation_lock_hash != lock["source"]["confirmation_protocol_sha256"]:
        raise ValueError("original confirmation protocol lock mismatch")
    if digest(bundle / "manifest.json") != lock["source"]["bundle_manifest_sha256"]:
        raise ValueError("original bundle manifest hash differs from frozen audit input")
    manifest = load(bundle / "manifest.json")
    for relative, expected in manifest["files"].items():
        if digest(bundle / relative) != expected:
            raise ValueError(f"bundle hash mismatch: {relative}")
    summary = load(bundle / "summary.json")
    if summary["model_revision"] != lock["inputs"]["model_revision"]:
        raise ValueError("model revision does not match the frozen audit")
    observed_model_hashes = {name: digest(model_dir / name) for name in lock["inputs"]["model_files"]}
    if observed_model_hashes != summary["model_file_sha256"]:
        raise ValueError("cached model files do not match the original run")

    examples = load(bundle / "examples.json")
    records = load(bundle / "seed_records.json")
    expected_seeds = [str(seed) for seed in lock["inputs"]["seeds"]]
    if (list(records) != expected_seeds or len(examples["train"]) != lock["inputs"]["train_examples"] or
            len(examples["heldout"]) != lock["inputs"]["heldout_examples"]):
        raise ValueError("example or seed count differs from the locked bundle contract")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(lock["resources"]["threads"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_a = tokenizer.encode(" A", add_special_tokens=False)
    token_b = tokenizer.encode(" B", add_special_tokens=False)
    token_ids = summary["response_token_ids"]
    if lock["inputs"]["response_tokens"] != [" A", " B"]:
        raise ValueError("unexpected response token contract")
    if len(token_a) != 1 or len(token_b) != 1 or token_ids != {"A": token_a[0], "B": token_b[0]}:
        raise ValueError("response token IDs differ from the original run")

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=torch.float32)
    model.to("cpu").eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("model did not load entirely on CPU")
    if time.monotonic() - started > lock["resources"]["max_wall_seconds"]:
        raise TimeoutError("model load exceeded the frozen wall-time cap")
    if rss_bytes() > lock["resources"]["max_peak_rss_bytes"]:
        raise MemoryError("model load exceeded the frozen RSS cap")
    output_rows = {}
    reconstructed_margins = {"train": {}, "heldout": {}}
    max_errors = {"base_margin": 0.0, "updated_margin": 0.0}

    for split in ("train", "heldout"):
        rows = examples[split]
        all_hidden, all_base_logits = [], []
        for start in range(0, len(rows), lock["resources"]["batch_size"]):
            batch_rows = rows[start:start + lock["resources"]["batch_size"]]
            batch = tokenizer([row["prompt"] for row in batch_rows], return_tensors="pt", padding=True,
                              add_special_tokens=True, truncation=False)
            last_indices = batch["attention_mask"].sum(dim=1) - 1
            with torch.inference_mode():
                final_hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                           use_cache=False).last_hidden_state
                index = torch.arange(final_hidden.shape[0])
                hidden = final_hidden[index, last_indices].to(dtype=torch.float32).cpu().numpy()
                rows_w = model.lm_head.weight[token_ids["A"], :], model.lm_head.weight[token_ids["B"], :]
                base_logits = np.column_stack((hidden @ rows_w[0].detach().cpu().numpy(),
                                               hidden @ rows_w[1].detach().cpu().numpy()))
            all_hidden.append(hidden)
            all_base_logits.append(base_logits)
            if time.monotonic() - started > lock["resources"]["max_wall_seconds"]:
                raise TimeoutError("frozen forensic audit exceeded its wall-time cap")
            if rss_bytes() > lock["resources"]["max_peak_rss_bytes"]:
                raise MemoryError("frozen forensic audit exceeded its RSS cap")
        hidden = np.concatenate(all_hidden, axis=0).astype(np.float32, copy=False)
        base_logits = np.concatenate(all_base_logits, axis=0).astype(np.float32, copy=False)
        base_margin = (base_logits[:, 0] - base_logits[:, 1]).astype(np.float64)
        signs = [1 if row["correct_label"] == "A" else -1 for row in rows]
        split_records = {}
        for seed in expected_seeds:
            record = records[seed]
            recorded_base = np.asarray(record[f"{split}_base_margins"], dtype=np.float64)
            recorded_updated = np.asarray(record[f"{split}_updated_margins"], dtype=np.float64)
            base_error = np.abs(base_margin - recorded_base)
            max_errors["base_margin"] = max(max_errors["base_margin"], float(base_error.max()))

            adapter_a = np.asarray(record["adapter"]["A_final"], dtype=np.float32)
            adapter_b = np.asarray(record["adapter"]["B_final"], dtype=np.float32)
            # Independent NumPy implementation of the frozen residual: delta = (h A) B.
            residual_logits = (hidden @ adapter_a) @ adapter_b
            updated_margin = base_margin + residual_logits[:, 0].astype(np.float64) - residual_logits[:, 1].astype(np.float64)
            updated_error = np.abs(updated_margin - recorded_updated)
            max_errors["updated_margin"] = max(max_errors["updated_margin"], float(updated_error.max()))
            reconstructed_margins[split][seed] = (base_margin.tolist(), updated_margin.tolist())
            base_metrics = summarize(base_margin.tolist(), signs)
            updated_metrics = summarize(updated_margin.tolist(), signs)
            kl = bernoulli_kl(base_margin.tolist(), updated_margin.tolist())
            recorded_metrics = record["metrics"]
            expected_base = recorded_metrics[f"{split}_base"]
            expected_updated = recorded_metrics[f"{split}_updated"]
            metric_errors = {
                "base_nll": abs(base_metrics["nll"] - expected_base["mean_conditional_preference_nll"]),
                "updated_nll": abs(updated_metrics["nll"] - expected_updated["mean_conditional_preference_nll"]),
                "base_accuracy": abs(base_metrics["accuracy"] - expected_base["preference_accuracy"]),
                "updated_accuracy": abs(updated_metrics["accuracy"] - expected_updated["preference_accuracy"]),
            }
            if split == "heldout":
                metric_errors["updated_to_base_kl"] = abs(
                    kl - recorded_metrics["heldout_mean_bernoulli_kl_updated_to_base"]
                )
            split_records[seed] = {
                "max_base_margin_abs_error": float(base_error.max()),
                "max_updated_margin_abs_error": float(updated_error.max()),
                "base": base_metrics,
                "updated": updated_metrics,
                "updated_to_base_bernoulli_kl": kl,
                "nll_change": updated_metrics["nll"] - base_metrics["nll"],
                "metric_abs_errors": metric_errors,
            }
        output_rows[split] = split_records
        del hidden, base_logits, all_hidden, all_base_logits

    tolerance = lock["success_criteria"]["max_abs_margin_error"]
    all_margins_match = all(value <= tolerance for value in max_errors.values())
    all_metric_errors = [error for split in output_rows.values() for record in split.values()
                         for error in record["metric_abs_errors"].values()]
    max_metric_error = max(all_metric_errors)
    heldout_changes = [output_rows["heldout"][seed]["nll_change"] for seed in expected_seeds]
    original_changes = [summary["per_seed"][i]["metrics"]["heldout_updated"]["mean_conditional_preference_nll"] -
                        summary["per_seed"][i]["metrics"]["heldout_base"]["mean_conditional_preference_nll"]
                        for i in range(len(expected_seeds))]
    max_seed_nll_change_error = max(abs(a - b) for a, b in zip(heldout_changes, original_changes, strict=True))
    all_changes = []
    signs = [1 if row["correct_label"] == "A" else -1 for row in examples["heldout"]]
    for seed in expected_seeds:
        base_margins, updated_margins = reconstructed_margins["heldout"][seed]
        # Recompute paired row-level loss differences from the independently reconstructed margins.
        # A/B label signs are fixed by the retained, hash-verified examples.
        for before, after, sign in zip(base_margins, updated_margins, signs, strict=True):
            base_loss = max(-sign * before, 0.0) + math.log1p(math.exp(-abs(sign * before)))
            updated_loss = max(-sign * after, 0.0) + math.log1p(math.exp(-abs(sign * after)))
            all_changes.append((int(seed), updated_loss - base_loss))
    bootstrap_seed = lock["inputs"]["bootstrap_seed"]
    bootstrap_resamples = lock["inputs"]["bootstrap_resamples"]
    rng = __import__("random").Random(bootstrap_seed)
    grouped = {int(seed): [] for seed in expected_seeds}
    for seed, value in all_changes:
        grouped[seed].append(value)
    interval_draws = []
    for _ in range(bootstrap_resamples):
        interval_draws.append(statistics.fmean(
            statistics.fmean(rng.choices(values, k=len(values))) for values in grouped.values()
        ))
    interval_draws.sort()
    reconstructed_interval = [interval_draws[math.floor(.025 * (bootstrap_resamples - 1))],
                              interval_draws[math.floor(.975 * (bootstrap_resamples - 1))]]
    recorded_interval = summary["paired_seed_stratified_bootstrap_95pct"]
    bootstrap_error = max(abs(a - b) for a, b in zip(reconstructed_interval, recorded_interval, strict=True))
    max_metric_error = max(max_metric_error, max_seed_nll_change_error, bootstrap_error)
    resource_ok = time.monotonic() - started <= lock["resources"]["max_wall_seconds"] and rss_bytes() <= lock["resources"]["max_peak_rss_bytes"]
    status = "pass" if (all_margins_match and max_metric_error <= lock["success_criteria"]["max_abs_metric_error"]
                        and resource_ok) else "non_pass"
    result = {
        "protocol_id": lock["protocol_id"],
        "status": status,
        "protocol_sha256": lock_hash,
        "bundle": str(bundle),
        "model_revision": lock["inputs"]["model_revision"],
        "model_file_sha256": observed_model_hashes,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "numpy": np.__version__},
        "device": "cpu",
        "network_controls": {"hf_hub_offline": os.environ["HF_HUB_OFFLINE"],
                             "hf_datasets_offline": os.environ["HF_DATASETS_OFFLINE"],
                             "transformers_offline": os.environ["TRANSFORMERS_OFFLINE"],
                             "local_files_only": True},
        "max_abs_margin_error": max_errors,
        "max_abs_metric_error": max_metric_error,
        "metric_error_components": {"condition_metrics": max(all_metric_errors),
                                    "per_seed_nll_change": max_seed_nll_change_error,
                                    "bootstrap_interval": bootstrap_error},
        "reconstructed_bootstrap_interval": reconstructed_interval,
        "recorded_bootstrap_interval": recorded_interval,
        "tolerance": {"margin": tolerance, "metric": lock["success_criteria"]["max_abs_metric_error"]},
        "all_margins_match": all_margins_match,
        "reconstructed": output_rows,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "limitations": [
            "Forensic replay of already consumed rows; no training, tuning, or new capability claim.",
            "Uses a different installed PyTorch/NumPy version than the original confirmation runtime.",
            "Checks the adapter-to-margin and metric path, not independent human reproduction or task generalization.",
        ],
    }
    result["resource_limits_respected"] = resource_ok
    result_path = args.output / "summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "max_abs_margin_error", "max_abs_metric_error",
                                                  "runtime", "wall_seconds", "peak_rss_bytes",
                                                  "resource_limits_respected")}, indent=2, sort_keys=True))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
