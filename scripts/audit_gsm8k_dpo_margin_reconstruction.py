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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/cpu_lm_gsm8k_dpo_margin_reconstruction_v1.lock.json"
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
                        default=ROOT / "results/cpu-lm-gsm8k-dpo-margin-reconstruction-v1/run-1")
    args = parser.parse_args()
    started = time.monotonic()

    lock = load(PROTOCOL)
    lock_hash = lock.pop("sha256", None)
    if lock_hash != hashlib.sha256(canonical(lock)).hexdigest():
        raise ValueError("protocol lock hash mismatch")
    if digest(Path(__file__)) != lock["reproducer"]["sha256"]:
        raise ValueError("reproducer source hash mismatch")
    if args.output.exists():
        raise FileExistsError(f"output must be new: {args.output}")
    args.output.mkdir(parents=True)

    os.environ.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    bundle = args.bundle.resolve()
    model_dir = args.model_dir.resolve()
    manifest = load(bundle / "manifest.json")
    for relative, expected in manifest["files"].items():
        if digest(bundle / relative) != expected:
            raise ValueError(f"bundle hash mismatch: {relative}")
    summary = load(bundle / "summary.json")
    if summary["model_revision"] != lock["model_revision"]:
        raise ValueError("model revision does not match the frozen audit")
    observed_model_hashes = {name: digest(model_dir / name) for name in lock["model_files"]}
    if observed_model_hashes != summary["model_file_sha256"]:
        raise ValueError("cached model files do not match the original run")

    examples = load(bundle / "examples.json")
    records = load(bundle / "seed_records.json")
    expected_seeds = [str(seed) for seed in lock["seeds"]]
    if list(records) != expected_seeds or len(examples["train"]) != lock["train_examples"] or len(examples["heldout"]) != lock["heldout_examples"]:
        raise ValueError("example or seed count differs from the locked bundle contract")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(lock["threads"])
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_a = tokenizer.encode(" A", add_special_tokens=False)
    token_b = tokenizer.encode(" B", add_special_tokens=False)
    token_ids = summary["response_token_ids"]
    if len(token_a) != 1 or len(token_b) != 1 or token_ids != {"A": token_a[0], "B": token_b[0]}:
        raise ValueError("response token IDs differ from the original run")

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=torch.float32)
    model.to("cpu").eval()
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("model did not load entirely on CPU")
    output_rows = {}
    max_errors = {"base_margin": 0.0, "updated_margin": 0.0}

    for split in ("train", "heldout"):
        rows = examples[split]
        all_hidden, all_base_logits = [], []
        for start in range(0, len(rows), lock["batch_size"]):
            batch_rows = rows[start:start + lock["batch_size"]]
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
            base_metrics = summarize(base_margin.tolist(), signs)
            updated_metrics = summarize(updated_margin.tolist(), signs)
            kl = bernoulli_kl(base_margin.tolist(), updated_margin.tolist())
            split_records[seed] = {
                "max_base_margin_abs_error": float(base_error.max()),
                "max_updated_margin_abs_error": float(updated_error.max()),
                "base": base_metrics,
                "updated": updated_metrics,
                "updated_to_base_bernoulli_kl": kl,
                "nll_change": updated_metrics["nll"] - base_metrics["nll"],
            }
        output_rows[split] = split_records
        del hidden, base_logits, all_hidden, all_base_logits

    tolerance = lock["max_abs_margin_error"]
    all_margins_match = all(value <= tolerance for value in max_errors.values())
    heldout_changes = [output_rows["heldout"][seed]["nll_change"] for seed in expected_seeds]
    original_changes = [summary["per_seed"][i]["metrics"]["heldout_updated"]["mean_conditional_preference_nll"] -
                        summary["per_seed"][i]["metrics"]["heldout_base"]["mean_conditional_preference_nll"]
                        for i in range(len(expected_seeds))]
    max_seed_nll_change_error = max(abs(a - b) for a, b in zip(heldout_changes, original_changes, strict=True))
    status = "pass" if all_margins_match and max_seed_nll_change_error <= lock["max_abs_metric_error"] else "non_pass"
    result = {
        "protocol_id": lock["protocol_id"],
        "status": status,
        "protocol_sha256": lock_hash,
        "bundle": str(bundle),
        "model_revision": lock["model_revision"],
        "model_file_sha256": observed_model_hashes,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "numpy": np.__version__},
        "device": "cpu",
        "network_disabled": True,
        "max_abs_margin_error": max_errors,
        "max_abs_metric_error": max_seed_nll_change_error,
        "tolerance": {"margin": tolerance, "metric": lock["max_abs_metric_error"]},
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
    result["resource_limits_respected"] = (
        result["wall_seconds"] <= lock["max_wall_seconds"] and
        result["peak_rss_bytes"] <= lock["max_peak_rss_bytes"]
    )
    result_path = args.output / "summary.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "max_abs_margin_error", "max_abs_metric_error",
                                                  "runtime", "wall_seconds", "peak_rss_bytes",
                                                  "resource_limits_respected")}, indent=2, sort_keys=True))
    return 0 if status == "pass" and result["resource_limits_respected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
