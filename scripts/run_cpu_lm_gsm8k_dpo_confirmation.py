#!/usr/bin/env python3
"""Frozen CPU-only DPO-style confirmation on the GSM8K official test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from typing import Any

from gsm8k_preference_task import make_split

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_lm_gsm8k_dpo_confirmation_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_lm_gsm8k_dpo_confirmation_v1.lock.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest(output: Path) -> None:
    files = {p.relative_to(output).as_posix(): sha256_file(p)
             for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": files})


def preference_metrics(margins: list[float], signs: list[int]) -> dict[str, float]:
    nll = [max(-s * m, 0.0) + math.log1p(math.exp(-abs(s * m))) for m, s in zip(margins, signs)]
    accuracy = [1.0 if s * m > 0 else 0.5 if m == 0 else 0.0 for m, s in zip(margins, signs)]
    return {"mean_conditional_preference_nll": statistics.fmean(nll),
            "preference_accuracy": statistics.fmean(accuracy)}


def kl_to_base(base: list[float], updated: list[float]) -> float:
    total = 0.0
    for before, after in zip(base, updated):
        p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, before))))
        q = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, after))))
        total += q * math.log(max(q, 1e-300) / max(p, 1e-300))
        total += (1.0 - q) * math.log(max(1.0 - q, 1e-300) / max(1.0 - p, 1e-300))
    return total / len(base)


def paired_bootstrap(values: list[tuple[int, float]], seed: int, resamples: int) -> list[float]:
    by_seed: dict[int, list[float]] = {}
    for seed_value, delta in values:
        by_seed.setdefault(seed_value, []).append(delta)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(statistics.fmean(statistics.fmean(rng.choices(group, k=len(group)))
                                      for group in by_seed.values()))
    means.sort()
    return [means[math.floor(.025 * (resamples - 1))], means[math.floor(.975 * (resamples - 1))]]


def locked_spec() -> tuple[dict[str, Any], str]:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked_hash = lock.pop("sha256", None)
    observed_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if locked_hash != observed_hash or canonical(lock) != canonical(spec):
        raise ValueError("confirmation protocol differs from its lock")
    return spec, observed_hash


def run(output: Path) -> dict[str, Any]:
    spec, protocol_hash = locked_spec()
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    for source in (Path(__file__), Path(__file__).with_name("gsm8k_preference_task.py")):
        (output / (source.stem + ".snapshot.py")).write_bytes(source.read_bytes())
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        os.environ[name] = "false" if name == "TOKENIZERS_PARALLELISM" else "1"
    try:
        import datasets
        import numpy
        import torch
        import transformers
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        runtime = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
                   "transformers": transformers.__version__, "numpy": numpy.__version__, "datasets": datasets.__version__}
        if runtime != spec["runtime"]:
            raise RuntimeError(f"runtime differs from lock: {runtime}")
        torch.set_num_threads(spec["compute_limits"]["threads"])
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must not be initialized")
        if MODEL_DIR.name != spec["model"]["revision"]:
            raise ValueError("cached model snapshot revision mismatch")
        if sha256_file(DATA_DIR / "gsm8k-train.arrow") != spec["dataset"]["cached_train_arrow_sha256"]:
            raise ValueError("GSM8K training split hash mismatch")
        if sha256_file(DATA_DIR / "gsm8k-test.arrow") != spec["dataset"]["cached_test_arrow_sha256"]:
            raise ValueError("GSM8K test split hash mismatch")

        train_ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="train")
        if train_ds._fingerprint != spec["dataset"]["cached_fingerprint"]:
            raise ValueError(f"GSM8K train fingerprint mismatch: {train_ds._fingerprint}")
        # This is the first point at which the locked confirmation runner reads test rows.
        test_ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="test")
        if len(test_ds) != spec["dataset"]["confirmation_examples"]:
            raise ValueError(f"GSM8K test row count mismatch: {len(test_ds)}")
        train_candidates = make_split(train_ds, "train", 544)
        train_rows = train_candidates[288:544]
        test_rows = make_split(test_ds, "test", len(test_ds))
        if len(train_rows) != 256 or len(test_rows) != 1319:
            raise ValueError("data selection did not produce locked example counts")
        train_hashes = {row["question_sha256"] for row in train_rows}
        test_hashes = {row["question_sha256"] for row in test_rows}
        if len(train_hashes) != len(train_rows) or len(test_hashes) != len(test_rows):
            raise ValueError("duplicate questions inside a split")
        if train_hashes & test_hashes:
            raise ValueError("train and confirmation question sets overlap")

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("tokenizer has no pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32)
        model.to("cpu").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            if parameter.device.type != "cpu":
                raise RuntimeError("non-CPU model parameter detected")
        token_a = tokenizer.encode(" A", add_special_tokens=False)
        token_b = tokenizer.encode(" B", add_special_tokens=False)
        if len(token_a) != 1 or len(token_b) != 1 or token_a[0] == token_b[0]:
            raise RuntimeError("A/B completions are not distinct single tokens")
        token_ids = [token_a[0], token_b[0]]

        def encode(rows):
            hidden_batches, base_batches = [], []
            batch_size = spec["compute_limits"]["inference_batch_size"]
            for start in range(0, len(rows), batch_size):
                batch_rows = rows[start:start + batch_size]
                batch = tokenizer([row["prompt"] for row in batch_rows], return_tensors="pt", padding=True,
                                  add_special_tokens=True, truncation=False)
                lengths = batch["attention_mask"].sum(dim=1) - 1
                with torch.no_grad():
                    hidden_all = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                             use_cache=False).last_hidden_state
                    indices = torch.arange(hidden_all.shape[0])
                    hidden = hidden_all[indices, lengths].to(dtype=torch.float32).contiguous()
                    base = torch.nn.functional.linear(hidden, model.lm_head.weight[token_ids].to(dtype=torch.float32))
                hidden_batches.append(hidden)
                base_batches.append(base)
            return torch.cat(hidden_batches, dim=0), torch.cat(base_batches, dim=0)

        train_h, train_base = encode(train_rows)
        test_h, test_base = encode(test_rows)
        train_sign = torch.tensor([1.0 if row["correct_label"] == "A" else -1.0 for row in train_rows])
        test_signs = [1 if row["correct_label"] == "A" else -1 for row in test_rows]
        beta, updates, learning_rate, rank = (spec["learner"]["beta"], spec["learner"]["updates"],
                                               spec["learner"]["optimizer"]["learning_rate"],
                                               spec["learner"]["adapter_rank"])
        seed_records = []
        for seed in spec["learner"]["seeds"]:
            seed_start = time.monotonic()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            hidden_size = train_h.shape[1]
            adapter_a = torch.nn.Parameter(torch.randn((hidden_size, rank), generator=generator) / math.sqrt(hidden_size))
            adapter_b = torch.nn.Parameter(torch.zeros((rank, 2)))
            initial_a, initial_b = adapter_a.detach().clone(), adapter_b.detach().clone()
            optimizer = torch.optim.SGD([adapter_a, adapter_b], lr=learning_rate,
                                        weight_decay=spec["learner"]["optimizer"]["weight_decay"])
            train_losses = []
            for _ in range(updates):
                optimizer.zero_grad(set_to_none=True)
                delta = (train_h @ adapter_a) @ adapter_b
                delta_margin = delta[:, 0] - delta[:, 1]
                loss = torch.nn.functional.softplus(-beta * train_sign * delta_margin).mean()
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in (adapter_a, adapter_b)):
                    raise FloatingPointError("non-finite DPO loss or gradient")
                train_losses.append(float(loss.detach()))
                optimizer.step()
            with torch.inference_mode():
                train_delta = (train_h @ adapter_a) @ adapter_b
                test_delta = (test_h @ adapter_a) @ adapter_b
                train_base_margins = (train_base[:, 0] - train_base[:, 1]).tolist()
                train_updated_margins = (train_base[:, 0] - train_base[:, 1] + train_delta[:, 0] - train_delta[:, 1]).tolist()
                test_base_margins = (test_base[:, 0] - test_base[:, 1]).tolist()
                test_updated_margins = (test_base[:, 0] - test_base[:, 1] + test_delta[:, 0] - test_delta[:, 1]).tolist()
                final_a, final_b = adapter_a.detach().clone(), adapter_b.detach().clone()
            train_signs = [int(x) for x in train_sign.tolist()]
            base_train_metrics = preference_metrics(train_base_margins, train_signs)
            updated_train_metrics = preference_metrics(train_updated_margins, train_signs)
            base_test_metrics = preference_metrics(test_base_margins, test_signs)
            updated_test_metrics = preference_metrics(test_updated_margins, test_signs)
            seed_records.append({
                "seed": seed,
                "train_base_margins": train_base_margins,
                "train_updated_margins": train_updated_margins,
                "heldout_base_margins": test_base_margins,
                "heldout_updated_margins": test_updated_margins,
                "metrics": {
                    "train_base": base_train_metrics,
                    "train_updated": updated_train_metrics,
                    "heldout_base": base_test_metrics,
                    "heldout_updated": updated_test_metrics,
                    "heldout_mean_bernoulli_kl_updated_to_base": kl_to_base(test_base_margins, test_updated_margins),
                    "train_objective_first_update": train_losses[0],
                    "train_objective_last_update": train_losses[-1],
                    "adapter_l2": float(torch.sqrt(final_a.square().sum() + final_b.square().sum())),
                    "adapter_update_l2": float(torch.sqrt((final_a-initial_a).square().sum() + (final_b-initial_b).square().sum())),
                    "updates": updates, "wall_seconds": time.monotonic() - seed_start,
                    "peak_rss_bytes": rss_bytes(), "device": "cpu"
                },
                "adapter": {"A_initial": initial_a.tolist(), "B_initial": initial_b.tolist(),
                            "A_final": final_a.tolist(), "B_final": final_b.tolist()}
            })
            if rss_bytes() > spec["compute_limits"]["max_peak_rss_bytes"]:
                raise MemoryError("peak RSS exceeded locked limit")
            if time.monotonic() - started > spec["compute_limits"]["max_wall_seconds"]:
                raise TimeoutError("run exceeded locked wall limit")

        paired_changes = []
        for record in seed_records:
            seed = record["seed"]
            signs = test_signs
            for before, after, sign in zip(record["heldout_base_margins"], record["heldout_updated_margins"], signs):
                before_nll = max(-sign * before, 0.0) + math.log1p(math.exp(-abs(sign * before)))
                after_nll = max(-sign * after, 0.0) + math.log1p(math.exp(-abs(sign * after)))
                paired_changes.append((seed, after_nll - before_nll))
        mean_change = statistics.fmean(value for _, value in paired_changes)
        interval = paired_bootstrap(paired_changes, spec["metrics"]["paired_bootstrap"]["seed"],
                                    spec["metrics"]["paired_bootstrap"]["resamples"])
        all_improve = all(r["metrics"]["heldout_updated"]["mean_conditional_preference_nll"] < r["metrics"]["heldout_base"]["mean_conditional_preference_nll"] for r in seed_records)
        all_kl_ok = all(r["metrics"]["heldout_mean_bernoulli_kl_updated_to_base"] <= 0.5 for r in seed_records)
        passed = all_improve and interval[1] < 0 and all_kl_ok
        summary = {
            "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_hash,
            "runner_sha256": sha256_file(Path(__file__)),
            "task_helper_sha256": sha256_file(Path(__file__).with_name("gsm8k_preference_task.py")),
            "model_id": spec["model"]["id"], "model_revision": MODEL_DIR.name,
            "model_file_sha256": {name: sha256_file(MODEL_DIR / name) for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")},
            "dataset_train_sha256": sha256_file(DATA_DIR / "gsm8k-train.arrow"),
            "dataset_test_sha256": sha256_file(DATA_DIR / "gsm8k-test.arrow"),
            "dataset_train_fingerprint": train_ds._fingerprint,
            "dataset_test_fingerprint": test_ds._fingerprint,
            "response_token_ids": {"A": token_ids[0], "B": token_ids[1]},
            "runtime": runtime, "platform": platform.platform(), "device": "cpu",
            "paid_compute": False, "network_disabled": True,
            "train_examples": len(train_rows), "heldout_examples": len(test_rows),
            "per_seed": [{"seed": r["seed"], "metrics": r["metrics"]} for r in seed_records],
            "mean_heldout_nll_change_updated_minus_base": mean_change,
            "paired_seed_stratified_bootstrap_95pct": interval,
            "all_seeds_improve": all_improve, "all_seed_kl_within_limit": all_kl_ok,
            "decision": "positive_small_model_preference_result" if passed else "non_pass_or_null",
            "total_wall_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
            "compute_limits_respected": rss_bytes() <= spec["compute_limits"]["max_peak_rss_bytes"] and time.monotonic()-started <= spec["compute_limits"]["max_wall_seconds"]
        }
        write_json(output / "examples.json", {"train": train_rows, "heldout": test_rows})
        write_json(output / "seed_records.json", {str(r["seed"]): r for r in seed_records})
        write_json(output / "summary.json", summary)
        write_json(output / "environment.json", {"python_executable": sys.executable,
            "environment": {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM")},
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.strip()})
        return summary
    except BaseException as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__, "message": str(exc),
                                              "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes()})
        raise
    finally:
        write_manifest(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1")
    args = parser.parse_args()
    result = run(args.output.expanduser().resolve())
    print(json.dumps({"decision": result["decision"],
                      "mean_heldout_nll_change": result.get("mean_heldout_nll_change_updated_minus_base"),
                      "interval": result.get("paired_seed_stratified_bootstrap_95pct"),
                      "wall_seconds": result["total_wall_seconds"], "peak_rss_bytes": result["peak_rss_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
