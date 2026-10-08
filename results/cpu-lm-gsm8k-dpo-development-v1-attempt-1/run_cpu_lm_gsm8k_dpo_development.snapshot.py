#!/usr/bin/env python3
"""Training-split-only development for a bounded GSM8K preference update."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from typing import Any

from gsm8k_preference_task import make_split

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_lm_gsm8k_dpo_development_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_lm_gsm8k_dpo_development_v1.lock.json"
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


def metrics(margins: list[float], signs: list[int]) -> dict[str, float]:
    nll = [max(-s * m, 0.0) + math.log1p(math.exp(-abs(s * m))) for m, s in zip(margins, signs)]
    accuracy = [1.0 if s * m > 0 else 0.5 if m == 0 else 0.0 for m, s in zip(margins, signs)]
    return {"mean_preference_nll": statistics.fmean(nll), "preference_accuracy": statistics.fmean(accuracy)}


def mean_bernoulli_kl(base: list[float], updated: list[float]) -> float:
    total = 0.0
    for before, after in zip(base, updated):
        p = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, before))))
        q = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, after))))
        total += q * math.log(max(q, 1e-300) / max(p, 1e-300))
        total += (1.0 - q) * math.log(max(1.0 - q, 1e-300) / max(1.0 - p, 1e-300))
    return total / len(base)


def main() -> int:
    started = time.monotonic()
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    lock_hash = lock.pop("sha256", None)
    spec_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_hash != spec_hash or canonical(lock) != canonical(spec):
        raise ValueError("development protocol differs from its lock")
    output = ROOT / "results/cpu-lm-gsm8k-dpo-development-v1"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    for name in (Path(__file__), Path(__file__).with_name("gsm8k_preference_task.py")):
        (output / (name.stem + ".snapshot.py")).write_bytes(name.read_bytes())
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"

    try:
        import numpy
        import torch
        import transformers
        import datasets
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        versions = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
                    "transformers": transformers.__version__, "numpy": numpy.__version__,
                    "datasets": datasets.__version__}
        if versions != spec["runtime"]:
            raise RuntimeError(f"runtime differs from protocol: {versions}")
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must not be initialized")
        torch.set_num_threads(spec["compute_limits"]["threads"])
        if not MODEL_DIR.is_dir() or MODEL_DIR.name != spec["model"]["revision"]:
            raise FileNotFoundError("the protocol-pinned model snapshot is not cached")
        if sha256_file(DATA_DIR / "gsm8k-train.arrow") != spec["dataset"]["cached_train_arrow_sha256"]:
            raise ValueError("cached GSM8K training data hash differs from protocol")
        ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="train")
        if ds._fingerprint != spec["dataset"]["cached_fingerprint"]:
            raise ValueError(f"cached dataset fingerprint differs: {ds._fingerprint}")
        train_rows = make_split(ds, "train", spec["dataset"]["train_examples"])
        val_rows_all = make_split(ds, "validation", spec["dataset"]["train_examples"] + spec["dataset"]["validation_examples"])
        selected_hashes = {row["question_sha256"] for row in train_rows}
        val_rows = [row for row in val_rows_all if row["question_sha256"] not in selected_hashes]
        val_rows = val_rows[:spec["dataset"]["validation_examples"]]
        if len(val_rows) != spec["dataset"]["validation_examples"]:
            raise ValueError("could not form disjoint training-only development split")
        if selected_hashes & {row["question_sha256"] for row in val_rows}:
            raise ValueError("development train/validation overlap")

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32)
        model.to("cpu").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            if parameter.device.type != "cpu":
                raise RuntimeError("non-CPU model parameter detected")
        token_a, token_b = tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)
        if len(token_a) != 1 or len(token_b) != 1 or token_a[0] == token_b[0]:
            raise RuntimeError("A/B completions are not distinct single tokens")
        token_ids = [token_a[0], token_b[0]]

        def encode(rows):
            batch = tokenizer([r["prompt"] for r in rows], return_tensors="pt", padding=True,
                              add_special_tokens=True, truncation=False)
            lengths = batch["attention_mask"].sum(1) - 1
            with torch.no_grad():
                hidden_all = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                         use_cache=False).last_hidden_state
                indices = torch.arange(hidden_all.shape[0])
                hidden = hidden_all[indices, lengths].to(torch.float32).contiguous()
                base = torch.nn.functional.linear(hidden, model.lm_head.weight[token_ids].to(torch.float32))
            return hidden, base

        train_h, train_base = encode(train_rows)
        val_h, val_base = encode(val_rows)
        train_sign = torch.tensor([1.0 if row["correct_label"] == "A" else -1.0 for row in train_rows])
        val_signs = [1 if row["correct_label"] == "A" else -1 for row in val_rows]
        checkpoints = spec["learner"]["checkpoints"]
        seed_records = []
        for seed in spec["learner"]["seeds"]:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            hidden_size = train_h.shape[1]
            rank = int(spec["learner"]["adapter_rank"])
            adapter_a = torch.nn.Parameter(torch.randn((hidden_size, rank), generator=generator) / math.sqrt(hidden_size))
            adapter_b = torch.nn.Parameter(torch.zeros((rank, 2)))
            initial = {"A": adapter_a.detach().tolist(), "B": adapter_b.detach().tolist()}
            optimizer = torch.optim.SGD([adapter_a, adapter_b], lr=spec["learner"]["optimizer"]["learning_rate"],
                                        weight_decay=spec["learner"]["optimizer"]["weight_decay"])
            beta = spec["learner"]["beta"]
            row = {"seed": seed, "train_examples": train_rows, "validation_examples": val_rows,
                   "train_base_margins": (train_base[:, 0] - train_base[:, 1]).tolist(),
                   "validation_base_margins": (val_base[:, 0] - val_base[:, 1]).tolist(),
                   "checkpoints": {}}
            for target_update in checkpoints:
                current_step = 0
                if target_update > 0:
                    previous = max(v for v in checkpoints if v < target_update)
                    current_step = previous
                    while current_step < target_update:
                        optimizer.zero_grad(set_to_none=True)
                        delta = (train_h @ adapter_a) @ adapter_b
                        margin = delta[:, 0] - delta[:, 1]
                        loss = torch.nn.functional.softplus(-beta * train_sign * margin).mean()
                        loss.backward()
                        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in (adapter_a, adapter_b)):
                            raise FloatingPointError("non-finite DPO loss or gradient")
                        optimizer.step()
                        current_step += 1
                with torch.inference_mode():
                    val_delta = (val_h @ adapter_a) @ adapter_b
                    base_margins = (val_base[:, 0] - val_base[:, 1]).tolist()
                    updated_margins = (val_base[:, 0] - val_base[:, 1] + val_delta[:, 0] - val_delta[:, 1]).tolist()
                    train_delta = (train_h @ adapter_a) @ adapter_b
                    train_updated = (train_base[:, 0] - train_base[:, 1] + train_delta[:, 0] - train_delta[:, 1]).tolist()
                    adapter_snap = {"A": adapter_a.detach().tolist(), "B": adapter_b.detach().tolist()}
                row["checkpoints"][str(target_update)] = {
                    "validation_margins": updated_margins,
                    "validation_metrics": metrics(updated_margins, val_signs),
                    "validation_mean_bernoulli_kl_to_base": mean_bernoulli_kl(base_margins, updated_margins),
                    "train_margins": train_updated,
                    "train_metrics": metrics(train_updated, [1 if r["correct_label"] == "A" else -1 for r in train_rows]),
                    "adapter": adapter_snap,
                }
            row["adapter_initial"] = initial
            seed_records.append(row)
            if time.monotonic() - started > spec["compute_limits"]["max_wall_seconds"]:
                raise TimeoutError("development run exceeded frozen wall limit")
            if rss_bytes() > spec["compute_limits"]["max_peak_rss_bytes"]:
                raise MemoryError("development run exceeded frozen memory limit")

        summary_by_step = {}
        for step in checkpoints:
            rows = [record["checkpoints"][str(step)] for record in seed_records]
            base_vals = [metrics(r["validation_base_margins"], [1 if x["correct_label"] == "A" else -1 for x in r["validation_examples"]])["mean_preference_nll"] for r in seed_records]
            mean_nll = statistics.fmean(r["validation_metrics"]["mean_preference_nll"] for r in rows)
            mean_kl = statistics.fmean(r["validation_mean_bernoulli_kl_to_base"] for r in rows)
            summary_by_step[str(step)] = {"mean_validation_nll": mean_nll, "mean_base_validation_nll": statistics.fmean(base_vals),
                                          "mean_validation_kl_to_base": mean_kl,
                                          "per_seed_validation_nll": [r["validation_metrics"]["mean_preference_nll"] for r in rows]}
        eligible = [s for s in checkpoints if s > 0 and summary_by_step[str(s)]["mean_validation_nll"] < summary_by_step[str(s)]["mean_base_validation_nll"] and summary_by_step[str(s)]["mean_validation_kl_to_base"] <= 0.5]
        selected = min(eligible, key=lambda s: (summary_by_step[str(s)]["mean_validation_nll"], s)) if eligible else 0
        result = {
            "protocol_id": spec["protocol_id"], "protocol_sha256": spec_hash,
            "runner_sha256": sha256_file(Path(__file__)), "task_helper_sha256": sha256_file(Path(__file__).with_name("gsm8k_preference_task.py")),
            "model_id": spec["model"]["id"], "model_revision": MODEL_DIR.name,
            "model_file_sha256": {name: sha256_file(MODEL_DIR / name) for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")},
            "dataset_train_sha256": sha256_file(DATA_DIR / "gsm8k-train.arrow"), "dataset_fingerprint": ds._fingerprint,
            "response_token_ids": {"A": token_ids[0], "B": token_ids[1]}, "runtime": versions,
            "platform": platform.platform(), "device": "cpu", "paid_compute": False, "network_disabled": True,
            "per_seed": seed_records, "checkpoints": summary_by_step, "selected_update_budget": selected,
            "decision": "development_candidate_selected" if selected else "development_non_pass",
            "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
        }
        write_json(output / "development.json", result)
        write_manifest(output)
        print(json.dumps({"decision": result["decision"], "selected_update_budget": selected,
                          "checkpoints": summary_by_step, "elapsed_seconds": result["elapsed_seconds"],
                          "peak_rss_bytes": result["peak_rss_bytes"]}, sort_keys=True))
        return 0
    except BaseException as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__, "message": str(exc),
                                              "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes()})
        write_manifest(output)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
