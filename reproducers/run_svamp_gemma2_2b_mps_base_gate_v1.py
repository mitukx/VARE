#!/usr/bin/env python3
"""Run the frozen Gemma 2 2B / fresh-SVAMP-cohort MPS base gate."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from decimal import Decimal
from pathlib import Path
from urllib.request import urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_svamp_qwen25_05b_mps_base_gate_v2 as frozen_helpers


DATA_REVISION = frozen_helpers.DATA_REVISION
DATA_SHA256 = frozen_helpers.DATA_SHA256
MODEL_REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--google--gemma-2-2b-it/snapshots" / MODEL_REVISION
V2_PREDS = Path("results/svamp-qwen25-05b-mps-base-gate-v2/run-1/predictions.jsonl")
OUTPUT_DIR = Path("results/svamp-gemma2-2b-mps-base-gate-v1/run-1")
PROMPT_TEMPLATE = (
    "Solve this math word problem. Give only the numeric answer, with no words, "
    "units, or explanation.\n\n{body}\nQuestion: {question}\nAnswer:"
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rank(rows: list[dict], label: str) -> list[dict]:
    return sorted(rows, key=lambda row: digest(f"VARE-SVAMP-Gemma2B-v1/{label}:{row['ID']}"))


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def generate(model, tokenizer, row: dict, *, sampled: bool, seed: int | None) -> dict:
    prompt = PROMPT_TEMPLATE.format(body=row["Body"].strip(), question=row["Question"].strip())
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("mps")
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "max_new_tokens": 32,
        "do_sample": sampled,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sampled:
        assert seed is not None
        torch.manual_seed(seed)
        torch.mps.manual_seed(seed)
        kwargs.update({"temperature": 0.8, "top_p": 0.95})
    with torch.inference_mode():
        output = model.generate(**kwargs)
    torch.mps.synchronize()
    raw = tokenizer.decode(output[0, input_ids.shape[-1] :], skip_special_tokens=True)
    parsed = frozen_helpers.parse_answer(raw)
    target = Decimal(str(row["Answer"]))
    return {
        "id": row["ID"],
        "prompt_sha256": digest(prompt),
        "sampled": sampled,
        "seed": seed,
        "raw_output": raw,
        "parsed_answer": str(parsed) if parsed is not None else None,
        "target_answer": str(target),
        "parseable": parsed is not None,
        "reward": int(parsed == target),
    }


def main() -> None:
    if torch.cuda.is_available() or not torch.backends.mps.is_available():
        raise RuntimeError("requires local MPS and no CUDA")
    if not MODEL_PATH.is_dir() or not V2_PREDS.is_file():
        raise FileNotFoundError("pinned Gemma snapshot or prior cohort manifest is missing")
    v2_scored = {json.loads(line)["id"] for line in V2_PREDS.read_text(encoding="utf-8").splitlines()}

    source = f"https://huggingface.co/datasets/ChilleD/SVAMP/resolve/{DATA_REVISION}/train.json?download=true"
    with urlopen(source, timeout=60) as response:
        train_bytes = response.read()
    if hashlib.sha256(train_bytes).hexdigest() != DATA_SHA256:
        raise ValueError("SVAMP train.json digest mismatch")
    rows = json.loads(train_bytes)
    post_v1_v2_train, _v2_dev, _v2_base, _v2_variance, excluded = frozen_helpers.valid_train_rows(rows)
    pool_ids = {row["ID"] for row in post_v1_v2_train}
    if pool_ids & v2_scored or len(post_v1_v2_train) != 499:
        raise ValueError("invalid fresh-pool manifest")
    dev = rank(post_v1_v2_train, "dev")[:100]
    training_only = [row for row in post_v1_v2_train if row["ID"] not in {item["ID"] for item in dev}]
    base = rank(dev, "base")[:64]
    variance = rank(base, "variance")[:16]
    if len(training_only) != 399:
        raise ValueError("unexpected Gemma development/training allocation")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, torch_dtype=torch.float16
    ).to("mps")
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - t0
    model.eval()
    greedy = [generate(model, tokenizer, row, sampled=False, seed=None) for row in base]
    groups = []
    for group_rank, row in enumerate(variance):
        members = [
            generate(model, tokenizer, row, sampled=True, seed=20261011 + 100 * group_rank + sample_rank)
            for sample_rank in range(4)
        ]
        groups.append({"id": row["ID"], "members": members})
    elapsed = time.perf_counter() - t0

    parseable = sum(item["parseable"] for item in greedy)
    exact = sum(item["reward"] for item in greedy)
    mixed = sum(len({item["reward"] for item in group["members"]}) > 1 for group in groups)
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss_raw if sys.platform == "darwin" else rss_raw * 1024
    passed = exact >= 13 and parseable >= 61 and mixed >= 4 and elapsed <= 1800 and rss_bytes <= 12 * 1024**3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT_DIR / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as stream:
        for item in greedy:
            stream.write(json.dumps({"arm": "greedy", **item}, sort_keys=True) + "\n")
        for group in groups:
            for sample_index, item in enumerate(group["members"]):
                stream.write(json.dumps({"arm": "reward_variance", "sample_index": sample_index, **item}, sort_keys=True) + "\n")
    summary = {
        "protocol": "svamp-gemma2-2b-mps-base-gate-v1",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATA_REVISION,
        "train_file_sha256": DATA_SHA256,
        "official_test_accessed": False,
        "previous_v1_v2_cohorts_excluded": True,
        "training_only_rows": len(training_only),
        "development_rows": len(dev),
        "base_greedy_n": len(greedy),
        "base_exact": exact,
        "base_exact_rate": exact / len(greedy),
        "base_exact_wilson_95": wilson(exact, len(greedy)),
        "base_parseable": parseable,
        "base_parse_rate": parseable / len(greedy),
        "base_parse_wilson_95": wilson(parseable, len(greedy)),
        "sampled_response_n": 64,
        "sampled_parseable": sum(item["parseable"] for group in groups for item in group["members"]),
        "sampled_exact": sum(item["reward"] for group in groups for item in group["members"]),
        "reward_variance_groups": len(groups),
        "mixed_reward_groups": mixed,
        "mixed_reward_group_rate": mixed / len(groups),
        "excluded_label_rows": excluded,
        "model_load_seconds": load_seconds,
        "total_wall_seconds": elapsed,
        "peak_rss_bytes": rss_bytes,
        "peak_rss_gib": rss_bytes / 1024**3,
        "predictions_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "gate_passed": passed,
        "gate": {"greedy_exact_min": 13, "greedy_parseable_min": 61, "mixed_groups_min": 4, "wall_seconds_max": 1800, "peak_rss_bytes_max": 12884901888},
        "limitations": ["base-only feasibility gate", "SVAMP is elementary arithmetic", "official test data was not opened"],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
