#!/usr/bin/env python3
"""Run the frozen SVAMP base-only feasibility gate on cached Qwen MPS.

This runner downloads only the pinned SVAMP train.json. It never requests the
official test split. It writes development prompts, predictions and metrics
under results/svamp-qwen25-05b-mps-base-gate-v1/run-1/.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import os
import re
import resource
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.request import urlopen

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DATA_REVISION = "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef"
DATA_SHA256 = "bb3b4cd2957f07643bbf9f8f8f5faba3bfb47fcb1d9034655b03060a76675e2b"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / MODEL_REVISION
OUTPUT_DIR = Path("results/svamp-qwen25-05b-mps-base-gate-v1/run-1")
MAX_NEW_TOKENS = 160
PROMPT_TEMPLATE = (
    "Solve this arithmetic word problem. Show concise arithmetic reasoning. "
    "End with exactly one line <answer>NUMBER</answer>, with an integer or "
    "decimal and no thousands separators.\n\n{body}\n{question}"
)
ANSWER_RE = re.compile(
    r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)
TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_rank(label: str, row: dict) -> str:
    return digest(f"VARE-SVAMP-v1/{label}:{row['ID']}".encode())


def eval_decimal(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return eval_decimal(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = eval_decimal(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](eval_decimal(node.left), eval_decimal(node.right))
    raise ValueError(f"disallowed equation syntax: {ast.dump(node)}")


def audit_and_partition(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    seen_ids: set[str] = set()
    valid: list[dict] = []
    excluded: list[dict] = []
    for row in rows:
        if row["ID"] in seen_ids:
            raise ValueError(f"duplicate train ID: {row['ID']}")
        seen_ids.add(row["ID"])
        computed = eval_decimal(ast.parse(row["Equation"], mode="eval"))
        stored = Decimal(str(row["Answer"]))
        if computed != stored:
            excluded.append({"id": row["ID"], "equation_value": str(computed), "stored_answer": str(stored)})
        else:
            valid.append(row)
    if len(rows) != 700 or len(excluded) != 1 or excluded[0]["id"] != "chal-680":
        raise ValueError(f"unexpected pinned-source audit: rows={len(rows)} excluded={excluded}")
    by_dev_hash = sorted(valid, key=lambda row: hash_rank("dev", row))
    dev, training_only = by_dev_hash[:100], by_dev_hash[100:]
    by_base_hash = sorted(dev, key=lambda row: hash_rank("base", row))
    base = by_base_hash[:64]
    by_variance_hash = sorted(base, key=lambda row: hash_rank("variance", row))
    variance = by_variance_hash[:16]
    return training_only, base, variance, excluded


def parse_answer(text: str) -> Decimal | None:
    tags = TAG_RE.findall(text)
    if len(tags) != 1:
        return None
    value = tags[0].strip()
    if not ANSWER_RE.fullmatch(value):
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed.is_finite() else None


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def generate(model, tokenizer, row: dict, *, sampled: bool, seed: int | None) -> dict:
    user_text = PROMPT_TEMPLATE.format(body=row["Body"].strip(), question=row["Question"].strip())
    messages = [{"role": "user", "content": user_text}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("mps")
    attention_mask = torch.ones_like(input_ids)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": sampled,
        "pad_token_id": tokenizer.eos_token_id,
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
    prediction = parse_answer(raw)
    target = Decimal(str(row["Answer"]))
    return {
        "id": row["ID"],
        "prompt_sha256": digest(user_text.encode()),
        "sampled": sampled,
        "seed": seed,
        "raw_output": raw,
        "parsed_answer": str(prediction) if prediction is not None else None,
        "target_answer": str(target),
        "parseable": prediction is not None,
        "reward": int(prediction == target),
    }


def main() -> None:
    if torch.cuda.is_available() or not torch.backends.mps.is_available():
        raise RuntimeError("protocol requires MPS available and CUDA unavailable")
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"pinned model snapshot is not cached: {MODEL_PATH}")

    source_url = (
        f"https://huggingface.co/datasets/ChilleD/SVAMP/resolve/"
        f"{DATA_REVISION}/train.json?download=true"
    )
    with urlopen(source_url, timeout=60) as response:
        train_bytes = response.read()
    if digest(train_bytes) != DATA_SHA256:
        raise ValueError("SVAMP train.json digest mismatch")
    rows = json.loads(train_bytes)
    training_only, base_rows, variance_rows, excluded = audit_and_partition(rows)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to("mps")
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - t0
    model.eval()

    predictions = []
    for row in base_rows:
        predictions.append(generate(model, tokenizer, row, sampled=False, seed=None))
    group_results = []
    for group_rank, row in enumerate(variance_rows):
        members = [
            generate(
                model,
                tokenizer,
                row,
                sampled=True,
                seed=20261009 + 100 * group_rank + sample_rank,
            )
            for sample_rank in range(4)
        ]
        group_results.append({"id": row["ID"], "members": members})

    elapsed = time.perf_counter() - t0
    parsed = sum(item["parseable"] for item in predictions)
    exact = sum(item["reward"] for item in predictions)
    mixed_groups = sum(len({item["reward"] for item in group["members"]}) > 1 for group in group_results)
    sampled_parseable = sum(item["parseable"] for group in group_results for item in group["members"])
    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = peak_rss_raw if sys.platform == "darwin" else peak_rss_raw * 1024
    passed = (
        exact >= 13
        and parsed >= 61
        and mixed_groups >= 4
        and elapsed <= 5400
        and peak_rss_bytes <= 12 * 1024**3
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_DIR / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for item in predictions:
            stream.write(json.dumps({"arm": "greedy", **item}, sort_keys=True) + "\n")
        for group in group_results:
            for sample_index, item in enumerate(group["members"]):
                stream.write(json.dumps({"arm": "reward_variance", "sample_index": sample_index, **item}, sort_keys=True) + "\n")

    summary = {
        "protocol": "svamp-qwen25-05b-mps-base-gate-v1",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATA_REVISION,
        "train_file_sha256": DATA_SHA256,
        "test_split_accessed": False,
        "training_only_rows": len(training_only),
        "development_rows": 100,
        "base_greedy_n": len(predictions),
        "base_exact": exact,
        "base_exact_rate": exact / len(predictions),
        "base_exact_wilson_95": wilson(exact, len(predictions)),
        "base_parseable": parsed,
        "base_parse_rate": parsed / len(predictions),
        "base_parse_wilson_95": wilson(parsed, len(predictions)),
        "reward_variance_groups": len(group_results),
        "mixed_reward_groups": mixed_groups,
        "mixed_reward_group_rate": mixed_groups / len(group_results),
        "sampled_parseable": sampled_parseable,
        "sampled_response_n": len(group_results) * 4,
        "train_label_exclusions": excluded,
        "model_load_seconds": load_seconds,
        "total_wall_seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes,
        "peak_rss_gib": peak_rss_bytes / 1024**3,
        "predictions_sha256": digest(prediction_path.read_bytes()),
        "gate_passed": passed,
        "gate": {
            "greedy_exact_min": 13,
            "greedy_parseable_min": 61,
            "mixed_groups_min": 4,
            "wall_seconds_max": 5400,
            "peak_rss_bytes_max": 12884901888,
        },
        "limitations": [
            "base-only feasibility screen; no policy update or capability claim",
            "one train-split label inconsistency excluded under frozen rule",
            "official test data intentionally not accessed",
        ],
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
