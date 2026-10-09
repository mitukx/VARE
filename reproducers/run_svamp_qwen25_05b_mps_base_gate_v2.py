#!/usr/bin/env python3
"""Run the frozen fresh-cohort SVAMP/Qwen MPS base gate v2.

Downloads only the pinned train.json. It does not open the official test split.
V2 uses IDs outside the v1 development allocation and a numeric-only prompt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
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
V1_PREDICTIONS = Path("results/svamp-qwen25-05b-mps-base-gate-v1/run-1/predictions.jsonl")
OUTPUT_DIR = Path("results/svamp-qwen25-05b-mps-base-gate-v2/run-1")
PROMPT_TEMPLATE = (
    "Solve this math word problem. Give only the numeric answer, with no words, "
    "units, or explanation.\n\n{body}\nQuestion: {question}\nAnswer:"
)
NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
NUMBER_RE = re.compile(rf"^{NUMBER}$")
ANSWER_LINE_RE = re.compile(rf"^(?:final\s+)?answer\s*:\s*({NUMBER})\.?$", re.IGNORECASE)
TAG_RE = re.compile(rf"<answer>\s*({NUMBER})\s*</answer>", re.IGNORECASE)
BOX_RE = re.compile(rf"\\boxed\s*\{{\s*({NUMBER})\s*\}}")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ranked(rows: list[dict], label: str) -> list[dict]:
    return sorted(rows, key=lambda row: sha(f"VARE-SVAMP-v2/{label}:{row['ID']}"))


def evaluate_equation(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return evaluate_equation(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = evaluate_equation(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate_equation(node.left), evaluate_equation(node.right))
    raise ValueError(f"disallowed equation AST: {ast.dump(node)}")


def valid_train_rows(
    rows: list[dict],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    seen, valid, excluded = set(), [], []
    for row in rows:
        if row["ID"] in seen:
            raise ValueError(f"duplicate ID: {row['ID']}")
        seen.add(row["ID"])
        equation_value = evaluate_equation(ast.parse(row["Equation"], mode="eval"))
        answer_value = Decimal(str(row["Answer"]))
        if equation_value != answer_value:
            excluded.append({"id": row["ID"], "equation": str(equation_value), "answer": str(answer_value)})
        else:
            valid.append(row)
    if len(valid) != 699 or len(excluded) != 1 or excluded[0]["id"] != "chal-680":
        raise ValueError(f"unexpected pinned training data: {len(valid)} valid, exclusions={excluded}")
    # V1 used its own namespace and hash, so reproduce that exact cohort
    # definition before allocating any v2 rows.
    v1_dev = sorted(
        valid,
        key=lambda row: sha(f"VARE-SVAMP-v1/dev:{row['ID']}"),
    )[:100]
    v1_ids = {row["ID"] for row in v1_dev}
    remaining = [row for row in valid if row["ID"] not in v1_ids]
    dev = ranked(remaining, "dev")[:100]
    training_only = [row for row in remaining if row["ID"] not in {x["ID"] for x in dev}]
    base64 = ranked(dev, "base")[:64]
    variance16 = ranked(base64, "variance")[:16]
    return training_only, dev, base64, variance16, excluded


def parse_answer(text: str) -> Decimal | None:
    stripped = text.strip()
    candidates: list[str] = []
    if NUMBER_RE.fullmatch(stripped):
        candidates.append(stripped)
    elif re.fullmatch(rf"{NUMBER}\.", stripped):
        candidates.append(stripped[:-1])
    else:
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines:
            m = ANSWER_LINE_RE.fullmatch(lines[-1])
            if m:
                candidates.append(m.group(1))
        tags = TAG_RE.findall(stripped)
        boxes = BOX_RE.findall(stripped)
        if len(tags) == 1:
            candidates.append(tags[0])
        if len(boxes) == 1:
            candidates.append(boxes[0])
    if len(candidates) != 1:
        return None
    try:
        answer = Decimal(candidates[0])
    except InvalidOperation:
        return None
    return answer if answer.is_finite() else None


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [0.0, 0.0]
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
    predicted = parse_answer(raw)
    target = Decimal(str(row["Answer"]))
    return {
        "id": row["ID"],
        "prompt_sha256": sha(prompt),
        "sampled": sampled,
        "seed": seed,
        "raw_output": raw,
        "parsed_answer": str(predicted) if predicted is not None else None,
        "target_answer": str(target),
        "parseable": predicted is not None,
        "reward": int(predicted == target),
    }


def main() -> None:
    if torch.cuda.is_available() or not torch.backends.mps.is_available():
        raise RuntimeError("requires MPS available and CUDA unavailable")
    if not V1_PREDICTIONS.exists():
        raise FileNotFoundError(f"required prior-run manifest missing: {V1_PREDICTIONS}")
    v1_rows = [json.loads(line) for line in V1_PREDICTIONS.read_text(encoding="utf-8").splitlines()]
    v1_ids = {row["id"] for row in v1_rows}
    source_url = f"https://huggingface.co/datasets/ChilleD/SVAMP/resolve/{DATA_REVISION}/train.json?download=true"
    with urlopen(source_url, timeout=60) as response:
        train_bytes = response.read()
    if sha(train_bytes.decode()) != DATA_SHA256:
        raise ValueError("SVAMP training split hash mismatch")
    rows = json.loads(train_bytes)
    training_only, dev, base_rows, variance_rows, excluded = valid_train_rows(rows)
    if v1_ids & {row["ID"] for row in dev}:
        raise ValueError("v2 dev overlaps v1 scored IDs")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True, torch_dtype=torch.float16).to("mps")
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - t0
    model.eval()
    greedy = [generate(model, tokenizer, row, sampled=False, seed=None) for row in base_rows]
    groups = []
    for group_rank, row in enumerate(variance_rows):
        samples = [
            generate(model, tokenizer, row, sampled=True, seed=20261010 + 100 * group_rank + sample_rank)
            for sample_rank in range(4)
        ]
        groups.append({"id": row["ID"], "members": samples})
    elapsed = time.perf_counter() - t0

    parseable = sum(row["parseable"] for row in greedy)
    exact = sum(row["reward"] for row in greedy)
    mixed = sum(len({row["reward"] for row in group["members"]}) > 1 for group in groups)
    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss_raw if sys.platform == "darwin" else rss_raw * 1024
    passed = exact >= 13 and parseable >= 61 and mixed >= 4 and elapsed <= 1800 and rss_bytes <= 12 * 1024**3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT_DIR / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as stream:
        for row in greedy:
            stream.write(json.dumps({"arm": "greedy", **row}, sort_keys=True) + "\n")
        for group in groups:
            for sample_index, row in enumerate(group["members"]):
                stream.write(json.dumps({"arm": "reward_variance", "sample_index": sample_index, **row}, sort_keys=True) + "\n")
    summary = {
        "protocol": "svamp-qwen25-05b-mps-base-gate-v2",
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATA_REVISION,
        "train_file_sha256": DATA_SHA256,
        "official_test_accessed": False,
        "v1_and_v2_dev_ids_disjoint": True,
        "training_only_rows_after_v2_gate": len(training_only),
        "v2_development_rows": len(dev),
        "base_greedy_n": len(greedy),
        "base_exact": exact,
        "base_exact_rate": exact / len(greedy),
        "base_exact_wilson_95": wilson(exact, len(greedy)),
        "base_parseable": parseable,
        "base_parse_rate": parseable / len(greedy),
        "base_parse_wilson_95": wilson(parseable, len(greedy)),
        "reward_variance_groups": len(groups),
        "mixed_reward_groups": mixed,
        "mixed_reward_group_rate": mixed / len(groups),
        "sampled_response_n": sum(len(group["members"]) for group in groups),
        "sampled_parseable": sum(row["parseable"] for group in groups for row in group["members"]),
        "excluded_label_rows": excluded,
        "model_load_seconds": load_seconds,
        "total_wall_seconds": elapsed,
        "peak_rss_bytes": rss_bytes,
        "peak_rss_gib": rss_bytes / 1024**3,
        "predictions_sha256": sha(pred_path.read_text(encoding="utf-8")),
        "gate_passed": passed,
        "gate": {"greedy_exact_min": 13, "greedy_parseable_min": 61, "mixed_groups_min": 4, "wall_seconds_max": 1800, "peak_rss_bytes_max": 12884901888},
        "limitations": ["base-only gate; no update or capability claim", "SVAMP is elementary arithmetic reasoning", "official test data was not opened"],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
