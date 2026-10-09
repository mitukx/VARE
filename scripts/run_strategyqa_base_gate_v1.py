#!/usr/bin/env python3
"""Frozen CPU base-feasibility screen for StrategyQA."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/cpu_strategyqa_grpo_shift_v1.lock.json"
DATA = Path(os.environ.get("VARE_STRATEGYQA_TRAIN_JSON", ""))
MODEL = Path(os.environ.get("VARE_QWEN_05B_INSTRUCT", ""))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def selected_ids(rows: list[dict], salt: str, n: int) -> list[str]:
    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{salt}|{r['qid']}".encode()).hexdigest())
    return [str(r["qid"]) for r in ordered[:n]]


def extract_final(text: str) -> str | None:
    marker = "final answer:"
    lower = text.lower()
    at = lower.rfind(marker)
    if at < 0:
        return None
    tail = lower[at + len(marker):].strip()
    token = tail.split()[0].strip(" .!?,;:*`'\"()[]") if tail else ""
    return token if token in {"yes", "no"} else None


def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    lock = json.loads(LOCK.read_text())
    if not DATA.is_file() or not MODEL.is_dir():
        raise SystemExit("Set VARE_STRATEGYQA_TRAIN_JSON and VARE_QWEN_05B_INSTRUCT to existing local artifacts")
    if sha256(DATA) != lock["dataset"]["train_json_sha256"]:
        raise SystemExit("dataset hash mismatch")
    if MODEL.name != lock["model"]["revision"]:
        raise SystemExit("model snapshot revision mismatch")
    weight = MODEL / "model.safetensors"
    if sha256(weight) != lock["model"]["weights_sha256"]:
        raise SystemExit("model weight hash mismatch")
    if sha256(Path(__file__)) != lock["frozen_sources"]["runner_sha256"]:
        raise SystemExit("runner source hash mismatch")
    raw = json.loads(DATA.read_text())
    rows = raw if isinstance(raw, list) else list(raw.values())
    gate_ids = selected_ids(rows, lock["selection"]["salt"], lock["selection"]["base_gate_n"])
    confirmation_ids = selected_ids(
        [r for r in rows if str(r["qid"]) not in set(gate_ids)],
        lock["selection"]["salt"] + "|confirmation",
        lock["selection"]["confirmation_n"],
    )
    train_ids = selected_ids(
        [r for r in rows if str(r["qid"]) not in set(gate_ids + confirmation_ids)],
        lock["selection"]["salt"] + "|train",
        lock["selection"]["train_n"],
    )
    if len(set(gate_ids + confirmation_ids + train_ids)) != len(gate_ids + confirmation_ids + train_ids):
        raise SystemExit("selection overlap")
    if gate_ids != lock["selection"]["base_gate_ids"] or confirmation_ids != lock["selection"]["confirmation_ids"] or train_ids != lock["selection"]["train_ids"]:
        raise SystemExit("row selection differs from frozen protocol")

    row_by_id = {str(r["qid"]): r for r in rows}
    torch.set_num_threads(lock["runtime"]["threads"])
    torch.manual_seed(lock["runtime"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True, torch_dtype=torch.float32).to("cpu")
    model.eval()
    records = []
    start = time.time()
    for qid in gate_ids:
        row = row_by_id[qid]
        user = f"Question: {row['question']}\nThink briefly, then finish with exactly one line: Final answer: yes OR Final answer: no."
        messages = [{"role": "user", "content": user}]
        inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        with torch.inference_mode():
            ids = model.generate(input_ids=inputs, max_new_tokens=lock["generation"]["max_new_tokens"], do_sample=False, pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(ids[0, inputs.shape[1]:], skip_special_tokens=True)
        pred = extract_final(completion)
        gold = "yes" if bool(row["answer"]) else "no"
        records.append({
            "qid": qid,
            "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
            "completion": completion,
            "parsed": pred,
            "gold": gold,
            "correct": pred == gold,
        })
    n = len(records)
    k = sum(r["correct"] for r in records)
    acc = k / n
    parsed_n = sum(r["parsed"] is not None for r in records)
    parse_rate = parsed_n / n
    elapsed = time.time() - start
    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if platform.system() == "Darwin" else 1024)
    passed = (
        acc >= lock["gate"]["minimum_accuracy"]
        and wilson_lower(k, n) > lock["gate"]["null_accuracy"]
        and parse_rate >= lock["gate"]["minimum_parse_rate"]
        and elapsed <= lock["resources"]["wall_seconds_cap"]
        and peak_rss_mib <= lock["resources"]["peak_rss_mib_cap"]
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "protocol_id": lock["protocol_id"],
        "status": "pass" if passed else "non_pass",
        "n": n,
        "correct": k,
        "accuracy": acc,
        "wilson_95_lower": wilson_lower(k, n),
        "parsed": parsed_n,
        "parse_rate": parse_rate,
        "runtime_seconds": elapsed,
        "peak_rss_mib": peak_rss_mib,
        "device": "cpu",
        "paid_spend_usd": 0,
        "confirmation_ids_sha256": hashlib.sha256("\n".join(confirmation_ids).encode()).hexdigest(),
        "train_ids_sha256": hashlib.sha256("\n".join(train_ids).encode()).hexdigest(),
        "selection_ids": {"base_gate": gate_ids, "train": train_ids, "confirmation": confirmation_ids},
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"records", "selection_ids"}}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
