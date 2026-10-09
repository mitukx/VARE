#!/usr/bin/env python3
"""Evaluate archived GSM8K DPO adapters on a frozen ASDiv transfer set."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import time
import re
import unicodedata
from typing import Any
from decimal import Decimal

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/asdiv_dpo_semantic_transfer_v1.json"
LOCK = ROOT / "protocols/asdiv_dpo_semantic_transfer_v1.lock.json"
DATA = ROOT.parent / "work/private/asdiv-dpo-transfer-v1/asdiv.parquet"
ADAPTERS = ROOT / "results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1/seed_records.json"
MODEL = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if os.uname().sysname == "Darwin" else value * 1024)


def load_locked() -> tuple[dict[str, Any], str]:
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    locked_hash = lock.pop("sha256", None)
    actual_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if locked_hash != actual_hash or canonical(lock) != canonical(spec):
        raise ValueError("protocol differs from its lock")
    runner_hash = spec["implementation"]["runner_sha256"]
    if sha256_file(Path(__file__)) != runner_hash:
        raise ValueError("runner source differs from frozen implementation hash")
    return spec, actual_hash


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    spec, protocol_hash = load_locked()
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)

    if sha256_file(DATA) != spec["source"]["sha256"]:
        raise ValueError("ASDiv source parquet hash mismatch")
    if sha256_file(ADAPTERS) != spec["models"]["adapter_bundle_sha256"]:
        raise ValueError("archived adapter bundle hash mismatch")
    for name, expected in spec["models"]["model_file_sha256"].items():
        if sha256_file(MODEL / name) != expected:
            raise ValueError(f"model file hash mismatch: {name}")

    for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[variable] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.__version__.split("+")[0] != spec["runtime"]["torch"] or transformers.__version__ != spec["runtime"]["transformers"] or np.__version__ != spec["runtime"]["numpy"]:
        raise RuntimeError("runtime differs from frozen protocol")
    torch.set_num_threads(spec["compute"]["threads"])
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must remain uninitialized")

    source = pq.read_table(DATA).to_pylist()
    selected = spec["selection"]["rows"]
    questions = []
    for item in selected:
        row = source[item["index"]]
        text = (row["body"].strip() + " " + row["question"].strip()).strip()
        normalized = " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower()))
        question_hash = hashlib.sha256(normalized.encode()).hexdigest()
        if question_hash != item["problem_sha256"]:
            raise ValueError(f"source row differs from frozen selection: {item['index']}")
        answer_string = row["answer"].strip()
        # Eligibility and answer/formula consistency were checked before freezing.
        match = re.match(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))", answer_string)
        if not match:
            raise ValueError(f"selected answer is no longer numeric: {item['index']}")
        answer = Decimal(match.group(1))
        direction = 1.0 if int(question_hash[:2], 16) % 2 == 0 else -1.0
        wrong = answer + Decimal(str(direction))
        def format_number(value: Decimal) -> str:
            value = value.normalize()
            return format(value.quantize(Decimal(1)), "f") if value == value.to_integral_value() else format(value, "f")
        prompts = []
        for correct_is_a in (True, False):
            a, b = (answer, wrong) if correct_is_a else (wrong, answer)
            prompts.append(
                "Solve the following word problem. Choose the correct final numeric answer.\n\n"
                f"Problem: {text}\n\nA) {format_number(a)}\nB) {format_number(b)}\n\nAnswer with A or B:"
            )
        questions.append({"index": item["index"], "problem_sha256": question_hash,
                          "answer": format_number(answer), "wrong": format_number(wrong), "prompts": prompts})

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    token_ids = {label: tokenizer.encode(" " + label, add_special_tokens=False) for label in ("A", "B")}
    if token_ids != spec["models"]["response_token_ids"]:
        raise ValueError("A/B response-token encoding changed")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), local_files_only=True, torch_dtype=torch.float32)
    model.to("cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    records = json.loads(ADAPTERS.read_text(encoding="utf-8"))
    adapter_tensors = {str(seed): (
        torch.tensor(records[str(seed)]["adapter"]["A_final"], dtype=torch.float32),
        torch.tensor(records[str(seed)]["adapter"]["B_final"], dtype=torch.float32),
    ) for seed in spec["models"]["adapter_seeds"]}

    prompt_rows = [(qi, orientation, prompt)
                   for qi, question in enumerate(questions)
                   for orientation, prompt in enumerate(question["prompts"])]
    base_margins: list[float] = []
    adapter_margins: dict[str, list[float]] = {str(seed): [] for seed in spec["models"]["adapter_seeds"]}
    batch_size = spec["compute"]["batch_size"]
    for start in range(0, len(prompt_rows), batch_size):
        batch_rows = prompt_rows[start:start + batch_size]
        batch = tokenizer([row[2] for row in batch_rows], return_tensors="pt", padding=True,
                          add_special_tokens=True, truncation=False)
        lengths = batch["attention_mask"].sum(dim=1) - 1
        with torch.inference_mode():
            hidden_all = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                     use_cache=False).last_hidden_state
            hidden = hidden_all[torch.arange(hidden_all.shape[0]), lengths].to(dtype=torch.float32).contiguous()
            base_logits = torch.nn.functional.linear(hidden, model.lm_head.weight[[token_ids['A'][0], token_ids['B'][0]]])
            margins = (base_logits[:, 0] - base_logits[:, 1]).cpu().tolist()
            base_margins.extend(float(value) for value in margins)
            for seed in spec["models"]["adapter_seeds"]:
                a, b = adapter_tensors[str(seed)]
                delta = (hidden @ a) @ b
                updated = (base_logits[:, 0] - base_logits[:, 1] + delta[:, 0] - delta[:, 1]).cpu().tolist()
                adapter_margins[str(seed)].extend(float(value) for value in updated)

    score_rows = []
    for qi, question in enumerate(questions):
        index = qi * 2
        score_rows.append({
            "index": question["index"], "problem_sha256": question["problem_sha256"],
            "answer": question["answer"], "wrong": question["wrong"],
            "base_margin_correct_a": base_margins[index], "base_margin_correct_b": base_margins[index + 1],
            "adapter_margins": {seed: {"correct_a": values[index], "correct_b": values[index + 1]}
                                for seed, values in adapter_margins.items()},
        })
    (output / "scores.json").write_text(json.dumps(score_rows, separators=(",", ":")) + "\n", encoding="utf-8")
    summary = {
        "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_hash,
        "runner_sha256": sha256_file(Path(__file__)), "adapter_bundle_sha256": sha256_file(ADAPTERS),
        "asdiv_parquet_sha256": sha256_file(DATA), "model_file_sha256": spec["models"]["model_file_sha256"],
        "device": "cpu", "paid_compute": False, "network_disabled": True,
        "runtime_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
        "n_items": len(score_rows), "n_prompts": len(prompt_rows),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    (output / "manifest.json").write_text(json.dumps({"algorithm": "sha256", "files": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary["runtime_seconds"] > spec["compute"]["max_wall_seconds"] or summary["peak_rss_bytes"] > spec["compute"]["max_peak_rss_bytes"]:
        raise RuntimeError("frozen resource limit exceeded")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
