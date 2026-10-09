#!/usr/bin/env python3
"""Run a frozen CPU base-feasibility gate on ARC-Challenge validation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/qwen_arc_challenge_base_gate_v1.lock.json"
DATASET_REPO = "allenai/ai2_arc"
DATASET_FILE = "ARC-Challenge/validation-00000-of-00001.parquet"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    without_digest = {key: item for key, item in value.items() if key != "sha256"}
    payload = json.dumps(without_digest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def check_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if canonical_sha256(protocol) != protocol.get("sha256"):
        raise RuntimeError("protocol digest mismatch")
    if sha256_file(Path(__file__).resolve()) != protocol["source"]["runner_sha256"]:
        raise RuntimeError("runner digest mismatch")
    return protocol


def package_versions(names: list[str]) -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in names}


def check_runtime(protocol: dict) -> dict:
    import torch
    import transformers

    expected = protocol["runtime"]["packages"]
    actual = package_versions(list(expected))
    if actual != expected:
        raise RuntimeError(f"package version mismatch: actual={actual}, expected={expected}")
    if sys.version.split()[0] != protocol["runtime"]["python"]:
        raise RuntimeError(f"Python mismatch: {sys.version.split()[0]}")
    if transformers.__version__ != expected["transformers"] or torch.__version__ != expected["torch"]:
        raise RuntimeError("imported framework versions disagree with frozen versions")
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": actual,
        "cpu_count": os.cpu_count(),
        "torch_num_threads": protocol["runtime"]["torch_num_threads"],
        "device": "cpu",
    }


def rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return value / (1024 * 1024)
    return value / 1024


def choices_from_row(row: dict) -> list[tuple[str, str]]:
    choices = row["choices"]
    if isinstance(choices, dict):
        return list(zip(choices["label"], choices["text"], strict=True))
    return [(item["label"], item["text"]) for item in choices]


def selection_key(row: dict, protocol: dict) -> str:
    item = f"{protocol['sample']['selection_salt']}|{row['id']}".encode()
    return hashlib.sha256(item).hexdigest()


def format_prompt(tokenizer, row: dict, choices: list[tuple[str, str]], template: dict) -> tuple[str, str]:
    options = "\n".join(f"{label}. {text}" for label, text in choices)
    messages = [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": template["user"].format(question=row["question"], choices=options)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, hashlib.sha256(prompt.encode()).hexdigest()


def parse_answer(raw: str) -> str | None:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    text = re.sub(r"(?i)^(?:the\s+)?(?:correct\s+)?answer(?:\s+is)?\s*[:\-]?\s*", "", text)
    match = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", text.strip())
    return match.group(1).upper() if match else None


def wilson_lower(successes: int, total: int, z: float) -> float:
    if total <= 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    numerator = p + z * z / (2 * total) - z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new directory for the frozen run bundle.")
    args = parser.parse_args()
    protocol = check_protocol()
    runtime = check_runtime(protocol)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)

    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(protocol["runtime"]["torch_num_threads"])
    torch.manual_seed(protocol["sample"]["seed"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    started = time.monotonic()
    rows_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=DATASET_FILE,
        revision=protocol["dataset"]["revision"],
    )
    data_hash = sha256_file(Path(rows_path))
    if data_hash != protocol["dataset"]["validation_parquet_sha256"]:
        raise RuntimeError(f"dataset file hash mismatch: {data_hash}")
    frame = pd.read_parquet(rows_path)
    data_rows = frame.to_dict(orient="records")
    if len(data_rows) != protocol["dataset"]["validation_rows"]:
        raise RuntimeError(f"validation row count mismatch: {len(data_rows)}")

    eligible = []
    for row in data_rows:
        choices = choices_from_row(row)
        labels = [str(label) for label, _ in choices]
        if len(choices) == 4 and labels == ["A", "B", "C", "D"]:
            eligible.append(row)
    selected = sorted(eligible, key=lambda row: selection_key(row, protocol))[: protocol["sample"]["n"]]
    if len(selected) != protocol["sample"]["n"]:
        raise RuntimeError(f"eligible sample shortage: {len(selected)}")
    ids = [str(row["id"]) for row in selected]
    if len(set(ids)) != len(ids):
        raise RuntimeError("selected task IDs are not unique")

    model_dir = snapshot_download(
        repo_id=MODEL_REPO,
        revision=protocol["model"]["revision"],
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float32,
    ).to("cpu")
    model.eval()

    records_path = output / "responses.jsonl"
    metrics = {"parsed": 0, "exact": 0, "unparsed": 0}
    with records_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(selected):
            choices = choices_from_row(row)
            prompt, prompt_hash = format_prompt(tokenizer, row, choices, protocol["prompt"])
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=protocol["generation"]["max_input_tokens"],
            )
            if inputs["input_ids"].shape[1] >= protocol["generation"]["max_input_tokens"]:
                raise RuntimeError(f"input reached truncation boundary for row {row['id']}")
            input_ids = inputs["input_ids"].to("cpu")
            attention_mask = inputs["attention_mask"].to("cpu")
            generated_at = time.monotonic()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=protocol["generation"]["max_new_tokens"],
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            elapsed = time.monotonic() - generated_at
            new_tokens = output_ids[0, input_ids.shape[1] :]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed = parse_answer(raw)
            answer_key = str(row["answerKey"]).upper()
            record = {
                "index": index,
                "task_id": str(row["id"]),
                "prompt_sha256": prompt_hash,
                "input_tokens": int(input_ids.shape[1]),
                "generated_token_ids": [int(token) for token in new_tokens.tolist()],
                "raw_generation": raw,
                "parsed_label": parsed,
                "answer_key": answer_key,
                "exact": parsed == answer_key,
                "generation_seconds": elapsed,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            metrics["parsed"] += parsed is not None
            metrics["exact"] += parsed == answer_key
            metrics["unparsed"] += parsed is None

            if time.monotonic() - started > protocol["resources"]["wall_seconds_cap"]:
                raise TimeoutError("frozen wall-time cap exceeded")
            if rss_mib() > protocol["resources"]["peak_rss_mib_cap"]:
                raise MemoryError("frozen RSS cap exceeded")

    n = protocol["sample"]["n"]
    accuracy = metrics["exact"] / n
    parse_rate = metrics["parsed"] / n
    lower = wilson_lower(metrics["exact"], n, protocol["metrics"]["wilson_z"])
    completed = sum(1 for _ in records_path.open(encoding="utf-8"))
    wall_seconds = time.monotonic() - started
    gate_checks = {
        "all_rows_completed": completed == n,
        "minimum_exact_accuracy": accuracy >= protocol["acceptance"]["minimum_accuracy"],
        "accuracy_wilson_lower_above_random_chance": lower > protocol["acceptance"]["random_chance"],
        "minimum_parse_rate": parse_rate >= protocol["acceptance"]["minimum_parse_rate"],
        "within_wall_time": wall_seconds <= protocol["resources"]["wall_seconds_cap"],
        "within_peak_rss": rss_mib() <= protocol["resources"]["peak_rss_mib_cap"],
    }
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["sha256"],
        "runner_sha256": protocol["source"]["runner_sha256"],
        "model": {"repo_id": MODEL_REPO, "revision": protocol["model"]["revision"], "dtype": "float32"},
        "dataset": {
            "repo_id": DATASET_REPO,
            "revision": protocol["dataset"]["revision"],
            "file": DATASET_FILE,
            "file_sha256": data_hash,
            "split": "validation",
            "selected_ids": ids,
            "selection_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        },
        "runtime": runtime,
        "n": n,
        "exact_successes": metrics["exact"],
        "accuracy": accuracy,
        "accuracy_wilson_95_lower": lower,
        "random_chance": protocol["acceptance"]["random_chance"],
        "parsed": metrics["parsed"],
        "parse_rate": parse_rate,
        "unparsed": metrics["unparsed"],
        "wall_seconds": wall_seconds,
        "peak_rss_mib": rss_mib(),
        "gate_checks": gate_checks,
        "gate": "pass" if all(gate_checks.values()) else "fail",
        "next_step_authorized": "update_cost_smoke_only" if all(gate_checks.values()) else "retire_exact_model_task_pairing",
        "interpretation": "Base-only feasibility screen on an opened validation sample; not an update result or capability claim.",
    }
    (output / "environment.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(gate_checks.values()) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        output_arg = None
        for position, value in enumerate(sys.argv[:-1]):
            if value == "--output":
                output_arg = Path(sys.argv[position + 1]).expanduser().resolve()
                break
        if output_arg is not None and output_arg.is_dir() and not (output_arg / "summary.json").exists():
            completed = 0
            responses = output_arg / "responses.jsonl"
            if responses.exists():
                completed = sum(1 for _ in responses.open(encoding="utf-8"))
            failure = {
                "protocol_id": "qwen_arc_challenge_base_gate_v1",
                "status": "execution_failure",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "completed_rows": completed,
                "partial_responses_sha256": sha256_file(responses) if responses.exists() else None,
            }
            (output_arg / "failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
