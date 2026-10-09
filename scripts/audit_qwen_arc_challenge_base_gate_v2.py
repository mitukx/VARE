#!/usr/bin/env python3
"""Independently audit a retained ARC-Challenge CPU base-gate bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/qwen_arc_challenge_base_gate_v2.lock.json"
RUNNER_PATH = ROOT / "scripts/run_qwen_arc_challenge_base_gate_v2.py"
AUDITOR_PATH = Path(__file__).resolve()
DATASET_REPO = "allenai/ai2_arc"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    payload = {key: item for key, item in value.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def choices_for(row: dict) -> list[tuple[str, str]]:
    value = row["choices"]
    if isinstance(value, dict):
        return list(zip(value["label"], value["text"], strict=True))
    return [(item["label"], item["text"]) for item in value]


def selected_rows(rows: list[dict], protocol: dict) -> list[dict]:
    excluded = set(protocol["sample"]["excluded_task_ids"])
    eligible = [
        row for row in rows
        if [str(label) for label, _ in choices_for(row)] == ["A", "B", "C", "D"]
        and str(row["id"]) not in excluded
    ]
    def rank(row: dict) -> str:
        text = protocol["sample"]["selection_salt"] + "|" + str(row["id"])
        return hashlib.sha256(text.encode()).hexdigest()
    return sorted(eligible, key=rank)[: protocol["sample"]["n"]]


def render_prompt(tokenizer, row: dict, protocol: dict) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in choices_for(row))
    msgs = [
        {"role": "system", "content": protocol["prompt"]["system"]},
        {"role": "user", "content": protocol["prompt"]["user"].format(question=row["question"], choices=options)},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def independent_parse(raw: str) -> str | None:
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        bits = value.splitlines()
        if len(bits) < 2:
            return None
        value = "\n".join(bits[1:-1]).strip()
    prefixes = (
        "the correct answer is", "the answer is:", "answer:",
    )
    low = value.lower()
    for prefix in prefixes:
        if low.startswith(prefix):
            value = value[len(prefix):].strip()
            if value.startswith((":", "-")):
                value = value[1:].strip()
            break
    match = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", value)
    return match.group(1).upper() if match else None


def wilson_low(x: int, n: int, z: float) -> float:
    p = x / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if canonical_sha256(protocol) != protocol.get("sha256"):
        raise RuntimeError("frozen protocol digest mismatch")
    expected_runner_hash = protocol["source"]["runner_sha256"]
    expected_auditor_hash = protocol["source"]["auditor_sha256"]
    if sha256_file(RUNNER_PATH) != expected_runner_hash:
        raise RuntimeError("runner digest mismatch")
    if sha256_file(AUDITOR_PATH) != expected_auditor_hash:
        raise RuntimeError("auditor digest mismatch")

    from huggingface_hub import hf_hub_download, snapshot_download
    import pandas as pd
    from transformers import AutoTokenizer

    parquet = Path(hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=protocol["dataset"]["file"],
        revision=protocol["dataset"]["revision"],
    ))
    if sha256_file(parquet) != protocol["dataset"]["validation_parquet_sha256"]:
        raise RuntimeError("validation parquet digest mismatch")
    rows = pd.read_parquet(parquet).to_dict(orient="records")
    if len(rows) != protocol["dataset"]["validation_rows"]:
        raise RuntimeError("validation split row count mismatch")
    selected = selected_rows(rows, protocol)
    if len(selected) != protocol["sample"]["n"]:
        raise RuntimeError("frozen selection has fewer than the required rows")

    model_dir = snapshot_download(
        repo_id=MODEL_REPO,
        revision=protocol["model"]["revision"],
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    records = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records) != protocol["sample"]["n"]:
        raise RuntimeError("raw response count mismatch")
    if [record["index"] for record in records] != list(range(len(records))):
        raise RuntimeError("response ordering/index mismatch")

    parsed_n = exact_n = 0
    for record, row in zip(records, selected, strict=True):
        answer = str(row["answerKey"]).upper()
        prompt = render_prompt(tokenizer, row, protocol)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        raw = tokenizer.decode(record["generated_token_ids"], skip_special_tokens=True)
        parsed = independent_parse(raw)
        if str(row["id"]) != record["task_id"] or prompt_hash != record["prompt_sha256"]:
            raise RuntimeError(f"task identity or prompt hash mismatch at row {record['index']}")
        if answer != record["answer_key"] or raw != record["raw_generation"]:
            raise RuntimeError(f"answer key or decoded generation mismatch at row {record['index']}")
        if parsed != record["parsed_label"] or (parsed == answer) != record["exact"]:
            raise RuntimeError(f"independent parser/scorer mismatch at row {record['index']}")
        if record["input_tokens"] > protocol["generation"]["max_input_tokens"]:
            raise RuntimeError("input length exceeds frozen limit")
        parsed_n += parsed is not None
        exact_n += parsed == answer

    n = len(records)
    accuracy = exact_n / n
    lower = wilson_low(exact_n, n, protocol["metrics"]["wilson_z"])
    parse_rate = parsed_n / n
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    expected_gate = {
        "all_rows_completed": n == protocol["sample"]["n"],
        "minimum_exact_accuracy": accuracy >= protocol["acceptance"]["minimum_accuracy"],
        "accuracy_wilson_lower_above_random_chance": lower > protocol["acceptance"]["random_chance"],
        "minimum_parse_rate": parse_rate >= protocol["acceptance"]["minimum_parse_rate"],
        "within_wall_time": summary["wall_seconds"] <= protocol["resources"]["wall_seconds_cap"],
        "within_peak_rss": summary["peak_rss_mib"] <= protocol["resources"]["peak_rss_mib_cap"],
    }
    if summary["protocol_sha256"] != protocol["sha256"] or summary["runner_sha256"] != expected_runner_hash:
        raise RuntimeError("summary source/protocol provenance mismatch")
    if summary["exact_successes"] != exact_n or abs(summary["accuracy"] - accuracy) > 1e-12:
        raise RuntimeError("primary metric mismatch")
    if summary["gate_checks"] != expected_gate:
        raise RuntimeError("gate decision mismatch")
    audit = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["sha256"],
        "runner_sha256": expected_runner_hash,
        "auditor_sha256": expected_auditor_hash,
        "audit_method": "separate source-hash-locked implementation; reloads pinned validation rows and tokenizer, reconstructs task selection and prompt digests, decodes retained token IDs, independently parses and scores every response",
        "n": n,
        "exact_successes": exact_n,
        "accuracy": accuracy,
        "accuracy_wilson_95_lower": lower,
        "parsed": parsed_n,
        "gate_checks_match": expected_gate == summary["gate_checks"],
        "status": "pass",
        "interpretation": "Same-host independent implementation audit; not external human reproduction.",
    }
    (run_dir / "independent_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
