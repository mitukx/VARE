#!/usr/bin/env python3
"""Read-only tokenizer/trajectory alignment audit for the OpenBookQA study."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
V5_RUNNER = ROOT / "scripts/run_openbookqa_qwen_grpo_sft_v5.py"
V5_PROTOCOL = ROOT / "protocols/openbookqa_qwen_grpo_sft_v5.lock.json"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--train-parquet", type=Path, required=True)
    p.add_argument("--test-parquet", type=Path, required=True)
    p.add_argument("--rollouts-json", type=Path, required=True)
    p.add_argument("--progress-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    spec = importlib.util.spec_from_file_location("openbookqa_v5_runner", V5_RUNNER)
    runner = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    lock = runner.load_lock()
    if {name: sha_file(args.model_path / name) for name in lock["model"]["files_sha256"]} != lock["model"]["files_sha256"]:
        raise RuntimeError("base model file hashes differ from the v5 protocol")

    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    train = runner.select(runner.load_rows(args.train_parquet, "train", lock), "train", lock["selection"]["train_n"])
    test = runner.select(runner.load_rows(args.test_parquet, "test", lock), "test", lock["selection"]["test_n"])
    progress = json.loads(args.progress_json.read_text(encoding="utf-8"))
    rollouts = json.loads(args.rollouts_json.read_text(encoding="utf-8"))
    if set(progress["training"]) != {f"{a}-{s}" for a in ("sft", "grpo") for s in (11, 23, 37)}:
        raise RuntimeError("expected six completed checkpoints")

    cohorts = {"train": train, "test": test}
    template_hash = hashlib.sha256((base.chat_template or "").encode()).hexdigest()
    prompt_checks = {}
    for key, record in progress["training"].items():
        folder = args.progress_json.parent / "checkpoints" / key
        file_hashes = {x.name: sha_file(x) for x in sorted(folder.iterdir()) if x.is_file()}
        if file_hashes != record["model_files_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {key}")
        tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
        same_template = hashlib.sha256((tok.chat_template or "").encode()).hexdigest() == template_hash
        counts = {}
        for split, rows in cohorts.items():
            mismatch = 0
            for row in rows:
                prompt = runner.render(base, row)
                candidate_prompt = runner.render(tok, row)
                base_ids = base(prompt, add_special_tokens=False)["input_ids"]
                candidate_ids = tok(candidate_prompt, add_special_tokens=False)["input_ids"]
                mismatch += int(candidate_prompt != prompt or candidate_ids != base_ids)
            counts[split] = {"n": len(rows), "prompt_or_token_mismatches": mismatch}
        prompt_checks[key] = {"template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
                              "template_matches_base": same_template, "cohorts": counts}
        if not same_template or any(x["prompt_or_token_mismatches"] for x in counts.values()):
            raise RuntimeError(f"tokenizer/template mismatch: {key}")
        del tok

    train_prompt_by_id = {str(row["id"]): runner.render(base, row) for row in train}
    prompt_id_mismatches = response_decode_mismatches = checked = 0
    for seed, groups in rollouts["seed_groups"].items():
        if len(groups) != len(train):
            raise RuntimeError(f"rollout group count mismatch for seed {seed}")
        for row, group in zip(train, groups):
            qid = str(row["id"])
            if len(group) != 4:
                raise RuntimeError(f"bad group size for seed {seed}, task {qid}")
            expected_prompt_ids = base(train_prompt_by_id[qid], add_special_tokens=False)["input_ids"]
            for item in group:
                checked += 1
                prompt_id_mismatches += int(item["prompt_token_ids"] != expected_prompt_ids)
                decoded = base.decode(item["response_token_ids"], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
                response_decode_mismatches += int(decoded != item["raw_generation"])
    result = {
        "status": "pass" if prompt_id_mismatches == 0 and response_decode_mismatches == 0 else "fail",
        "protocol_id": "openbookqa_qwen_grpo_sft_tokenizer_audit_v1",
        "audit_script_sha256": sha_file(Path(__file__)),
        "v5_runner_sha256": sha_file(V5_RUNNER),
        "v5_protocol_sha256": sha_file(V5_PROTOCOL),
        "base_tokenizer_class": type(base).__name__,
        "base_chat_template_sha256": template_hash,
        "checkpoint_tokenizer_prompt_checks": prompt_checks,
        "rollout_records_checked": checked,
        "rollout_prompt_token_id_mismatches": prompt_id_mismatches,
        "rollout_text_token_decode_mismatches": response_decode_mismatches,
        "rollouts_sha256": sha_file(args.rollouts_json),
        "training_progress_sha256": sha_file(args.progress_json),
        "note": "Read-only integrity audit; it does not rescore outcomes or authorize changing the consumed confirmation cohort.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
