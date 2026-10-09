#!/usr/bin/env python3
"""Evaluate frozen v5 OpenBookQA checkpoints on the sealed confirmation cohort."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/openbookqa_qwen_grpo_sft_v6-evaluation.lock.json"
V5_RUNNER = ROOT / "scripts/run_openbookqa_qwen_grpo_sft_v5.py"
V5_PROTOCOL = ROOT / "protocols/openbookqa_qwen_grpo_sft_v5.lock.json"
V5_PROGRESS = ROOT / "artifacts/openbookqa_qwen_grpo_sft_v1/study-v5/progress.json"
CHECKPOINTS = ROOT / "artifacts/openbookqa_qwen_grpo_sft_v1/study-v5/checkpoints"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != "lock_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def fingerprint(state: dict) -> str:
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("vare_openbookqa_v5_runner", V5_RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_frozen_inputs(lock: dict, runner) -> tuple[dict, dict]:
    if canonical_sha(lock) != lock.get("lock_sha256"):
        raise RuntimeError("v6 evaluation protocol digest mismatch")
    if sha_file(Path(__file__)) != lock["source_sha256"]:
        raise RuntimeError("v6 evaluator source differs from frozen protocol")
    if sha_file(V5_RUNNER) != lock["v5_runner_sha256"] or sha_file(V5_PROTOCOL) != lock["v5_protocol_sha256"]:
        raise RuntimeError("v5 training source/protocol changed")
    progress = json.loads(V5_PROGRESS.read_text(encoding="utf-8"))
    if sha_file(V5_PROGRESS) != lock["v5_progress_sha256"]:
        raise RuntimeError("v5 training record changed")
    if progress.get("status") != "training" or progress.get("evaluation") != {}:
        raise RuntimeError("v5 must contain completed training and no evaluation results")
    expected = {f"{arm}-{seed}" for arm in ("sft", "grpo") for seed in (11, 23, 37)}
    training = progress.get("training", {})
    if set(training) != expected:
        raise RuntimeError("v5 does not contain all six trained arms")
    if not progress.get("confirmation_opened") or progress.get("confirmation_n") != 128:
        raise RuntimeError("v5 confirmation stage record differs from expected completed run")
    if progress.get("confirmation_ids_sha256") != lock["confirmation_ids_sha256"]:
        raise RuntimeError("v5 confirmation cohort receipt differs")
    if training != lock["training_records"]:
        raise RuntimeError("v5 training records differ from frozen evaluator lock")
    for key, expected_files in lock["checkpoint_file_hashes"].items():
        folder = CHECKPOINTS / key
        actual = {p.name: sha_file(p) for p in sorted(folder.iterdir()) if p.is_file()}
        if actual != expected_files:
            raise RuntimeError(f"checkpoint file hash mismatch: {key}")
    return lock, progress


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--dev-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    runner = load_runner()
    lock, progress = verify_frozen_inputs(lock, runner)
    v5lock = runner.load_lock()
    runner.check_source_and_model(args.model_path, args.rvl_source, v5lock)
    if args.output.exists():
        raise FileExistsError(args.output)

    # Test rows are opened only after the evaluator code, v5 run, and all checkpoints pass hash checks.
    test_all = runner.load_rows(args.test_parquet, "test", v5lock)
    test = runner.select(test_all, "test", v5lock["selection"]["test_n"])
    test_ids_hash = hashlib.sha256("\n".join(str(x["id"]) for x in test).encode()).hexdigest()
    if test_ids_hash != progress["confirmation_ids_sha256"]:
        raise RuntimeError("selected confirmation IDs do not match the v5 receipt")
    dev_all = runner.load_rows(args.dev_parquet, "validation", v5lock)
    dev = runner.select(dev_all, "validation", v5lock["selection"]["dev_n"])
    if [str(x["id"]) for x in dev] != v5lock["selection"]["dev_ids"]:
        raise RuntimeError("development cohort mismatch")

    args.output.mkdir(parents=True)
    report = {"protocol_id": lock["protocol_id"], "status": "evaluating", "evaluation": {},
              "confirmation_opened": True, "confirmation_n": len(test),
              "confirmation_ids_sha256": test_ids_hash,
              "training_record_sha256": lock["v5_progress_sha256"],
              "checkpoint_file_hashes_verified": True, "test_parquet_sha256": lock["test_parquet_sha256"],
              "start_unix": time.time()}
    write_json(args.output / "progress.json", report)
    torch, HFLocalBackend, *_ = runner.load_runtime(args.model_path, args.rvl_source, v5lock)
    started = time.monotonic()
    print("Loading unchanged base model and evaluating confirmation cohort", flush=True)
    base_backend = HFLocalBackend(model_name=str(args.model_path), max_new_tokens=8, device="cpu", precision="fp32")
    base_model, base_tokenizer = base_backend.model, base_backend.tokenizer
    if next(base_model.parameters()).device.type != "cpu":
        raise RuntimeError("base evaluation model is not on CPU")
    base_logprobs = runner.label_log_distributions(base_model, base_tokenizer, dev, torch)
    base_fp = fingerprint(base_model.state_dict())
    base_records = runner.eval_rows(base_backend, base_tokenizer, test, "test", v5lock["evaluation"]["test_seed"])
    report["evaluation"]["base"] = {"arm": "base", "seed": 0, "fingerprint": base_fp,
        "accuracy": sum(x["exact"] for x in base_records)/len(base_records),
        "parse_rate": sum(x["parsed_label"] is not None for x in base_records)/len(base_records),
        "mean_latency_s": sum(x["latency_s"] for x in base_records)/len(base_records),
        "dev_answer_label_kl_from_base": 0.0, "results": base_records}
    del base_model, base_tokenizer, base_backend
    import gc; gc.collect()
    write_json(args.output / "progress.json", report)
    print(f"base done: accuracy={report['evaluation']['base']['accuracy']:.4f}; elapsed={time.monotonic()-started:.1f}s", flush=True)

    for seed in v5lock["training"]["seeds"]:
        for arm in ("sft", "grpo"):
            key = f"{arm}-{seed}"
            print(f"Evaluating {key}", flush=True)
            evaluated = runner.evaluate_saved(v5lock, CHECKPOINTS/key, args.model_path, args.rvl_source,
                test, arm, int(seed), dev, base_logprobs)
            if evaluated["fingerprint"] != progress["training"][key]["candidate_fingerprint"]:
                raise RuntimeError(f"reloaded checkpoint fingerprint mismatch: {key}")
            report["evaluation"][key] = evaluated
            write_json(args.output / "progress.json", report)
            print(f"{key} done: accuracy={evaluated['accuracy']:.4f}; parse={evaluated['parse_rate']:.4f}; elapsed={time.monotonic()-started:.1f}s", flush=True)

    report["status"] = "complete"
    report["wall_seconds"] = time.monotonic()-started
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)
    report["evaluation_summary"] = runner.summarize(v5lock, report["evaluation"])
    report["resource_gate"] = report["wall_seconds"] <= v5lock["resources"]["wall_cap_seconds_per_stage"] and report["peak_rss_mib"] <= v5lock["resources"]["peak_rss_cap_mib"]
    report["decision"] = "CONTINUE" if report["evaluation_summary"]["success_gate"] and report["resource_gate"] else ("STOP_RESOURCE_GATE" if not report["resource_gate"] else "STOP_NO_PREDECLARED_SUCCESS")
    report["limitations"] = ["single 0.5B model and one public science benchmark", "possible benchmark pretraining overlap", "128-item confirmation subset has limited power", "three training seeds; task bootstrap is conditional on these seeds", "no production impact or general capability improvement established"]
    updates=[m for seed in v5lock["training"]["seeds"] for m in progress["training"][f"grpo-{seed}"]["update_metrics"]]
    report["mean_grpo_behavior_kl_estimate"] = sum(float(m.get("behavior_kl_estimate", 0.0)) for m in updates)/len(updates)
    write_json(args.output/"summary.json",report)
    print(json.dumps({"status":report["status"],"decision":report["decision"],"evaluation_summary":report["evaluation_summary"],"wall_seconds":report["wall_seconds"],"peak_rss_mib":report["peak_rss_mib"]},indent=2,sort_keys=True),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
