#!/usr/bin/env python3
"""Frozen, no-update CPU feasibility screen for binary SNLI entailment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

try:
    from .snli_entailment_task import balanced_accuracy, class_recall, denylist_hashes, option_order, render_prompt, row_hash, select_validation_rows
except ImportError:
    from snli_entailment_task import balanced_accuracy, class_recall, denylist_hashes, option_order, render_prompt, row_hash, select_validation_rows


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/snli_entailment_base_feasibility_v1.json"
LOCK_PATH = ROOT / "protocols/snli_entailment_base_feasibility_v1.lock.json"
DENYLIST_PATH = ROOT / "protocols/snli_entailment_base_feasibility_v1.denylist.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/datasets/stanfordnlp___snli/plain_text/0.0.0/cdb5c3d5eed6ead6e5a341c8e56e669bb666725b"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def write_manifest(output: Path, *, include_audit: bool = False) -> None:
    files = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json" and (include_audit or path.name != "audit.json")
    }
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": files})


def locked_inputs():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    digest = lock.pop("sha256", None)
    expected = hashlib.sha256(canonical(spec)).hexdigest()
    if digest != expected or canonical(lock) != canonical(spec):
        raise ValueError("protocol JSON differs from its SHA-256 lock")
    deny = json.loads(DENYLIST_PATH.read_text(encoding="utf-8"))
    deny_digest = deny.pop("sha256", None)
    if deny_digest != hashlib.sha256(canonical(deny)).hexdigest():
        raise ValueError("pilot denylist differs from its SHA-256 digest")
    if deny_digest != spec["dataset"]["denylist_sha256"]:
        raise ValueError("pilot denylist digest differs from protocol")
    excluded_text_hashes, excluded_premise_hashes = denylist_hashes(deny)
    return spec, expected, deny, excluded_text_hashes, excluded_premise_hashes


def git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status:
        raise RuntimeError("source tree must be clean before the frozen feasibility screen")
    return {"commit": commit, "working_tree_clean": True}


def check_cache(spec):
    expected_data = spec["dataset"]["validation_split"]["arrow_sha256"]
    arrow = DATA_DIR / spec["dataset"]["validation_split"]["arrow_file"]
    if not arrow.is_file() or sha256_file(arrow) != expected_data:
        raise ValueError("pinned SNLI validation Arrow cache missing or hash mismatch")
    cache_hashes = {"validation_arrow": sha256_file(arrow)}
    for filename, digest in spec["model"]["snapshot_file_sha256"].items():
        path = MODEL_DIR / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"pinned model cache missing or hash mismatch: {filename}")
        cache_hashes[f"model_snapshot/{filename}"] = sha256_file(path)
    return cache_hashes


def evaluate_base(dataset, selected, spec, output: Path, deadline: float):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    action_ids = {
        "A": tokenizer.encode(" A", add_special_tokens=False),
        "B": tokenizer.encode(" B", add_special_tokens=False),
    }
    if any(len(value) != 1 for value in action_ids.values()) or action_ids["A"] == action_ids["B"]:
        raise ValueError("A/B actions are not distinct single tokens")
    action_ids = {key: value[0] for key, value in action_ids.items()}
    if action_ids != spec["action_token_ids"]:
        raise ValueError("tokenizer A/B action IDs differ from the frozen protocol")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.float32,
    ).to("cpu").eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("base model is not on CPU")
    if rss_bytes() > int(spec["compute_limits"]["max_peak_rss_bytes_process"]):
        raise RuntimeError("model load exceeded frozen memory budget")
    prompts = []
    examples = []
    for meta in selected:
        source = dataset[int(meta["dataset_index"])]
        digest = row_hash(source["premise"], source["hypothesis"])
        if digest != meta["prompt_sha256"]:
            raise ValueError("prompt row changed between cohort selection and scoring")
        prompts.append(render_prompt(source["premise"], source["hypothesis"], digest))
        examples.append({**meta, "premise": source["premise"], "hypothesis": source["hypothesis"], "prompt": prompts[-1]})
    if len(examples) != 512:
        raise ValueError("frozen feasibility cohort must contain exactly 512 rows")
    scores = []
    with torch.inference_mode():
        batch_size = int(spec["compute_limits"]["feature_batch_size"])
        for start in range(0, len(prompts), batch_size):
            if time.monotonic() >= deadline or rss_bytes() > int(spec["compute_limits"]["max_peak_rss_bytes_process"]):
                raise RuntimeError("base scoring exceeded frozen time or memory budget")
            encoded = tokenizer(
                prompts[start:start + batch_size], return_tensors="pt", padding=True,
                truncation=False, add_special_tokens=True,
            )
            logits = model(**encoded).logits[:, -1, :]
            for offset, row in enumerate(examples[start:start + batch_size]):
                logit_a = float(logits[offset, action_ids["A"]].item())
                logit_b = float(logits[offset, action_ids["B"]].item())
                option_a, option_b = option_order(row["prompt_sha256"])
                predicted_letter = "A" if logit_a >= logit_b else "B"
                gold_letter = ("A" if option_a == "entailment" else "B") if row["label"] == 0 else ("A" if option_a != "entailment" else "B")
                predicted_label = 0 if (option_a if predicted_letter == "A" else option_b) == "entailment" else 1
                scores.append({
                    "dataset_index": row["dataset_index"],
                    "prompt_sha256": row["prompt_sha256"],
                    "label": row["label"],
                    "gold_letter": gold_letter,
                    "predicted_letter": predicted_letter,
                    "predicted_label": predicted_label,
                    "logit_a": logit_a,
                    "logit_b": logit_b,
                    "entailment_minus_not_entailment_logit": logit_a - logit_b if option_a == "entailment" else logit_b - logit_a,
                    "action_token_ids": action_ids,
                })
    if len(scores) != len(examples):
        raise ValueError("scoring did not return one record per selected example")
    (output / "selected_examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_json(output / "scores.json", scores)
    return scores, action_ids


def metrics(scores, spec):
    labels = [int(row["label"]) for row in scores]
    predictions = [int(row["predicted_label"]) for row in scores]
    correct = [int(pred == label) for pred, label in zip(predictions, labels)]
    n = len(labels)
    tp = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fn = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    ba = balanced_accuracy(predictions, labels)
    import random
    rng = random.Random(int(spec["metrics"]["paired_bootstrap"]["seed"]))
    by_class = {cls: [i for i, label in enumerate(labels) if label == cls] for cls in (0, 1)}
    values = []
    for _ in range(int(spec["metrics"]["paired_bootstrap"]["resamples"])):
        sampled = [i for cls in (0, 1) for i in rng.choices(by_class[cls], k=len(by_class[cls]))]
        recalls = [sum(predictions[i] == cls for i in sampled if labels[i] == cls) / len(by_class[cls]) for cls in (0, 1)]
        values.append(sum(recalls) / 2)
    values.sort()
    lo = values[int(0.025 * len(values))]
    hi = values[min(len(values) - 1, int(0.975 * len(values)))]
    gate = spec["metrics"]["feasibility_gate"]
    checks = {
        "minimum_balanced_accuracy_pass": ba >= float(gate["minimum_balanced_accuracy"]),
        "minimum_entailment_recall_pass": class_recall(predictions, labels, 0) >= float(gate["minimum_recall_per_class"]),
        "minimum_not_entailment_recall_pass": class_recall(predictions, labels, 1) >= float(gate["minimum_recall_per_class"]),
        "bootstrap_lower_bound_above_chance_pass": lo > float(gate["bootstrap_95_percent_lower_bound_above"]),
    }
    return {
        "examples": n,
        "accuracy": sum(correct) / n,
        "balanced_accuracy": ba,
        "entailment_recall": class_recall(predictions, labels, 0),
        "not_entailment_recall": class_recall(predictions, labels, 1),
        "confusion_matrix_true_rows_predicted_columns": [[tp, fn], [fp, tn]],
        "balanced_accuracy_bootstrap_95_percent_interval": [lo, hi],
        "checks": checks,
        "result": "pass" if all(checks.values()) else "non-pass",
    }


def run(output: Path):
    started = time.monotonic()
    spec, protocol_hash, deny, excluded_text_hashes, excluded_premise_hashes = locked_inputs()
    source = git_state()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for path in (SPEC_PATH, LOCK_PATH, DENYLIST_PATH, Path(__file__), Path(__file__).with_name("snli_entailment_task.py"), Path(__file__).with_name("audit_snli_entailment_base_feasibility_v1.py")):
        (output / f"source-{path.name}").write_bytes(path.read_bytes())
    (output / "protocol.snapshot.json").write_bytes(SPEC_PATH.read_bytes())
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    (output / "pilot-denylist.snapshot.json").write_bytes(DENYLIST_PATH.read_bytes())
    compute = spec["compute_limits"]
    os.environ.update({"HF_HUB_OFFLINE":"1", "HF_DATASETS_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "HF_HUB_DISABLE_TELEMETRY":"1", "TOKENIZERS_PARALLELISM":"false"})
    deadline = started + int(compute["max_wall_seconds_process"])
    try:
        import datasets, safetensors, tokenizers, torch, transformers
        from datasets import load_dataset
        actual_runtime = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__, "numpy": np.__version__, "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__}
        if actual_runtime != spec["runtime_lock"]:
            raise RuntimeError(f"runtime differs from protocol: {actual_runtime}")
        torch.set_num_threads(int(compute["threads"]))
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU-only screen refuses an initialized CUDA runtime")
        cache_hashes = check_cache(spec)
        dataset_spec = spec["dataset"]
        dataset = load_dataset(dataset_spec["id"], dataset_spec["configuration"], split="validation", revision=dataset_spec["revision"])
        if len(dataset) != dataset_spec["validation_split"]["rows"] or dataset._fingerprint != dataset_spec["validation_split"]["fingerprint"]:
            raise ValueError("cached SNLI validation split differs from frozen source")
        selected = select_validation_rows(
            dataset, 256, excluded_text_hashes=excluded_text_hashes,
            excluded_premise_hashes=excluded_premise_hashes,
        )
        for row in selected:
            source_row = dataset[int(row["dataset_index"])]
            if row_hash(source_row["premise"], source_row["hypothesis"]) != row["prompt_sha256"]:
                raise ValueError("selected prompt hash mismatch")
        write_json(output / "cohort_selection.json", selected)
        scores, action_ids = evaluate_base(dataset, selected, spec, output, deadline)
        result = metrics(scores, spec)
        elapsed = time.monotonic() - started
        peak = rss_bytes()
        resources = {"wall_seconds": elapsed, "peak_rss_bytes": peak, "cpu_threads": torch.get_num_threads(), "device": "cpu", "network_offline": True, "model_updates": 0, "paid_compute": False}
        resource_pass = elapsed <= compute["max_wall_seconds_process"] and peak <= compute["max_peak_rss_bytes_process"]
        result["checks"]["resource_limits_pass"] = resource_pass
        result["checks"]["tokenizer_action_ids_pass"] = action_ids == spec["action_token_ids"]
        result["result"] = "pass" if all(result["checks"].values()) else "non-pass"
        fingerprints = {
            "protocol_sha256": protocol_hash,
            "denylist_sha256": spec["dataset"]["denylist_sha256"],
            "source": source,
            "dataset_revision": dataset_spec["revision"],
            "dataset_split": "validation",
            "dataset_fingerprint": dataset._fingerprint,
            "cached_dataset_and_model_sha256": cache_hashes,
            "runtime": actual_runtime,
            "action_token_ids": action_ids,
        }
        write_json(output / "input_fingerprints.json", fingerprints)
        write_json(output / "summary.json", {"protocol_id": spec["protocol_id"], "protocol_sha256": protocol_hash, "source_commit": source["commit"], "cohort_size": len(selected), "class_counts": {"entailment": sum(row["label"] == 0 for row in selected), "not_entailment": sum(row["label"] == 1 for row in selected)}, "metrics": result, "resource_measurements": resources, "interpretation": spec["claim_boundary"]})
        write_manifest(output)
        if not resource_pass:
            raise RuntimeError("feasibility process exceeded its frozen resource ceiling")
        subprocess.check_call([sys.executable, str(Path(__file__).with_name("audit_snli_entailment_base_feasibility_v1.py")), str(output)], cwd=ROOT)
        write_manifest(output, include_audit=True)
    except Exception as exc:
        write_json(output / "failure.json", {"protocol_id": spec["protocol_id"], "source_commit": source["commit"], "exception_type": type(exc).__name__, "message": str(exc), "wall_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes()})
        write_manifest(output)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
