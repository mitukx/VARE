#!/usr/bin/env python3
"""Independent same-host reconstruction of SNLI binary feasibility v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/snli_entailment_base_feasibility_v1.json"
LOCK_PATH = ROOT / "protocols/snli_entailment_base_feasibility_v1.lock.json"
DENYLIST_PATH = ROOT / "protocols/snli_entailment_base_feasibility_v1.denylist.json"
SOURCE_MAP = {
    "protocols/snli_entailment_base_feasibility_v1.json": "source-snli_entailment_base_feasibility_v1.json",
    "protocols/snli_entailment_base_feasibility_v1.lock.json": "source-snli_entailment_base_feasibility_v1.lock.json",
    "protocols/snli_entailment_base_feasibility_v1.denylist.json": "source-snli_entailment_base_feasibility_v1.denylist.json",
    "scripts/run_snli_entailment_base_feasibility_v1.py": "source-run_snli_entailment_base_feasibility_v1.py",
    "scripts/audit_snli_entailment_base_feasibility_v1.py": "source-audit_snli_entailment_base_feasibility_v1.py",
    "scripts/snli_entailment_task.py": "source-snli_entailment_task.py",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def close(actual, expected, tolerance=1e-6):
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"reconstructed value differs: {actual!r} != {expected!r}")


def verify_manifest(bundle):
    manifest = read_json(bundle / "manifest.json").get("files")
    if not isinstance(manifest, dict):
        raise ValueError("manifest file map is missing")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file() and p.name not in {"manifest.json", "audit.json"}}
    if actual != set(manifest):
        raise ValueError("pre-audit manifest inventory differs from bundle")
    for name, digest in manifest.items():
        if sha256_file(bundle / name) != digest:
            raise ValueError(f"manifest mismatch: {name}")
    return len(manifest)


def verify_locks_and_sources(bundle, commit):
    spec = read_json(bundle / "protocol.snapshot.json")
    lock = read_json(bundle / "protocol.lock.snapshot.json")
    lock_digest = lock.pop("sha256", None)
    protocol_digest = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_digest != protocol_digest or canonical(lock) != canonical(spec):
        raise ValueError("protocol snapshot does not match lock")
    deny = read_json(bundle / "pilot-denylist.snapshot.json")
    deny_digest = deny.pop("sha256", None)
    if deny_digest != hashlib.sha256(canonical(deny)).hexdigest() or deny_digest != spec["dataset"]["denylist_sha256"]:
        raise ValueError("pilot denylist snapshot does not match protocol")
    if canonical(spec) != canonical(read_json(SPEC_PATH)) or canonical(deny) != canonical({k:v for k,v in read_json(DENYLIST_PATH).items() if k != "sha256"}):
        raise ValueError("bundle protocol/denylist differs from current source")
    git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, text=True).strip()
    if Path(git_root) != ROOT:
        raise ValueError("unexpected auditor repository root")
    subprocess.check_output(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT)
    for relative, snapshot in SOURCE_MAP.items():
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if (bundle / snapshot).read_bytes() != committed:
            raise ValueError(f"source snapshot differs from frozen source: {relative}")
    return spec, protocol_digest, deny


def prompt_digest(premise, hypothesis):
    return hashlib.sha256((premise + "\n" + hypothesis).encode("utf-8")).hexdigest()


def premise_digest(premise):
    return hashlib.sha256(premise.encode("utf-8")).hexdigest()


def option_order(digest):
    flip = int(hashlib.sha256(("snli-binary-order-v1:" + digest).encode()).hexdigest()[-1], 16) & 1
    return ("not_entailment", "entailment") if flip else ("entailment", "not_entailment")


def prompt_text(premise, hypothesis, digest):
    a, b = option_order(digest)
    meanings = {"entailment": "the hypothesis follows from the premise", "not_entailment": "the hypothesis does not follow from the premise"}
    return (f"Premise: {premise}\nHypothesis: {hypothesis}\n\nWhich relation holds? Choose the best answer.\n"
            f"A. {meanings[a]}\nB. {meanings[b]}\n\nAnswer with one letter only: A or B.")


def reconstruct_rows(view, denylist, per_class):
    blocked_text = set()
    blocked_premise = set()
    for item in denylist.get("hashes", []):
        if item.get("field") == "text_hash":
            blocked_text.add(item["sha256"])
        elif item.get("field") == "premise_hash":
            blocked_premise.add(item["sha256"])
        else:
            raise ValueError("denylist contains an unknown hash field")
    if not blocked_text or not blocked_premise:
        raise ValueError("typed pilot denylist is incomplete")
    candidates = {0: [], 1: []}
    for index, source in enumerate(view):
        label = int(source["label"])
        if label not in (0, 1, 2):
            continue
        digest = prompt_digest(source["premise"], source["hypothesis"])
        premise_sha256 = premise_digest(source["premise"])
        if digest in blocked_text or premise_sha256 in blocked_premise:
            continue
        candidates[int(label != 0)].append((digest, premise_sha256, index, label))
    for rows in candidates.values():
        rows.sort(key=lambda row: (row[0], row[1]))
    selected = []
    for cls in (0, 1):
        for rank, (digest, premise_sha256, index, source_label) in enumerate(candidates[cls][:per_class]):
            selected.append({"dataset_split": "validation", "dataset_index": index, "prompt_sha256": digest,
                             "premise_sha256": premise_sha256,
                             "source_label": source_label, "label": cls, "class_rank": rank})
    selected.sort(key=lambda row: (row["label"], row["class_rank"]))
    if len(selected) != 2 * per_class or len({r["prompt_sha256"] for r in selected}) != len(selected):
        raise ValueError("reconstructed cohort size/uniqueness differs")
    return selected


def audit_model(bundle, spec, selected, view, deadline):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / spec["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    actions = {name: tokenizer.encode(f" {name}", add_special_tokens=False) for name in ("A", "B")}
    if any(len(ids) != 1 for ids in actions.values()) or {k:v[0] for k,v in actions.items()} != spec["action_token_ids"]:
        raise ValueError("independently reconstructed action token IDs differ")
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float32).to("cpu").eval()
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("audit model is not on CPU")
    if rss_bytes() > int(spec["compute_limits"]["max_peak_rss_bytes_process"]):
        raise RuntimeError("audit model load exceeded frozen memory budget")
    actual_rows = []
    prompts = []
    for meta in selected:
        source = view[int(meta["dataset_index"])]
        digest = prompt_digest(source["premise"], source["hypothesis"])
        if digest != meta["prompt_sha256"] or int(source["label"]) != meta["source_label"]:
            raise ValueError("selected source row text/label differs")
        actual_rows.append({**meta, "premise": source["premise"], "hypothesis": source["hypothesis"], "prompt": prompt_text(source["premise"], source["hypothesis"], digest)})
        prompts.append(actual_rows[-1]["prompt"])
    stored_rows = read_json(bundle / "selected_examples.json")
    if actual_rows != stored_rows:
        raise ValueError("serialized selected prompts differ from independently rebuilt rows")
    saved_scores = read_json(bundle / "scores.json")
    if len(saved_scores) != len(actual_rows):
        raise ValueError("stored score count differs from cohort")
    logits_saved = []
    with torch.inference_mode():
        size = int(spec["compute_limits"]["feature_batch_size"])
        for start in range(0, len(prompts), size):
            if time.monotonic() >= deadline or rss_bytes() > int(spec["compute_limits"]["max_peak_rss_bytes_process"]):
                raise TimeoutError("independent base-forward audit exceeded a frozen resource budget")
            batch = tokenizer(prompts[start:start + size], return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
            logits = model(**batch).logits[:, -1, :]
            for offset, row in enumerate(actual_rows[start:start + size]):
                a, b = float(logits[offset, spec["action_token_ids"]["A"]]), float(logits[offset, spec["action_token_ids"]["B"]])
                option_a, option_b = option_order(row["prompt_sha256"])
                pred_letter = "A" if a >= b else "B"
                pred_label = 0 if (option_a if pred_letter == "A" else option_b) == "entailment" else 1
                gold_letter = ("A" if option_a == "entailment" else "B") if row["label"] == 0 else ("A" if option_a != "entailment" else "B")
                logits_saved.append({"dataset_index": row["dataset_index"], "prompt_sha256": row["prompt_sha256"], "label": row["label"],
                                    "gold_letter": gold_letter, "predicted_letter": pred_letter, "predicted_label": pred_label,
                                    "logit_a": a, "logit_b": b, "entailment_minus_not_entailment_logit": a-b if option_a == "entailment" else b-a,
                                    "action_token_ids": spec["action_token_ids"]})
    for index, (actual, stored) in enumerate(zip(logits_saved, saved_scores)):
        for key in ("dataset_index", "prompt_sha256", "label", "gold_letter", "predicted_letter", "predicted_label", "action_token_ids"):
            if actual[key] != stored[key]:
                raise ValueError(f"independent score record differs at {index}/{key}")
        for key in ("logit_a", "logit_b", "entailment_minus_not_entailment_logit"):
            close(actual[key], stored[key], tolerance=2e-5)
    return logits_saved


def independent_metrics(scores, spec):
    labels = [int(row["label"]) for row in scores]
    preds = [int(row["predicted_label"]) for row in scores]
    supports = {cls: [i for i, y in enumerate(labels) if y == cls] for cls in (0, 1)}
    recalls = {cls: sum(preds[i] == cls for i in supports[cls]) / len(supports[cls]) for cls in (0, 1)}
    rng = random.Random(int(spec["metrics"]["paired_bootstrap"]["seed"]))
    bootstrap = []
    for _ in range(int(spec["metrics"]["paired_bootstrap"]["resamples"])):
        sample = [i for cls in (0, 1) for i in rng.choices(supports[cls], k=len(supports[cls]))]
        class_acc = [sum(preds[i] == cls for i in sample if labels[i] == cls) / len(supports[cls]) for cls in (0, 1)]
        bootstrap.append(sum(class_acc) / 2)
    bootstrap.sort()
    lo, hi = bootstrap[int(0.025 * len(bootstrap))], bootstrap[min(len(bootstrap)-1, int(0.975 * len(bootstrap)))]
    ba = sum(recalls.values()) / 2
    gate = spec["metrics"]["feasibility_gate"]
    checks = {
        "minimum_balanced_accuracy_pass": ba >= float(gate["minimum_balanced_accuracy"]),
        "minimum_entailment_recall_pass": recalls[0] >= float(gate["minimum_recall_per_class"]),
        "minimum_not_entailment_recall_pass": recalls[1] >= float(gate["minimum_recall_per_class"]),
        "bootstrap_lower_bound_above_chance_pass": lo > float(gate["bootstrap_95_percent_lower_bound_above"]),
    }
    confusion = [[sum(y == 0 and p == 0 for y,p in zip(labels,preds)), sum(y == 0 and p == 1 for y,p in zip(labels,preds))],
                 [sum(y == 1 and p == 0 for y,p in zip(labels,preds)), sum(y == 1 and p == 1 for y,p in zip(labels,preds))]]
    return {"examples": len(labels), "accuracy": sum(p == y for p,y in zip(preds,labels))/len(labels),
            "balanced_accuracy": ba, "entailment_recall": recalls[0], "not_entailment_recall": recalls[1],
            "confusion_matrix_true_rows_predicted_columns": confusion,
            "balanced_accuracy_bootstrap_95_percent_interval": [lo,hi], "checks": checks,
            "result": "pass" if all(checks.values()) else "non-pass"}


def audit(bundle):
    started = time.monotonic()
    bundle = bundle.resolve()
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if (bundle / "failure.json").exists():
        raise ValueError("study runner contains a failure record")
    manifest_count = verify_manifest(bundle)
    fingerprints = read_json(bundle / "input_fingerprints.json")
    summary = read_json(bundle / "summary.json")
    spec, protocol_digest, denylist = verify_locks_and_sources(bundle, fingerprints["source"]["commit"])
    if summary["protocol_sha256"] != protocol_digest or fingerprints["protocol_sha256"] != protocol_digest:
        raise ValueError("protocol digest differs across result records")
    import datasets, safetensors, tokenizers, torch, transformers
    runtime = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
               "datasets": datasets.__version__, "numpy": np.__version__, "tokenizers": tokenizers.__version__, "safetensors": safetensors.__version__}
    if runtime != spec["runtime_lock"] or runtime != fingerprints["runtime"]:
        raise ValueError("auditor runtime differs from locked runtime")
    torch.set_num_threads(int(spec["compute_limits"]["threads"]))
    if torch.cuda.is_initialized():
        raise RuntimeError("independent audit refuses an initialized CUDA runtime")
    if fingerprints["source"].get("working_tree_clean") is not True or summary["source_commit"] != fingerprints["source"]["commit"]:
        raise ValueError("source commit provenance differs")
    from datasets import load_dataset
    dataset_spec = spec["dataset"]
    if fingerprints.get("dataset_split") != "validation":
        raise ValueError("recorded dataset split is not the locked validation split")
    dataset = load_dataset(dataset_spec["id"], dataset_spec["configuration"], split="validation", revision=dataset_spec["revision"])
    if len(dataset) != dataset_spec["validation_split"]["rows"] or dataset._fingerprint != dataset_spec["validation_split"]["fingerprint"]:
        raise ValueError("validation split fingerprint differs")
    selected = reconstruct_rows(dataset, denylist, 256)
    recorded = read_json(bundle / "cohort_selection.json")
    if selected != recorded:
        raise ValueError("rank-selected validation cohort differs")
    data_dir = Path.home() / ".cache/huggingface/datasets/stanfordnlp___snli/plain_text/0.0.0" / dataset_spec["revision"]
    model_dir = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots" / spec["model"]["revision"]
    expected_cache = {"validation_arrow": dataset_spec["validation_split"]["arrow_sha256"]}
    for filename, digest in spec["model"]["snapshot_file_sha256"].items():
        expected_cache[f"model_snapshot/{filename}"] = digest
    if fingerprints["cached_dataset_and_model_sha256"] != expected_cache:
        raise ValueError("source hash inventory differs from protocol")
    actual_paths = {"validation_arrow": data_dir / dataset_spec["validation_split"]["arrow_file"]}
    actual_paths.update({f"model_snapshot/{name}": model_dir / name for name in spec["model"]["snapshot_file_sha256"]})
    for key,path in actual_paths.items():
        if not path.is_file() or sha256_file(path) != expected_cache[key]:
            raise ValueError(f"pinned local input hash mismatch: {key}")
    deadline = started + int(spec["compute_limits"]["max_wall_seconds_process"])
    replayed = audit_model(bundle, spec, selected, dataset, deadline)
    metrics = independent_metrics(replayed, spec)
    stored = summary["metrics"]
    resources = summary["resource_measurements"]
    limits = spec["compute_limits"]
    recorded_resource_pass = (
        resources.get("device") == "cpu"
        and resources.get("network_offline") is True
        and resources.get("model_updates") == 0
        and resources.get("paid_compute") is False
        and resources.get("cpu_threads") == int(limits["threads"])
        and resources.get("wall_seconds", float("inf")) <= limits["max_wall_seconds_process"]
        and resources.get("peak_rss_bytes", float("inf")) <= limits["max_peak_rss_bytes_process"]
    )
    metrics["checks"]["resource_limits_pass"] = recorded_resource_pass
    metrics["checks"]["tokenizer_action_ids_pass"] = fingerprints.get("action_token_ids") == spec["action_token_ids"]
    metrics["result"] = "pass" if all(metrics["checks"].values()) else "non-pass"
    for key,value in metrics.items():
        if isinstance(value, dict) or isinstance(value, list) or isinstance(value, str) or isinstance(value, int):
            if stored.get(key) != value:
                raise ValueError(f"reconstructed summary metric differs: {key}")
        else:
            close(stored.get(key), value)
    if stored["checks"] != metrics["checks"] or stored["result"] != metrics["result"]:
        raise ValueError("frozen feasibility gate differs")
    if not recorded_resource_pass or not metrics["checks"]["tokenizer_action_ids_pass"]:
        raise ValueError("recorded study violated a frozen compute or tokenizer check")
    audit_wall = time.monotonic() - started
    peak_bytes = rss_bytes()
    if audit_wall > limits["max_wall_seconds_process"] or peak_bytes > limits["max_peak_rss_bytes_process"]:
        raise RuntimeError("independent audit exceeded its resource ceiling")
    return {"audit":"pass", "protocol_id":spec["protocol_id"], "protocol_sha256":protocol_digest,
            "source_commit":summary["source_commit"], "manifest_files_verified":manifest_count,
            "cohort_rows_reconstructed":len(selected), "selected_prompt_scores_replayed":len(replayed),
            "metrics_gate_reconstructed":metrics, "model_forward_replay":True,
            "study_resource_record_verified":True, "audit_wall_seconds":audit_wall, "audit_peak_rss_bytes":peak_bytes,
            "audit_scope":"Same-host replay with a separate audit implementation; not external reproduction."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = audit(args.bundle)
    (args.bundle / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = {p.relative_to(args.bundle).as_posix(): sha256_file(p) for p in sorted(args.bundle.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (args.bundle / "manifest.json").write_text(json.dumps({"algorithm":"sha256", "files":files}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
