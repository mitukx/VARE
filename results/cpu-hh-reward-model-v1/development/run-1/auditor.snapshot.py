#!/usr/bin/env python3
"""Independently audit HH reward-model selection, provenance, metrics, and gate."""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, platform, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hh_reward_task import canonical, load_locked_protocol, sha256_file, write_json

SPEC_PATH = ROOT / "protocols/cpu_hh_reward_model_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_hh_reward_model_v1.lock.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/hub/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-base"
MARKER = "\n\nAssistant:"
DOMAIN = b"vare-hh-helpful-context-v1\0"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b, tol=1e-8):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def audited_metrics(margins):
    if not margins:
        raise ValueError("empty metric cohort")
    probs = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(x))))) for x in margins]
    result = {"pairwise_accuracy": sum(1.0 if x > 0 else 0.0 if x < 0 else 0.5 for x in margins) / len(margins),
              "logistic_nll": sum(max(-x, 0.0) + math.log1p(math.exp(-abs(x))) for x in margins) / len(margins),
              "brier_score": sum((p - 1.0) ** 2 for p in probs) / len(probs),
              "expected_calibration_error": 0.0}
    for i in range(10):
        low, high = i / 10.0, (i + 1) / 10.0
        values = [p for p in probs if low <= p < high or (i == 9 and p == 1.0)]
        if values:
            result["expected_calibration_error"] += len(values) / len(probs) * abs(sum(values) / len(values) - 1.0)
    return result


def audited_bootstrap(values, seed, resamples):
    import numpy as np
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(data), size=(resamples, len(data)))
    means = data[choices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975], method="linear")]


def scan(path: Path, start: int, end: int):
    with gzip.open(str(path), "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= end:
                return
            if index < start:
                continue
            yield index, json.loads(line)


def parse_independent(raw, tokenizer, maximum):
    chosen, rejected = raw.get("chosen"), raw.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    left, right = chosen.rfind(MARKER), rejected.rfind(MARKER)
    if left < 0 or right < 0:
        return None
    context = chosen[:left + len(MARKER)]
    if context != rejected[:right + len(MARKER)]:
        return None
    if not chosen[left + len(MARKER):] or not rejected[right + len(MARKER):]:
        return None
    try:
        a = tokenizer(chosen, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
        b = tokenizer(rejected, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    except Exception:
        return None
    if len(a["input_ids"]) > maximum or len(b["input_ids"]) > maximum:
        return None
    apos = [i for i, (s, e) in enumerate(a["offset_mapping"]) if e > s and s >= len(context)]
    bpos = [i for i, (s, e) in enumerate(b["offset_mapping"]) if e > s and s >= len(context)]
    if not apos or not bpos:
        return None
    digest = hashlib.sha256(DOMAIN + context.encode("utf-8")).hexdigest()
    return {"source_index": None, "context": context, "context_hash": digest,
            "chosen_tokens": len(apos), "rejected_tokens": len(bpos)}


def reconstruct_training(spec, tokenizer):
    n = spec["dataset"]["expected_split_rows"]["train"]
    rows = list(scan(DATA_DIR / "train.jsonl.gz", 0, n))
    pilot_ids = set(spec["dataset"]["manual_pilot_train_row_indices_excluded"])
    pilot_hashes = set()
    for index in pilot_ids:
        item = parse_independent(rows[index][1], tokenizer, spec["compute_limits"]["max_sequence_tokens"])
        if item is None:
            raise ValueError("manual pilot row is no longer eligible")
        pilot_hashes.add(item["context_hash"])
    unique = {}
    invalid = 0
    for index, raw in rows:
        item = parse_independent(raw, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
        if item is None:
            invalid += 1
            continue
        if item["context_hash"] in pilot_hashes or item["context_hash"] in unique:
            continue
        item["source_index"] = index
        unique[item["context_hash"]] = item
    ranked = sorted(unique.values(), key=lambda x: (
        hashlib.sha256(DOMAIN + x["context"].encode("utf-8")).digest(), x["context"].encode("utf-8")))
    if len(ranked) < 408:
        raise ValueError("training source cannot reconstruct all fixed cohorts")
    selected = ranked[24:408]
    return [selected[i:i+128] for i in (0, 128, 256)], pilot_hashes, invalid


def reconstruct_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, dev_rows=None):
    start, end = spec["dataset"]["selection"][stage + "_test_indices"]
    excluded = set(pilot_hashes)
    for cohort in cohorts:
        excluded.update(row["context_hash"] for row in cohort)
    if dev_rows is not None:
        excluded.update(row["context_hash"] for row in dev_rows)
    chosen, seen = [], set()
    for index, raw in scan(DATA_DIR / "test.jsonl.gz", start, end):
        item = parse_independent(raw, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
        if item is None or item["context_hash"] in excluded or item["context_hash"] in seen:
            continue
        item["source_index"] = index
        seen.add(item["context_hash"])
        chosen.append(item)
        if len(chosen) == spec["dataset"]["selection"][stage + "_test_pairs"]:
            break
    if len(chosen) != spec["dataset"]["selection"][stage + "_test_pairs"]:
        raise ValueError("fixed evaluation range cannot reconstruct its required cohort")
    return chosen


def audit(bundle: Path):
    bundle = bundle.resolve()
    spec, protocol_hash = load_locked_protocol(SPEC_PATH, LOCK_PATH)
    manifest = load(bundle / "manifest.json")
    observed = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
                if p.is_file() and p.name not in ("manifest.json", "audit.json")}
    if set(manifest.get("files", {})) != observed:
        raise ValueError("manifest inventory differs from bundle")
    for name, digest in manifest["files"].items():
        if sha256_file(bundle / name) != digest:
            raise ValueError("bundle file hash mismatch: %s" % name)
    stage = bundle.parent.name
    if stage not in ("development", "confirmation"):
        raise ValueError("bundle path must be under development or confirmation")
    failure = bundle / "failure.json"
    expected_files = {"protocol.snapshot.json", "protocol.lock.snapshot.json",
                      "runner.snapshot.py", "helper.snapshot.py", "auditor.snapshot.py"}
    if failure.exists():
        expected_files.add("failure.json")
        if (bundle / "development-audit.snapshot.json").exists():
            expected_files.add("development-audit.snapshot.json")
    else:
        expected_files.update(("pairs.json", "summary.json", "environment.json"))
        if stage == "confirmation":
            expected_files.add("development-audit.snapshot.json")
    if observed != expected_files:
        raise ValueError("bundle contains missing or unapproved files")
    if failure.exists():
        if any((bundle / name).exists() for name in ("summary.json", "pairs.json")):
            raise ValueError("failure bundle must not also contain model metrics")
        record = load(failure)
        report = {"status": "failed_attempt_preserved", "protocol_sha256": protocol_hash,
                  "manifest_sha256": sha256_file(bundle / "manifest.json"), "stage": stage,
                  "exception_type": record.get("exception_type"), "message": record.get("message"),
                  "decision": {"pass": False}, "audit_limit": "No held-out result is claimed from this failed attempt."}
        write_json(bundle / "audit.json", report)
        return report

    summary = load(bundle / "summary.json")
    pair_file = load(bundle / "pairs.json")
    if set(pair_file) != {"rows"}:
        raise ValueError("pair artifact contains unknown top-level data")
    if summary.get("protocol_sha256") != protocol_hash or summary.get("stage") != stage:
        raise ValueError("summary is not bound to the current protocol and stage")
    if canonical(load(bundle / "protocol.snapshot.json")) != canonical(spec):
        raise ValueError("protocol snapshot differs from the repository lock")
    lock_snapshot = load(bundle / "protocol.lock.snapshot.json")
    current_lock = load(LOCK_PATH)
    if lock_snapshot != current_lock:
        raise ValueError("protocol lock snapshot differs from the current repository lock")
    if summary.get("runner_sha256") != sha256_file(bundle / "runner.snapshot.py"):
        raise ValueError("runner source hash mismatch")
    if summary.get("helper_sha256") != sha256_file(bundle / "helper.snapshot.py"):
        raise ValueError("helper source hash mismatch")
    if (bundle / "runner.snapshot.py").read_bytes() != (ROOT / "scripts/run_cpu_hh_reward_model.py").read_bytes():
        raise ValueError("run was not produced by the current frozen runner source")
    if (bundle / "helper.snapshot.py").read_bytes() != (ROOT / "scripts/hh_reward_task.py").read_bytes():
        raise ValueError("run was not produced by the current frozen helper source")
    if summary.get("auditor_sha256") != sha256_file(bundle / "auditor.snapshot.py"):
        raise ValueError("auditor source hash mismatch")
    if (bundle / "auditor.snapshot.py").read_bytes() != Path(__file__).read_bytes():
        raise ValueError("run was not audited with the current auditor source")
    if summary.get("model_revision") != spec["feature_extractor"]["model_revision"]:
        raise ValueError("model revision mismatch")
    if summary.get("device") != "cpu" or summary.get("cuda_initialized") is not False or summary.get("paid_compute") is not False:
        raise ValueError("compute mode violates the frozen protocol")
    if summary.get("network_disabled") is not True or summary.get("torch_threads") != spec["compute_limits"]["threads"]:
        raise ValueError("offline/thread constraints were not recorded")
    environment = load(bundle / "environment.json")
    if environment.get("runtime") != summary.get("runtime") or environment.get("model_file_sha256") != summary.get("model_file_sha256") or environment.get("dataset_file_sha256") != summary.get("dataset_file_sha256"):
        raise ValueError("environment provenance differs from summary")
    if environment.get("device") != "cpu" or environment.get("network_disabled") is not True or environment.get("torch_threads") != spec["compute_limits"]["threads"]:
        raise ValueError("environment record violates the frozen compute mode")
    if summary.get("runtime") != spec["runtime"]:
        raise ValueError("runtime version mismatch")
    expected_model_hashes = {
        name: sha256_file(MODEL_DIR / name)
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    }
    expected_data_hashes = {
        name: sha256_file(DATA_DIR / name) for name in ("train.jsonl.gz", "test.jsonl.gz")
    }
    if summary.get("model_file_sha256") != expected_model_hashes or summary.get("dataset_file_sha256") != expected_data_hashes:
        raise ValueError("cached model or dataset source hash mismatch")
    if summary.get("peak_rss_bytes", 0) > spec["compute_limits"]["max_peak_rss_bytes"] or summary.get("total_wall_seconds", 0) > spec["compute_limits"]["max_wall_seconds"]:
        raise ValueError("resource ceiling exceeded")
    if summary.get("compute_limits_respected") is not True:
        raise ValueError("resource limits were not confirmed")
    if summary.get("test_confirmation_rows_read") is not (stage == "confirmation"):
        raise ValueError("confirmation source access does not match stage")
    expected_range = spec["dataset"]["selection"][stage + "_test_indices"]
    if summary.get("test_source_range") != expected_range:
        raise ValueError("test range differs from frozen protocol")
    if stage == "confirmation":
        development_audit = load(bundle / "development-audit.snapshot.json")
        if (development_audit.get("status") != "pass" or
                development_audit.get("protocol_sha256") != protocol_hash or
                development_audit.get("decision", {}).get("pass") is not True or
                summary.get("development_audit_sha256") != sha256_file(bundle / "development-audit.snapshot.json")):
            raise ValueError("confirmation is not bound to a passing development audit")

    import os
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
    cohorts, pilot_hashes, _invalid_count = reconstruct_training(spec, tokenizer)
    expected_training = [[{"source_index": r["source_index"], "context_hash": r["context_hash"]} for r in c]
                         for c in cohorts]
    if summary.get("training_selection") != expected_training:
        raise ValueError("training cohort rows differ from independently reconstructed hash ranks")
    if summary.get("pilot_context_hashes") != sorted(pilot_hashes):
        raise ValueError("manual pilot prompt exclusions differ from locked training source")
    dev_rows = reconstruct_evaluation(spec, tokenizer, "development", cohorts, pilot_hashes) if stage == "confirmation" else None
    eval_rows = reconstruct_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, dev_rows)
    rows = pair_file.get("rows")
    if not isinstance(rows, list) or len(rows) != len(eval_rows):
        raise ValueError("result pair count differs from frozen selection")
    allowed_keys = {"source_index", "context_hash", "chosen_response_tokens", "rejected_response_tokens", "cohorts"}
    cohort_keys = {"reward_margin", "reward_probability", "reward_accuracy", "baseline_margin", "baseline_probability", "baseline_accuracy"}
    for actual, expected in zip(rows, eval_rows):
        if set(actual) != allowed_keys:
            raise ValueError("result pair has missing/unknown or potentially sensitive fields")
        if actual["source_index"] != expected["source_index"] or actual["context_hash"] != expected["context_hash"]:
            raise ValueError("evaluation source order/context hash differs from the frozen selection")
        if actual["chosen_response_tokens"] != expected["chosen_tokens"] or actual["rejected_response_tokens"] != expected["rejected_tokens"]:
            raise ValueError("recorded response token counts differ from source")
        if len(actual["cohorts"]) != 3:
            raise ValueError("each evaluation pair must have predictions from all three heads")
        for prediction in actual["cohorts"]:
            if set(prediction) != cohort_keys:
                raise ValueError("pair prediction schema mismatch")
            for name in ("reward_margin", "reward_probability", "baseline_margin", "baseline_probability"):
                if not math.isfinite(float(prediction[name])):
                    raise ValueError("non-finite per-pair metric")
            for prefix in ("reward", "baseline"):
                margin = float(prediction[prefix + "_margin"])
                probability = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, margin))))
                accuracy = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
                if not close(prediction[prefix + "_probability"], probability) or not close(prediction[prefix + "_accuracy"], accuracy):
                    raise ValueError("pair probability/accuracy does not follow its recorded margin")

    if len(summary.get("cohorts", [])) != 3:
        raise ValueError("summary must contain exactly three fitted heads")
    cohort_metrics = []
    deltas = []
    for i in range(3):
        reward_margins = [float(row["cohorts"][i]["reward_margin"]) for row in rows]
        baseline_margins = [float(row["cohorts"][i]["baseline_margin"]) for row in rows]
        reward_metric = audited_metrics(reward_margins)
        baseline_metric = audited_metrics(baseline_margins)
        listed = summary["cohorts"][i]
        if any(not close(a, b) for a, b in zip(listed.get("reward_margins", []), reward_margins)):
            raise ValueError("summary reward margins differ from per-pair records")
        if any(not close(a, b) for a, b in zip(listed.get("baseline_margins", []), baseline_margins)):
            raise ValueError("summary baseline margins differ from per-pair records")
        if len(listed.get("reward_margins", [])) != len(reward_margins) or len(listed.get("baseline_margins", [])) != len(baseline_margins):
            raise ValueError("summary margin vector length differs from pair records")
        for source_name, expected in (("reward_metrics", reward_metric), ("baseline_metrics", baseline_metric)):
            for key, value in expected.items():
                if not close(listed[source_name][key], value):
                    raise ValueError("cohort %d %s.%s does not reconstruct" % (i, source_name, key))
        acc_delta = [float(a["cohorts"][i]["reward_accuracy"]) - float(a["cohorts"][i]["baseline_accuracy"]) for a in rows]
        nll_delta = []
        for rm, base in zip(reward_margins, baseline_margins):
            nll_delta.append(max(-rm, 0.0) + math.log1p(math.exp(-abs(rm))) -
                             (max(-base, 0.0) + math.log1p(math.exp(-abs(base)))))
        if len(listed["accuracy_delta_per_pair"]) != len(rows) or len(listed["nll_delta_per_pair"]) != len(rows):
            raise ValueError("paired delta arrays have the wrong length")
        if any(not close(a, b) for a, b in zip(listed["accuracy_delta_per_pair"], acc_delta)):
            raise ValueError("paired accuracy deltas do not reconstruct")
        if any(not close(a, b) for a, b in zip(listed["nll_delta_per_pair"], nll_delta)):
            raise ValueError("paired NLL deltas do not reconstruct")
        cohort_metrics.append({"cohort": i, "reward": reward_metric, "baseline": baseline_metric})
        deltas.append({"accuracy": acc_delta, "nll": nll_delta})

    n = len(rows)
    avg_acc = [sum(d["accuracy"][j] for d in deltas) / 3 for j in range(n)]
    avg_nll = [sum(d["nll"][j] for d in deltas) / 3 for j in range(n)]
    bootstrap = spec["metrics"]["bootstrap"]
    acc_ci = audited_bootstrap(avg_acc, bootstrap["seed"], bootstrap["resamples"])
    nll_ci = audited_bootstrap(avg_nll, bootstrap["seed"], bootstrap["resamples"])
    mean_acc = sum(m["reward"]["pairwise_accuracy"] for m in cohort_metrics) / 3
    mean_gain = sum(avg_acc) / n
    mean_rm_nll = sum(m["reward"]["logistic_nll"] for m in cohort_metrics) / 3
    mean_base_nll = sum(m["baseline"]["logistic_nll"] for m in cohort_metrics) / 3
    passed = (mean_acc > 0.5 and mean_gain >= 0.02 and acc_ci[0] > 0 and mean_rm_nll < mean_base_nll and
              nll_ci[1] < 0 and sum(m["reward"]["pairwise_accuracy"] > 0.5 for m in cohort_metrics) >= 2)
    expected_decision = {"pass": bool(passed), "mean_reward_accuracy": mean_acc, "mean_accuracy_gain": mean_gain,
                         "accuracy_gain_interval_95pct": acc_ci, "mean_reward_nll": mean_rm_nll,
                         "mean_baseline_nll": mean_base_nll, "nll_difference_interval_95pct": nll_ci,
                         "confirmation_allowed": bool(stage == "development" and passed)}
    observed_decision = summary.get("decision", {})
    if observed_decision.keys() != expected_decision.keys():
        raise ValueError("decision field inventory differs")
    for key, value in expected_decision.items():
        if isinstance(value, list):
            if len(observed_decision[key]) != len(value) or any(not close(a, b) for a, b in zip(observed_decision[key], value)):
                raise ValueError("decision interval does not reconstruct")
        elif isinstance(value, float):
            if not close(observed_decision[key], value):
                raise ValueError("decision metric does not reconstruct: %s" % key)
        elif observed_decision[key] != value:
            raise ValueError("decision differs from frozen gate")
    report = {"status": "pass", "protocol_sha256": protocol_hash,
              "manifest_sha256": sha256_file(bundle / "manifest.json"), "stage": stage,
              "decision": expected_decision, "cohort_metrics": cohort_metrics,
              "evaluation_source_indices": [row["source_index"] for row in eval_rows],
              "audit_limit": "Independently reconstructs split selection, prompt exclusions, response token counts, per-pair metrics, bootstrap intervals, and the gate. It does not rerun transformer forward passes or fit the reward heads; recorded model margins remain runner-produced evidence."}
    write_json(bundle / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    report = audit(args.bundle)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
