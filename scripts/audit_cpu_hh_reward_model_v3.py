#!/usr/bin/env python3
"""Independently replay the frozen HH v3 selection, fit, calibration and gate."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cpu_hh_reward_model as shared
from hh_reward_task import load_locked_protocol, sha256_file, write_json

SPEC_PATH = ROOT / "protocols/cpu_hh_reward_model_v3.json"
LOCK_PATH = ROOT / "protocols/cpu_hh_reward_model_v3.lock.json"
HISTORICAL_BUNDLES = {
    "v1_development": ROOT / "results/cpu-hh-reward-model-v1/development/run-1",
    "v2_development": ROOT / "results/cpu-hh-reward-model-v2/development/run-1",
    "v2_confirmation": ROOT / "results/cpu-hh-reward-model-v2/confirmation/run-1",
}
MARKER = "\n\nAssistant:"
DOMAIN = b"vare-hh-helpful-context-v1\0"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(value)))))


def close(left, right, tolerance=2e-4):
    return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance


def parse_pair(raw, tokenizer, maximum):
    chosen, rejected = raw.get("chosen"), raw.get("rejected")
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    a, b = chosen.rfind(MARKER), rejected.rfind(MARKER)
    if a < 0 or b < 0:
        return None
    context = chosen[:a + len(MARKER)]
    if context != rejected[:b + len(MARKER)]:
        return None
    if not chosen[a + len(MARKER):] or not rejected[b + len(MARKER):]:
        return None
    try:
        left = tokenizer(chosen, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
        right = tokenizer(rejected, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    except Exception:
        return None
    if len(left["input_ids"]) > maximum or len(right["input_ids"]) > maximum:
        return None
    lp = [i for i, (start, end) in enumerate(left["offset_mapping"])
          if end > start and start >= len(context)]
    rp = [i for i, (start, end) in enumerate(right["offset_mapping"])
          if end > start and start >= len(context)]
    if not lp or not rp:
        return None
    digest = hashlib.sha256(DOMAIN + context.encode("utf-8")).hexdigest()
    return {"context": context, "context_hash": digest, "chosen": chosen, "rejected": rejected,
        "chosen_tokens": len(lp), "rejected_tokens": len(rp), "chosen_ids": left["input_ids"],
        "rejected_ids": right["input_ids"], "chosen_positions": lp, "rejected_positions": rp}


def historical_hashes(spec):
    selection = spec["dataset"]["selection"]
    observed = set()
    for name, path in HISTORICAL_BUNDLES.items():
        expected = selection["historical_bundles"][name]
        if (sha256_file(path / "manifest.json") != expected["manifest_sha256"] or
                sha256_file(path / "pairs.json") != expected["pairs_sha256"]):
            raise ValueError("historical HH bundle differs from the frozen inventory: %s" % name)
        manifest = load(path / "manifest.json")
        if manifest.get("files", {}).get("pairs.json") != expected["pairs_sha256"]:
            raise ValueError("historical manifest does not bind its pair inventory: %s" % name)
        for row in load(path / "pairs.json")["rows"]:
            observed.add(row["context_hash"])
        for cohort in load(path / "summary.json")["training_selection"]:
            observed.update(row["context_hash"] for row in cohort)
    observed.update(selection["manual_pilot_context_hashes"])
    if observed != set(selection["historical_context_hashes"]):
        raise ValueError("historical context-hash inventory mismatch")
    serialized = json.dumps(sorted(observed), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if hashlib.sha256(serialized).hexdigest() != selection["historical_context_inventory_sha256"]:
        raise ValueError("historical context inventory digest mismatch")
    return observed


def select_training(spec, tokenizer, old_hashes):
    count = spec["dataset"]["expected_split_rows"]["train"]
    max_tokens = spec["compute_limits"]["max_sequence_tokens"]
    pilot_ids = set(spec["dataset"]["manual_pilot_train_row_indices_excluded"])
    pilot_hashes, unique = set(), {}
    with gzip.open(str(shared.DATA_DIR / "train.jsonl.gz"), "rt", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index >= count:
                break
            row = json.loads(line)
            item = parse_pair(row, tokenizer, max_tokens)
            if item is None:
                if index in pilot_ids:
                    raise ValueError("a manual pilot row is no longer eligible")
                continue
            if index in pilot_ids:
                pilot_hashes.add(item["context_hash"])
            if item["context_hash"] in old_hashes or index in pilot_ids:
                continue
            if item["context_hash"] not in unique:
                item["source_index"] = index
                unique[item["context_hash"]] = item
    for digest in pilot_hashes:
        unique.pop(digest, None)
    ranked = sorted(unique.values(), key=lambda row: (
        hashlib.sha256(DOMAIN + row["context"].encode("utf-8")).digest(),
        row["context"].encode("utf-8")))
    per_head = spec["learner"]["pairs_per_head"]
    total = per_head * spec["learner"]["heads"]
    if len(ranked) < total:
        raise ValueError("not enough fresh eligible training pairs")
    picked = ranked[:total]
    return [picked[i:i + per_head] for i in range(0, total, per_head)], pilot_hashes


def select_evaluation(spec, tokenizer, stage, cohorts, old_hashes, pilot_hashes, development=None):
    start, end = spec["dataset"]["selection"][stage + "_test_indices"]
    excluded = set(old_hashes) | set(pilot_hashes)
    for cohort in cohorts:
        excluded.update(row["context_hash"] for row in cohort)
    if development:
        excluded.update(row["context_hash"] for row in development)
    seen, accepted = set(), []
    with gzip.open(str(shared.DATA_DIR / "test.jsonl.gz"), "rt", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index < start:
                continue
            if index >= end:
                break
            row = json.loads(line)
            item = parse_pair(row, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
            if item is None or item["context_hash"] in excluded or item["context_hash"] in seen:
                continue
            item["source_index"] = index
            seen.add(item["context_hash"])
            accepted.append(item)
    if len(accepted) < spec["dataset"]["selection"]["minimum_test_pairs"]:
        raise ValueError("fixed %s range has fewer than the locked minimum eligible contexts" % stage)
    return accepted


def extract_features(rows, model, torch):
    output = []
    with torch.no_grad():
        for row in rows:
            sides = []
            for ids, positions in ((row["chosen_ids"], row["chosen_positions"]),
                                   (row["rejected_ids"], row["rejected_positions"])):
                input_ids = torch.tensor([ids], dtype=torch.long)
                hidden = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                     use_cache=False).last_hidden_state[0]
                sides.append(hidden[torch.tensor(positions, dtype=torch.long)].mean(0).float())
            output.append(torch.stack(sides))
    return torch.stack(output)


def fit_head(features, lengths, spec, torch):
    import torch.nn.functional as F
    flat = features.reshape(-1, features.shape[-1])
    mean, scale = flat.mean(0), flat.std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    x = (features[:, 0] - mean) / scale - (features[:, 1] - mean) / scale
    opt = spec["learner"]["optimizer"]
    config = {"lr": 1.0, "max_iter": opt["max_iterations"], "tolerance_grad": opt["tolerance_grad"],
        "tolerance_change": opt["tolerance_change"], "history_size": opt["history_size"],
        "line_search_fn": opt["line_search"]}
    weight = torch.nn.Parameter(torch.zeros(x.shape[-1]))
    optimizer = torch.optim.LBFGS([weight], **config)
    response_length = lengths[:, 0] - lengths[:, 1]
    length_mean, length_scale = response_length.mean(), response_length.std(unbiased=False)
    if length_scale < 1e-6:
        length_scale = torch.tensor(1.0)
    y = (response_length - length_mean) / length_scale
    baseline_weight = torch.nn.Parameter(torch.zeros(1))
    baseline_optimizer = torch.optim.LBFGS([baseline_weight], **config)
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.softplus(-(x @ weight)).mean() + 0.5 * spec["learner"]["regularization"] * (weight @ weight)
        loss.backward()
        return loss
    def baseline_closure():
        baseline_optimizer.zero_grad(set_to_none=True)
        loss = F.softplus(-(y * baseline_weight[0])).mean() + 0.5 * spec["learner"]["regularization"] * baseline_weight[0] ** 2
        loss.backward()
        return loss
    optimizer.step(closure)
    baseline_optimizer.step(baseline_closure)
    return (weight.detach(), mean.detach(), scale.detach(), float(baseline_weight.detach()[0]),
            float(length_mean), float(length_scale))


def fit_alpha(margins, spec, torch):
    import torch.nn.functional as F
    cfg = spec["learner"]["calibration"]
    theta = torch.nn.Parameter(torch.tensor(cfg["initial_theta"], dtype=torch.float32))
    optimizer = torch.optim.LBFGS([theta], lr=cfg["learning_rate"], max_iter=cfg["max_iterations"],
        tolerance_grad=cfg["tolerance_grad"], tolerance_change=cfg["tolerance_change"],
        history_size=cfg["history_size"], line_search_fn=cfg["line_search"])
    calls = [0]
    def closure():
        optimizer.zero_grad(set_to_none=True)
        alpha = cfg["maximum_inverse_temperature"] * torch.sigmoid(theta)
        loss = F.softplus(-alpha * margins).mean()
        loss.backward()
        calls[0] += 1
        return loss
    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    alpha = cfg["maximum_inverse_temperature"] * torch.sigmoid(theta)
    after = F.softplus(-alpha * margins).mean()
    after.backward()
    before = F.softplus(-margins).mean()
    return float(alpha.detach()), {"nll_before": float(before.detach()), "nll_after": float(after.detach()),
        "iterations": int(optimizer.state[theta].get("n_iter", 0)), "closure_calls": calls[0],
        "gradient_norm": float(torch.linalg.vector_norm(theta.grad).item())}


def metric_summary(margins):
    if not margins:
        raise ValueError("empty margins")
    probs = [sigmoid(value) for value in margins]
    accuracy = sum(1.0 if x > 0 else 0.0 if x < 0 else 0.5 for x in margins) / len(margins)
    nll = sum(max(-x, 0.0) + math.log1p(math.exp(-abs(x))) for x in margins) / len(margins)
    brier = sum((p - 1.0) ** 2 for p in probs) / len(probs)
    ece = 0.0
    for index in range(10):
        chosen = [p for p in probs if index / 10 <= p < (index + 1) / 10 or (index == 9 and p == 1.0)]
        if chosen:
            ece += len(chosen) / len(probs) * abs(sum(chosen) / len(chosen) - 1.0)
    return {"pairwise_accuracy": accuracy, "logistic_nll": nll,
        "brier_score": brier, "expected_calibration_error": ece}


def replay(spec, cohorts, eval_rows, tokenizer, model, torch):
    train_features = [extract_features(cohort, model, torch) for cohort in cohorts]
    eval_features = extract_features(eval_rows, model, torch)
    per_head = []
    fit_count = spec["learner"]["head_fit_pairs"]
    for head, rows in enumerate(cohorts):
        features = train_features[head]
        fit_features, cal_features = features[:fit_count], features[fit_count:]
        fit_lengths = torch.tensor([[r["chosen_tokens"], r["rejected_tokens"]] for r in rows[:fit_count]], dtype=torch.float32)
        w, mean, scale, base_weight, length_mean, length_scale = fit_head(fit_features, fit_lengths, spec, torch)
        cal_diff = ((cal_features[:, 0] - mean) / scale - (cal_features[:, 1] - mean) / scale)
        alpha, calibration = fit_alpha(cal_diff @ w, spec, torch)
        eval_diff = ((eval_features[:, 0] - mean) / scale - (eval_features[:, 1] - mean) / scale)
        raw = (eval_diff @ w).tolist()
        calibrated = [alpha * float(value) for value in raw]
        eval_lengths = torch.tensor([[row["chosen_tokens"], row["rejected_tokens"]] for row in eval_rows], dtype=torch.float32)
        baseline = ((((eval_lengths[:, 0] - eval_lengths[:, 1]) - length_mean) / length_scale) * base_weight).tolist()
        per_head.append({"alpha": alpha, "calibration": calibration, "raw": raw,
            "calibrated": calibrated, "baseline": baseline})
    return per_head


def bootstrap(values, seed, resamples):
    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(data), size=(resamples, len(data)))
    means = data[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975], method="linear")]


def audit(bundle: Path):
    bundle = bundle.resolve()
    stage = bundle.parent.name
    if stage not in ("development", "confirmation"):
        raise ValueError("bundle must be under development or confirmation")
    spec, protocol_hash = load_locked_protocol(SPEC_PATH, LOCK_PATH)
    manifest = load(bundle / "manifest.json")
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
              if p.is_file() and p.name not in ("manifest.json", "audit.json")}
    expected = {"protocol.snapshot.json", "protocol.lock.snapshot.json", "runner.snapshot.py",
        "auditor.snapshot.py", "shared_runner.snapshot.py", "shared_helper.snapshot.py"}
    if (bundle / "failure.json").exists():
        expected.add("failure.json")
    else:
        expected.update(("pairs.json", "summary.json", "environment.json"))
        if stage == "confirmation":
            expected.add("development-audit.snapshot.json")
    if actual != expected or set(manifest.get("files", {})) != actual:
        raise ValueError("result bundle has missing or unapproved files")
    for name, digest in manifest["files"].items():
        if sha256_file(bundle / name) != digest:
            raise ValueError("manifest digest mismatch: %s" % name)
    if load(bundle / "protocol.snapshot.json") != spec or load(bundle / "protocol.lock.snapshot.json") != load(LOCK_PATH):
        raise ValueError("protocol snapshot differs from current frozen protocol")
    current = {"runner.snapshot.py": ROOT / "scripts/run_cpu_hh_reward_model_v3.py",
        "auditor.snapshot.py": Path(__file__), "shared_runner.snapshot.py": Path(shared.__file__),
        "shared_helper.snapshot.py": ROOT / "scripts/hh_reward_task.py"}
    for snapshot, source in current.items():
        if (bundle / snapshot).read_bytes() != source.read_bytes():
            raise ValueError("code snapshot differs from current source: %s" % snapshot)
    if (bundle / "failure.json").exists():
        failure = load(bundle / "failure.json")
        report = {"status": "failed_attempt_preserved", "stage": stage, "protocol_sha256": protocol_hash,
            "manifest_sha256": sha256_file(bundle / "manifest.json"), "decision": {"pass": False},
            "failure_type": failure.get("exception_type"), "failure_message": failure.get("message")}
        write_json(bundle / "audit.json", report)
        return report
    summary, pair_file = load(bundle / "summary.json"), load(bundle / "pairs.json")
    if summary.get("protocol_sha256") != protocol_hash or summary.get("stage") != stage:
        raise ValueError("summary protocol/stage differs")
    if summary.get("evaluation_source_range") != spec["dataset"]["selection"][stage + "_test_indices"]:
        raise ValueError("evaluation source block differs from lock")
    if summary.get("test_confirmation_row_records_parsed") is not (stage == "confirmation"):
        raise ValueError("confirmation access indicator differs from stage")
    if summary.get("network_disabled") is not True or summary.get("paid_compute") is not False or summary.get("device") != "cpu":
        raise ValueError("runtime policy differs from lock")
    if (summary.get("test_source_container_hashed_opaque") is not True or
            summary.get("cuda_initialized") is not False or
            summary.get("torch_threads") != spec["compute_limits"]["threads"] or
            summary.get("compute_limits_respected") is not True):
        raise ValueError("data-access or compute provenance differs from the lock")
    if summary.get("peak_rss_bytes", 0) > spec["compute_limits"]["max_peak_rss_bytes"] or summary.get("total_wall_seconds", 0) > spec["compute_limits"]["max_wall_seconds"]:
        raise ValueError("resource limit exceeded")
    if stage == "confirmation":
        dev_audit = load(bundle / "development-audit.snapshot.json")
        if (dev_audit.get("status") != "pass" or dev_audit.get("protocol_sha256") != protocol_hash or
            dev_audit.get("decision", {}).get("pass") is not True or
            summary.get("development_audit_sha256") != sha256_file(bundle / "development-audit.snapshot.json")):
            raise ValueError("confirmation lacks a passing audited development")

    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    old_hashes = historical_hashes(spec)
    tokenizer, model, torch, runtime, model_hashes, data_hashes = shared.load_runtime(spec)
    if (summary.get("runtime") != runtime or summary.get("model_file_sha256") != model_hashes or
        summary.get("dataset_file_sha256") != data_hashes):
        raise ValueError("runtime, model, or source data hashes differ")
    environment = load(bundle / "environment.json")
    if (environment.get("runtime") != runtime or environment.get("model_file_sha256") != model_hashes or
            environment.get("dataset_file_sha256") != data_hashes or environment.get("device") != "cpu" or
            environment.get("network_disabled") is not True or
            environment.get("torch_threads") != spec["compute_limits"]["threads"]):
        raise ValueError("environment record differs from the independently observed runtime")
    cohorts, pilot_hashes = select_training(spec, tokenizer, old_hashes)
    expected_train = [[{"source_index": r["source_index"], "context_hash": r["context_hash"]} for r in c] for c in cohorts]
    frozen_train = spec["dataset"]["selection"]["frozen_training_selection"]
    frozen_digest = hashlib.sha256(json.dumps(frozen_train, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if expected_train != frozen_train or frozen_digest != spec["dataset"]["selection"]["training_selection_sha256"]:
        raise ValueError("independent training selection differs from frozen inventory")
    if summary.get("training_selection") != expected_train or summary.get("pilot_context_hashes") != sorted(pilot_hashes):
        raise ValueError("fresh training selection differs")
    dev_rows = select_evaluation(spec, tokenizer, "development", cohorts, old_hashes, pilot_hashes) if stage == "confirmation" else None
    eval_rows = select_evaluation(spec, tokenizer, stage, cohorts, old_hashes, pilot_hashes, dev_rows)
    result_rows = pair_file.get("rows", [])
    if len(result_rows) != len(eval_rows) or len(eval_rows) < spec["dataset"]["selection"]["minimum_test_pairs"]:
        raise ValueError("eligible evaluation count differs")
    replayed = replay(spec, cohorts, eval_rows, tokenizer, model, torch)
    cohort_summaries, per_prompt = [], [[] for _ in eval_rows]
    for head, (expected_head, actual_head) in enumerate(zip(replayed, summary["cohorts"])):
        alpha = expected_head["alpha"]
        if not close(actual_head["temperature_alpha"], alpha, 1e-6):
            raise ValueError("replayed calibration scalar differs")
        calibration = actual_head.get("calibration", {})
        if calibration.get("pairs") != spec["learner"]["calibration_fit_pairs"]:
            raise ValueError("calibration sample count differs")
        for field in ("nll_before", "nll_after", "iterations", "closure_calls", "gradient_norm"):
            if not math.isfinite(float(calibration[field])):
                raise ValueError("invalid calibration diagnostic: %s" % field)
            if not close(calibration[field], expected_head["calibration"][field], 1e-6):
                raise ValueError("replayed calibration diagnostic differs: %s" % field)
        raw_summary, cal_summary = metric_summary(expected_head["raw"]), metric_summary(expected_head["calibrated"])
        baseline_summary = metric_summary(expected_head["baseline"])
        for field, recomputed in (("raw_metrics", raw_summary), ("calibrated_metrics", cal_summary)):
            for key, value in recomputed.items():
                if not close(actual_head[field][key], value, 1e-8):
                    raise ValueError("per-head metric mismatch: %s.%s" % (field, key))
        for key, value in baseline_summary.items():
            if not close(actual_head["length_baseline_metrics"][key], value, 1e-8):
                raise ValueError("length baseline metric mismatch: %s" % key)
        raw_vector = actual_head.get("uncalibrated_reward_margins", [])
        cal_vector = actual_head.get("reward_margins", [])
        if len(raw_vector) != len(eval_rows) or len(cal_vector) != len(eval_rows):
            raise ValueError("head margin arrays have wrong size")
        deltas = []
        baseline_vector = actual_head.get("baseline_margins", [])
        if len(baseline_vector) != len(eval_rows):
            raise ValueError("length baseline margin vector has wrong size")
        for i, (raw, cal) in enumerate(zip(expected_head["raw"], expected_head["calibrated"])):
            if not close(raw_vector[i], raw) or not close(cal_vector[i], cal) or not close(cal, alpha * raw, 1e-6):
                raise ValueError("raw/calibrated margins do not reconstruct")
            if (raw > 0) != (cal > 0):
                raise ValueError("positive calibration changed pairwise ranking")
            delta = max(-cal, 0.0) + math.log1p(math.exp(-abs(cal))) - (max(-raw, 0.0) + math.log1p(math.exp(-abs(raw))))
            deltas.append(delta)
            if not close(baseline_vector[i], expected_head["baseline"][i]):
                raise ValueError("length baseline margin does not reconstruct")
            per_prompt[i].append(delta)
            row = result_rows[i]
            if row.get("source_index") != eval_rows[i]["source_index"] or row.get("context_hash") != eval_rows[i]["context_hash"]:
                raise ValueError("evaluation source row/hash differs")
            if (row.get("chosen_response_tokens") != eval_rows[i]["chosen_tokens"] or
                    row.get("rejected_response_tokens") != eval_rows[i]["rejected_tokens"]):
                raise ValueError("evaluation response token counts differ")
            pred = row.get("cohorts", [])[head]
            for key, value in (("raw_margin", raw), ("calibrated_margin", cal),
                               ("raw_probability", sigmoid(raw)), ("calibrated_probability", sigmoid(cal))):
                if not close(pred.get(key), value, 1e-8):
                    raise ValueError("per-prompt prediction mismatch: %s" % key)
            if pred.get("raw_accuracy") != pred.get("calibrated_accuracy"):
                raise ValueError("raw/calibrated accuracy changed")
        delta_mean = sum(deltas) / len(deltas)
        if len(actual_head.get("nll_delta_per_pair", [])) != len(deltas) or any(
                not close(a, b, 1e-8) for a, b in zip(actual_head["nll_delta_per_pair"], deltas)):
            raise ValueError("per-prompt NLL deltas differ")
        if not close(actual_head.get("mean_calibrated_minus_raw_nll"), delta_mean, 1e-8):
            raise ValueError("per-head NLL delta mismatch")
        cohort_summaries.append({"head": head, "alpha": alpha, "raw_metrics": raw_summary,
            "calibrated_metrics": cal_summary, "length_baseline_metrics": baseline_summary,
            "mean_calibrated_minus_raw_nll": delta_mean})
    prompt_delta = [sum(values) / len(values) for values in per_prompt]
    interval = bootstrap(prompt_delta, spec["metrics"]["bootstrap"]["seed"], spec["metrics"]["bootstrap"]["resamples"])
    mean_delta = sum(prompt_delta) / len(prompt_delta)
    passed = (len(prompt_delta) >= spec["dataset"]["selection"]["minimum_test_pairs"] and
        mean_delta <= -spec["metrics"]["minimum_mean_nll_improvement"] and interval[1] < 0.0)
    decision = {"pass": bool(passed), "prompt_count": len(prompt_delta),
        "mean_calibrated_minus_raw_nll": mean_delta, "paired_prompt_bootstrap_95pct": interval,
        "confirmation_allowed": bool(stage == "development" and passed)}
    if summary.get("decision") != decision:
        for key, value in decision.items():
            observed = summary.get("decision", {}).get(key)
            if isinstance(value, list):
                if len(observed or []) != 2 or any(not close(a, b, 1e-10) for a, b in zip(observed, value)):
                    raise ValueError("frozen decision interval mismatch")
            elif isinstance(value, float):
                if not close(observed, value, 1e-10):
                    raise ValueError("frozen decision metric mismatch")
            elif observed != value:
                raise ValueError("frozen decision mismatch: %s" % key)
    report = {"status": "pass", "protocol_sha256": protocol_hash,
        "audit_implementation_sha256": sha256_file(Path(__file__)),
        "run_auditor_snapshot_sha256": summary["auditor_sha256"],
        "manifest_sha256": sha256_file(bundle / "manifest.json"), "stage": stage,
        "decision": decision, "cohort_metrics": cohort_summaries,
        "model_score_replay": "passed; separately implemented transcript parsing, eligibility, context selection, frozen-feature extraction, reward-head fitting, temperature fitting and per-prompt NLL reproduced on the same local model/runtime",
        "audit_limit": "Local replay shares the pinned model weights, tokenizer, runtime, and host; it is not external reproduction or a different model implementation."}
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
