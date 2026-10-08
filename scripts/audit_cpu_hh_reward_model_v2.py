#!/usr/bin/env python3
"""Audit v2 data selection, metrics, calibration, and replayed CPU model scores."""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, os, platform, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cpu_hh_reward_model as shared
from hh_reward_task import (canonical, load_locked_protocol, sha256_file, write_json,
                            paired_bootstrap_interval)

SPEC_PATH = ROOT / "protocols/cpu_hh_reward_model_v2.json"
LOCK_PATH = ROOT / "protocols/cpu_hh_reward_model_v2.lock.json"
MODEL_DIR = shared.MODEL_DIR
DATA_DIR = shared.DATA_DIR
MARKER = "\n\nAssistant:"
DOMAIN = b"vare-hh-helpful-context-v1\0"
PRIOR_BUNDLE = ROOT / "results/cpu-hh-reward-model-v1/development/run-1"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_snapshot_matches(path, expected):
    """Compare protocol data independent of harmless JSON Unicode escaping."""
    return load(path) == expected


def close(a, b, tolerance=2e-4):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tolerance


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(value)))))


def metrics(margins):
    if not margins:
        raise ValueError("empty evaluation margins")
    probabilities = [sigmoid(x) for x in margins]
    accuracy = sum(1.0 if x > 0 else 0.0 if x < 0 else 0.5 for x in margins) / len(margins)
    nll = sum(max(-x, 0.0) + math.log1p(math.exp(-abs(x))) for x in margins) / len(margins)
    brier = sum((p - 1.0) ** 2 for p in probabilities) / len(probabilities)
    ece = 0.0
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        selected = [p for p in probabilities if lo <= p < hi or (i == 9 and p == 1.0)]
        if selected:
            ece += len(selected) / len(probabilities) * abs(sum(selected) / len(selected) - 1.0)
    return {"pairwise_accuracy": accuracy, "logistic_nll": nll, "brier_score": brier,
            "expected_calibration_error": ece}


def read_range(path, start, end):
    if start < 0 or start > end:
        raise ValueError("requested row range violates the frozen protocol")
    if start == end:
        return
    with gzip.open(str(path), "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index < start:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("source row is not a JSON object")
            yield index, row
            # Do not request/decode the next JSONL row past the locked range.
            if index + 1 >= end:
                return


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
    left_positions = [i for i, (start, end) in enumerate(left["offset_mapping"])
                      if end > start and start >= len(context)]
    right_positions = [i for i, (start, end) in enumerate(right["offset_mapping"])
                       if end > start and start >= len(context)]
    if not left_positions or not right_positions:
        return None
    digest = hashlib.sha256(DOMAIN + context.encode("utf-8")).hexdigest()
    return {"context": context, "context_hash": digest, "chosen": chosen, "rejected": rejected,
            "chosen_tokens": len(left_positions), "rejected_tokens": len(right_positions),
            "chosen_ids": left["input_ids"], "rejected_ids": right["input_ids"],
            "chosen_positions": left_positions, "rejected_positions": right_positions}


def build_training(spec, tokenizer):
    count = spec["dataset"]["expected_split_rows"]["train"]
    pilot_ids = set(spec["dataset"]["manual_pilot_train_row_indices_excluded"])
    pilot_hashes = set()
    unique = {}
    with gzip.open(str(DATA_DIR / "train.jsonl.gz"), "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= count:
                break
            row = json.loads(line)
            item = parse_pair(row, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
            if item is None:
                if index in pilot_ids:
                    raise ValueError("a manually reviewed pilot row is no longer eligible")
                continue
            if index in pilot_ids:
                pilot_hashes.add(item["context_hash"])
            if item["context_hash"] in pilot_hashes or index in pilot_ids:
                continue
            if item["context_hash"] in unique:
                continue
            item["source_index"] = index
            unique[item["context_hash"]] = item
    # Pilot contexts may recur before their pilot index. Remove them after all six are known.
    for digest in pilot_hashes:
        unique.pop(digest, None)
    ranked = sorted(unique.values(), key=lambda x: (
        hashlib.sha256(DOMAIN + x["context"].encode("utf-8")).digest(), x["context"].encode("utf-8")))
    if len(ranked) < 408:
        raise ValueError("not enough unique training pairs for frozen ranks")
    selected = ranked[24:408]
    return [selected[i:i+128] for i in (0, 128, 256)], pilot_hashes


def build_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, prior_hashes, development=None):
    start, end = spec["dataset"]["selection"][stage + "_test_indices"]
    excluded = set(pilot_hashes) | set(prior_hashes)
    for cohort in cohorts:
        excluded.update(row["context_hash"] for row in cohort)
    if development:
        excluded.update(row["context_hash"] for row in development)
    seen, accepted = set(), []
    required = spec["dataset"]["selection"][stage + "_test_pairs"]
    for index, row in read_range(DATA_DIR / "test.jsonl.gz", start, end):
        item = parse_pair(row, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
        if item is None or item["context_hash"] in excluded or item["context_hash"] in seen:
            continue
        item["source_index"] = index
        seen.add(item["context_hash"])
        accepted.append(item)
        if len(accepted) == required:
            break
    if len(accepted) != required:
        raise ValueError("fixed v2 evaluation range does not contain 256 eligible prompts")
    return accepted


def extract_features(rows, model, torch):
    output = []
    with torch.no_grad():
        for row in rows:
            pair = []
            for ids, positions in ((row["chosen_ids"], row["chosen_positions"]),
                                   (row["rejected_ids"], row["rejected_positions"])):
                input_ids = torch.tensor([ids], dtype=torch.long)
                hidden = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                     use_cache=False).last_hidden_state[0]
                pair.append(hidden[torch.tensor(positions, dtype=torch.long)].mean(0).float())
            output.append(torch.stack(pair))
    return torch.stack(output)


def fit_head_independent(features, lengths, spec, torch):
    import torch.nn.functional as F
    flat = features.reshape(-1, features.shape[-1])
    mean, scale = flat.mean(0), flat.std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    x = (features[:, 0] - mean) / scale - (features[:, 1] - mean) / scale
    y = lengths[:, 0] - lengths[:, 1]
    ym, ys = y.mean(), y.std(unbiased=False)
    if ys < 1e-6:
        ys = torch.tensor(1.0)
    y = (y - ym) / ys
    opt = spec["learner"]["optimizer"]
    cfg = {"lr": 1.0, "max_iter": opt["max_iterations"], "tolerance_grad": opt["tolerance_grad"],
           "tolerance_change": opt["tolerance_change"], "history_size": opt["history_size"],
           "line_search_fn": opt["line_search"]}
    penalty = spec["learner"]["regularization"]
    reward_w = torch.nn.Parameter(torch.zeros(x.shape[-1]))
    base_w = torch.nn.Parameter(torch.zeros(1))
    reward_optimizer, base_optimizer = torch.optim.LBFGS([reward_w], **cfg), torch.optim.LBFGS([base_w], **cfg)
    def reward_closure():
        reward_optimizer.zero_grad(set_to_none=True)
        value = F.softplus(-(x @ reward_w)).mean() + 0.5 * penalty * (reward_w @ reward_w)
        value.backward()
        return value
    def base_closure():
        base_optimizer.zero_grad(set_to_none=True)
        value = F.softplus(-(y * base_w[0])).mean() + 0.5 * penalty * base_w[0] ** 2
        value.backward()
        return value
    reward_optimizer.step(reward_closure)
    base_optimizer.step(base_closure)
    return reward_w.detach(), mean.detach(), scale.detach(), float(base_w.detach()[0]), float(ym), float(ys)


def calibrate_independent(oof, spec, torch):
    import torch.nn.functional as F
    calibration = spec["learner"]["calibration"]
    theta = torch.nn.Parameter(torch.tensor(calibration["initial_theta"], dtype=torch.float32))
    optimizer = torch.optim.LBFGS([theta], lr=1.0, max_iter=100, tolerance_grad=1e-8,
        tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(set_to_none=True)
        alpha = calibration["maximum_inverse_temperature"] * torch.sigmoid(theta)
        loss = F.softplus(-alpha * oof).mean()
        loss.backward()
        return loss
    optimizer.step(closure)
    alpha = float(calibration["maximum_inverse_temperature"] * torch.sigmoid(theta).detach())
    before = float(F.softplus(-oof).mean().detach())
    after = float(F.softplus(-alpha * oof).mean().detach())
    return alpha, before, after


def replay_predictions(spec, cohorts, eval_rows, tokenizer, model, torch):
    train_features = [extract_features(cohort, model, torch) for cohort in cohorts]
    eval_features = extract_features(eval_rows, model, torch)
    eval_lengths = torch.tensor([[r["chosen_tokens"], r["rejected_tokens"]] for r in eval_rows], dtype=torch.float32)
    result = []
    for cohort_index, cohort in enumerate(cohorts):
        features = train_features[cohort_index]
        train_lengths = torch.tensor([[r["chosen_tokens"], r["rejected_tokens"]] for r in cohort], dtype=torch.float32)
        margins = torch.empty(128, dtype=torch.float32)
        for held_fold in (0, 1):
            held = [i for i in range(128) if i % 2 == held_fold]
            fit = [i for i in range(128) if i % 2 != held_fold]
            fit_ids, held_ids = torch.tensor(fit), torch.tensor(held)
            w, mean, scale, _, _, _ = fit_head_independent(features[fit_ids], train_lengths[fit_ids], spec, torch)
            held_diff = ((features[held_ids, 0] - mean) / scale -
                         (features[held_ids, 1] - mean) / scale)
            margins[held_ids] = held_diff @ w
        alpha, before, after = calibrate_independent(margins, spec, torch)
        w, mean, scale, base_w, ym, ys = fit_head_independent(features, train_lengths, spec, torch)
        eval_diff = (eval_features[:, 0] - mean) / scale - (eval_features[:, 1] - mean) / scale
        reward = (eval_diff @ w) * alpha
        base_diff = eval_lengths[:, 0] - eval_lengths[:, 1]
        baseline = ((base_diff - ym) / ys) * base_w
        result.append({"alpha": alpha, "oof_nll_before": before, "oof_nll_after": after,
                       "reward": [float(x) for x in reward.tolist()],
                       "baseline": [float(x) for x in baseline.tolist()]})
    return result


def audit(bundle: Path):
    bundle = bundle.resolve()
    stage = bundle.parent.name
    if stage not in ("development", "confirmation"):
        raise ValueError("bundle path must be under development or confirmation")
    spec, protocol_hash = load_locked_protocol(SPEC_PATH, LOCK_PATH)
    manifest = load(bundle / "manifest.json")
    actual_files = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
                    if p.is_file() and p.name not in ("manifest.json", "audit.json")}
    expected_files = {"protocol.snapshot.json", "protocol.lock.snapshot.json", "runner.snapshot.py",
        "auditor.snapshot.py", "shared_runner.snapshot.py", "shared_helper.snapshot.py"}
    if (bundle / "failure.json").exists():
        expected_files.add("failure.json")
        if (bundle / "development-audit.snapshot.json").exists():
            expected_files.add("development-audit.snapshot.json")
    else:
        expected_files.update(("pairs.json", "summary.json", "environment.json"))
        if stage == "confirmation":
            expected_files.add("development-audit.snapshot.json")
    if actual_files != expected_files or set(manifest.get("files", {})) != actual_files:
        raise ValueError("result bundle contains missing or unapproved files")
    for name, expected in manifest["files"].items():
        if sha256_file(bundle / name) != expected:
            raise ValueError("bundle manifest hash mismatch: %s" % name)
    if not json_snapshot_matches(bundle / "protocol.snapshot.json", spec):
        raise ValueError("protocol snapshot differs from the frozen v2 protocol")
    if not json_snapshot_matches(bundle / "protocol.lock.snapshot.json", load(LOCK_PATH)):
        raise ValueError("lock snapshot differs from the frozen v2 lock")
    source_map = {"runner.snapshot.py": ROOT / "scripts/run_cpu_hh_reward_model_v2.py",
                  "shared_runner.snapshot.py": Path(shared.__file__),
                  "shared_helper.snapshot.py": ROOT / "scripts/hh_reward_task.py"}
    for snapshot, source in source_map.items():
        if (bundle / snapshot).read_bytes() != source.read_bytes():
            raise ValueError("source snapshot differs from committed source: %s" % snapshot)
    if (bundle / "failure.json").exists():
        failure = load(bundle / "failure.json")
        report = {"status": "failed_attempt_preserved", "stage": stage, "protocol_sha256": protocol_hash,
                  "manifest_sha256": sha256_file(bundle / "manifest.json"),
                  "decision": {"pass": False}, "failure_type": failure.get("exception_type"),
                  "failure_message": failure.get("message")}
        write_json(bundle / "audit.json", report)
        return report
    summary, pair_file = load(bundle / "summary.json"), load(bundle / "pairs.json")
    if set(pair_file) != {"rows"}:
        raise ValueError("pair artifact contains unapproved top-level fields")
    if summary.get("auditor_sha256") != sha256_file(bundle / "auditor.snapshot.py"):
        raise ValueError("run-time auditor snapshot differs from its recorded digest")
    if summary.get("protocol_sha256") != protocol_hash or summary.get("stage") != stage:
        raise ValueError("summary protocol/stage mismatch")
    if summary.get("runner_sha256") != sha256_file(bundle / "runner.snapshot.py"):
        raise ValueError("runner digest mismatch")
    if summary.get("auditor_sha256") != sha256_file(bundle / "auditor.snapshot.py"):
        raise ValueError("auditor digest mismatch")
    if summary.get("shared_runner_sha256") != sha256_file(bundle / "shared_runner.snapshot.py"):
        raise ValueError("shared runner digest mismatch")
    if summary.get("helper_sha256") != sha256_file(bundle / "shared_helper.snapshot.py"):
        raise ValueError("shared helper digest mismatch")
    if summary.get("model_revision") != spec["feature_extractor"]["model_revision"]:
        raise ValueError("model revision mismatch")
    if summary.get("runtime") != spec["runtime"] or summary.get("device") != "cpu":
        raise ValueError("runtime/device mismatch")
    if summary.get("cuda_initialized") is not False or summary.get("network_disabled") is not True or summary.get("paid_compute") is not False:
        raise ValueError("compute mode violates protocol")
    if summary.get("torch_threads") != spec["compute_limits"]["threads"]:
        raise ValueError("thread count differs from protocol")
    if summary.get("peak_rss_bytes", 0) > spec["compute_limits"]["max_peak_rss_bytes"] or summary.get("total_wall_seconds", 0) > spec["compute_limits"]["max_wall_seconds"] or summary.get("compute_limits_respected") is not True:
        raise ValueError("resource limits failed")
    env = load(bundle / "environment.json")
    if (env.get("runtime") != summary.get("runtime") or env.get("model_file_sha256") != summary.get("model_file_sha256") or
            env.get("dataset_file_sha256") != summary.get("dataset_file_sha256") or env.get("device") != "cpu" or
            env.get("network_disabled") is not True or env.get("torch_threads") != spec["compute_limits"]["threads"]):
        raise ValueError("environment record differs from summary or protocol")
    if summary.get("test_source_container_hashed_opaque") is not True:
        raise ValueError("whole-container source hashing provenance is missing")
    if summary.get("test_confirmation_row_records_parsed") is not (stage == "confirmation"):
        raise ValueError("test confirmation access does not match stage")
    if summary.get("evaluation_source_range") != spec["dataset"]["selection"][stage + "_test_indices"]:
        raise ValueError("evaluation range differs from frozen v2 protocol")

    prior = spec["dataset"]["selection"]["prior_development_exclusion"]
    if prior["source_range"] != [0, 512] or prior["context_hashes"] != [x["context_hash"] for x in load(PRIOR_BUNDLE / "pairs.json")["rows"]]:
        raise ValueError("prior v1 development context exclusion list differs")
    if prior["bundle_manifest_sha256"] != sha256_file(PRIOR_BUNDLE / "manifest.json") or prior["pairs_file_sha256"] != sha256_file(PRIOR_BUNDLE / "pairs.json"):
        raise ValueError("prior development bundle hash differs from v2 protocol")
    old_manifest = load(PRIOR_BUNDLE / "manifest.json")
    if old_manifest.get("files", {}).get("pairs.json") != prior["pairs_file_sha256"]:
        raise ValueError("v1 development bundle does not bind its pair artifact")
    if stage == "confirmation":
        dev_audit = load(bundle / "development-audit.snapshot.json")
        if (dev_audit.get("status") != "pass" or dev_audit.get("protocol_sha256") != protocol_hash or
                dev_audit.get("decision", {}).get("pass") is not True or
                summary.get("development_audit_sha256") != sha256_file(bundle / "development-audit.snapshot.json")):
            raise ValueError("confirmation lacks a passing v2 development audit")

    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "1"
    tokenizer, model, torch, runtime, model_hashes, data_hashes = shared.load_runtime(spec)
    if summary.get("model_file_sha256") != model_hashes or summary.get("dataset_file_sha256") != data_hashes:
        raise ValueError("model or pinned data bytes differ from the recorded source")
    cohorts, pilot_hashes = build_training(spec, tokenizer)
    expected_training = [[{"source_index": x["source_index"], "context_hash": x["context_hash"]} for x in c] for c in cohorts]
    if summary.get("training_selection") != expected_training or summary.get("pilot_context_hashes") != sorted(pilot_hashes):
        raise ValueError("training cohort selection or pilot exclusion differs")
    prior_hashes = set(prior["context_hashes"])
    dev_rows = build_evaluation(spec, tokenizer, "development", cohorts, pilot_hashes, prior_hashes) if stage == "confirmation" else None
    exclusions = (dev_rows or []) + [{"context_hash": x} for x in prior_hashes]
    eval_rows = build_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, prior_hashes, dev_rows)
    rows = pair_file["rows"]
    if len(rows) != len(eval_rows):
        raise ValueError("evaluation result count differs from deterministic selection")
    allowed = {"source_index", "context_hash", "chosen_response_tokens", "rejected_response_tokens", "cohorts"}
    prediction_keys = {"reward_margin", "reward_probability", "reward_accuracy",
                       "baseline_margin", "baseline_probability", "baseline_accuracy"}
    for actual, expected in zip(rows, eval_rows):
        if set(actual) != allowed or actual["source_index"] != expected["source_index"] or actual["context_hash"] != expected["context_hash"]:
            raise ValueError("evaluation row provenance/schema mismatch")
        if actual["chosen_response_tokens"] != expected["chosen_tokens"] or actual["rejected_response_tokens"] != expected["rejected_tokens"]:
            raise ValueError("response token counts differ from independent source parse")
        if len(actual["cohorts"]) != 3:
            raise ValueError("missing per-head result")
        for pred in actual["cohorts"]:
            if set(pred) != prediction_keys:
                raise ValueError("pair result includes unknown or sensitive fields")
            for prefix in ("reward", "baseline"):
                margin = float(pred[prefix + "_margin"])
                expected_p = sigmoid(margin)
                expected_accuracy = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
                if not close(pred[prefix + "_probability"], expected_p, 1e-8) or not close(pred[prefix + "_accuracy"], expected_accuracy, 1e-8):
                    raise ValueError("pair probability or accuracy does not match margin")

    replay = replay_predictions(spec, cohorts, eval_rows, tokenizer, model, torch)
    cohort_metrics, acc_deltas, nll_deltas = [], [], []
    for i in range(3):
        item = summary["cohorts"][i]
        reward = [float(row["cohorts"][i]["reward_margin"]) for row in rows]
        baseline = [float(row["cohorts"][i]["baseline_margin"]) for row in rows]
        if len(item.get("reward_margins", [])) != len(rows) or len(item.get("baseline_margins", [])) != len(rows):
            raise ValueError("aggregate margin arrays have wrong length")
        if any(not close(a, b) for a, b in zip(item["reward_margins"], reward)) or any(not close(a, b) for a, b in zip(item["baseline_margins"], baseline)):
            raise ValueError("aggregate margins differ from per-pair values")
        if len(item.get("uncalibrated_reward_margins", [])) != len(rows):
            raise ValueError("uncalibrated prediction vector has wrong length")
        if not close(item["temperature_alpha"], replay[i]["alpha"]):
            raise ValueError("replayed training-only temperature does not match")
        for key in ("oof_nll_before", "oof_nll_after"):
            if not close(item["calibration"][key], replay[i][key], 1e-7):
                raise ValueError("cross-fit calibration diagnostic does not reconstruct: %s" % key)
        if item["calibration"]["pairs"] != 128 or item["calibration"]["fold_pairs"] != 64:
            raise ValueError("cross-fit calibration cohort sizes differ from the lock")
        for field in ("oof_nll_before", "oof_nll_after", "iterations", "closure_calls", "gradient_norm"):
            if not math.isfinite(float(item["calibration"][field])):
                raise ValueError("non-finite calibration diagnostic: %s" % field)
        for got, want in zip(item["uncalibrated_reward_margins"], replay[i]["reward"]):
            if not close(item["temperature_alpha"] * got, want):
                raise ValueError("unscaled reward margins do not reproduce calibrated model output")
        for got, want in zip(reward, replay[i]["reward"]):
            if not close(got, want):
                raise ValueError("replayed transformer/head reward margin differs")
        for got, want in zip(baseline, replay[i]["baseline"]):
            if not close(got, want):
                raise ValueError("replayed length baseline margin differs")
        rm_metrics, base_metrics = metrics(reward), metrics(baseline)
        for field, expected in (("reward_metrics", rm_metrics), ("baseline_metrics", base_metrics)):
            for name, value in expected.items():
                if not close(item[field][name], value, 1e-8):
                    raise ValueError("metric does not reconstruct: %s.%s" % (field, name))
        acc_delta = [float(row["cohorts"][i]["reward_accuracy"]) - float(row["cohorts"][i]["baseline_accuracy"]) for row in rows]
        nll_delta = [max(-a, 0.0) + math.log1p(math.exp(-abs(a))) -
                     (max(-b, 0.0) + math.log1p(math.exp(-abs(b)))) for a, b in zip(reward, baseline)]
        if any(not close(a, b, 1e-8) for a, b in zip(item["accuracy_delta_per_pair"], acc_delta)) or any(not close(a, b, 1e-8) for a, b in zip(item["nll_delta_per_pair"], nll_delta)):
            raise ValueError("paired per-row deltas do not reconstruct")
        cohort_metrics.append({"cohort": i, "reward": rm_metrics, "baseline": base_metrics,
                               "temperature_alpha": replay[i]["alpha"]})
        acc_deltas.append(acc_delta)
        nll_deltas.append(nll_delta)

    n = len(rows)
    mean_acc_delta = [sum(acc_deltas[k][j] for k in range(3)) / 3 for j in range(n)]
    mean_nll_delta = [sum(nll_deltas[k][j] for k in range(3)) / 3 for j in range(n)]
    boot = spec["metrics"]["bootstrap"]
    acc_ci = paired_bootstrap_interval(mean_acc_delta, boot["seed"], boot["resamples"])
    nll_ci = paired_bootstrap_interval(mean_nll_delta, boot["seed"], boot["resamples"])
    mean_acc = sum(x["reward"]["pairwise_accuracy"] for x in cohort_metrics) / 3
    mean_gain = sum(mean_acc_delta) / n
    mean_nll = sum(x["reward"]["logistic_nll"] for x in cohort_metrics) / 3
    base_nll = sum(x["baseline"]["logistic_nll"] for x in cohort_metrics) / 3
    passed = (mean_acc > 0.5 and mean_gain >= 0.02 and acc_ci[0] > 0 and mean_nll < base_nll and
              nll_ci[1] < 0 and sum(x["reward"]["pairwise_accuracy"] > 0.5 for x in cohort_metrics) >= 2)
    decision = {"pass": bool(passed), "mean_reward_accuracy": mean_acc, "mean_accuracy_gain": mean_gain,
        "accuracy_gain_interval_95pct": acc_ci, "mean_reward_nll": mean_nll,
        "mean_baseline_nll": base_nll, "nll_difference_interval_95pct": nll_ci,
        "confirmation_allowed": bool(stage == "development" and passed)}
    observed = summary.get("decision", {})
    if observed.keys() != decision.keys():
        raise ValueError("decision field set mismatch")
    for key, value in decision.items():
        if isinstance(value, list):
            if any(not close(a, b, 1e-8) for a, b in zip(observed[key], value)):
                raise ValueError("bootstrap interval differs from independent recomputation")
        elif isinstance(value, float):
            if not close(observed[key], value, 1e-8):
                raise ValueError("decision metric differs from independent recomputation")
        elif observed[key] != value:
            raise ValueError("frozen gate decision differs")
    report = {"status": "pass", "protocol_sha256": protocol_hash,
        "audit_implementation_sha256": sha256_file(Path(__file__)),
        "run_auditor_snapshot_sha256": summary["auditor_sha256"],
        "manifest_sha256": sha256_file(bundle / "manifest.json"), "stage": stage,
        "decision": decision, "cohort_metrics": cohort_metrics,
        "model_score_replay": "passed; separately implemented token extraction, frozen-feature scoring, two-fold out-of-fold head fitting, temperature fit, full-cohort head fit, and baseline fit reproduced per-pair margins within 2e-4",
        "audit_limit": "Replay uses the same pinned local model weights, tokenizer, framework and machine as the run. It is a deterministic local replay, not an external reproduction or a different model implementation."}
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
