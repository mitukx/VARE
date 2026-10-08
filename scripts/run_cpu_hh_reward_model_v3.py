#!/usr/bin/env python3
"""Run the frozen fresh-cohort fixed-head HH calibration replication."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cpu_hh_reward_model as shared
from hh_reward_task import (load_locked_protocol, metric_summary,
                            paired_bootstrap_interval, sha256_file,
                            write_json, write_manifest)

SPEC_PATH = ROOT / "protocols/cpu_hh_reward_model_v3.json"
LOCK_PATH = ROOT / "protocols/cpu_hh_reward_model_v3.lock.json"
OUTPUT_ROOT = ROOT / "results/cpu-hh-reward-model-v3"
HISTORICAL_BUNDLES = {
    "v1_development": ROOT / "results/cpu-hh-reward-model-v1/development/run-1",
    "v2_development": ROOT / "results/cpu-hh-reward-model-v2/development/run-1",
    "v2_confirmation": ROOT / "results/cpu-hh-reward-model-v2/confirmation/run-1",
}


def fit_temperature(margins, spec, torch):
    import torch.nn.functional as F
    cfg = spec["learner"]["calibration"]
    theta = torch.nn.Parameter(torch.tensor(cfg["initial_theta"], dtype=torch.float32))
    optimizer = torch.optim.LBFGS([theta], lr=cfg["learning_rate"],
        max_iter=cfg["max_iterations"], tolerance_grad=cfg["tolerance_grad"],
        tolerance_change=cfg["tolerance_change"], history_size=cfg["history_size"],
        line_search_fn=cfg["line_search"])
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
    value = float(alpha.detach())
    if not math.isfinite(value) or not (0.0 < value < cfg["maximum_inverse_temperature"]):
        raise ValueError("temperature fit produced a non-positive or non-finite scale")
    return value, {"pairs": int(margins.numel()), "nll_before": float(before.detach()),
        "nll_after": float(after.detach()), "iterations": int(optimizer.state[theta].get("n_iter", 0)),
        "closure_calls": calls[0], "gradient_norm": float(torch.linalg.vector_norm(theta.grad).item())}


def verify_historical_inventory(spec):
    selection = spec["dataset"]["selection"]
    observed = set()
    for name, path in HISTORICAL_BUNDLES.items():
        expected = selection["historical_bundles"][name]
        if sha256_file(path / "manifest.json") != expected["manifest_sha256"]:
            raise ValueError("historical HH bundle manifest changed: %s" % name)
        if sha256_file(path / "pairs.json") != expected["pairs_sha256"]:
            raise ValueError("historical HH pair inventory changed: %s" % name)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("files", {}).get("pairs.json") != expected["pairs_sha256"]:
            raise ValueError("historical HH manifest does not bind pair inventory: %s" % name)
        pairs = json.loads((path / "pairs.json").read_text(encoding="utf-8"))["rows"]
        observed.update(row["context_hash"] for row in pairs)
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        for cohort in summary["training_selection"]:
            observed.update(row["context_hash"] for row in cohort)
    observed.update(selection["manual_pilot_context_hashes"])
    if observed != set(selection["historical_context_hashes"]):
        raise ValueError("locked historical HH context exclusion inventory differs")
    inventory = json.dumps(sorted(observed), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if hashlib.sha256(inventory).hexdigest() != selection["historical_context_inventory_sha256"]:
        raise ValueError("historical HH context inventory digest differs")


def select_training(spec, tokenizer):
    dataset = spec["dataset"]
    maximum = spec["compute_limits"]["max_sequence_tokens"]
    rows = shared.read_rows(shared.DATA_DIR / "train.jsonl.gz", 0,
                            dataset["expected_split_rows"]["train"],
                            dataset["expected_split_rows"]["train"])
    pilot_ids = set(dataset["manual_pilot_train_row_indices_excluded"])
    pilot_hashes = set()
    for index in pilot_ids:
        item = shared.eligible_pair(rows[index][1], tokenizer, maximum)
        if item is None:
            raise ValueError("a manually inspected train row is no longer eligible")
        pilot_hashes.add(item["context_hash"])
    prior_hashes = set(dataset["selection"]["historical_context_hashes"])
    unique = {}
    counts = {"invalid": 0, "historical_context": 0, "duplicate": 0, "eligible": 0}
    for index, raw in rows:
        item = shared.eligible_pair(raw, tokenizer, maximum)
        if item is None:
            counts["invalid"] += 1
            continue
        counts["eligible"] += 1
        digest = item["context_hash"]
        if digest in prior_hashes or digest in pilot_hashes:
            counts["historical_context"] += 1
            continue
        if digest in unique:
            counts["duplicate"] += 1
            continue
        item["source_index"] = index
        unique[digest] = item
    domain = b"vare-hh-helpful-context-v1\0"
    ranked = sorted(unique.values(), key=lambda item: (
        __import__("hashlib").sha256(domain + item["context"].encode("utf-8")).digest(),
        item["context"].encode("utf-8")))
    size = spec["learner"]["pairs_per_head"]
    needed = size * spec["learner"]["heads"]
    if len(ranked) < needed:
        raise ValueError("fewer fresh eligible unique training contexts than locked requirement")
    selected = ranked[:needed]
    cohorts = [selected[index:index + size] for index in range(0, needed, size)]
    return cohorts, pilot_hashes, counts


def select_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, development=None):
    rule = spec["dataset"]["selection"]
    start, end = rule[stage + "_test_indices"]
    excluded = set(rule["historical_context_hashes"]) | set(pilot_hashes)
    for cohort in cohorts:
        excluded.update(row["context_hash"] for row in cohort)
    if development:
        excluded.update(row["context_hash"] for row in development)
    seen, selected = set(), []
    for index, raw in shared.iter_rows(shared.DATA_DIR / "test.jsonl.gz", start, end):
        item = shared.eligible_pair(raw, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
        if item is None or item["context_hash"] in excluded or item["context_hash"] in seen:
            continue
        item["source_index"] = index
        seen.add(item["context_hash"])
        selected.append(item)
    if len(selected) < rule["minimum_test_pairs"]:
        raise ValueError("fixed %s source block yielded %d eligible unique prompts; minimum is %d" %
            (stage, len(selected), rule["minimum_test_pairs"]))
    return selected


def verify_development_audit(path: Path, protocol_hash: str):
    report = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = path.parent / "manifest.json"
    if (report.get("status") != "pass" or report.get("protocol_sha256") != protocol_hash or
            report.get("decision", {}).get("pass") is not True or
            report.get("manifest_sha256") != sha256_file(manifest_path)):
        raise ValueError("development audit is not a passing audit of the current v3 bundle")


def run(stage: str, output: Path, dev_audit: Optional[Path]):
    spec, protocol_hash = load_locked_protocol(SPEC_PATH, LOCK_PATH)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    (output / "runner.snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "auditor.snapshot.py").write_bytes(Path(__file__).with_name("audit_cpu_hh_reward_model_v3.py").read_bytes())
    (output / "shared_runner.snapshot.py").write_bytes(Path(shared.__file__).read_bytes())
    (output / "shared_helper.snapshot.py").write_bytes(Path(__file__).with_name("hh_reward_task.py").read_bytes())
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    try:
        verify_historical_inventory(spec)
        if stage == "confirmation":
            if dev_audit is None:
                raise ValueError("confirmation requires --development-audit")
            verify_development_audit(dev_audit, protocol_hash)
            (output / "development-audit.snapshot.json").write_bytes(dev_audit.read_bytes())
            from audit_cpu_hh_reward_model_v3 import audit as audit_development
            fresh = audit_development(dev_audit.parent)
            if fresh.get("status") != "pass" or fresh.get("decision", {}).get("pass") is not True:
                raise ValueError("fresh v3 development replay/audit did not pass")
            (output / "development-audit.snapshot.json").write_bytes(dev_audit.read_bytes())
        with shared.ResourceGuard(spec["compute_limits"]["max_wall_seconds"],
                                  spec["compute_limits"]["max_peak_rss_bytes"]):
            tokenizer, model, torch, runtime, model_hashes, data_hashes = shared.load_runtime(spec)
            cohorts, pilot_hashes, train_counts = select_training(spec, tokenizer)
            selected_train = [[{"source_index": row["source_index"], "context_hash": row["context_hash"]}
                for row in cohort] for cohort in cohorts]
            frozen_train = spec["dataset"]["selection"]["frozen_training_selection"]
            frozen_digest = hashlib.sha256(json.dumps(frozen_train, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            if selected_train != frozen_train or frozen_digest != spec["dataset"]["selection"]["training_selection_sha256"]:
                raise ValueError("fresh training selection differs from the pre-run frozen inventory")
            development = (select_evaluation(spec, tokenizer, "development", cohorts, pilot_hashes)
                           if stage == "confirmation" else None)
            eval_rows = select_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, development)
            train_features = [shared.encode_rows(cohort, tokenizer, model, torch,
                spec["compute_limits"]["max_sequence_tokens"]) for cohort in cohorts]
            eval_features = shared.encode_rows(eval_rows, tokenizer, model, torch,
                spec["compute_limits"]["max_sequence_tokens"])
            eval_lengths = torch.tensor([[row["chosen_tokens"], row["rejected_tokens"]]
                                         for row in eval_rows], dtype=torch.float32)
            cohort_results = []
            for cohort_index, cohort in enumerate(cohorts):
                fit_n = spec["learner"]["head_fit_pairs"]
                fit_features = train_features[cohort_index][:fit_n]
                fit_lengths = torch.tensor([[row["chosen_tokens"], row["rejected_tokens"]]
                    for row in cohort[:fit_n]], dtype=torch.float32)
                calibration_features = train_features[cohort_index][fit_n:]
                calibration_lengths = torch.tensor([[row["chosen_tokens"], row["rejected_tokens"]]
                    for row in cohort[fit_n:]], dtype=torch.float32)
                w, mean, scale, base_w, y_mean, y_scale, training = shared.fit_head(
                    fit_features, fit_lengths, spec, torch)
                cal_diff = ((calibration_features[:, 0] - mean) / scale -
                            (calibration_features[:, 1] - mean) / scale)
                alpha, calibration = fit_temperature(cal_diff @ w, spec, torch)
                eval_diff = ((eval_features[:, 0] - mean) / scale -
                             (eval_features[:, 1] - mean) / scale)
                raw_margins = (eval_diff @ w).tolist()
                calibrated_margins = [alpha * float(value) for value in raw_margins]
                if any((raw > 0) != (cal > 0) for raw, cal in zip(raw_margins, calibrated_margins)):
                    raise ValueError("positive temperature changed a reward-margin sign")
                base_diff = eval_lengths[:, 0] - eval_lengths[:, 1]
                baseline_margins = (((base_diff - y_mean) / y_scale) * base_w).tolist()
                delta = [max(-cal, 0.0) + math.log1p(math.exp(-abs(cal))) -
                         (max(-raw, 0.0) + math.log1p(math.exp(-abs(raw))))
                         for raw, cal in zip(raw_margins, calibrated_margins)]
                cohort_results.append({"cohort": cohort_index, "temperature_alpha": alpha,
                    "calibration": calibration, "training": training,
                    "raw_metrics": metric_summary(raw_margins),
                    "calibrated_metrics": metric_summary(calibrated_margins),
                    "length_baseline_metrics": metric_summary(baseline_margins),
                    "uncalibrated_reward_margins": raw_margins,
                    "reward_margins": calibrated_margins,
                    "baseline_margins": baseline_margins,
                    "nll_delta_per_pair": delta,
                    "mean_calibrated_minus_raw_nll": sum(delta) / len(delta)})
            n = len(eval_rows)
            prompt_delta = [sum(cohort["nll_delta_per_pair"][index] for cohort in cohort_results) /
                            len(cohort_results) for index in range(n)]
            boot = spec["metrics"]["bootstrap"]
            interval = paired_bootstrap_interval(prompt_delta, boot["seed"], boot["resamples"])
            mean_delta = sum(prompt_delta) / n
            passed = mean_delta <= -spec["metrics"]["minimum_mean_nll_improvement"] and interval[1] < 0.0
            decision = {"pass": bool(passed), "prompt_count": n,
                "mean_calibrated_minus_raw_nll": mean_delta,
                "paired_prompt_bootstrap_95pct": interval,
                "confirmation_allowed": bool(stage == "development" and passed)}
            records = []
            for index, row in enumerate(eval_rows):
                predictions = []
                for cohort in cohort_results:
                    raw = cohort["uncalibrated_reward_margins"][index]
                    calibrated = cohort["reward_margins"][index]
                    predictions.append({"raw_margin": raw, "calibrated_margin": calibrated,
                        "raw_probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw)))),
                        "calibrated_probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, calibrated)))),
                        "raw_accuracy": 1.0 if raw > 0 else 0.0 if raw < 0 else 0.5,
                        "calibrated_accuracy": 1.0 if calibrated > 0 else 0.0 if calibrated < 0 else 0.5})
                records.append({"source_index": row["source_index"], "context_hash": row["context_hash"],
                    "chosen_response_tokens": row["chosen_tokens"], "rejected_response_tokens": row["rejected_tokens"],
                    "cohorts": predictions})
            summary = {"protocol_sha256": protocol_hash, "stage": stage,
                "runner_sha256": sha256_file(Path(__file__)),
                "auditor_sha256": sha256_file(Path(__file__).with_name("audit_cpu_hh_reward_model_v3.py")),
                "shared_runner_sha256": sha256_file(Path(shared.__file__)),
                "helper_sha256": sha256_file(Path(__file__).with_name("hh_reward_task.py")),
                "model_revision": spec["feature_extractor"]["model_revision"],
                "model_file_sha256": model_hashes, "dataset_file_sha256": data_hashes,
                "runtime": runtime, "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
                "network_disabled": True, "paid_compute": False, "torch_threads": torch.get_num_threads(),
                "decision": decision, "cohorts": cohort_results,
                "training_selection": [[{"source_index": row["source_index"], "context_hash": row["context_hash"]}
                    for row in cohort] for cohort in cohorts],
                "pilot_context_hashes": sorted(pilot_hashes), "training_selection_diagnostics": train_counts,
                "historical_context_inventory_sha256": spec["dataset"]["selection"]["historical_context_inventory_sha256"],
                "evaluation_source_range": spec["dataset"]["selection"][stage + "_test_indices"],
                "test_source_container_hashed_opaque": True,
                "test_confirmation_row_records_parsed": stage == "confirmation",
                "development_audit_sha256": sha256_file(dev_audit) if dev_audit else None,
                "peak_rss_bytes": shared.rss_bytes(), "total_wall_seconds": time.monotonic() - started,
                "compute_limits_respected": shared.rss_bytes() <= spec["compute_limits"]["max_peak_rss_bytes"] and
                    time.monotonic() - started <= spec["compute_limits"]["max_wall_seconds"]}
            write_json(output / "pairs.json", {"rows": records})
            write_json(output / "summary.json", summary)
            write_json(output / "environment.json", {"runtime": runtime, "model_file_sha256": model_hashes,
                "dataset_file_sha256": data_hashes, "device": "cpu", "network_disabled": True,
                "torch_threads": torch.get_num_threads()})
        write_manifest(output)
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__,
            "message": str(exc), "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": shared.rss_bytes()})
        write_manifest(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-audit", type=Path)
    args = parser.parse_args()
    output = args.output or OUTPUT_ROOT / args.stage / "run-1"
    report = run(args.stage, output, args.development_audit)
    print(json.dumps({"stage": args.stage, "bundle": str(output.resolve()),
        "decision": report["decision"], "wall_seconds": report["total_wall_seconds"],
        "peak_rss_bytes": report["peak_rss_bytes"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
