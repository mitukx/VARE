#!/usr/bin/env python3
"""Run the frozen out-of-fold-calibrated HH reward-model v2 study."""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_cpu_hh_reward_model as shared
from hh_reward_task import (load_locked_protocol, metric_summary, paired_bootstrap_interval,
                            sha256_file, write_json, write_manifest)

SPEC_PATH = ROOT / "protocols/cpu_hh_reward_model_v2.json"
LOCK_PATH = ROOT / "protocols/cpu_hh_reward_model_v2.lock.json"
OUTPUT_ROOT = ROOT / "results/cpu-hh-reward-model-v2"
PRIOR_DEV_BUNDLE = ROOT / "results/cpu-hh-reward-model-v1/development/run-1"


def fit_temperature(margins, spec, torch):
    import torch.nn.functional as F
    calibration = spec["learner"]["calibration"]
    theta = torch.nn.Parameter(torch.tensor(calibration["initial_theta"], dtype=torch.float32))
    optimizer = torch.optim.LBFGS([theta], lr=1.0, max_iter=100, tolerance_grad=1e-8,
                                  tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")
    calls = [0]
    def closure():
        optimizer.zero_grad(set_to_none=True)
        alpha = calibration["maximum_inverse_temperature"] * torch.sigmoid(theta)
        loss = F.softplus(-(alpha * margins)).mean()
        loss.backward()
        calls[0] += 1
        return loss
    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    alpha = calibration["maximum_inverse_temperature"] * torch.sigmoid(theta)
    final_loss = F.softplus(-(alpha * margins)).mean()
    final_loss.backward()
    raw_nll = F.softplus(-margins).mean()
    grad_norm = float(torch.linalg.vector_norm(theta.grad).item())
    if (not math.isfinite(float(alpha.detach())) or float(alpha.detach()) <= 0 or
            float(alpha.detach()) >= calibration["maximum_inverse_temperature"]):
        raise ValueError("temperature fit produced a non-positive or non-finite scale")
    return float(alpha.detach()), {"oof_nll_before": float(raw_nll.detach()),
        "oof_nll_after": float(final_loss.detach()), "iterations": int(optimizer.state[theta].get("n_iter", 0)),
        "closure_calls": calls[0], "gradient_norm": grad_norm}


def calibrate_cohort(features, lengths, spec, torch):
    n = features.shape[0]
    if n != spec["learner"]["training_pairs_per_cohort"] or n % 2:
        raise ValueError("cross-fitting requires the exact even-sized locked training cohort")
    oof = torch.empty(n, dtype=torch.float32)
    indices = list(range(n))
    for held_fold in (0, 1):
        held = [index for index in indices if index % 2 == held_fold]
        train = [index for index in indices if index % 2 != held_fold]
        ix_train = torch.tensor(train, dtype=torch.long)
        ix_held = torch.tensor(held, dtype=torch.long)
        w, mean, scale, _bw, _ym, _ys, _diagnostics = shared.fit_head(
            features[ix_train], lengths[ix_train], spec, torch)
        normalized = ((features[ix_held, 0] - mean) / scale -
                      (features[ix_held, 1] - mean) / scale)
        oof[ix_held] = normalized @ w
    alpha, diagnostics = fit_temperature(oof, spec, torch)
    diagnostics.update({"pairs": n, "fold_pairs": n // 2})
    return alpha, diagnostics


def verify_development_audit(path: Path, protocol_hash: str):
    report = json.loads(path.read_text(encoding="utf-8"))
    manifest_path = path.parent / "manifest.json"
    if (report.get("status") != "pass" or report.get("protocol_sha256") != protocol_hash or
            report.get("decision", {}).get("pass") is not True or
            report.get("manifest_sha256") != sha256_file(manifest_path)):
        raise ValueError("development audit is not a passing audit of the current v2 bundle")


def verify_prior_development(spec):
    prior = spec["dataset"]["selection"]["prior_development_exclusion"]
    pairs_path = PRIOR_DEV_BUNDLE / "pairs.json"
    manifest_path = PRIOR_DEV_BUNDLE / "manifest.json"
    if (sha256_file(pairs_path) != prior["pairs_file_sha256"] or
            sha256_file(manifest_path) != prior["bundle_manifest_sha256"]):
        raise ValueError("historical v1 development bundle differs from the v2 frozen exclusions")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("files", {}).get("pairs.json") != prior["pairs_file_sha256"]:
        raise ValueError("v1 manifest does not bind the prior development pair artifact")
    rows = json.loads(pairs_path.read_text(encoding="utf-8"))["rows"]
    if [row.get("context_hash") for row in rows] != prior["context_hashes"]:
        raise ValueError("v1 development context hashes differ from the v2 frozen exclusion list")


def run(stage: str, output: Path, dev_audit: Optional[Path]):
    spec, protocol_hash = load_locked_protocol(SPEC_PATH, LOCK_PATH)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    (output / "runner.snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "auditor.snapshot.py").write_bytes(Path(__file__).with_name("audit_cpu_hh_reward_model_v2.py").read_bytes())
    (output / "shared_runner.snapshot.py").write_bytes(Path(shared.__file__).read_bytes())
    (output / "shared_helper.snapshot.py").write_bytes(Path(__file__).with_name("hh_reward_task.py").read_bytes())
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    try:
        verify_prior_development(spec)
        if stage == "confirmation":
            if dev_audit is None:
                raise ValueError("confirmation requires --development-audit")
            verify_development_audit(dev_audit, protocol_hash)
            (output / "development-audit.snapshot.json").write_bytes(dev_audit.read_bytes())
        with shared.ResourceGuard(spec["compute_limits"]["max_wall_seconds"], spec["compute_limits"]["max_peak_rss_bytes"]):
            if stage == "confirmation":
                from audit_cpu_hh_reward_model_v2 import audit as audit_development
                fresh = audit_development(dev_audit.parent)
                if fresh.get("status") != "pass" or fresh.get("decision", {}).get("pass") is not True:
                    raise ValueError("fresh v2 development replay/audit did not pass")
                (output / "development-audit.snapshot.json").write_bytes(dev_audit.read_bytes())
            tokenizer, model, torch, runtime, model_hashes, data_hashes = shared.load_runtime(spec)
            cohorts, pilot_hashes, train_counts = shared.select_training(spec, tokenizer)
            selection = spec["dataset"]["selection"]
            old_hashes = selection["prior_development_exclusion"]["context_hashes"]
            prior_rows = [{"context_hash": value} for value in old_hashes]
            if len(old_hashes) != 256 or len(set(old_hashes)) != 256:
                raise ValueError("v1 historical development exclusions differ from their frozen inventory")
            if stage == "confirmation":
                dev_rows = shared.select_evaluation(spec, tokenizer, "development", cohorts, pilot_hashes, prior_rows)
                exclusion_rows = prior_rows + dev_rows
            else:
                dev_rows = None
                exclusion_rows = prior_rows
            eval_rows = shared.select_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, exclusion_rows)
            train_features = [shared.encode_rows(cohort, tokenizer, model, torch, spec["compute_limits"]["max_sequence_tokens"])
                              for cohort in cohorts]
            eval_features = shared.encode_rows(eval_rows, tokenizer, model, torch, spec["compute_limits"]["max_sequence_tokens"])
            eval_lengths = torch.tensor([[r["chosen_tokens"], r["rejected_tokens"]] for r in eval_rows], dtype=torch.float32)
            cohort_results = []
            for cohort_index, cohort in enumerate(cohorts):
                train_lengths = torch.tensor([[r["chosen_tokens"], r["rejected_tokens"]] for r in cohort], dtype=torch.float32)
                alpha, calibration = calibrate_cohort(train_features[cohort_index], train_lengths, spec, torch)
                w, mean, scale, bw, y_mean, y_scale, training = shared.fit_head(
                    train_features[cohort_index], train_lengths, spec, torch)
                normalized = (eval_features[:, 0] - mean) / scale - (eval_features[:, 1] - mean) / scale
                raw_margins = (normalized @ w).tolist()
                reward_margins = [alpha * float(value) for value in raw_margins]
                length_margin = (eval_lengths[:, 0] - eval_lengths[:, 1] - y_mean) / y_scale
                baseline_margins = (length_margin * bw).tolist()
                accuracy_delta = [(1.0 if a > 0 else 0.0 if a < 0 else 0.5) -
                                  (1.0 if b > 0 else 0.0 if b < 0 else 0.5)
                                  for a, b in zip(reward_margins, baseline_margins)]
                nll_delta = [max(-a, 0.0) + math.log1p(math.exp(-abs(a))) -
                             (max(-b, 0.0) + math.log1p(math.exp(-abs(b))))
                             for a, b in zip(reward_margins, baseline_margins)]
                cohort_results.append({"cohort": cohort_index, "temperature_alpha": alpha,
                    "calibration": calibration, "training": training,
                    "reward_metrics": metric_summary(reward_margins),
                    "baseline_metrics": metric_summary(baseline_margins),
                    "reward_margins": reward_margins, "uncalibrated_reward_margins": raw_margins,
                    "baseline_margins": baseline_margins, "accuracy_delta_per_pair": accuracy_delta,
                    "nll_delta_per_pair": nll_delta})
            n = len(eval_rows)
            mean_acc_delta = [sum(c["accuracy_delta_per_pair"][i] for c in cohort_results) / 3 for i in range(n)]
            mean_nll_delta = [sum(c["nll_delta_per_pair"][i] for c in cohort_results) / 3 for i in range(n)]
            boot = spec["metrics"]["bootstrap"]
            acc_ci = paired_bootstrap_interval(mean_acc_delta, boot["seed"], boot["resamples"])
            nll_ci = paired_bootstrap_interval(mean_nll_delta, boot["seed"], boot["resamples"])
            mean_accuracy = sum(c["reward_metrics"]["pairwise_accuracy"] for c in cohort_results) / 3
            mean_gain = sum(mean_acc_delta) / n
            mean_nll = sum(c["reward_metrics"]["logistic_nll"] for c in cohort_results) / 3
            baseline_nll = sum(c["baseline_metrics"]["logistic_nll"] for c in cohort_results) / 3
            passed = (mean_accuracy > 0.5 and mean_gain >= 0.02 and acc_ci[0] > 0 and
                      mean_nll < baseline_nll and nll_ci[1] < 0 and
                      sum(c["reward_metrics"]["pairwise_accuracy"] > 0.5 for c in cohort_results) >= 2)
            decision = {"pass": bool(passed), "mean_reward_accuracy": mean_accuracy,
                "mean_accuracy_gain": mean_gain, "accuracy_gain_interval_95pct": acc_ci,
                "mean_reward_nll": mean_nll, "mean_baseline_nll": baseline_nll,
                "nll_difference_interval_95pct": nll_ci,
                "confirmation_allowed": bool(stage == "development" and passed)}
            records = []
            for index, row in enumerate(eval_rows):
                predictions = []
                for result in cohort_results:
                    reward = result["reward_margins"][index]
                    baseline = result["baseline_margins"][index]
                    predictions.append({"reward_margin": reward,
                        "reward_probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, reward)))),
                        "reward_accuracy": 1.0 if reward > 0 else 0.0 if reward < 0 else 0.5,
                        "baseline_margin": baseline,
                        "baseline_probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, baseline)))),
                        "baseline_accuracy": 1.0 if baseline > 0 else 0.0 if baseline < 0 else 0.5})
                records.append({"source_index": row["source_index"], "context_hash": row["context_hash"],
                    "chosen_response_tokens": row["chosen_tokens"], "rejected_response_tokens": row["rejected_tokens"],
                    "cohorts": predictions})
            summary = {"protocol_sha256": protocol_hash, "stage": stage,
                "runner_sha256": sha256_file(Path(__file__)),
                "auditor_sha256": sha256_file(Path(__file__).with_name("audit_cpu_hh_reward_model_v2.py")),
                "shared_runner_sha256": sha256_file(Path(shared.__file__)),
                "helper_sha256": sha256_file(Path(__file__).with_name("hh_reward_task.py")),
                "model_revision": spec["feature_extractor"]["model_revision"],
                "model_file_sha256": model_hashes, "dataset_file_sha256": data_hashes,
                "runtime": runtime, "device": "cpu", "cuda_initialized": torch.cuda.is_initialized(),
                "network_disabled": True, "paid_compute": False, "torch_threads": torch.get_num_threads(),
                "decision": decision, "cohorts": cohort_results,
                "training_selection": [[{"source_index": r["source_index"], "context_hash": r["context_hash"]} for r in c] for c in cohorts],
                "pilot_context_hashes": sorted(pilot_hashes), "training_selection_diagnostics": train_counts,
                "prior_development_exclusion_sha256": selection["prior_development_exclusion"]["pairs_file_sha256"],
                "evaluation_source_range": selection[stage + "_test_indices"],
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
