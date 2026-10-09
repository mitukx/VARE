#!/usr/bin/env python3
"""Run the frozen HH-RLHF reward-model study on cached weights and CPU only."""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, os, platform, resource, signal, sys, threading, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hh_reward_task import (eligible_pair, load_locked_protocol, metric_summary,
                            paired_bootstrap_interval, sha256_file, write_json, write_manifest)

SPEC_PATH = ROOT / "protocols/cpu_hh_reward_model_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_hh_reward_model_v1.lock.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/hub/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-base"
OUTPUT_ROOT = ROOT / "results/cpu-hh-reward-model-v1"


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class ResourceGuard:
    def __init__(self, seconds: int, memory: int):
        self.seconds, self.memory = seconds, memory
        self.stop = threading.Event()

    def __enter__(self):
        def abort(signum, _frame):
            raise RuntimeError("resource guard aborted run with signal %s" % signum)
        self.old_usr1 = signal.signal(signal.SIGUSR1, abort)
        self.old_alarm = signal.signal(signal.SIGALRM, abort)
        signal.alarm(self.seconds)
        def monitor():
            while not self.stop.wait(0.25):
                if rss_bytes() > self.memory:
                    os.kill(os.getpid(), signal.SIGUSR1)
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        self.thread.join(timeout=1)
        signal.alarm(0)
        signal.signal(signal.SIGUSR1, self.old_usr1)
        signal.signal(signal.SIGALRM, self.old_alarm)


def read_rows(path: Path, start: int, end: int, total: int) -> List[Tuple[int, Dict[str, Any]]]:
    if end > total or start < 0 or start > end:
        raise ValueError("requested row range violates the frozen protocol")
    result = []
    with gzip.open(str(path), "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= end:
                break
            if index < start:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("source row is not a JSON object")
            result.append((index, row))
    if len(result) != end - start:
        raise ValueError("source ended inside the locked row range")
    return result


def iter_rows(path: Path, start: int, end: int):
    if start < 0 or start > end:
        raise ValueError("requested row range violates the frozen protocol")
    with gzip.open(str(path), "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= end:
                return
            if index < start:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("source row is not a JSON object")
            yield index, row


def load_runtime(spec):
    import torch, numpy, transformers, datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer
    actual = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
              "transformers": transformers.__version__, "numpy": numpy.__version__, "datasets": datasets.__version__}
    if actual != spec["runtime"]:
        raise RuntimeError("runtime differs from lock: %r" % actual)
    if MODEL_DIR.name != spec["feature_extractor"]["model_revision"] or not MODEL_DIR.is_dir():
        raise FileNotFoundError("the exact model revision must already be cached")
    if DATA_DIR.parent.name != spec["dataset"]["revision"] or not DATA_DIR.is_dir():
        raise FileNotFoundError("the exact data revision must already be cached")
    for name, size in spec["dataset"]["file_bytes_at_lock"].items():
        path = DATA_DIR / Path(name).name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError("cached source size differs from lock: %s" % name)
    model_files = {}
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        path = MODEL_DIR / name
        if not path.is_file():
            raise FileNotFoundError("missing cached model file: %s" % name)
        model_files[name] = sha256_file(path)
    dataset_files = {name: sha256_file(DATA_DIR / name) for name in ("train.jsonl.gz", "test.jsonl.gz")}
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("fast tokenizer is required by the protocol")
    torch.set_num_threads(spec["compute_limits"]["threads"])
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must not be initialized")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32)
    model.to(torch.device("cpu"))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
        if param.device.type != "cpu":
            raise RuntimeError("model contains a non-CPU parameter")
    return tokenizer, model, torch, actual, model_files, dataset_files


def select_training(spec, tokenizer):
    total = spec["dataset"]["expected_split_rows"]["train"]
    rows = read_rows(DATA_DIR / "train.jsonl.gz", 0, total, total)
    pilot_indices = set(spec["dataset"]["manual_pilot_train_row_indices_excluded"])
    pilot_hashes = set()
    for index in pilot_indices:
        item = eligible_pair(rows[index][1], tokenizer, spec["compute_limits"]["max_sequence_tokens"])
        if item is None:
            raise ValueError("manually inspected pilot row is not eligible")
        pilot_hashes.add(item["context_hash"])
    unique = {}
    counts = {"invalid": 0, "pilot_context": 0, "duplicate": 0, "eligible": 0}
    for index, raw in rows:
        item = eligible_pair(raw, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
        if item is None:
            counts["invalid"] += 1
            continue
        counts["eligible"] += 1
        if item["context_hash"] in pilot_hashes:
            counts["pilot_context"] += 1
            continue
        if item["context_hash"] in unique:
            counts["duplicate"] += 1
            continue
        unique[item["context_hash"]] = {
            "source_index": index, "context": item["context"], "context_hash": item["context_hash"],
            "chosen": item["chosen"], "rejected": item["rejected"],
            "chosen_tokens": item["chosen_tokens"], "rejected_tokens": item["rejected_tokens"],
        }
    domain = b"vare-hh-helpful-context-v1\0"
    ranked = sorted(unique.values(), key=lambda row: (
        hashlib.sha256(domain + row["context"].encode("utf-8")).digest(), row["context"].encode("utf-8")))
    if len(ranked) < 408:
        raise ValueError("fewer than 408 eligible unique training contexts")
    selected = ranked[24:408]
    return [selected[i:i+128] for i in (0, 128, 256)], pilot_hashes, counts


def select_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, development=None):
    rule = spec["dataset"]["selection"]
    start, end = rule[stage + "_test_indices"]
    total = spec["dataset"]["expected_split_rows"]["test"]
    excluded = set(pilot_hashes)
    for cohort in cohorts:
        excluded.update(row["context_hash"] for row in cohort)
    if development is not None:
        excluded.update(row["context_hash"] for row in development)
    selected, seen = [], set()
    for index, raw in iter_rows(DATA_DIR / "test.jsonl.gz", start, end):
        item = eligible_pair(raw, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
        if item is None or item["context_hash"] in excluded or item["context_hash"] in seen:
            continue
        item["source_index"] = index
        seen.add(item["context_hash"])
        selected.append(item)
        if len(selected) == rule[stage + "_test_pairs"]:
            break
    required = rule[stage + "_test_pairs"]
    if len(selected) != required:
        raise ValueError("fixed %s source block yielded %d eligible rows; requires %d" % (stage, len(selected), required))
    return selected


def encode_rows(rows, tokenizer, model, torch, maximum):
    tensors = []
    with torch.no_grad():
        for row in rows:
            tokenized = eligible_pair(row, tokenizer, maximum)
            if tokenized is None:
                raise ValueError("selected pair changed eligibility before feature extraction")
            sides = []
            for ids, positions in ((tokenized["chosen_ids"], tokenized["chosen_feature_positions"]),
                                   (tokenized["rejected_ids"], tokenized["rejected_feature_positions"])):
                input_ids = torch.tensor([ids], dtype=torch.long)
                attention = torch.ones_like(input_ids)
                hidden = model.model(input_ids=input_ids,
                                     attention_mask=attention,
                                     use_cache=False).last_hidden_state[0]
                sides.append(hidden[torch.tensor(positions, dtype=torch.long)].mean(0).float())
            tensors.append(torch.stack(sides))
    return torch.stack(tensors)


def fit_head(features, lengths, spec, torch):
    import torch.nn.functional as F
    learner = spec["learner"]
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(0)
    scale = flat.std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    x = (features[:, 0] - mean) / scale - (features[:, 1] - mean) / scale
    y = lengths[:, 0] - lengths[:, 1]
    y_mean, y_scale = y.mean(), y.std(unbiased=False)
    if y_scale < 1e-6:
        y_scale = torch.tensor(1.0)
    y = (y - y_mean) / y_scale
    opt = learner["optimizer"]
    settings = {"lr": 1.0, "max_iter": opt["max_iterations"], "tolerance_grad": opt["tolerance_grad"],
                "tolerance_change": opt["tolerance_change"], "history_size": opt["history_size"],
                "line_search_fn": opt["line_search"]}
    reg = learner["regularization"]
    w = torch.nn.Parameter(torch.zeros(x.shape[-1]))
    bw = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([w], **settings)
    base_optimizer = torch.optim.LBFGS([bw], **settings)
    calls, base_calls = [0], [0]
    losses, base_losses = [float("nan")], [float("nan")]
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = F.softplus(-(x @ w)).mean() + 0.5 * reg * (w @ w)
        loss.backward()
        calls[0] += 1
        losses[0] = float(loss.detach())
        return loss
    def base_closure():
        base_optimizer.zero_grad(set_to_none=True)
        loss = F.softplus(-(y * bw[0])).mean() + 0.5 * reg * (bw[0] ** 2)
        loss.backward()
        base_calls[0] += 1
        base_losses[0] = float(loss.detach())
        return loss
    optimizer.step(closure)
    base_optimizer.step(base_closure)
    optimizer.zero_grad(set_to_none=True)
    final_reward_loss = F.softplus(-(x @ w)).mean() + 0.5 * reg * (w @ w)
    final_reward_loss.backward()
    base_optimizer.zero_grad(set_to_none=True)
    final_base_loss = F.softplus(-(y * bw[0])).mean() + 0.5 * reg * (bw[0] ** 2)
    final_base_loss.backward()
    diag = {"reward_training_loss": float(final_reward_loss.detach()), "reward_iterations": int(optimizer.state[w].get("n_iter", 0)),
            "reward_closure_calls": calls[0], "reward_gradient_norm": float(torch.linalg.vector_norm(w.grad).item()),
            "baseline_training_loss": float(final_base_loss.detach()), "baseline_iterations": int(base_optimizer.state[bw].get("n_iter", 0)),
            "baseline_closure_calls": base_calls[0], "baseline_gradient_norm": float(torch.linalg.vector_norm(bw.grad).item())}
    return w.detach(), mean.detach(), scale.detach(), float(bw.detach()[0]), float(y_mean), float(y_scale), diag


def run(stage: str, output: Path, dev_audit: Optional[Path]):
    spec, protocol_hash = load_locked_protocol(SPEC_PATH, LOCK_PATH)
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    (output / "runner.snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "helper.snapshot.py").write_bytes(Path(__file__).with_name("hh_reward_task.py").read_bytes())
    (output / "auditor.snapshot.py").write_bytes(Path(__file__).with_name("audit_cpu_hh_reward_model.py").read_bytes())
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    try:
        if stage == "confirmation":
            if dev_audit is None:
                raise ValueError("confirmation requires --development-audit")
            audit_record = json.loads(dev_audit.read_text(encoding="utf-8"))
            manifest_hash = sha256_file(dev_audit.parent / "manifest.json")
            if (audit_record.get("status") != "pass" or audit_record.get("protocol_sha256") != protocol_hash or
                    audit_record.get("manifest_sha256") != manifest_hash or
                    audit_record.get("decision", {}).get("pass") is not True):
                raise ValueError("development audit is not a passing audit of the current protocol bundle")
            (output / "development-audit.snapshot.json").write_bytes(dev_audit.read_bytes())
        with ResourceGuard(spec["compute_limits"]["max_wall_seconds"], spec["compute_limits"]["max_peak_rss_bytes"]):
            if stage == "confirmation":
                from audit_cpu_hh_reward_model import audit as audit_development
                fresh_audit = audit_development(dev_audit.parent)
                if fresh_audit.get("status") != "pass" or fresh_audit.get("decision", {}).get("pass") is not True:
                    raise ValueError("fresh independent development audit did not pass its locked gate")
                (output / "development-audit.snapshot.json").write_bytes(dev_audit.read_bytes())
            tokenizer, model, torch, runtime, model_hashes, data_hashes = load_runtime(spec)
            cohorts, pilot_hashes, train_counts = select_training(spec, tokenizer)
            dev_rows = select_evaluation(spec, tokenizer, "development", cohorts, pilot_hashes) if stage == "confirmation" else None
            eval_rows = select_evaluation(spec, tokenizer, stage, cohorts, pilot_hashes, dev_rows)
            maximum = spec["compute_limits"]["max_sequence_tokens"]
            train_features = [encode_rows(cohort, tokenizer, model, torch, maximum) for cohort in cohorts]
            eval_features = encode_rows(eval_rows, tokenizer, model, torch, maximum)
            eval_lengths = torch.tensor([[r["chosen_tokens"], r["rejected_tokens"]] for r in eval_rows], dtype=torch.float32)
            cohort_results = []
            for i, cohort in enumerate(cohorts):
                train_lengths = torch.tensor([[r["chosen_tokens"], r["rejected_tokens"]] for r in cohort], dtype=torch.float32)
                w, mean, scale, bw, y_mean, y_scale, diagnostics = fit_head(train_features[i], train_lengths, spec, torch)
                normalized = (eval_features[:, 0] - mean) / scale - (eval_features[:, 1] - mean) / scale
                margins = (normalized @ w).tolist()
                lengths = (eval_lengths[:, 0] - eval_lengths[:, 1] - y_mean) / y_scale
                baseline = (lengths * bw).tolist()
                accuracy_delta = [(1.0 if a > 0 else 0.0 if a < 0 else 0.5) -
                                  (1.0 if b > 0 else 0.0 if b < 0 else 0.5) for a, b in zip(margins, baseline)]
                nll_delta = [max(-a, 0.0) + math.log1p(math.exp(-abs(a))) -
                             (max(-b, 0.0) + math.log1p(math.exp(-abs(b)))) for a, b in zip(margins, baseline)]
                cohort_results.append({"cohort": i, "reward_metrics": metric_summary(margins),
                                       "baseline_metrics": metric_summary(baseline), "reward_margins": margins,
                                       "baseline_margins": baseline, "accuracy_delta_per_pair": accuracy_delta,
                                       "nll_delta_per_pair": nll_delta, "training": diagnostics})
            n = len(eval_rows)
            avg_acc = [sum(c["accuracy_delta_per_pair"][j] for c in cohort_results) / len(cohort_results) for j in range(n)]
            avg_nll = [sum(c["nll_delta_per_pair"][j] for c in cohort_results) / len(cohort_results) for j in range(n)]
            acc_ci = paired_bootstrap_interval(avg_acc, spec["metrics"]["bootstrap"]["seed"], spec["metrics"]["bootstrap"]["resamples"])
            nll_ci = paired_bootstrap_interval(avg_nll, spec["metrics"]["bootstrap"]["seed"], spec["metrics"]["bootstrap"]["resamples"])
            mean_acc = sum(c["reward_metrics"]["pairwise_accuracy"] for c in cohort_results) / 3
            mean_gain = sum(avg_acc) / n
            mean_nll = sum(c["reward_metrics"]["logistic_nll"] for c in cohort_results) / 3
            mean_base_nll = sum(c["baseline_metrics"]["logistic_nll"] for c in cohort_results) / 3
            passed = (mean_acc > 0.5 and mean_gain >= 0.02 and acc_ci[0] > 0 and mean_nll < mean_base_nll and
                      nll_ci[1] < 0 and sum(c["reward_metrics"]["pairwise_accuracy"] > 0.5 for c in cohort_results) >= 2)
            decision = {"pass": bool(passed), "mean_reward_accuracy": mean_acc, "mean_accuracy_gain": mean_gain,
                        "accuracy_gain_interval_95pct": acc_ci, "mean_reward_nll": mean_nll,
                        "mean_baseline_nll": mean_base_nll, "nll_difference_interval_95pct": nll_ci,
                        "confirmation_allowed": bool(stage == "development" and passed)}
            pair_records = []
            for j, row in enumerate(eval_rows):
                records = []
                for c in cohort_results:
                    m, b = c["reward_margins"][j], c["baseline_margins"][j]
                    records.append({"reward_margin": m, "reward_probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, m)))),
                                    "reward_accuracy": 1.0 if m > 0 else 0.0 if m < 0 else 0.5,
                                    "baseline_margin": b, "baseline_probability": 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, b)))),
                                    "baseline_accuracy": 1.0 if b > 0 else 0.0 if b < 0 else 0.5})
                pair_records.append({"source_index": row["source_index"], "context_hash": row["context_hash"],
                                     "chosen_response_tokens": row["chosen_tokens"],
                                     "rejected_response_tokens": row["rejected_tokens"], "cohorts": records})
            summary = {"protocol_sha256": protocol_hash, "stage": stage,
                       "runner_sha256": sha256_file(Path(__file__)),
                       "helper_sha256": sha256_file(Path(__file__).with_name("hh_reward_task.py")),
                       "auditor_sha256": sha256_file(Path(__file__).with_name("audit_cpu_hh_reward_model.py")),
                       "model_revision": spec["feature_extractor"]["model_revision"], "model_file_sha256": model_hashes,
                       "dataset_file_sha256": data_hashes, "runtime": runtime, "device": "cpu",
                       "cuda_initialized": torch.cuda.is_initialized(), "network_disabled": True, "paid_compute": False,
                       "torch_threads": torch.get_num_threads(), "decision": decision, "cohorts": cohort_results,
                       "training_selection": [[{"source_index": r["source_index"], "context_hash": r["context_hash"]} for r in c] for c in cohorts],
                       "pilot_context_hashes": sorted(pilot_hashes), "training_selection_diagnostics": train_counts,
                       "test_source_range": spec["dataset"]["selection"][stage + "_test_indices"],
                       "test_confirmation_rows_read": stage == "confirmation",
                       "development_audit_sha256": sha256_file(dev_audit) if dev_audit else None,
                       "peak_rss_bytes": rss_bytes(), "total_wall_seconds": time.monotonic() - started,
                       "compute_limits_respected": rss_bytes() <= spec["compute_limits"]["max_peak_rss_bytes"] and
                           time.monotonic() - started <= spec["compute_limits"]["max_wall_seconds"]}
            write_json(output / "pairs.json", {"rows": pair_records})
            write_json(output / "summary.json", summary)
            write_json(output / "environment.json", {"runtime": runtime, "model_file_sha256": model_hashes,
                "dataset_file_sha256": data_hashes, "device": "cpu", "network_disabled": True,
                "torch_threads": torch.get_num_threads()})
        write_manifest(output)
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__, "message": str(exc),
            "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes()})
        write_manifest(output)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-audit", type=Path)
    args = parser.parse_args()
    output = args.output or OUTPUT_ROOT / args.stage / "run-1"
    result = run(args.stage, output, args.development_audit)
    print(json.dumps({"stage": args.stage, "bundle": str(output.resolve()),
        "decision": result["decision"], "wall_seconds": result["total_wall_seconds"],
        "peak_rss_bytes": result["peak_rss_bytes"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
