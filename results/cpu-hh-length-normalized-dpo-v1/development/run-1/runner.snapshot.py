#!/usr/bin/env python3
"""Outcome-informed CPU comparison of sequence-sum and length-normalized DPO."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
import os
import platform
import random
import resource
import signal
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hh_reward_task import canonical, eligible_pair, sha256_file, write_json, write_manifest

SPEC_PATH = ROOT / "protocols/cpu_hh_length_normalized_dpo_development_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_hh_length_normalized_dpo_development_v1.lock.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/hub/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-base"
OUTPUT_ROOT = ROOT / "results/cpu-hh-length-normalized-dpo-v1/development"


def digest_file(path: Path) -> str:
    return sha256_file(path)


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class ResourceGuard:
    def __init__(self, seconds: int, memory: int):
        self.seconds, self.memory = seconds, memory
        self.stop = threading.Event()

    def __enter__(self):
        self.old_alarm = signal.signal(signal.SIGALRM,
            lambda *_: (_ for _ in ()).throw(TimeoutError("HH DPO development exceeded wall limit")))
        self.old_usr1 = signal.signal(signal.SIGUSR1,
            lambda *_: (_ for _ in ()).throw(MemoryError("HH DPO development exceeded RSS limit")))
        signal.alarm(self.seconds)
        def monitor():
            while not self.stop.wait(0.25):
                if rss_bytes() > self.memory:
                    signal.raise_signal(signal.SIGUSR1)
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=1)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self.old_alarm)
        signal.signal(signal.SIGUSR1, self.old_usr1)


def load_protocol():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    locked = lock.pop("sha256", None)
    actual = hashlib.sha256(canonical(spec)).hexdigest()
    if locked != actual or canonical(lock) != canonical(spec):
        raise ValueError("HH human-DPO development protocol differs from frozen lock")
    for rel, key in ((spec["freeze"]["runner"], "runner_sha256"),
                     (spec["freeze"]["auditor"], "auditor_sha256"),
                     (spec["freeze"]["objective_test"], "objective_test_sha256")):
        if digest_file(ROOT / rel) != spec["freeze"][key]:
            raise ValueError("frozen code hash differs: %s" % rel)
    return spec, actual


def verify_assets(spec):
    import numpy
    import torch
    import transformers
    import datasets
    runtime = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
               "transformers": transformers.__version__, "numpy": numpy.__version__, "datasets": datasets.__version__}
    if runtime != spec["runtime"]:
        raise RuntimeError("runtime differs from the frozen protocol: %r" % runtime)
    if MODEL_DIR.name != spec["model"]["revision"] or not MODEL_DIR.is_dir():
        raise FileNotFoundError("the exact cached model revision is required")
    if DATA_DIR.parent.name != spec["dataset"]["revision"] or not DATA_DIR.is_dir():
        raise FileNotFoundError("the exact cached HH revision is required")
    for rel, size in spec["dataset"]["file_bytes_at_lock"].items():
        path = DATA_DIR / Path(rel).name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError("HH source file size differs from lock: %s" % rel)
    for name, expected in spec["dataset"]["file_sha256_at_lock"].items():
        path = DATA_DIR / Path(name).name
        if digest_file(path) != expected:
            raise ValueError("HH source hash differs from lock: %s" % name)
    for name, expected in spec["model"]["files_sha256_at_lock"].items():
        if digest_file(MODEL_DIR / name) != expected:
            raise ValueError("model asset hash differs from lock: %s" % name)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must not be initialized")
    torch.set_num_threads(spec["compute_limits"]["threads"])
    return runtime


def select_rows(spec, tokenizer):
    selection = spec["dataset"]["selection"]
    training_n = selection["training_pairs"]
    development_n = selection["development_pairs"]
    rank_start = selection["rank_start"]
    rank_end = selection["rank_end"]
    keep_n = rank_end
    excluded = set(spec["dataset"]["exclusions"]["historical_context_hashes"])
    excluded.update(spec["dataset"]["exclusions"]["private_feasibility_pilot_context_hashes"])
    seen = set()
    heap = []
    counts = {"source_rows": 0, "invalid": 0, "historical_or_pilot": 0,
              "duplicate_context": 0, "eligible_unique": 0}
    source = DATA_DIR / "train.jsonl.gz"
    with gzip.open(str(source), "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            counts["source_rows"] += 1
            raw = json.loads(line)
            item = eligible_pair(raw, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
            if item is None:
                counts["invalid"] += 1
                continue
            context_hash = item["context_hash"]
            if context_hash in excluded:
                counts["historical_or_pilot"] += 1
                continue
            if context_hash in seen:
                counts["duplicate_context"] += 1
                continue
            seen.add(context_hash)
            counts["eligible_unique"] += 1
            context_bytes = item["context"].encode("utf-8")
            domain = selection["domain"].encode("utf-8") + b"\0"
            rank = int.from_bytes(hashlib.sha256(domain + context_bytes).digest(), "big")
            compact = {"source_index": index, "context_hash": context_hash,
                       "context": item["context"], "chosen": item["chosen"], "rejected": item["rejected"],
                       "chosen_tokens": item["chosen_tokens"], "rejected_tokens": item["rejected_tokens"],
                       "rank": rank}
            entry = (-rank, context_bytes, compact)
            if len(heap) < keep_n:
                heapq.heappush(heap, entry)
            else:
                worst = (-heap[0][0], heap[0][1])
                if (rank, context_bytes) < worst:
                    heapq.heapreplace(heap, entry)
    if counts["source_rows"] != spec["dataset"]["expected_split_rows"]["train"]:
        raise ValueError("HH training split row count differs from lock")
    if len(heap) != keep_n:
        raise ValueError("not enough eligible fresh unique HH prompts")
    ranked = sorted((entry[2] for entry in heap), key=lambda row: (row["rank"], row["context"].encode("utf-8")))
    train = ranked[rank_start:rank_start + training_n]
    development = ranked[rank_start + training_n:rank_end]
    if len(train) != training_n or len(development) != development_n:
        raise ValueError("frozen ranked intervals are incomplete")
    if {x["context_hash"] for x in train} & {x["context_hash"] for x in development}:
        raise ValueError("HH train/development context overlap")
    for row in train + development:
        row.pop("context", None)
        row.pop("rank", None)
    return train, development, counts


def encode_rows(rows, tokenizer, model, torch, max_tokens):
    encoded_pairs = []
    for row in rows:
        raw = {"chosen": row["chosen"], "rejected": row["rejected"]}
        item = eligible_pair(raw, tokenizer, max_tokens)
        if item is None or item["context_hash"] != row["context_hash"]:
            raise ValueError("selected HH pair changed eligibility or context identity")
        candidates = []
        for side in ("chosen", "rejected"):
            ids = item[side + "_ids"]
            response_positions = item[side + "_feature_positions"]
            if not response_positions or min(response_positions) < 1:
                raise ValueError("response positions do not have causal predictor states")
            targets = [ids[pos] for pos in response_positions] + [tokenizer.eos_token_id]
            feature_positions = [pos - 1 for pos in response_positions] + [len(ids) - 1]
            input_ids = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                hidden = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                     use_cache=False).last_hidden_state[0]
            candidates.append({"features": hidden[torch.tensor(feature_positions, dtype=torch.long)]
                               .to(dtype=torch.float32).contiguous(),
                               "targets": torch.tensor(targets, dtype=torch.long),
                               "response_token_count": len(response_positions)})
        encoded_pairs.append(candidates)
    return encoded_pairs


def score_candidate(candidate, model, adapter_a, adapter_b, torch, chunk_size, compute_kl=False):
    import torch.nn.functional as F
    features = candidate["features"]
    targets = candidate["targets"]
    total_logp = 0.0
    total_kl = 0.0
    count = 0
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        features_chunk = features[start:stop]
        targets_chunk = targets[start:stop]
        base_logits = F.linear(features_chunk, model.lm_head.weight)
        logp_base = torch.log_softmax(base_logits, dim=-1)
        if adapter_a is None:
            logp_policy = logp_base
        else:
            logits = base_logits + (features_chunk @ adapter_a) @ adapter_b
            logp_policy = torch.log_softmax(logits, dim=-1)
        total_logp += float(logp_policy.gather(1, targets_chunk[:, None]).squeeze(1).sum())
        if compute_kl and adapter_a is not None:
            probabilities = torch.softmax(logits, dim=-1)
            total_kl += float((probabilities * (logp_policy - logp_base)).sum(dim=-1).sum())
            count += len(targets_chunk)
    return total_logp, total_kl / count if count else 0.0


def evaluate(rows, model, adapter_a, adapter_b, torch, chunk_size):
    margins, normalized_margins, kl_values, length_accuracy = [], [], [], []
    with torch.no_grad():
        for pair in rows:
            chosen, chosen_kl = score_candidate(pair[0], model, adapter_a, adapter_b, torch, chunk_size, adapter_a is not None)
            rejected, rejected_kl = score_candidate(pair[1], model, adapter_a, adapter_b, torch, chunk_size, adapter_a is not None)
            margins.append(chosen - rejected)
            chosen_length = pair[0]["response_token_count"]
            rejected_length = pair[1]["response_token_count"]
            chosen_target_count = len(pair[0]["targets"])
            rejected_target_count = len(pair[1]["targets"])
            normalized_margins.append(chosen / chosen_target_count - rejected / rejected_target_count)
            length_accuracy.append(1.0 if chosen_length > rejected_length else
                                   0.5 if chosen_length == rejected_length else 0.0)
            if adapter_a is not None:
                kl_values.extend((chosen_kl, rejected_kl))
    n = len(margins)
    accuracies = [1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in margins]
    nlls = [max(-value, 0.0) + math.log1p(math.exp(-abs(value))) for value in margins]
    normalized_accuracies = [1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in normalized_margins]
    normalized_nlls = [max(-value, 0.0) + math.log1p(math.exp(-abs(value))) for value in normalized_margins]
    return {"n": n, "pair_accuracy": statistics.fmean(accuracies),
            "pair_nll": statistics.fmean(nlls),
            "mean_preferred_minus_rejected_logp": statistics.fmean(margins),
            "mean_token_normalized_pair_accuracy": statistics.fmean(normalized_accuracies),
            "mean_token_normalized_pair_nll": statistics.fmean(normalized_nlls),
            "mean_full_vocab_token_kl_to_base": statistics.fmean(kl_values) if kl_values else 0.0,
            "length_only_pair_accuracy": statistics.fmean(length_accuracy),
            "per_prompt": [{"margin": float(m), "accuracy": float(a), "nll": float(l),
                            "token_normalized_margin": float(nm), "token_normalized_accuracy": float(na),
                            "length_accuracy": float(length)}
                           for m, a, l, nm, na, length in zip(margins, accuracies, nlls, normalized_margins,
                                                               normalized_accuracies, length_accuracy)]}


def score_with_gradient_checkpoint(candidate, model, adapter_a, adapter_b, torch, chunk_size):
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    features, targets = candidate["features"], candidate["targets"]
    total = torch.zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        x, y = features[start:stop], targets[start:stop]
        def chunk_logp(x_chunk, y_chunk, a, b):
            logits = F.linear(x_chunk, model.lm_head.weight) + (x_chunk @ a) @ b
            return torch.log_softmax(logits, dim=-1).gather(1, y_chunk[:, None]).sum()
        total = total + checkpoint(chunk_logp, x, y, adapter_a, adapter_b, use_reentrant=False)
    return total


def length_normalized_relative_margin(relative_margin, chosen_target_count, rejected_target_count):
    if chosen_target_count <= 0 or rejected_target_count <= 0:
        raise ValueError("response target counts, including EOS, must be positive")
    return relative_margin / (0.5 * (chosen_target_count + rejected_target_count))


def train_fixed(train_pairs, model, spec, method, seed, torch):
    learner = spec["learner"]
    chunk_size = spec["compute_limits"]["logit_token_chunk_size"]
    hidden = train_pairs[0][0]["features"].shape[-1]
    rank = learner["adapter_rank"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    adapter_a = torch.nn.Parameter(torch.randn((hidden, rank), generator=generator) / math.sqrt(hidden))
    adapter_b = torch.nn.Parameter(torch.zeros((rank, model.lm_head.weight.shape[0])))
    optimizer = torch.optim.AdamW([adapter_a, adapter_b], lr=learner["optimizer"]["learning_rate"],
                                  weight_decay=learner["optimizer"]["weight_decay"])
    order_rng = random.Random(seed + 99000)
    order = list(range(len(train_pairs)))
    epoch_losses = []
    for epoch in range(1, learner["epochs"] + 1):
        order_rng.shuffle(order)
        losses = []
        for index in order:
            chosen, rejected = train_pairs[index]
            optimizer.zero_grad(set_to_none=True)
            if method in ("dpo", "length_normalized_dpo"):
                with torch.no_grad():
                    chosen_ref, _ = score_candidate(chosen, model, None, None, torch, chunk_size)
                    rejected_ref, _ = score_candidate(rejected, model, None, None, torch, chunk_size)
                chosen_logp = score_with_gradient_checkpoint(chosen, model, adapter_a, adapter_b, torch, chunk_size)
                rejected_logp = score_with_gradient_checkpoint(rejected, model, adapter_a, adapter_b, torch, chunk_size)
                relative_margin = (chosen_logp - rejected_logp) - (chosen_ref - rejected_ref)
                if method == "length_normalized_dpo":
                    relative_margin = length_normalized_relative_margin(
                        relative_margin, len(chosen["targets"]), len(rejected["targets"]))
                loss = torch.nn.functional.softplus(-learner["beta"] * relative_margin)
            elif method == "chosen_sft":
                chosen_logp = score_with_gradient_checkpoint(chosen, model, adapter_a, adapter_b, torch, chunk_size)
                loss = -chosen_logp / len(chosen["targets"])
            else:
                raise ValueError("unknown HH preference training method")
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all()
                                                for p in (adapter_a, adapter_b)):
                raise FloatingPointError("non-finite HH training objective or gradient")
            torch.nn.utils.clip_grad_norm_([adapter_a, adapter_b], learner["optimizer"]["gradient_norm_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        epoch_losses.append(statistics.fmean(losses))
        print("training", method, seed, epoch, "loss", epoch_losses[-1], flush=True)
    return adapter_a.detach(), adapter_b.detach(), epoch_losses


def paired_bootstrap(left, right, n_resamples, seed):
    import numpy as np
    if len(left) != len(right) or not left:
        raise ValueError("paired metric vectors differ or are empty")
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(delta), size=(n_resamples, len(delta)))
    distribution = delta[sampled].mean(axis=1)
    return [float(x) for x in np.quantile(distribution, [0.025, 0.975], method="linear")]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def run(output: Path):
    spec, protocol_hash = load_protocol()
    if output.exists():
        raise FileExistsError("output run path already exists")
    output.mkdir(parents=True)
    start = time.monotonic()
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    (output / "runner.snapshot.py").write_bytes(Path(__file__).read_bytes())
    (output / "auditor.snapshot.py").write_bytes(Path(__file__).with_name("audit_cpu_hh_length_normalized_dpo_development_v1.py").read_bytes())
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[key] = "1"
    runtime = verify_assets(spec)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
    if not tokenizer.is_fast or tokenizer.eos_token_id is None:
        raise RuntimeError("pinned fast tokenizer and EOS token are required")
    with ResourceGuard(spec["compute_limits"]["max_wall_seconds"], spec["compute_limits"]["max_peak_rss_bytes"]):
        train_rows, dev_rows, selection_counts = select_rows(spec, tokenizer)
        for record in train_rows + dev_rows:
            record.pop("chosen", None)
            record.pop("rejected", None)
        # Reload only the frozen selected source rows; no other prompt text is retained.
        selected_indices = {x["source_index"]: x for x in train_rows + dev_rows}
        source_selected = {}
        with gzip.open(str(DATA_DIR / "train.jsonl.gz"), "rt", encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index in selected_indices:
                    source_selected[index] = json.loads(line)
                if len(source_selected) == len(selected_indices):
                    break
        if set(source_selected) != set(selected_indices):
            raise ValueError("selected HH source rows could not be reopened")
        # Reattach source text only in memory for the feature pass.
        for record in train_rows + dev_rows:
            raw = source_selected[record["source_index"]]
            record["chosen"], record["rejected"] = raw["chosen"], raw["rejected"]
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=True).to(torch.device("cpu")).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        train_pairs = encode_rows(train_rows, tokenizer, model, torch, spec["compute_limits"]["max_sequence_tokens"])
        dev_pairs = encode_rows(dev_rows, tokenizer, model, torch, spec["compute_limits"]["max_sequence_tokens"])
        chunk_size = spec["compute_limits"]["logit_token_chunk_size"]
        base = evaluate(dev_pairs, model, None, None, torch, chunk_size)
        methods = spec["learner"]["methods"]
        method_results = {"base": {"per_prompt": base.pop("per_prompt"), "metrics": base},
                          **{method: {} for method in methods}}
        adapter_dir = output / "adapters"
        adapter_dir.mkdir()
        learner_cfg = spec["learner"]
        for method in methods:
            for seed in learner_cfg["seeds"]:
                adapter_a, adapter_b, training_loss = train_fixed(train_pairs, model, spec, method, seed, torch)
                metrics = evaluate(dev_pairs, model, adapter_a, adapter_b, torch, chunk_size)
                safe_method = method.replace("/", "_")
                adapter_path = adapter_dir / ("%s-seed-%d.pt" % (safe_method, seed))
                torch.save({"A": adapter_a.detach().cpu(), "B": adapter_b.detach().cpu()}, adapter_path)
                method_results[method][str(seed)] = {"metrics": {k:v for k,v in metrics.items() if k!="per_prompt"},
                    "per_prompt": metrics["per_prompt"], "adapter_file": adapter_path.relative_to(output).as_posix(),
                    "adapter_sha256": digest_file(adapter_path),
                    "training_loss_by_epoch": training_loss,
                    "adapter_a_l2": float(torch.linalg.vector_norm(adapter_a).detach()),
                    "adapter_b_l2": float(torch.linalg.vector_norm(adapter_b).detach())}
                print("completed", method, seed, json.dumps(method_results[method][str(seed)]["metrics"]), flush=True)
        n = len(dev_rows)
        metric_by_method = {}
        for method in methods:
            metric_by_method[method] = {}
            for metric in ("pair_accuracy", "pair_nll", "mean_full_vocab_token_kl_to_base"):
                metric_by_method[method][metric] = [method_results[method][str(seed)]["metrics"][metric]
                    for seed in learner_cfg["seeds"]]
        # Average fixed-seed per-prompt outcomes first, then resample independent prompts.
        candidate = "length_normalized_dpo"
        candidate_acc_rows = [statistics.fmean([method_results[candidate][str(s)]["per_prompt"][i]["accuracy"]
                                                for s in learner_cfg["seeds"]]) for i in range(n)]
        dpo_acc_rows = [statistics.fmean([method_results["dpo"][str(s)]["per_prompt"][i]["accuracy"]
                                          for s in learner_cfg["seeds"]]) for i in range(n)]
        sft_acc_rows = [statistics.fmean([method_results["chosen_sft"][str(s)]["per_prompt"][i]["accuracy"]
                                          for s in learner_cfg["seeds"]]) for i in range(n)]
        base_acc_rows = [method_results["base"]["per_prompt"][i]["accuracy"] for i in range(n)]
        cfg = spec["metrics"]["bootstrap"]
        candidate_base_ci = paired_bootstrap(candidate_acc_rows, base_acc_rows, cfg["resamples"], cfg["seed"])
        candidate_dpo_ci = paired_bootstrap(candidate_acc_rows, dpo_acc_rows, cfg["resamples"], cfg["seed"] + 1)
        candidate_sft_ci = paired_bootstrap(candidate_acc_rows, sft_acc_rows, cfg["resamples"], cfg["seed"] + 2)
        base_acc = statistics.fmean(base_acc_rows)
        candidate_acc = statistics.fmean(candidate_acc_rows)
        dpo_acc = statistics.fmean(dpo_acc_rows)
        sft_acc = statistics.fmean(sft_acc_rows)
        gates = spec["metrics"]["development_gate"]
        candidate_seed_gains_base = [method_results[candidate][str(s)]["metrics"]["pair_accuracy"] -
                                     method_results["base"]["metrics"]["pair_accuracy"] for s in learner_cfg["seeds"]]
        candidate_seed_gains_dpo = [method_results[candidate][str(s)]["metrics"]["pair_accuracy"] -
                                    method_results["dpo"][str(s)]["metrics"]["pair_accuracy"] for s in learner_cfg["seeds"]]
        kl_seed_pass = all(method_results[candidate][str(s)]["metrics"]["mean_full_vocab_token_kl_to_base"] <=
                           gates["maximum_mean_full_vocab_token_kl_per_seed"] for s in learner_cfg["seeds"])
        nll_guard = statistics.fmean(metric_by_method[candidate]["pair_nll"]) <= \
                    statistics.fmean([method_results["base"]["metrics"]["pair_nll"] for _ in learner_cfg["seeds"]]) + gates["nll_guard_vs_base_nats"]
        decision = {
            "base_floor_pass": base_acc >= gates["minimum_base_pair_accuracy"],
            "candidate_gain_vs_base_pass": candidate_acc - base_acc >= gates["minimum_length_normalized_dpo_gain_vs_base"] and candidate_base_ci[0] > 0.0,
            "candidate_gain_vs_standard_dpo_pass": candidate_acc - dpo_acc >= gates["minimum_length_normalized_dpo_gain_vs_standard_dpo"] and candidate_dpo_ci[0] > 0.0,
            "candidate_seed_consistency_vs_base_pass": sum(x > 0 for x in candidate_seed_gains_base) >= gates["minimum_seed_gains_vs_base"],
            "candidate_seed_consistency_vs_standard_dpo_pass": sum(x > 0 for x in candidate_seed_gains_dpo) >= gates["minimum_seed_gains_vs_standard_dpo"],
            "candidate_sft_noninferiority_pass": candidate_acc - sft_acc >= -gates["maximum_pair_accuracy_regression_vs_chosen_sft"] and candidate_sft_ci[0] > -gates["maximum_pair_accuracy_regression_vs_chosen_sft"],
            "kl_pass": kl_seed_pass,
            "nll_guard_pass": nll_guard,
        }
        decision["pass"] = all(decision.values())
        summary = {"protocol_sha256": protocol_hash, "status": "completed", "decision": decision,
            "selection_counts": selection_counts, "train_selection": [{k:r[k] for k in ("source_index","context_hash","chosen_tokens","rejected_tokens")} for r in train_rows],
            "development_selection": [{k:r[k] for k in ("source_index","context_hash","chosen_tokens","rejected_tokens")} for r in dev_rows],
            "base_metrics": method_results["base"]["metrics"],
            "base_per_prompt": method_results["base"]["per_prompt"],
            "arms": {k:v for k,v in method_results.items() if k!="base"},
            "aggregate": {"base_pair_accuracy": base_acc, "standard_dpo_pair_accuracy": dpo_acc,
                "length_normalized_dpo_pair_accuracy": candidate_acc, "chosen_sft_pair_accuracy": sft_acc,
                "length_normalized_dpo_minus_base_accuracy": candidate_acc-base_acc,
                "length_normalized_dpo_minus_base_accuracy_prompt_bootstrap_95":candidate_base_ci,
                "length_normalized_dpo_minus_standard_dpo_accuracy": candidate_acc-dpo_acc,
                "length_normalized_dpo_minus_standard_dpo_accuracy_prompt_bootstrap_95":candidate_dpo_ci,
                "length_normalized_dpo_minus_sft_accuracy": candidate_acc-sft_acc,
                "length_normalized_dpo_minus_sft_accuracy_prompt_bootstrap_95":candidate_sft_ci,
                "length_normalized_dpo_seed_gains_vs_base":candidate_seed_gains_base,
                "length_normalized_dpo_seed_gains_vs_standard_dpo":candidate_seed_gains_dpo,
                "length_normalized_dpo_mean_pair_nll":statistics.fmean(metric_by_method[candidate]['pair_nll']),
                "standard_dpo_mean_pair_nll":statistics.fmean(metric_by_method['dpo']['pair_nll']),
                "chosen_sft_mean_pair_nll":statistics.fmean(metric_by_method['chosen_sft']['pair_nll']),
                "length_normalized_dpo_mean_token_kl_by_seed":metric_by_method[candidate]['mean_full_vocab_token_kl_to_base'],
                "base_length_only_pair_accuracy":method_results['base']['metrics']['length_only_pair_accuracy'],
                "length_normalized_dpo_mean_token_pair_accuracy":metric_by_method[candidate]['mean_token_normalized_pair_accuracy'],
                "length_normalized_dpo_length_only_pair_accuracy_by_seed":[method_results[candidate][str(s)]['metrics']['length_only_pair_accuracy'] for s in learner_cfg['seeds']]},
            "runtime":runtime,"inputs_sha256":{"train.jsonl.gz":spec['dataset']['file_sha256_at_lock']['helpful-base/train.jsonl.gz'],
                "test.jsonl.gz_opaque_only":spec['dataset']['file_sha256_at_lock']['helpful-base/test.jsonl.gz'],
                "model_files":spec['model']['files_sha256_at_lock']},
            "wall_seconds":round(time.monotonic()-start,2),"peak_rss_bytes":rss_bytes()}
        summary_path=output/"summary.json"
        write_json(summary_path,summary)
        # Release raw text and tensor features before hashing the final bundle.
        del train_rows, dev_rows, train_pairs, dev_pairs, source_selected, model
        write_manifest(output)
        print(json.dumps({"status":summary['status'],"decision":decision,"aggregate":summary['aggregate'],
            "wall_seconds":summary['wall_seconds'],"peak_rss_bytes":summary['peak_rss_bytes'],
            "output":str(output)},indent=2),flush=True)
        return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUTPUT_ROOT/"run-1")
    args=parser.parse_args()
    run(args.output)


if __name__=="__main__":
    main()
