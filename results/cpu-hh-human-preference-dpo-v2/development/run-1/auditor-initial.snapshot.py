#!/usr/bin/env python3
"""Independent replay audit for HH human-preference DPO development v1."""
from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC_PATH = ROOT / "protocols/cpu_hh_human_dpo_development_v2.json"
LOCK_PATH = ROOT / "protocols/cpu_hh_human_dpo_development_v2.lock.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/hub/datasets--Anthropic--hh-rlhf/snapshots/09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa/helpful-base"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def locked_protocol():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    digest = lock.pop("sha256", None)
    actual = hashlib.sha256(canonical(spec)).hexdigest()
    if digest != actual or canonical(lock) != canonical(spec):
        raise ValueError("frozen protocol/lock mismatch")
    return spec, actual


def parse_and_tokenize(raw, tokenizer, maximum):
    chosen, rejected = raw.get("chosen"), raw.get("rejected")
    marker = "\n\nAssistant:"
    if not isinstance(chosen, str) or not isinstance(rejected, str):
        return None
    cpos, rpos = chosen.rfind(marker), rejected.rfind(marker)
    if cpos < 0 or rpos < 0:
        return None
    chosen_context = chosen[:cpos + len(marker)]
    rejected_context = rejected[:rpos + len(marker)]
    if chosen_context != rejected_context:
        return None
    ctext = chosen[cpos + len(marker):]
    rtext = rejected[rpos + len(marker):]
    if not ctext or not rtext:
        return None
    sides = []
    for transcript, response_start in ((chosen, len(chosen_context)), (rejected, len(rejected_context))):
        encoded = tokenizer(transcript, add_special_tokens=True, truncation=False,
                            return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) > maximum:
            return None
        positions = [i for i, (begin, end) in enumerate(offsets)
                     if end > begin and begin >= response_start]
        if not positions:
            return None
        sides.append({"ids": ids, "response_positions": positions, "response_tokens": len(positions)})
    domain = b"vare-hh-helpful-context-v1\0"
    context_digest = hashlib.sha256(domain + chosen_context.encode("utf-8")).hexdigest()
    return {"context": chosen_context, "context_hash": context_digest,
            "chosen": chosen, "rejected": rejected, "chosen_tokens": sides[0]["response_tokens"],
            "rejected_tokens": sides[1]["response_tokens"], "sides": sides}


def independently_select(spec, tokenizer):
    selection = spec["dataset"]["selection"]
    take = selection["training_pairs"] + selection["development_pairs"]
    exclusions = set(spec["dataset"]["exclusions"]["historical_context_hashes"])
    exclusions.update(spec["dataset"]["exclusions"]["private_feasibility_pilot_context_hashes"])
    seen, heap = set(), []
    source_rows = 0
    invalid = duplicate = excluded = unique_count = 0
    with gzip.open(str(DATA_DIR / "train.jsonl.gz"), "rt", encoding="utf-8") as source:
        for index, line in enumerate(source):
            source_rows += 1
            raw = json.loads(line)
            item = parse_and_tokenize(raw, tokenizer, spec["compute_limits"]["max_sequence_tokens"])
            if item is None:
                invalid += 1
                continue
            identity = item["context_hash"]
            if identity in exclusions:
                excluded += 1
                continue
            if identity in seen:
                duplicate += 1
                continue
            seen.add(identity)
            unique_count += 1
            ctx = item["context"].encode("utf-8")
            rank = int.from_bytes(hashlib.sha256(b"vare-hh-human-dpo-development-v1\0" + ctx).digest(), "big")
            compact = {"source_index": index, "context_hash": identity, "context": item["context"],
                       "chosen": item["chosen"], "rejected": item["rejected"],
                       "chosen_tokens": item["chosen_tokens"], "rejected_tokens": item["rejected_tokens"],
                       "rank": rank}
            slot = (-rank, ctx, compact)
            if len(heap) < take:
                heapq.heappush(heap, slot)
            elif (rank, ctx) < (-heap[0][0], heap[0][1]):
                heapq.heapreplace(heap, slot)
    if source_rows != spec["dataset"]["expected_split_rows"]["train"]:
        raise ValueError("source train count changed")
    ranked = sorted((entry[2] for entry in heap), key=lambda r: (r["rank"], r["context"].encode("utf-8")))
    train = ranked[:selection["training_pairs"]]
    dev = ranked[selection["training_pairs"]:take]
    inventory = {"source_rows": source_rows, "invalid": invalid, "historical_or_pilot": excluded,
                 "duplicate_context": duplicate, "eligible_unique": unique_count}
    return train, dev, inventory


def encode_independently(rows, tokenizer, model, torch, maximum):
    output = []
    with torch.no_grad():
        for row in rows:
            pair = parse_and_tokenize({"chosen": row["chosen"], "rejected": row["rejected"]}, tokenizer, maximum)
            if pair is None or pair["context_hash"] != row["context_hash"]:
                raise ValueError("reselected pair failed independent tokenization")
            candidates = []
            for side in pair["sides"]:
                ids = side["ids"]
                response_positions = side["response_positions"]
                if min(response_positions) < 1:
                    raise ValueError("response token has no causal feature")
                target_ids = [ids[position] for position in response_positions] + [tokenizer.eos_token_id]
                feature_positions = [position - 1 for position in response_positions] + [len(ids) - 1]
                input_ids = torch.tensor([ids], dtype=torch.long)
                states = model.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                     use_cache=False).last_hidden_state[0]
                candidates.append({"features": states[torch.tensor(feature_positions, dtype=torch.long)].float(),
                                   "targets": torch.tensor(target_ids, dtype=torch.long),
                                   "response_token_count": side["response_tokens"]})
            output.append(candidates)
    return output


def score_one(candidate, model, adapter, torch, chunk_size):
    import torch.nn.functional as F
    features, targets = candidate["features"], candidate["targets"]
    total = 0.0
    kl_total = 0.0
    token_count = 0
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        x, y = features[start:stop], targets[start:stop]
        reference_logits = F.linear(x, model.lm_head.weight)
        logp_ref = torch.log_softmax(reference_logits, dim=-1)
        if adapter is None:
            logp = logp_ref
        else:
            policy_logits = reference_logits + (x @ adapter["A"]) @ adapter["B"]
            logp = torch.log_softmax(policy_logits, dim=-1)
            probabilities = torch.softmax(policy_logits, dim=-1)
            kl_total += float((probabilities * (logp - logp_ref)).sum(dim=-1).sum())
            token_count += len(y)
        total += float(logp.gather(1, y[:, None]).squeeze(1).sum())
    return total, kl_total / token_count if token_count else 0.0


def evaluate_independently(rows, model, adapter, torch, chunk_size):
    margins, token_kls, lengths = [], [], []
    with torch.no_grad():
        for chosen, rejected in rows:
            chosen_logp, chosen_kl = score_one(chosen, model, adapter, torch, chunk_size)
            rejected_logp, rejected_kl = score_one(rejected, model, adapter, torch, chunk_size)
            margins.append(chosen_logp - rejected_logp)
            if adapter is not None:
                token_kls.extend((chosen_kl, rejected_kl))
            c_len, r_len = chosen["response_token_count"], rejected["response_token_count"]
            lengths.append(1.0 if c_len > r_len else 0.5 if c_len == r_len else 0.0)
    accuracy = [1.0 if x > 0 else 0.5 if x == 0 else 0.0 for x in margins]
    nll = [max(-x, 0.0) + math.log1p(math.exp(-abs(x))) for x in margins]
    return {"n": len(margins), "pair_accuracy": statistics.fmean(accuracy),
            "pair_nll": statistics.fmean(nll), "mean_preferred_minus_rejected_logp": statistics.fmean(margins),
            "mean_full_vocab_token_kl_to_base": statistics.fmean(token_kls) if token_kls else 0.0,
            "length_only_pair_accuracy": statistics.fmean(lengths),
            "per_prompt": [{"margin":m,"accuracy":a,"nll":l,"length_accuracy":z}
                           for m,a,l,z in zip(margins,accuracy,nll,lengths)]}


def paired_bootstrap(a, b, reps, seed):
    import numpy as np
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(reps, len(delta)))
    return [float(x) for x in np.quantile(delta[indices].mean(axis=1), [0.025,0.975], method="linear")]


def close(a, b, tol=1e-5):
    return abs(float(a)-float(b)) <= tol


def audit(bundle: Path):
    spec, protocol_hash = locked_protocol()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_hash = file_sha(manifest_path)
    for rel, expected in manifest.get("files", {}).items():
        path = bundle / rel
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError("bundle manifest mismatch: %s" % rel)
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    if summary.get("protocol_sha256") != protocol_hash:
        raise ValueError("bundle protocol digest differs")
    if json.loads((bundle / "protocol.snapshot.json").read_text(encoding="utf-8")) != spec:
        raise ValueError("bundle protocol snapshot differs from the frozen protocol")
    frozen_lock = json.loads((bundle / "protocol.lock.snapshot.json").read_text(encoding="utf-8"))
    if frozen_lock != json.loads(LOCK_PATH.read_text(encoding="utf-8")):
        raise ValueError("bundle frozen protocol lock differs")
    if summary.get("inputs_sha256", {}).get("train.jsonl.gz") != spec["dataset"]["file_sha256_at_lock"]["helpful-base/train.jsonl.gz"]:
        raise ValueError("bundle train data digest differs")
    if file_sha(DATA_DIR / "train.jsonl.gz") != spec["dataset"]["file_sha256_at_lock"]["helpful-base/train.jsonl.gz"]:
        raise ValueError("local train data changed")
    if file_sha(DATA_DIR / "test.jsonl.gz") != spec["dataset"]["file_sha256_at_lock"]["helpful-base/test.jsonl.gz"]:
        raise ValueError("local opaque test container changed")
    for rel, expected in spec["model"]["files_sha256_at_lock"].items():
        if file_sha(MODEL_DIR / rel) != expected:
            raise ValueError("local model asset changed: %s" % rel)
    # Ensure user data is not copied into public experiment bundles.
    forbidden = {"prompt", "context", "chosen", "rejected", "transcript", "input_ids", "token_ids", "hidden_states"}
    stack = [summary]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("bundle contains a prohibited raw-data field")
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, use_fast=True)
    train_rows, dev_rows, counts = independently_select(spec, tokenizer)
    expected_train = summary["train_selection"]
    expected_dev = summary["development_selection"]
    project = lambda rows: [{"source_index":r["source_index"],"context_hash":r["context_hash"],
                             "chosen_tokens":r["chosen_tokens"],"rejected_tokens":r["rejected_tokens"]} for r in rows]
    if project(train_rows) != expected_train or project(dev_rows) != expected_dev:
        raise ValueError("independent prompt selection differs from bundle")
    if counts != summary["selection_counts"]:
        raise ValueError("independent selection counters differ")
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to(torch.device("cpu")).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dev_features = encode_independently(dev_rows, tokenizer, model, torch,
        spec["compute_limits"]["max_sequence_tokens"])
    chunk_size=spec["compute_limits"]["logit_token_chunk_size"]
    base = evaluate_independently(dev_features, model, None, torch, chunk_size)
    for key in ("pair_accuracy","pair_nll","mean_preferred_minus_rejected_logp",
                "mean_full_vocab_token_kl_to_base","length_only_pair_accuracy"):
        if not close(base[key],summary["base_metrics"][key]):
            raise ValueError("independent base %s metric differs" % key)
    for fresh, retained in zip(base["per_prompt"],summary["base_per_prompt"]):
        for key in ("margin","accuracy","nll","length_accuracy"):
            if not close(fresh[key],retained[key]):
                raise ValueError("independent base per-prompt %s differs" % key)
    seed_results = {"base":base,"dpo":{},"chosen_sft":{}}
    for method in ("dpo","chosen_sft"):
        for seed in spec["learner"]["seeds"]:
            record = summary["arms"][method][str(seed)]
            adapter_path = bundle / record["adapter_file"]
            if file_sha(adapter_path) != record["adapter_sha256"]:
                raise ValueError("adapter digest mismatch")
            adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
            if set(adapter) != {"A","B"} or not adapter["A"].numel() or not adapter["B"].numel():
                raise ValueError("adapter file has invalid tensors")
            if float(torch.linalg.vector_norm(adapter["B"])) <= 0.0:
                raise ValueError("adapter is still the zero-update initialization")
            metrics = evaluate_independently(dev_features, model, adapter, torch, chunk_size)
            saved = record["metrics"]
            for key in ("pair_accuracy","pair_nll","mean_preferred_minus_rejected_logp",
                        "mean_full_vocab_token_kl_to_base","length_only_pair_accuracy"):
                if not close(metrics[key],saved[key]):
                    raise ValueError("independent %s metric differs for %s seed %s" % (key,method,seed))
            for fresh, retained in zip(metrics["per_prompt"],record["per_prompt"]):
                for key in ("margin","accuracy","nll","length_accuracy"):
                    if not close(fresh[key],retained[key]):
                        raise ValueError("independent per-prompt %s differs" % key)
            seed_results[method][str(seed)]=metrics
    n=len(dev_rows);seeds=spec["learner"]["seeds"];cfg=spec["metrics"]["bootstrap"]
    dpo_acc=[statistics.fmean([seed_results["dpo"][str(s)]["per_prompt"][i]["accuracy"] for s in seeds]) for i in range(n)]
    sft_acc=[statistics.fmean([seed_results["chosen_sft"][str(s)]["per_prompt"][i]["accuracy"] for s in seeds]) for i in range(n)]
    base_acc=[x["accuracy"] for x in base["per_prompt"]]
    dpo_base_ci=paired_bootstrap(dpo_acc,base_acc,cfg["resamples"],cfg["seed"])
    dpo_sft_ci=paired_bootstrap(dpo_acc,sft_acc,cfg["resamples"],cfg["seed"]+1)
    mean_base=statistics.fmean(base_acc);mean_dpo=statistics.fmean(dpo_acc);mean_sft=statistics.fmean(sft_acc)
    gates=spec["metrics"]["development_gate"]
    seed_gains=[seed_results["dpo"][str(s)]["pair_accuracy"]-base["pair_accuracy"] for s in seeds]
    kl_pass=all(seed_results["dpo"][str(s)]["mean_full_vocab_token_kl_to_base"]<=gates["maximum_mean_full_vocab_token_kl_per_seed"] for s in seeds)
    dpo_nll=statistics.fmean([seed_results["dpo"][str(s)]["pair_nll"] for s in seeds])
    decision={"base_floor_pass":mean_base>=gates["minimum_base_pair_accuracy"],
        "dpo_gain_pass":mean_dpo-mean_base>=gates["minimum_mean_dpo_accuracy_gain_vs_base"] and dpo_base_ci[0]>0,
        "seed_consistency_pass":sum(x>0 for x in seed_gains)>=gates["minimum_seed_gains"],
        "dpo_sft_noninferiority_pass":mean_dpo-mean_sft>=-gates["maximum_dpo_minus_sft_accuracy_regression"] and dpo_sft_ci[0]>-gates["maximum_dpo_minus_sft_accuracy_regression"],
        "kl_pass":kl_pass,
        "nll_guard_pass":dpo_nll<=base["pair_nll"]+1.0}
    decision["pass"]=all(decision.values())
    if decision != summary["decision"]:
        raise ValueError("independently recomputed advancement decision differs")
    if summary.get("status")!="completed":
        raise ValueError("bundle run status is not completed")
    report={"status":"pass","protocol_sha256":protocol_hash,"manifest_sha256":manifest_hash,
        "selection_replay":"pass","adapter_score_replay":"pass","decision_replay":"pass",
        "decision":decision,"recomputed":{"base_pair_accuracy":mean_base,"dpo_pair_accuracy":mean_dpo,
            "chosen_sft_pair_accuracy":mean_sft,"dpo_minus_base_accuracy":mean_dpo-mean_base,
            "dpo_minus_base_accuracy_prompt_bootstrap_95":dpo_base_ci,
            "dpo_minus_sft_accuracy":mean_dpo-mean_sft,"dpo_minus_sft_accuracy_prompt_bootstrap_95":dpo_sft_ci,
            "dpo_seed_gains_vs_base":seed_gains},
        "scope":"Same-host offline replay of the selected prompt inventory and retained adapters; not external reproduction or a retraining audit."}
    (bundle/"audit.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("bundle",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.bundle),indent=2,sort_keys=True))


if __name__=="__main__":
    main()
