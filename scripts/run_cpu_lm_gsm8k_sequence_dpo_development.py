#!/usr/bin/env python3
"""Train-only development of a sequence-level DPO answer adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from typing import Any

from gsm8k_preference_task import digest_text
from gsm8k_sequence_task import make_sequence_rows, parse_generated_number

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_lm_gsm8k_sequence_dpo_development_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_lm_gsm8k_sequence_dpo_development_v1.lock.json"
MODEL_DIR = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_DIR = Path.home() / ".cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_manifest(output: Path) -> None:
    files = {p.relative_to(output).as_posix(): sha256_file(p)
             for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    write_json(output / "manifest.json", {"algorithm": "sha256", "files": files})


def load_locked_spec(spec_path: Path = SPEC_PATH, lock_path: Path = LOCK_PATH):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_hash = lock.pop("sha256", None)
    spec_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_hash != spec_hash or canonical(lock) != canonical(spec):
        raise ValueError("sequence DPO development protocol differs from its lock")
    return spec, spec_hash


def sigmoid_nll(value: float) -> float:
    return max(-value, 0.0) + math.log1p(math.exp(-abs(value)))


def encode_candidates(rows, tokenizer, model, spec):
    import torch
    candidates = []
    messages = []
    for pair_index, row in enumerate(rows):
        messages.append((pair_index, "chosen", row["verifier_answer"], row["user_message"]))
        messages.append((pair_index, "rejected", row["rejected_answer"], row["user_message"]))
    batch_size = spec["compute_limits"]["feature_batch_size"]
    for offset in range(0, len(messages), batch_size):
        chunk = messages[offset:offset + batch_size]
        tokenized = []
        for pair_index, kind, answer, user_message in chunk:
            prefix_ids = tokenizer.apply_chat_template([{"role": "user", "content": user_message}],
                                                       tokenize=True, add_generation_prompt=True)
            prefix_text = tokenizer.apply_chat_template([{"role": "user", "content": user_message}],
                                                        tokenize=False, add_generation_prompt=True)
            if tokenizer.encode(prefix_text, add_special_tokens=False) != prefix_ids:
                raise ValueError("chat prefix text and token IDs do not reconstruct")
            completion = " " + answer
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            full_ids = tokenizer.encode(prefix_text + completion, add_special_tokens=False)
            if full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError("chat/completion tokenizer boundary changed the prefix")
            response_ids = full_ids[len(prefix_ids):]
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has no EOS token")
            response_ids = response_ids + [tokenizer.eos_token_id]
            if not response_ids or not completion_ids:
                raise ValueError("response completion tokenized to an empty sequence")
            input_ids = prefix_ids + response_ids
            feature_positions = list(range(len(prefix_ids) - 1, len(prefix_ids) + len(response_ids) - 1))
            tokenized.append({"pair_index": pair_index, "kind": kind, "input_ids": input_ids,
                              "response_ids": response_ids, "feature_positions": feature_positions,
                              "prefix_length": len(prefix_ids), "completion": completion})
        padded = tokenizer.pad({"input_ids": [row["input_ids"] for row in tokenized]},
                               padding=True, return_tensors="pt")
        lengths = padded["attention_mask"].sum(dim=1)
        if tokenizer.padding_side != "right":
            raise ValueError("feature extraction requires right padding")
        with torch.no_grad():
            hidden_all = model.model(input_ids=padded["input_ids"], attention_mask=padded["attention_mask"],
                                     use_cache=False).last_hidden_state
            for i, row in enumerate(tokenized):
                positions = torch.tensor(row["feature_positions"], dtype=torch.long)
                features = hidden_all[i, positions].to(dtype=torch.float32).contiguous()
                row["features"] = features
                row["targets"] = torch.tensor(row["response_ids"], dtype=torch.long)
                candidates.append(row)
    by_pair = {i: {} for i in range(len(rows))}
    for candidate in candidates:
        by_pair[candidate["pair_index"]][candidate["kind"]] = candidate
    return [[by_pair[i]["chosen"], by_pair[i]["rejected"]] for i in range(len(rows))]


def score_candidate_batch(candidates, model, adapter_a, adapter_b):
    import torch
    import torch.nn.functional as F
    features = torch.cat([candidate["features"] for candidate in candidates], dim=0)
    targets = torch.cat([candidate["targets"] for candidate in candidates], dim=0)
    base_logits = F.linear(features, model.lm_head.weight)
    delta_logits = torch.zeros_like(base_logits)
    if adapter_a is not None and adapter_b is not None:
        delta_logits = (features @ adapter_a) @ adapter_b
    policy_logits = base_logits + delta_logits
    log_probs = torch.log_softmax(policy_logits, dim=-1)
    chosen_logps = log_probs.gather(1, targets[:, None]).squeeze(1)
    offsets = [0]
    for candidate in candidates:
        offsets.append(offsets[-1] + len(candidate["targets"]))
    seq_logps = [chosen_logps[offsets[i]:offsets[i + 1]].sum() for i in range(len(candidates))]
    return seq_logps, base_logits, policy_logits, chosen_logps, offsets


def evaluate_pairs(pairs, model, adapter_a, adapter_b, beta, microbatch):
    import torch
    losses, correct, kls = [], [], []
    reference_margins, policy_margins, relative_margins = [], [], []
    for start in range(0, len(pairs), microbatch):
        chunk = pairs[start:start + microbatch]
        candidates = [candidate for pair in chunk for candidate in pair]
        with torch.no_grad():
            seq_logps, base_logits, policy_logits, _, offsets = score_candidate_batch(candidates, model, adapter_a, adapter_b)
            logp_ref = torch.log_softmax(base_logits, dim=-1)
            logp_pi = torch.log_softmax(policy_logits, dim=-1)
            probs_pi = torch.softmax(policy_logits, dim=-1)
            token_kl = (probs_pi * (logp_pi - logp_ref)).sum(dim=-1)
            kls.extend(token_kl.tolist())
            for pair_index in range(len(chunk)):
                i = 2 * pair_index
                chosen_ref = logp_ref[offsets[i]:offsets[i + 1]].gather(1, candidates[i]["targets"][:, None]).sum()
                rejected_ref = logp_ref[offsets[i + 1]:offsets[i + 2]].gather(1, candidates[i + 1]["targets"][:, None]).sum()
                ref_margin = chosen_ref - rejected_ref
                policy_margin = seq_logps[i] - seq_logps[i + 1]
                relative = float(policy_margin - ref_margin)
                reference_margins.append(float(ref_margin))
                policy_margins.append(float(policy_margin))
                relative_margins.append(relative)
                losses.append(sigmoid_nll(beta * relative))
                correct.append(1.0 if relative > 0 else 0.5 if relative == 0 else 0.0)
    return {"mean_dpo_preference_nll": statistics.fmean(losses),
            "preference_accuracy": statistics.fmean(correct),
            "mean_full_vocab_token_kl_to_base": statistics.fmean(kls),
            "reference_preference_margins": reference_margins,
            "policy_preference_margins": policy_margins,
            "relative_preference_margins": relative_margins}


def train_to_checkpoints(train_pairs, val_pairs, model, spec, seed):
    import torch
    import torch.nn.functional as F
    learner = spec["learner"]
    compute = spec["compute_limits"]
    hidden_size = train_pairs[0][0]["features"].shape[-1]
    rank = learner["adapter_rank"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    adapter_a = torch.nn.Parameter(torch.randn((hidden_size, rank), generator=generator) / math.sqrt(hidden_size))
    adapter_b = torch.nn.Parameter(torch.zeros((rank, model.lm_head.weight.shape[0])))
    initial = {"A": adapter_a.detach().clone(), "B": adapter_b.detach().clone()}
    optimizer = torch.optim.AdamW([adapter_a, adapter_b],
                                  lr=learner["optimizer"]["learning_rate"],
                                  weight_decay=learner["optimizer"]["weight_decay"])
    beta = learner["beta"]
    checkpoints = set(learner["checkpoints_epochs"])
    checkpoint_data = {}
    order_rng = __import__("random").Random(seed + 99000)
    examples = list(range(len(train_pairs)))
    first_epoch = 0
    if 0 in checkpoints:
        metrics = evaluate_pairs(val_pairs, model, None, None, beta, compute["pair_microbatch_size"])
        checkpoint_data["0"] = {"metrics": metrics,
                                "adapter": {"A": initial["A"].clone(), "B": initial["B"].clone()}}
    for epoch in range(1, learner["maximum_epochs"] + 1):
        order_rng.shuffle(examples)
        for start in range(0, len(examples), compute["pair_microbatch_size"]):
            selected = examples[start:start + compute["pair_microbatch_size"]]
            chunk_pairs = [train_pairs[index] for index in selected]
            candidates = [candidate for pair in chunk_pairs for candidate in pair]
            optimizer.zero_grad(set_to_none=True)
            seq_logps, base_logits, _, _, offsets = score_candidate_batch(candidates, model, adapter_a, adapter_b)
            ref_logp = torch.log_softmax(base_logits.detach(), dim=-1)
            pair_losses = []
            for pair_index, pair in enumerate(chunk_pairs):
                left = 2 * pair_index
                ref_chosen = ref_logp[offsets[left]:offsets[left + 1]].gather(1, pair[0]["targets"][:, None]).sum()
                ref_rejected = ref_logp[offsets[left + 1]:offsets[left + 2]].gather(1, pair[1]["targets"][:, None]).sum()
                policy_margin = seq_logps[left] - seq_logps[left + 1]
                relative = policy_margin - (ref_chosen - ref_rejected)
                pair_losses.append(F.softplus(-beta * relative))
            loss = torch.stack(pair_losses).mean()
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in (adapter_a, adapter_b)):
                raise FloatingPointError("non-finite sequence-DPO loss or gradient")
            torch.nn.utils.clip_grad_norm_([adapter_a, adapter_b], learner["optimizer"]["gradient_norm_clip"])
            optimizer.step()
        if epoch in checkpoints:
            metrics = evaluate_pairs(val_pairs, model, adapter_a, adapter_b, beta, compute["pair_microbatch_size"])
            checkpoint_data[str(epoch)] = {"metrics": metrics,
                                           "adapter": {"A": adapter_a.detach().clone(), "B": adapter_b.detach().clone()}}
    return checkpoint_data, initial


def generate_greedy(rows, tokenizer, model, spec, adapter_a=None, adapter_b=None):
    """Greedy decode with the frozen backbone and optional full-vocabulary head residual."""
    import torch
    tokenizer.padding_side = "left"
    batch_size = spec["compute_limits"]["generation_batch_size"]
    results = []
    max_new = spec["compute_limits"]["generation_max_new_tokens"]
    repetition_penalty = spec["generation"]["repetition_penalty"]
    eos_ids = set(spec["generation"]["eos_token_ids"])
    for offset in range(0, len(rows), batch_size):
        selected = rows[offset:offset + batch_size]
        messages = [[{"role": "user", "content": row["user_message"]}] for row in selected]
        batch = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              padding=True, return_tensors="pt", return_dict=True)
        input_ids = batch["input_ids"]
        attention = batch["attention_mask"]
        if not bool(attention.all()):
            raise ValueError("generation must use unpadded single prompts")
        position_ids = attention.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(attention == 0, 0)
        with torch.inference_mode():
            prompt_width = input_ids.shape[1]
            out = model.model(input_ids=input_ids, attention_mask=attention,
                              position_ids=position_ids, cache_position=torch.arange(prompt_width),
                              use_cache=True)
            past = out.past_key_values
            hidden = out.last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden)
            if adapter_a is not None:
                logits = logits + (hidden @ adapter_a) @ adapter_b
            tokens = [[] for _ in selected]
            finished = [False] * len(selected)
            for step in range(max_new):
                generation_logits = logits.clone()
                if repetition_penalty != 1.0:
                    for i, generated_ids in enumerate(tokens):
                        seen_ids = set(input_ids[i].tolist()) | set(generated_ids)
                        seen = torch.tensor(sorted(seen_ids), dtype=torch.long)
                        values = generation_logits[i, seen]
                        generation_logits[i, seen] = torch.where(
                            values < 0, values * repetition_penalty, values / repetition_penalty
                        )
                next_ids = generation_logits.argmax(dim=-1)
                for i, token in enumerate(next_ids.tolist()):
                    if not finished[i]:
                        tokens[i].append(token)
                        if token in eos_ids:
                            finished[i] = True
                if all(finished) or step + 1 >= max_new:
                    break
                attention = torch.cat([attention, torch.ones((len(selected), 1), dtype=attention.dtype)], dim=1)
                position_ids = (attention.sum(dim=1) - 1)[:, None]
                out = model.model(input_ids=next_ids[:, None], attention_mask=attention,
                                  position_ids=position_ids,
                                  cache_position=torch.tensor([prompt_width + step]),
                                  past_key_values=past, use_cache=True)
                past = out.past_key_values
                hidden = out.last_hidden_state[:, -1, :]
                logits = model.lm_head(hidden)
                if adapter_a is not None:
                    logits = logits + (hidden @ adapter_a) @ adapter_b
        for row, token_ids in zip(selected, tokens):
            text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            parsed = parse_generated_number(text)
            from decimal import Decimal
            exact = parsed is not None and parsed == Decimal(row["verifier_answer"])
            results.append({"dataset_index": row["dataset_index"], "generated_text": text,
                            "parsed_number": str(parsed) if parsed is not None else None,
                            "exact_match": bool(exact)})
    tokenizer.padding_side = "right"
    return results


def run(output: Path, spec_path: Path = SPEC_PATH, lock_path: Path = LOCK_PATH):
    started = time.monotonic()
    spec, protocol_hash = load_locked_spec(spec_path, lock_path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.snapshot.json", spec)
    (output / "protocol.lock.snapshot.json").write_bytes(LOCK_PATH.read_bytes())
    for source in (Path(__file__), Path(__file__).with_name("gsm8k_sequence_task.py")):
        (output / (source.stem + ".snapshot.py")).write_bytes(source.read_bytes())
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        os.environ[key] = "false" if key == "TOKENIZERS_PARALLELISM" else "1"
    try:
        import datasets
        import numpy
        import torch
        import transformers
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
        runtime = {"python": platform.python_version(), "torch": torch.__version__.split("+")[0],
                   "transformers": transformers.__version__, "numpy": numpy.__version__, "datasets": datasets.__version__}
        if runtime != spec["runtime"]:
            raise RuntimeError(f"runtime differs from frozen versions: {runtime}")
        torch.set_num_threads(spec["compute_limits"]["threads"])
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA must not be initialized")
        if MODEL_DIR.name != spec["model"]["revision"]:
            raise ValueError("pinned model snapshot is not present")
        if sha256_file(DATA_DIR / "gsm8k-train.arrow") != spec["dataset"]["cached_train_arrow_sha256"]:
            raise ValueError("cached training data hash differs from the lock")
        ds = load_dataset(spec["dataset"]["id"], spec["dataset"]["config"], split="train")
        if ds._fingerprint != spec["dataset"]["cached_fingerprint"]:
            raise ValueError(f"cached train fingerprint mismatch: {ds._fingerprint}")
        train_rows = make_sequence_rows(ds, "development_train", 608, 672)
        val_rows = make_sequence_rows(ds, "development_validation", 672, 736)
        if {r["question_sha256"] for r in train_rows} & {r["question_sha256"] for r in val_rows}:
            raise ValueError("development train/validation questions overlap")

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("tokenizer lacks pad and EOS IDs")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True, torch_dtype=torch.float32)
        model.to("cpu").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            if parameter.device.type != "cpu":
                raise RuntimeError("non-CPU parameter detected")
        train_pairs = encode_candidates(train_rows, tokenizer, model, spec)
        val_pairs = encode_candidates(val_rows, tokenizer, model, spec)
        train_reference = evaluate_pairs(train_pairs, model, None, None, spec["learner"]["beta"], spec["compute_limits"]["pair_microbatch_size"])
        base_val = evaluate_pairs(val_pairs, model, None, None, spec["learner"]["beta"], spec["compute_limits"]["pair_microbatch_size"])
        base_generations = generate_greedy(val_rows, tokenizer, model, spec)
        base_accuracy = statistics.fmean(float(row["exact_match"]) for row in base_generations)

        per_seed = []
        for seed in spec["learner"]["seeds"]:
            checkpoints, _ = train_to_checkpoints(train_pairs, val_pairs, model, spec, seed)
            per_seed.append({"seed": seed, "checkpoints": checkpoints})
            if rss_bytes() > spec["compute_limits"]["max_peak_rss_bytes"]:
                raise MemoryError("development exceeded its peak RSS ceiling")
            if time.monotonic() - started > spec["compute_limits"]["max_wall_seconds"]:
                raise TimeoutError("development exceeded its wall-time ceiling")

        summary = {}
        for epoch in spec["learner"]["checkpoints_epochs"]:
            records = [item["checkpoints"][str(epoch)]["metrics"] for item in per_seed]
            summary[str(epoch)] = {
                "mean_validation_dpo_preference_nll": statistics.fmean(r["mean_dpo_preference_nll"] for r in records),
                "mean_validation_preference_accuracy": statistics.fmean(r["preference_accuracy"] for r in records),
                "mean_validation_token_kl_to_base": statistics.fmean(r["mean_full_vocab_token_kl_to_base"] for r in records),
                "per_seed_nll": [r["mean_dpo_preference_nll"] for r in records],
            }
        eligible = [epoch for epoch in spec["learner"]["checkpoints_epochs"] if epoch > 0 and
                    summary[str(epoch)]["mean_validation_dpo_preference_nll"] < base_val["mean_dpo_preference_nll"] and
                    summary[str(epoch)]["mean_validation_token_kl_to_base"] <= 0.5]
        selected_epoch = min(eligible, key=lambda epoch: (summary[str(epoch)]["mean_validation_dpo_preference_nll"], epoch)) if eligible else 0
        import numpy as np
        stored_per_seed = []
        for seed_item in per_seed:
            stored_checkpoints = {}
            for epoch_text, checkpoint in seed_item["checkpoints"].items():
                adapter_path = output / "adapters" / f"seed-{seed_item['seed']}-epoch-{epoch_text}.npz"
                adapter_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(adapter_path, A=checkpoint["adapter"]["A"].numpy(),
                                    B=checkpoint["adapter"]["B"].numpy())
                stored_checkpoints[epoch_text] = {
                    "metrics": checkpoint["metrics"],
                    "adapter_artifact": {"path": adapter_path.relative_to(output).as_posix(),
                                         "sha256": sha256_file(adapter_path)},
                }
            stored_per_seed.append({"seed": seed_item["seed"], "checkpoints": stored_checkpoints})
        tokenized_artifact = {}
        for split_name, rows, pairs in (("train", train_rows, train_pairs),
                                        ("validation", val_rows, val_pairs)):
            tokenized_artifact[split_name] = [
                {"dataset_index": row["dataset_index"],
                 "chosen": {"input_ids": pair[0]["input_ids"], "response_ids": pair[0]["response_ids"],
                            "completion": pair[0]["completion"]},
                 "rejected": {"input_ids": pair[1]["input_ids"], "response_ids": pair[1]["response_ids"],
                              "completion": pair[1]["completion"]}}
                for row, pair in zip(rows, pairs)
            ]
        write_json(output / "tokenized_examples.json", tokenized_artifact)
        selected_rows = []
        generation_counts = []
        for seed_item in per_seed:
            ckpt = seed_item["checkpoints"][str(selected_epoch)]
            adapter_a = torch.tensor(ckpt["adapter"]["A"], dtype=torch.float32)
            adapter_b = torch.tensor(ckpt["adapter"]["B"], dtype=torch.float32)
            generated = generate_greedy(val_rows, tokenizer, model, spec, adapter_a, adapter_b)
            accuracy = statistics.fmean(float(row["exact_match"]) for row in generated)
            generation_counts.append({"seed": seed_item["seed"], "accuracy": accuracy,
                                      "correct": sum(row["exact_match"] for row in generated)})
            selected_rows.append({"seed": seed_item["seed"],
                                  "selected_adapter_artifact": {
                                      "path": f"adapters/seed-{seed_item['seed']}-epoch-{selected_epoch}.npz",
                                      "sha256": sha256_file(output / "adapters" / f"seed-{seed_item['seed']}-epoch-{selected_epoch}.npz")},
                                  "generated_validation": generated})
        mean_selected_accuracy = statistics.fmean(row["accuracy"] for row in generation_counts)
        at_least_two_not_worse = sum(row["accuracy"] >= base_accuracy for row in generation_counts) >= 2
        candidate_promoted = selected_epoch > 0 and mean_selected_accuracy >= base_accuracy and at_least_two_not_worse
        result = {
            "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_hash,
            "runner_sha256": sha256_file(Path(__file__)), "task_helper_sha256": sha256_file(Path(__file__).with_name("gsm8k_sequence_task.py")),
            "model_id": spec["model"]["id"], "model_revision": MODEL_DIR.name,
            "model_file_sha256": {name: sha256_file(MODEL_DIR / name) for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")},
            "dataset_train_sha256": sha256_file(DATA_DIR / "gsm8k-train.arrow"), "dataset_fingerprint": ds._fingerprint,
            "runtime": runtime, "platform": platform.platform(), "device": "cpu", "paid_compute": False,
            "network_disabled": True, "base_train_preference": train_reference,
            "base_validation_preference": base_val,
            "base_validation_generation": base_generations, "base_validation_exact_match_accuracy": base_accuracy,
            "per_seed": stored_per_seed, "checkpoint_summary": summary, "selected_epochs": selected_epoch,
            "selected_validation_generation": generation_counts,
            "selected_mean_exact_match_accuracy": mean_selected_accuracy,
            "at_least_two_seeds_not_worse": at_least_two_not_worse,
            "decision": "sequence_dpo_development_candidate" if candidate_promoted else "sequence_dpo_development_non_pass",
            "selected_adapters_and_generation": selected_rows,
            "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
        }
        write_json(output / "development.json", {"summary": result, "train_examples": train_rows, "validation_examples": val_rows})
        write_manifest(output)
        return result
    except BaseException as exc:
        write_json(output / "failure.json", {"exception_type": type(exc).__name__, "message": str(exc),
                                              "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes()})
        raise
    finally:
        write_manifest(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cpu-lm-gsm8k-sequence-dpo-development-v1/run-1")
    parser.add_argument("--protocol", type=Path, default=SPEC_PATH)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    args = parser.parse_args()
    result = run(args.output.expanduser().resolve(), args.protocol.resolve(), args.lock.resolve())
    print(json.dumps({"decision": result["decision"], "selected_epochs": result["selected_epochs"],
                      "base_accuracy": result["base_validation_exact_match_accuracy"],
                      "updated_accuracy": result["selected_mean_exact_match_accuracy"],
                      "checkpoint_summary": result["checkpoint_summary"],
                      "elapsed_seconds": result["elapsed_seconds"], "peak_rss_bytes": result["peak_rss_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
