#!/usr/bin/env python3
"""Frozen, CPU-only OpenBookQA comparison of one-epoch GRPO and SFT."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import signal
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "protocols/openbookqa_qwen_grpo_sft_v5.lock.json"


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(state: dict[str, Any]) -> str:
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def select(rows: list[dict], split: str, n: int) -> list[dict]:
    salt = f"vare-openbookqa-qwen-v1-{split}-20261010"
    return sorted(rows, key=lambda r: hashlib.sha256(f"{salt}|{r['id']}".encode()).hexdigest())[:n]


def choices(row: dict) -> list[tuple[str, str]]:
    raw = row["choices"]
    return list(zip(raw["label"], raw["text"]))


SYSTEM = "Answer the science question by choosing one listed option. Reply with exactly one option label: A, B, C, or D. Do not explain."


def render(tokenizer, row: dict) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in choices(row))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question:\n{row['question_stem']}\n\nOptions:\n{options}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_label(text: str) -> str | None:
    value = text.strip()
    match = re.match(r"^([A-D])(?=$|[.)\s])", value)
    return None if match is None else match.group(1)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rss_mib() -> float:
    # macOS ru_maxrss is in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def canonical_lock_sha(lock: dict) -> str:
    body = {k: v for k, v in lock.items() if k != "lock_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_lock() -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if canonical_lock_sha(lock) != lock["lock_sha256"]:
        raise RuntimeError("protocol lock digest mismatch")
    if sha_file(Path(__file__)) != lock["source_hashes"]["runner"]:
        raise RuntimeError("runner digest differs from frozen protocol")
    return lock


def load_rows(path: Path, split: str, lock: dict) -> list[dict]:
    import pandas as pd
    expected = lock["dataset"]["parquet_sha256"][split]
    if sha_file(path) != expected:
        raise RuntimeError(f"{split} parquet hash mismatch")
    rows = pd.read_parquet(path).to_dict(orient="records")
    if len({str(r["id"]) for r in rows}) != len(rows):
        raise RuntimeError(f"duplicate task IDs in {split}")
    if any(len(choices(r)) != 4 or [x[0] for x in choices(r)] != list("ABCD") or r["answerKey"] not in "ABCD" for r in rows):
        raise RuntimeError(f"malformed four-choice data in {split}")
    return rows


def load_runtime(model_path: Path, rvl_path: Path, lock: dict):
    import torch
    import transformers
    import datasets
    import pandas
    import pyarrow
    import huggingface_hub
    observed={"python":sys.version.split()[0],"torch":torch.__version__,"transformers":transformers.__version__,
        "datasets":datasets.__version__,"pandas":pandas.__version__,"pyarrow":pyarrow.__version__,
        "huggingface_hub":huggingface_hub.__version__}
    expected={key:lock["runtime"][key] for key in observed}
    if observed != expected:
        raise RuntimeError("runtime version differs from frozen protocol")
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is forbidden; this study is CPU-only")
    torch.set_num_threads(lock["runtime"]["torch_threads"])
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": str(lock["runtime"]["torch_threads"])})
    sys.path.insert(0, str(rvl_path / "src"))
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from rvl_systems.types import Generation, VerifiedGeneration
    return torch, HFLocalBackend, HFCausalLMGRPOTrainer, HFTTrainerConfig, Generation, VerifiedGeneration


def check_source_and_model(model_path: Path, rvl_path: Path, lock: dict) -> None:
    if sha_file(Path(__file__)) != lock["source_hashes"]["runner"]:
        raise RuntimeError("runner source hash mismatch")
    if sha_file(ROOT / "src/vare/integrations/rvl_grpo.py") != lock["source_hashes"]["vare_adapter"]:
        raise RuntimeError("VARE adapter source hash mismatch")
    actual_rvl = {p.relative_to(rvl_path).as_posix(): sha_file(p) for p in sorted((rvl_path / "src/rvl_systems").rglob("*.py"))}
    if actual_rvl != lock["source_hashes"]["rvl_python_files"]:
        raise RuntimeError("RVL source hashes differ from frozen protocol")
    actual_model = {name: sha_file(model_path / name) for name in lock["model"]["files_sha256"]}
    if actual_model != lock["model"]["files_sha256"]:
        raise RuntimeError("model files differ from frozen protocol")


def generate(backend, task_id: str, prompt: str, n: int, temperature: float, seed: int):
    return asyncio.run(backend.generate(task_id, prompt, n=n, temperature=temperature, seed=seed))


def eval_rows(backend, tokenizer, rows: list[dict], split: str, seed: int) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        qid = str(row["id"])
        prompt = render(tokenizer, row)
        gen = generate(backend, qid, prompt, 1, 0.0, seed + i)[0]
        if str(gen.prompt_id) != qid or str(gen.prompt) != prompt:
            raise RuntimeError("backend returned a generation bound to a different task")
        parsed = parse_label(gen.response)
        out.append({"task_id": qid, "split": split, "raw_generation": gen.response,
                    "parsed_label": parsed, "correct_label": str(row["answerKey"]),
                    "exact": parsed == str(row["answerKey"]),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "latency_s": float(gen.latency_s), "token_count": int(gen.token_count)})
    return out


def response_record(gen, row: dict, tokenizer) -> dict:
    parsed = parse_label(str(gen.response))
    return {"task_id": str(row["id"]), "raw_generation": str(gen.response),
        "answer": str(row["answerKey"]), "parsed_label": parsed,
        "reward": float(parsed == str(row["answerKey"])), "logprob": float(gen.logprob),
        "token_count": int(gen.token_count), "temperature": float(gen.metadata["sampling_temperature"]),
        "prompt_token_ids": list(gen.metadata["prompt_token_ids"]),
        "response_token_ids": list(gen.metadata["response_token_ids"]),
        "response_token_logprobs": list(gen.metadata["response_token_logprobs"]),
        "parse_failure": parsed is None}


def run_base_gate(args) -> int:
    lock = load_lock(); check_source_and_model(args.model_path, args.rvl_source, lock)
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch, HFLocalBackend, _, _, _, _ = load_runtime(args.model_path, args.rvl_source, lock)
    train_all = load_rows(args.train_parquet, "train", lock)
    dev_all = load_rows(args.dev_parquet, "validation", lock)
    train = select(train_all, "train", lock["selection"]["train_n"])
    dev = select(dev_all, "validation", lock["selection"]["dev_n"])
    if [str(x["id"]) for x in train] != lock["selection"]["train_ids"] or [str(x["id"]) for x in dev] != lock["selection"]["dev_ids"]:
        raise RuntimeError("frozen train/dev selection mismatch")
    started = time.monotonic()
    backend = HFLocalBackend(model_name=str(args.model_path), max_new_tokens=8, device="cpu", precision="fp32")
    model, tokenizer = backend.model, backend.tokenizer
    if next(model.parameters()).device.type != "cpu": raise RuntimeError("model is not on CPU")
    label_tokens = [tokenizer(label, add_special_tokens=False)["input_ids"] for label in "ABCD"]
    if any(len(ids) != 1 for ids in label_tokens) or len({ids[0] for ids in label_tokens}) != 4:
        raise RuntimeError("answer labels must map to distinct single tokens")
    prompts = {str(row["id"]): render(tokenizer, row) for row in train + dev}
    if args.dev_results_json is None:
        dev_eval = eval_rows(backend, tokenizer, dev, "validation", lock["evaluation"]["dev_seed"])
        dev_input_sha256 = None
    else:
        prior = json.loads(args.dev_results_json.read_text(encoding="utf-8"))
        prior_rows = prior.get("dev_results", [])
        expected_ids = {str(row["id"]): row for row in dev}
        if len(prior_rows) != len(dev) or {str(x["task_id"]) for x in prior_rows} != set(expected_ids):
            raise RuntimeError("development diagnostic input has unexpected task coverage")
        dev_eval = []
        for old in prior_rows:
            qid = str(old["task_id"]); row = expected_ids[qid]
            prompt = prompts[qid]
            if old.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest():
                raise RuntimeError("development diagnostic prompt hash mismatch")
            if old.get("correct_label") != str(row["answerKey"]):
                raise RuntimeError("development diagnostic answer key mismatch")
            raw = str(old["raw_generation"]); parsed = parse_label(raw)
            dev_eval.append({**old,"parsed_label":parsed,"exact":parsed==str(row["answerKey"])})
        dev_input_sha256 = sha_file(args.dev_results_json)
    groups = {}
    for seed in lock["training"]["seeds"]:
        samples = []
        for i, row in enumerate(train):
            qid = str(row["id"])
            gens = generate(backend, qid, prompts[qid], 4, lock["training"]["temperature"], seed * 1_000_000 + i)
            if len(gens) != 4 or any(str(g.prompt_id) != qid or str(g.prompt) != prompts[qid] for g in gens):
                raise RuntimeError("malformed rollout group or task identity")
            samples.append([response_record(g, row, tokenizer) for g in gens])
        groups[str(seed)] = samples
        write_json(args.output / "base-gate-progress.json", {"seed": seed, "completed_groups": len(samples), "wall_seconds": time.monotonic()-started})
    ready = {}
    for seed, seed_groups in groups.items():
        mixed = sum(len({int(x["reward"]) for x in g}) > 1 for g in seed_groups)
        positives = sum(int(x["reward"]) for g in seed_groups for x in g)
        ready[seed] = {"positive_traces": positives, "mixed_groups": mixed, "mixed_group_rate": mixed/len(seed_groups),
                       "parse_failures": sum(x["parse_failure"] for g in seed_groups for x in g)}
    parse_rate = sum(x["parsed_label"] is not None for x in dev_eval)/len(dev_eval)
    accuracy = sum(x["exact"] for x in dev_eval)/len(dev_eval)
    gates = lock["base_gate"]
    passed = (parse_rate >= gates["minimum_dev_parse_rate"] and accuracy >= gates["minimum_dev_accuracy"]
        and all(v["positive_traces"] >= gates["minimum_positive_traces_per_seed"] and v["mixed_group_rate"] >= gates["minimum_mixed_group_rate"] for v in ready.values())
        and time.monotonic()-started <= lock["resources"]["wall_cap_seconds_per_stage"]
        and rss_mib() <= lock["resources"]["peak_rss_cap_mib"])
    result = {"protocol_id": lock["protocol_id"], "stage":"base_gate", "status":"pass" if passed else "stop_pairing",
        "base_gate_passed": passed, "dev_accuracy": accuracy, "dev_parse_rate": parse_rate,
        "dev_n": len(dev_eval), "dev_results": dev_eval, "rollout_readiness": ready,
        "wall_seconds": time.monotonic()-started, "peak_rss_mib": rss_mib(),
        "device": str(next(model.parameters()).device), "base_fingerprint": fingerprint(model.state_dict()),
        "dataset_revision": lock["dataset"]["revision"], "model_revision": lock["model"]["revision"],
        "limitation":"validation is feasibility-only; it will not be used as confirmatory evidence"}
    result["development_diagnostic_input_sha256"] = dev_input_sha256
    result["development_diagnostic_source_protocol"] = "openbookqa_qwen_grpo_sft_v2" if dev_input_sha256 else None
    write_json(args.output / "training-rollouts.json", {"seed_groups":groups,"base_fingerprint":result["base_fingerprint"]})
    write_json(args.output / "base-gate.json", result)
    print(json.dumps({k:v for k,v in result.items() if k!="dev_results"},indent=2,sort_keys=True))
    return 0 if passed else 2


def label_log_distributions(model, tokenizer, rows: list[dict], torch) -> dict[str, list[float]]:
    model.eval(); result={}
    ids=[tokenizer(x,add_special_tokens=False)["input_ids"][0] for x in "ABCD"]
    with torch.inference_mode():
        for row in rows:
            prompt=render(tokenizer,row)
            encoded=tokenizer(prompt,return_tensors="pt")["input_ids"]
            logits=model(input_ids=encoded).logits[0,-1].float()
            result[str(row["id"])]=torch.log_softmax(logits[ids],dim=0).cpu().tolist()
    return result


def label_subset_kl(base_log_probs: dict[str, list[float]], candidate_log_probs: dict[str, list[float]]) -> float:
    vals=[]
    for qid, base in base_log_probs.items():
        cand=candidate_log_probs[qid]
        vals.append(sum(math.exp(p)*(p-q) for p,q in zip(base,cand)))
    return sum(vals)/len(vals)


def evaluate_saved(lock, model_dir: Path, model_path: Path, rvl_path: Path, rows: list[dict], arm: str, seed: int, dev_rows: list[dict], base_label_log_probs: dict[str, list[float]]):
    torch, HFLocalBackend, _, _, _, _ = load_runtime(model_path, rvl_path, lock)
    backend = HFLocalBackend(model_name=str(model_dir), max_new_tokens=8, device="cpu", precision="fp32")
    model, tokenizer = backend.model, backend.tokenizer
    observed_fingerprint = fingerprint(model.state_dict())
    records = eval_rows(backend, tokenizer, rows, "test", lock["evaluation"]["test_seed"])
    dev_label_log_probs=label_log_distributions(model,tokenizer,dev_rows,torch)
    out = {"arm":arm,"seed":seed,"fingerprint":observed_fingerprint,
        "accuracy":sum(x["exact"] for x in records)/len(records),
        "parse_rate":sum(x["parsed_label"] is not None for x in records)/len(records),
        "mean_latency_s":sum(x["latency_s"] for x in records)/len(records),
        "dev_answer_label_kl_from_base":label_subset_kl(base_label_log_probs,dev_label_log_probs),
        "results":records}
    del model, tokenizer, backend; gc.collect()
    return out


def _answer_distribution(model, tokenizer, prompt: str, label_ids: list[int], torch):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.inference_mode(): logits = model(input_ids=ids).logits[0, -1].float()
    return torch.log_softmax(logits[label_ids], dim=0).cpu()


def train_sft(model_path: Path, rvl_path: Path, rows: list[dict], seed: int, lock: dict, save_dir: Path) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _, _, Trainer, TrainerConfig, _, _ = load_runtime(model_path, rvl_path, lock)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, dtype=torch.float32).to("cpu")
    trainer = Trainer(model, config=TrainerConfig(learning_rate=lock["training"]["learning_rate"], clip_eps=0.2,
        max_grad_norm=1.0, disable_dropout=True, objective_backend="torch"))
    rng = random.Random(seed)
    order = list(range(len(rows))); rng.shuffle(order)
    batch_size = lock["training"]["batch_groups"]
    losses=[]; step_norms=[]
    base_fingerprint=fingerprint(model.state_dict())
    model.train()
    for start in range(0,len(order),batch_size):
        batch=[rows[i] for i in order[start:start+batch_size]]
        trainer.optimizer.zero_grad(set_to_none=True)
        for row in batch:
            prompt=render(tokenizer,row)
            pids=tokenizer(prompt,add_special_tokens=False)["input_ids"]
            target=tokenizer(str(row["answerKey"]),add_special_tokens=False)["input_ids"]+[tokenizer.eos_token_id]
            if len(target)!=2: raise RuntimeError("answer label tokenization is not one token plus EOS")
            ids=torch.tensor([pids+target],dtype=torch.long)
            logits=model(input_ids=ids).logits[0]
            pred=logits[len(pids)-1:len(pids)+1].float()
            labels=torch.tensor(target,dtype=torch.long)
            loss=F.cross_entropy(pred,labels,reduction="mean")/len(batch)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite SFT loss")
            losses.append(float(loss.detach())); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        trainer.optimizer.step(); step_norms.append(float(norm.detach()))
    base_sha=lock["model"]["files_sha256"]["model.safetensors"]
    model.save_pretrained(save_dir,safe_serialization=True);tokenizer.save_pretrained(save_dir)
    output={"arm":"sft","seed":seed,"optimizer_steps":len(step_norms),"mean_loss":sum(losses)/len(losses),
        "mean_grad_norm":sum(step_norms)/len(step_norms),"model_files_sha256":{p.name:sha_file(p) for p in save_dir.iterdir() if p.is_file()},
        "base_model_file_sha256":base_sha,"base_fingerprint":base_fingerprint,
        "candidate_fingerprint":fingerprint(model.state_dict())}
    del model,tokenizer,trainer;gc.collect()
    return output


def train_grpo(model_path: Path, rvl_path: Path, rows: list[dict], rollouts: dict, seed: int, lock: dict, save_dir: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch, _, Trainer, TrainerConfig, Generation, VerifiedGeneration = load_runtime(model_path,rvl_path,lock)
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer
    from rvl_systems.types import VerifiedGeneration as VG
    from rvl_systems.types import Generation as Gen
    torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,trust_remote_code=False,dtype=torch.float32).to("cpu")
    config=TrainerConfig(learning_rate=lock["training"]["learning_rate"],clip_eps=0.2,max_grad_norm=1.0,
        disable_dropout=True,objective_backend="torch")
    trainer=Trainer(model,config=config)
    base_fingerprint=fingerprint(model.state_dict())
    indexed={str(r["id"]):r for r in rows}
    groups=rollouts["seed_groups"][str(seed)]
    if len(groups)!=len(rows): raise RuntimeError("rollout/train row count mismatch")
    rng=random.Random(seed); order=list(range(len(groups)));rng.shuffle(order)
    batch_group=lock["training"]["batch_groups"]
    updates=[]
    model.train()
    for start in range(0,len(order),batch_group):
        batch_samples=[]; batch_adv=[]
        for gi in order[start:start+batch_group]:
            records=groups[gi]
            qid=records[0]["task_id"]
            if any(x["task_id"]!=qid for x in records) or qid not in indexed or len(records)!=4:
                raise RuntimeError("rollout group identity/size violation")
            row=indexed[qid];prompt=render(tokenizer,row)
            # RVL consumes the exact behavior tokens and token log probabilities.
            for rec in records:
                meta={"prompt_token_ids":rec["prompt_token_ids"],"response_token_ids":rec["response_token_ids"],
                      "response_token_logprobs":rec["response_token_logprobs"],"sampling_temperature":rec["temperature"]}
                g=Gen(prompt_id=qid,prompt=prompt,response=rec["raw_generation"],logprob=rec["logprob"],
                    token_count=rec["token_count"],latency_s=0.0,metadata=meta)
                batch_samples.append(VG(generation=g,reward=rec["reward"],verifier_latency_s=0.0,
                    verifier_version=0,metadata={"verifier":"openbookqa-answer-key"}))
            rewards=[x["reward"] for x in records];mean=sum(rewards)/4
            var=sum((x-mean)**2 for x in rewards)/4
            scale=math.sqrt(var+1e-6)
            batch_adv.extend(max(-5.0,min(5.0,(x-mean)/scale)) for x in rewards)
        updates.append(trainer.train_step(batch_samples,advantages=batch_adv))
    model.save_pretrained(save_dir,safe_serialization=True);tokenizer.save_pretrained(save_dir)
    out={"arm":"grpo","seed":seed,"optimizer_steps":len(updates),"update_metrics":updates,
        "model_files_sha256":{p.name:sha_file(p) for p in save_dir.iterdir() if p.is_file()},
        "base_fingerprint":base_fingerprint,"candidate_fingerprint":fingerprint(model.state_dict()),
        "rollout_policy_revision":lock["model"]["revision"],
        "update_note":"one epoch over fixed base-policy rollouts; RVL recomputes current log probabilities and applies clipped GRPO ratios"}
    del model,tokenizer,trainer;gc.collect()
    return out


def run_training(args) -> int:
    lock=load_lock();check_source_and_model(args.model_path,args.rvl_source,lock)
    gate_path=args.output/"base-gate.json";rollout_path=args.output/"training-rollouts.json"
    if not gate_path.is_file() or not rollout_path.is_file():raise RuntimeError("passing base gate is required")
    gate=json.loads(gate_path.read_text())
    if gate.get("protocol_id") != lock["protocol_id"]: raise RuntimeError("base gate protocol ID mismatch")
    if not gate.get("base_gate_passed"):raise RuntimeError("base gate did not pass")
    if args.final_output.exists():raise FileExistsError(args.final_output)
    args.final_output.mkdir(parents=True)
    torch,*_=load_runtime(args.model_path,args.rvl_source,lock)
    train_all=load_rows(args.train_parquet,"train",lock)
    train=select(train_all,"train",lock["selection"]["train_n"])
    if [str(x["id"]) for x in train]!=lock["selection"]["train_ids"]:raise RuntimeError("train cohort mismatch")
    rollouts=json.loads(rollout_path.read_text())
    started=time.monotonic();checkpoints=args.final_output/"checkpoints";checkpoints.mkdir()
    report={"protocol_id":lock["protocol_id"],"status":"training","training":{},"evaluation":{},
        "confirmation_opened":False,"start_unix":time.time()}
    write_json(args.final_output/"progress.json",report)
    try:
        for seed in lock["training"]["seeds"]:
            seed=str(seed)
            sft_dir=checkpoints/f"sft-{seed}";grpo_dir=checkpoints/f"grpo-{seed}"
            report["training"][f"sft-{seed}"]=train_sft(args.model_path,args.rvl_source,train,int(seed),lock,sft_dir)
            write_json(args.final_output/"progress.json",report)
            report["training"][f"grpo-{seed}"]=train_grpo(args.model_path,args.rvl_source,train,rollouts,int(seed),lock,grpo_dir)
            write_json(args.final_output/"progress.json",report)
        # The confirmation split is intentionally loaded only after every candidate checkpoint has been trained and saved.
        dev_all=load_rows(args.dev_parquet,"validation",lock)
        dev=select(dev_all,"validation",lock["selection"]["dev_n"])
        test_all=load_rows(args.test_parquet,"test",lock)
        test=select(test_all,"test",lock["selection"]["test_n"])
        report["confirmation_opened"]=True
        report["confirmation_ids_sha256"]=hashlib.sha256("\n".join(str(x["id"]) for x in test).encode()).hexdigest()
        report["confirmation_n"]=len(test)
        write_json(args.final_output/"progress.json",report)
        base_backend=HFLocalBackend(model_name=str(args.model_path),max_new_tokens=8,device="cpu",precision="fp32")
        base_label_log_probs=label_log_distributions(base_backend.model,base_backend.tokenizer,dev,torch)
        del base_backend;gc.collect()
        base=evaluate_saved(lock,args.model_path,args.model_path,args.rvl_source,test,"base",0,dev,base_label_log_probs)
        report["evaluation"]["base"]=base
        write_json(args.final_output/"progress.json",report)
        for seed in lock["training"]["seeds"]:
            seed=str(seed)
            for arm in ("sft","grpo"):
                train_record=report["training"][f"{arm}-{seed}"]
                evaluated=evaluate_saved(lock,checkpoints/f"{arm}-{seed}",args.model_path,args.rvl_source,test,arm,int(seed),dev,base_label_log_probs)
                if evaluated["fingerprint"] != train_record["candidate_fingerprint"]:
                    raise RuntimeError(f"{arm}/{seed} checkpoint reload fingerprint mismatch")
                report["evaluation"][f"{arm}-{seed}"]=evaluated
                write_json(args.final_output/"progress.json",report)
        report["status"]="complete";report["wall_seconds"]=time.monotonic()-started;report["peak_rss_mib"]=rss_mib()
        report["evaluation_summary"]=summarize(lock,report["evaluation"])
        report["resource_gate"]=(report["wall_seconds"] <= lock["resources"]["wall_cap_seconds_per_stage"] and report["peak_rss_mib"] <= lock["resources"]["peak_rss_cap_mib"])
        report["decision"]="CONTINUE" if report["evaluation_summary"]["success_gate"] and report["resource_gate"] else ("STOP_RESOURCE_GATE" if not report["resource_gate"] else "STOP_NO_PREDECLARED_SUCCESS")
        report["limitations"]=["single 0.5B model and one public science benchmark", "possible benchmark pretraining overlap", "128-item confirmation subset has limited power", "one training seed population of three; bootstrap is conditional on these seeds"]
        report["mean_grpo_behavior_kl_estimate"]=sum(float(m.get("behavior_kl_estimate",0.0)) for s in lock["training"]["seeds"] for m in report["training"][f"grpo-{s}"]["update_metrics"])/sum(len(report["training"][f"grpo-{s}"]["update_metrics"]) for s in lock["training"]["seeds"])
        write_json(args.final_output/"summary.json",report)
        return 0
    except BaseException as exc:
        report["status"]="execution_failure";report["exception"]=f"{type(exc).__name__}: {exc}";report["wall_seconds"]=time.monotonic()-started;report["peak_rss_mib"]=rss_mib()
        write_json(args.final_output/"summary.json",report);raise


def summarize(lock: dict, evaluations: dict) -> dict:
    seeds=[str(s) for s in lock["training"]["seeds"]]
    base=evaluations["base"]["results"]
    base_map={x["task_id"]:int(x["exact"]) for x in base}
    grpo={s:{x["task_id"]:int(x["exact"]) for x in evaluations[f"grpo-{s}"]["results"]} for s in seeds}
    sft={s:{x["task_id"]:int(x["exact"]) for x in evaluations[f"sft-{s}"]["results"]} for s in seeds}
    ids=list(base_map)
    per_seed={s:{"grpo_accuracy":sum(grpo[s].values())/len(ids),"sft_accuracy":sum(sft[s].values())/len(ids),
        "grpo_minus_sft":sum(grpo[s][q]-sft[s][q] for q in ids)/len(ids),
        "grpo_minus_base":sum(grpo[s][q]-base_map[q] for q in ids)/len(ids)} for s in seeds}
    rng=random.Random(lock["analysis"]["bootstrap_seed"])
    diffs_s=[];diffs_b=[];rep=lock["analysis"]["bootstrap_replicates"]
    for _ in range(rep):
        sample=[rng.choice(ids) for _ in ids]
        diffs_s.append(sum(sum(grpo[s][q]-sft[s][q] for q in sample)/len(sample) for s in seeds)/len(seeds))
        diffs_b.append(sum(sum(grpo[s][q]-base_map[q] for q in sample)/len(sample) for s in seeds)/len(seeds))
    def ci(v):
        v.sort();return [v[math.floor(.025*(len(v)-1))],v[math.floor(.975*(len(v)-1))]]
    mean_g=sum(sum(grpo[s].values())/len(ids) for s in seeds)/len(seeds)
    mean_s=sum(sum(sft[s].values())/len(ids) for s in seeds)/len(seeds)
    base_acc=sum(base_map.values())/len(ids)
    d_s=mean_g-mean_s;d_b=mean_g-base_acc;ci_s=ci(diffs_s);ci_b=ci(diffs_b)
    criteria=lock["analysis"]["success_criteria"]
    gate=(d_s>=criteria["minimum_grpo_minus_sft"] and ci_s[0]>0
        and sum(x["grpo_minus_sft"]>0 for x in per_seed.values())>=criteria["minimum_positive_seeds"]
        and d_b>=criteria["minimum_grpo_minus_base"] and ci_b[0]>0
        and sum(x["grpo_minus_base"]>0 for x in per_seed.values())>=criteria["minimum_positive_seeds"])
    return {"n_confirmation":len(ids),"base_accuracy":base_acc,"mean_sft_accuracy":mean_s,"mean_grpo_accuracy":mean_g,
        "grpo_minus_sft":d_s,"grpo_minus_sft_bootstrap_95":ci_s,"grpo_minus_base":d_b,"grpo_minus_base_bootstrap_95":ci_b,
        "per_seed":per_seed,"success_gate":bool(gate),"bootstrap_note":"Percentile bootstrap over confirmation tasks, conditional on three training seeds; seed uncertainty is shown separately."}


def main() -> int:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("base-gate");p.add_argument("--model-path",type=Path,required=True);p.add_argument("--rvl-source",type=Path,required=True);p.add_argument("--train-parquet",type=Path,required=True);p.add_argument("--dev-parquet",type=Path,required=True);p.add_argument("--dev-results-json",type=Path);p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("train-eval");p.add_argument("--model-path",type=Path,required=True);p.add_argument("--rvl-source",type=Path,required=True);p.add_argument("--train-parquet",type=Path,required=True);p.add_argument("--dev-parquet",type=Path,required=True);p.add_argument("--test-parquet",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--final-output",type=Path,required=True)
    args=parser.parse_args()
    cap=12*60*60
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError("frozen 12-hour wall-clock cap reached")))
    signal.alarm(cap)
    return run_base_gate(args) if args.command=="base-gate" else run_training(args)


if __name__=="__main__":
    raise SystemExit(main())
