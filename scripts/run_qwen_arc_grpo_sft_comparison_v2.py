#!/usr/bin/env python3
"""Frozen three-seed CPU comparison of ARC answer-trace SFT and VARE GRPO."""
from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import resource
import signal
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "protocols/qwen_arc_grpo_sft_comparison_v2.lock.json"
SYSTEM = "Answer the science question by choosing one of the listed options. Reply with exactly one option label: A, B, C, or D. Do not explain."


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(data: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in data.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fingerprint(state: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(json.dumps(list(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choices(row: dict) -> list[tuple[str, str]]:
    raw = row["choices"]
    pairs = zip(raw["label"], raw["text"], strict=True) if isinstance(raw, dict) else ((x["label"], x["text"]) for x in raw)
    return [(str(a), str(b)) for a, b in pairs]


def eligible(rows: list[dict]) -> list[dict]:
    return [r for r in rows if [a for a, _ in choices(r)] == ["A", "B", "C", "D"]]


def select(rows: list[dict], salt: str, n: int) -> list[dict]:
    return sorted(eligible(rows), key=lambda r: hashlib.sha256(f"{salt}|{r['id']}".encode()).hexdigest())[:n]


def render(tokenizer, row: dict) -> str:
    opts = "\n".join(f"{label}. {text}" for label, text in choices(row))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question:\n{row['question']}\n\nOptions:\n{opts}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_label(raw: str) -> str | None:
    value = raw.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 2:
            value = "\n".join(lines[1:-1]).strip()
    value = re.sub(r"(?i)^(?:answer:|the\s+answer\s+is:|the\s+correct\s+answer\s+is:?)\s*", "", value)
    match = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", value.strip())
    return match.group(1).upper() if match else None


def validate_schema(lock: dict) -> None:
    if canonical_sha(lock) != lock.get("sha256"):
        raise RuntimeError("canonical protocol digest mismatch")
    if lock.get("protocol_id") != "qwen_arc_grpo_sft_comparison_v2":
        raise RuntimeError("unexpected protocol ID")
    for section in ("sources", "source_hashes", "model", "dataset", "selection", "training", "evaluation", "analysis", "resources", "runtime"):
        if section not in lock:
            raise RuntimeError(f"protocol missing section {section}")
    if len(lock["selection"]["train_ids"]) != lock["selection"]["train_n"]:
        raise RuntimeError("training selection count mismatch")
    if len(lock["selection"]["validation_ids"]) != lock["selection"]["validation_n"]:
        raise RuntimeError("validation selection count mismatch")
    if set(lock["selection"]["validation_ids"]) & set(lock["selection"]["excluded_validation_ids"]):
        raise RuntimeError("validation selection overlaps consumed IDs")
    seeds = lock["training"].get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or any(type(seed) is not int or seed <= 0 for seed in seeds) or len(set(seeds)) != 3:
        raise RuntimeError("protocol requires three distinct positive integer training seeds")
    if len(lock["selection"]["train_ids"]) != 32 or len(lock["selection"]["validation_ids"]) != 115:
        raise RuntimeError("unexpected locked cohort sizes")
    if lock["training"].get("group_size") != 4 or lock["training"].get("optimizer_steps_per_arm") != 1:
        raise RuntimeError("unexpected locked update budget")
    if lock["resources"].get("external_spend_usd") != 0 or lock["resources"].get("gpu_hours") != 0:
        raise RuntimeError("protocol must forbid external spend and GPU use")
    rules = lock["analysis"].get("success_criteria", {})
    if rules.get("minimum_grpo_minus_sft", 0) <= 0 or rules.get("minimum_grpo_minus_base", 0) <= 0:
        raise RuntimeError("protocol must freeze positive effect thresholds")
    excluded_train = set(lock["selection"]["excluded_train_ids"])
    if set(lock["selection"]["train_ids"]) & excluded_train:
        raise RuntimeError("training selection overlaps consumed IDs")
    if len(set(lock["selection"]["train_ids"])) != len(lock["selection"]["train_ids"]):
        raise RuntimeError("duplicate training IDs")
    if len(set(lock["selection"]["validation_ids"])) != len(lock["selection"]["validation_ids"]):
        raise RuntimeError("duplicate validation IDs")


def validate_sources(lock: dict, model_path: Path, rvl_path: Path) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("protocol requires macOS arm64")
    if sha_file(Path(__file__)) != lock["source_hashes"]["runner"]:
        raise RuntimeError("runner source hash mismatch")
    if sha_file(ROOT / "src/vare/integrations/rvl_grpo.py") != lock["source_hashes"]["vare_adapter"]:
        raise RuntimeError("VARE adapter source hash mismatch")
    if sha_file(ROOT / "scripts/audit_qwen_arc_grpo_sft_comparison_v2.py") != lock["source_hashes"]["auditor"]:
        raise RuntimeError("independent auditor source hash mismatch")
    actual = {p.relative_to(rvl_path).as_posix(): sha_file(p) for p in sorted((rvl_path / "src/rvl_systems").rglob("*.py"))}
    if actual != lock["source_hashes"]["rvl_files"]:
        raise RuntimeError("pinned RVL source hash mismatch")
    for name, digest in lock["model"]["files_sha256"].items():
        if sha_file(model_path / name) != digest:
            raise RuntimeError(f"model file hash mismatch: {name}")


def generate_many(backend, specs: list[tuple[str, str]], *, n: int, temperature: float, seeds: list[int]) -> list[list[Any]]:
    async def go() -> list[list[Any]]:
        out = []
        for (task_id, prompt), seed in zip(specs, seeds, strict=True):
            out.append(await backend.generate(task_id, prompt, n=n, temperature=temperature, seed=seed))
        return out
    return asyncio.run(go())


def eval_many(backend, rows: list[dict], prompts: dict[str, str], seed: int) -> list[dict]:
    async def go() -> list[dict]:
        out = []
        for row in rows:
            qid = str(row["id"])
            g = (await backend.generate(qid, prompts[qid], n=1, temperature=0.0, seed=seed))[0]
            parsed = parse_label(g.response)
            answer = str(row["answerKey"]).upper()
            out.append({
                "task_id": qid, "prompt_sha256": hashlib.sha256(prompts[qid].encode()).hexdigest(),
                "answer_key": answer, "raw_generation": g.response,
                "generated_token_ids": list(g.metadata["response_token_ids"]), "parsed_label": parsed,
                "exact": parsed == answer,
            })
        return out
    return asyncio.run(go())


def gen_record(g: Any, answer: str) -> dict:
    parsed = parse_label(g.response)
    return {
        "prompt_id": str(g.prompt_id), "prompt": str(g.prompt), "raw_generation": str(g.response),
        "answer_key": answer, "parsed_label": parsed, "reward": float(parsed == answer),
        "logprob": float(g.logprob), "token_count": int(g.token_count),
        "response_token_ids": list(g.metadata["response_token_ids"]),
        "response_token_logprobs": list(g.metadata["response_token_logprobs"]),
        "prompt_token_ids": list(g.metadata["prompt_token_ids"]),
        "sampling_temperature": float(g.metadata["sampling_temperature"]),
    }


def generation_object(record: dict):
    from rvl_systems.types import Generation
    meta = {"response_token_ids": record["response_token_ids"],
            "response_token_logprobs": record["response_token_logprobs"],
            "prompt_token_ids": record["prompt_token_ids"],
            "sampling_temperature": record["sampling_temperature"]}
    return Generation(prompt_id=record["prompt_id"], prompt=record["prompt"], response=record["raw_generation"],
        logprob=record["logprob"], token_count=record["token_count"], latency_s=0.0, metadata=meta)


def paired_bootstrap(lock: dict, validation: list[dict], base: list[dict], grpo: dict, sft: dict) -> dict:
    seeds = lock["training"]["seeds"]
    ids = [str(x["id"]) for x in validation]
    answer = {str(x["id"]): str(x["answerKey"]).upper() for x in validation}
    truth = {item["task_id"]: bool(item["exact"]) for item in base}
    groups = {k: [qid for qid in ids if answer[qid] == k] for k in "ABCD"}
    rng = random.Random(lock["analysis"]["bootstrap_seed"])
    b_gs, b_gb = [], []
    for _ in range(lock["analysis"]["bootstrap_replicates"]):
        picked = [rng.choices(qids, k=len(qids)) for qids in groups.values()]
        sample = [qid for group in picked for qid in group]
        def acc(rows: dict, seed: int | None = None) -> float:
            selected = [rows[seed][qid] for qid in sample] if seed is not None else [truth[qid] for qid in sample]
            return sum(selected) / len(selected)
        g_s = sum(acc(grpo, str(seed)) - acc(sft, str(seed)) for seed in seeds) / len(seeds)
        g_b = sum(acc(grpo, str(seed)) - acc({None: truth}, None) for seed in seeds) / len(seeds)
        b_gs.append(g_s); b_gb.append(g_b)
    def ci(values: list[float]) -> list[float]:
        values.sort()
        n = len(values)
        return [values[math.floor(.025 * (n - 1))], values[math.floor(.975 * (n - 1))]]
    def accuracy(rows: dict, seed: int | None = None) -> float:
        vals = [truth[qid] for qid in ids] if seed is None else [rows[str(seed)][qid] for qid in ids]
        return sum(vals) / len(vals)
    per_seed = {}
    for seed in seeds:
        per_seed[str(seed)] = {
            "grpo_accuracy": accuracy(grpo, seed), "sft_accuracy": accuracy(sft, seed),
            "grpo_minus_sft": accuracy(grpo, seed) - accuracy(sft, seed),
            "grpo_minus_base": accuracy(grpo, seed) - accuracy({None: truth}, None),
            "sft_minus_base": accuracy(sft, seed) - accuracy({None: truth}, None),
        }
    mean_g = sum(x["grpo_accuracy"] for x in per_seed.values()) / len(seeds)
    mean_s = sum(x["sft_accuracy"] for x in per_seed.values()) / len(seeds)
    mean_b = accuracy({None: truth}, None)
    grpo_sft = mean_g - mean_s
    grpo_base = mean_g - mean_b
    ci_gs, ci_gb = ci(b_gs), ci(b_gb)
    rules = lock["analysis"]["success_criteria"]
    success = (grpo_sft >= rules["minimum_grpo_minus_sft"] and ci_gs[0] > 0
        and sum(x["grpo_minus_sft"] > 0 for x in per_seed.values()) >= rules["minimum_seeds_positive_vs_sft"]
        and grpo_base >= rules["minimum_grpo_minus_base"] and ci_gb[0] > 0
        and sum(x["grpo_minus_base"] > 0 for x in per_seed.values()) >= rules["minimum_seeds_positive_vs_base"])
    return {
        "n_validation": len(ids), "base_accuracy": mean_b, "mean_grpo_accuracy": mean_g,
        "mean_sft_accuracy": mean_s, "primary_grpo_minus_sft": grpo_sft,
        "primary_grpo_minus_sft_task_bootstrap_95_ci": ci_gs,
        "secondary_grpo_minus_base": grpo_base, "secondary_grpo_minus_base_task_bootstrap_95_ci": ci_gb,
        "per_seed": per_seed, "positive_evidence_gate": bool(success),
        "uncertainty_note": "Stratified task bootstrap over fixed answer-key classes, conditional on the three locked training seeds; seed variability is separately shown per seed.",
    }


def run(model_path: Path, rvl_path: Path, train_path: Path, val_path: Path, out: Path) -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    validate_schema(lock)
    validate_sources(lock, model_path, rvl_path)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"})
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(rvl_path / "src"))
    import pandas as pd
    import pyarrow
    import huggingface_hub
    import torch
    import transformers
    from transformers import AutoTokenizer
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from rvl_systems.types import Generation, VerifiedGeneration
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    from vare.types import Attempt, Experience, Task, Verification
    if (sys.version.split()[0] != lock["runtime"]["python"] or torch.__version__ != lock["runtime"]["torch"]
        or transformers.__version__ != lock["runtime"]["transformers"] or pd.__version__ != lock["runtime"]["pandas"]
        or pyarrow.__version__ != lock["runtime"]["pyarrow"] or huggingface_hub.__version__ != lock["runtime"]["huggingface_hub"]):
        raise RuntimeError("runtime version mismatch")
    if sha_file(train_path) != lock["dataset"]["train_sha256"] or sha_file(val_path) != lock["dataset"]["validation_sha256"]:
        raise RuntimeError("dataset parquet hash mismatch")
    torch.set_num_threads(lock["runtime"]["torch_threads"])
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is forbidden by this protocol")

    train_rows = pd.read_parquet(train_path).to_dict(orient="records")
    val_rows = pd.read_parquet(val_path).to_dict(orient="records")
    excluded_train = set(lock["selection"]["excluded_train_ids"])
    train = select([r for r in train_rows if str(r["id"]) not in excluded_train], lock["selection"]["train_salt"], lock["selection"]["train_n"])
    excluded_val = set(lock["selection"]["excluded_validation_ids"])
    validation = select([r for r in val_rows if str(r["id"]) not in excluded_val], lock["selection"]["validation_salt"], lock["selection"]["validation_n"])
    if [str(r["id"]) for r in train] != lock["selection"]["train_ids"] or [str(r["id"]) for r in validation] != lock["selection"]["validation_ids"]:
        raise RuntimeError("locked sample selection mismatch")

    started = time.monotonic()
    summary: dict[str, Any] = {
        "protocol_id": lock["protocol_id"], "protocol_sha256": sha_file(LOCK_PATH),
        "runner_sha256": sha_file(Path(__file__)), "status": "running",
        "vare_base_revision": lock["sources"]["vare_base_revision"], "rvl_revision": lock["sources"]["rvl_revision"],
        "model_revision": lock["model"]["revision"], "dataset": {"train_sha256": lock["dataset"]["train_sha256"], "validation_sha256": lock["dataset"]["validation_sha256"]},
        "base_eval": [], "rollouts_by_seed": {}, "arms": {"grpo": {}, "success_trace_sft": {}},
    }
    write_json(out / "progress.json", summary)
    backend = None
    try:
        backend = HFLocalBackend(model_name=str(model_path), max_new_tokens=lock["training"]["max_new_tokens"], device="cpu", precision="fp32")
        model, tokenizer = backend.model, backend.tokenizer
        if next(model.parameters()).device.type != "cpu":
            raise RuntimeError("base model is not on CPU")
        prompts = {str(r["id"]): render(tokenizer, r) for r in train + validation}
        over_limit = {qid: len(tokenizer(text, add_special_tokens=False)["input_ids"]) for qid, text in prompts.items()
                      if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > lock["prompt"]["max_input_tokens"]}
        if over_limit:
            raise RuntimeError(f"prompt exceeds frozen max_input_tokens: {sorted(over_limit)[:3]}")
        summary["base_model_fingerprint"] = fingerprint(model.state_dict())
        summary["base_eval"] = eval_many(backend, validation, prompts, lock["evaluation"]["seed"])
        summary["base_exact_successes"] = sum(bool(x["exact"]) for x in summary["base_eval"])
        summary["base_parse_rate"] = sum(x["parsed_label"] is not None for x in summary["base_eval"]) / len(validation)
        write_json(out / "progress.json", summary)

        for seed in lock["training"]["seeds"]:
            groups = []
            for i, row in enumerate(train):
                qid = str(row["id"]); answer = str(row["answerKey"]).upper()
                gens = asyncio.run(backend.generate(qid, prompts[qid], n=lock["training"]["group_size"], temperature=lock["training"]["temperature"], seed=seed * lock["training"]["rollout_seed_multiplier"] + i))
                records = [gen_record(g, answer) for g in gens]
                groups.append({"task_id": qid, "answer_key": answer, "prompt_sha256": hashlib.sha256(prompts[qid].encode()).hexdigest(), "members": records, "rewards": [x["reward"] for x in records]})
            summary["rollouts_by_seed"][str(seed)] = groups
            write_json(out / "progress.json", summary)
        readiness = {}
        for seed in lock["training"]["seeds"]:
            groups = summary["rollouts_by_seed"][str(seed)]
            positives = sum(int(r["reward"]) for g in groups for r in g["members"])
            mixed = sum(len(set(g["rewards"])) > 1 for g in groups)
            readiness[str(seed)] = {"positive_traces": positives, "mixed_groups": mixed}
        summary["training_readiness"] = readiness
        if any(v["positive_traces"] == 0 or v["mixed_groups"] == 0 for v in readiness.values()):
            summary["status"] = "completed_no_training_readiness_gate"
            summary["gate"] = False
            summary["wall_seconds"] = time.monotonic() - started
            summary["peak_rss_mib"] = rss_mib()
            write_json(out / "summary.json", summary)
            return summary
        del backend, model
        backend = None
        gc.collect()

        for seed in lock["training"]["seeds"]:
            seed_text = str(seed)
            groups = summary["rollouts_by_seed"][seed_text]
            summary["arms"]["grpo"][seed_text] = train_grpo_arm(lock, model_path, rvl_path, train, validation, prompts, groups, seed, out)
            write_json(out / "progress.json", summary)
            summary["arms"]["success_trace_sft"][seed_text] = train_sft_arm(lock, model_path, train, validation, prompts, groups, seed, out)
            write_json(out / "progress.json", summary)

        grpo = {seed: {x["task_id"]: bool(x["exact"]) for x in arm["eval"]} for seed, arm in summary["arms"]["grpo"].items()}
        sft = {seed: {x["task_id"]: bool(x["exact"]) for x in arm["eval"]} for seed, arm in summary["arms"]["success_trace_sft"].items()}
        summary["analysis"] = paired_bootstrap(lock, validation, summary["base_eval"], grpo, sft)
        summary["positive_evidence_gate"] = summary["analysis"]["positive_evidence_gate"]
        summary["status"] = "complete"
        summary["decision"] = "CONTINUE" if summary["positive_evidence_gate"] else "STOP_THIS_PAIRING_NO_POSITIVE_GATE"
        summary["interpretation"] = "Frozen three-seed small-model task-success comparison; positive result supports only this ARC setup, and a non-pass is retained without prompt or hyperparameter tuning."
        summary["wall_seconds"] = time.monotonic() - started
        summary["peak_rss_mib"] = rss_mib()
        summary["resource_gate"] = (summary["wall_seconds"] <= lock["resources"]["wall_seconds_cap"] and summary["peak_rss_mib"] <= lock["resources"]["peak_rss_mib_cap"])
        if not summary["resource_gate"]:
            summary["decision"] = "NONPASS_RESOURCE_GATE"
        write_json(out / "summary.json", summary)
        return summary
    except BaseException as exc:
        summary["status"] = "execution_failure"
        summary["exception"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        summary["wall_seconds"] = time.monotonic() - started
        summary["peak_rss_mib"] = rss_mib()
        write_json(out / "summary.json", summary)
        return summary


def _metric_config(lock: dict):
    from rvl_systems.hf_trainer import HFTTrainerConfig
    u = lock["training"]
    return HFTTrainerConfig(learning_rate=u["learning_rate"], clip_eps=u["clip_eps"], max_grad_norm=u["max_grad_norm"], advantage_eps=u["advantage_epsilon"], clip_advantage=u["advantage_clip"], objective_backend="torch")


def _evaluate_reload(lock: dict, candidate_dir: Path, model_path: Path, validation: list[dict], prompts: dict[str, str], seed: int, expected_fingerprint: str) -> dict:
    from rvl_systems.hf_backend import HFLocalBackend
    backend = HFLocalBackend(model_name=str(candidate_dir), max_new_tokens=lock["training"]["max_new_tokens"], device="cpu", precision="fp32")
    model = backend.model
    observed = fingerprint(model.state_dict())
    if observed != expected_fingerprint:
        raise RuntimeError("reloaded model fingerprint mismatch")
    tokenizer = backend.tokenizer
    from transformers import AutoTokenizer
    base_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    tokenizer_match = True
    for text in prompts.values():
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # Check canonical prompt tokenization against the pinned base tokenizer.
        if base_tokenizer(text, add_special_tokens=False)["input_ids"] != ids:
            tokenizer_match = False
            break
    if not tokenizer_match:
        raise RuntimeError("reloaded tokenizer differs on locked prompts")
    rows = eval_many(backend, validation, prompts, lock["evaluation"]["seed"])
    result = {"parameter_fingerprint": observed, "tokenizer_roundtrip_exact": tokenizer_match,
        "exact_successes": sum(bool(x["exact"]) for x in rows),
        "parse_rate": sum(x["parsed_label"] is not None for x in rows)/len(rows), "eval": rows}
    del base_tokenizer, tokenizer, model, backend
    gc.collect()
    return result


def train_grpo_arm(lock: dict, model_path: Path, rvl_path: Path, train: list[dict], validation: list[dict], prompts: dict[str, str], groups: list[dict], seed: int, out: Path) -> dict:
    import torch
    from transformers import AutoTokenizer
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer
    from rvl_systems.types import Generation, VerifiedGeneration
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    from vare.types import Attempt, Experience, Task, Verification
    torch.manual_seed(seed)
    backend = HFLocalBackend(model_name=str(model_path), max_new_tokens=lock["training"]["max_new_tokens"], device="cpu", precision="fp32")
    model, tokenizer = backend.model, backend.tokenizer
    trainer = HFCausalLMGRPOTrainer(model, config=_metric_config(lock))
    class RecordingTrainer(HFCausalLMGRPOTrainer):
        def __init__(self, inner):
            self.__dict__ = inner.__dict__; self.calls = 0; self.metrics = None; self.grads = None; self.advantages = None
        def train_step(self, samples, *, advantages=None):
            self.calls += 1; self.advantages = list(advantages) if advantages is not None else None
            self.metrics = super().train_step(samples, advantages=advantages)
            gs = [p.grad for p in self.model.parameters() if p.grad is not None]
            self.grads = {"count": len(gs), "finite": sum(int(bool(torch.isfinite(g).all())) for g in gs), "nonzero": sum(int(bool(torch.count_nonzero(g))) for g in gs)}
            if not gs or self.grads["finite"] != len(gs) or self.grads["nonzero"] == 0:
                raise RuntimeError("GRPO arm has no finite nonzero gradients")
            return self.metrics
    trainer = RecordingTrainer(trainer)
    experiences = []
    for group in groups:
        qid = group["task_id"]
        task = Task(id=qid, prompt=prompts[qid], family="arc-challenge-confirmation")
        for member_i, rec in enumerate(group["members"]):
            g = Generation(prompt_id=qid, prompt=rec["prompt"], response=rec["raw_generation"], logprob=rec["logprob"], token_count=rec["token_count"], latency_s=0.0,
                metadata={"response_token_ids": rec["response_token_ids"], "response_token_logprobs": rec["response_token_logprobs"], "prompt_token_ids": rec["prompt_token_ids"], "sampling_temperature": rec["sampling_temperature"]})
            raw = RVLGRPOHooks._generation_fields(g); reward = rec["reward"]
            attempt = Attempt(task=task, output=g.response, policy_id="policy-0", policy_version=0, created_step=0, logprob=g.logprob,
                latency_ms=0.0, metadata={"rvl_generation": raw, "vare_rollout_group": f"arc-confirm:{seed}:{qid}", "vare_rollout_group_size": lock["training"]["group_size"], "sampling_temperature": rec["sampling_temperature"], "member": member_i})
            verification = Verification(score=reward, passed=bool(reward), confidence=1.0, verifier_version=0, verifier_name="arc-answer-key", trusted=True)
            experiences.append(Experience(attempt=attempt, verification=verification, policy_lag=0, verifier_lag=0, shift_score=0.0))
    hooks = RVLGRPOHooks(backend=backend, trainer=trainer, eval_tasks=[], score_fn=lambda _t, _r: 0.0,
        config=RVLGRPOConfig(train_temperature=lock["training"]["temperature"], seed=seed), generation_factory=Generation, verified_generation_factory=VerifiedGeneration)
    base_fp = fingerprint(model.state_dict()); rng_before = torch.get_rng_state().clone()
    candidate_id = asyncio.run(hooks.train_candidate("policy-0", experiences))
    incumbent_state = hooks._states["policy-0"]
    restored = fingerprint(model.state_dict()) == base_fp and torch.equal(torch.get_rng_state(), rng_before)
    if not restored or trainer.calls != 1:
        raise RuntimeError("GRPO arm failed exact incumbent restore or optimizer step gate")
    grad_record = dict(trainer.grads or {})
    training_metrics = dict(trainer.metrics or {})
    observed_advantages = list(trainer.advantages or [])
    optimizer_step_calls = trainer.calls
    candidate = hooks._states[candidate_id]
    changed = sum(int(not torch.equal(candidate["model"][k], incumbent_state["model"][k])) for k in candidate["model"])
    delta = max(float((candidate["model"][k] - incumbent_state["model"][k]).abs().max()) for k in candidate["model"])
    trainer.restore_training_state(candidate)
    expected_fp = fingerprint(model.state_dict())
    import tempfile
    with tempfile.TemporaryDirectory(prefix=f"vare-arc-grpo-{seed}-") as td:
        ckpt = Path(td) / "candidate"; model.save_pretrained(ckpt, safe_serialization=True); tokenizer.save_pretrained(ckpt)
        hashes = {p.name: sha_file(p) for p in sorted(ckpt.iterdir()) if p.is_file()}
        hooks._states.clear(); del candidate, incumbent_state, hooks, trainer, model, backend, experiences; gc.collect()
        result = _evaluate_reload(lock, ckpt, model_path, validation, prompts, seed, expected_fp)
        result.update({"optimizer_step_calls": optimizer_step_calls,
            "gradient_tensor_count": grad_record.get("count", 0),
            "finite_gradient_tensor_count": grad_record.get("finite", 0),
            "nonzero_gradient_tensor_count": grad_record.get("nonzero", 0),
            "training_metrics": training_metrics,
            "observed_group_advantage_count": len(observed_advantages),
            "observed_group_advantages": observed_advantages})
        result["checkpoint_file_sha256"] = hashes
        result["candidate_parameter_fingerprint"] = expected_fp
        result["incumbent_restored_exact"] = restored
        result["changed_parameter_tensor_count"] = changed
        result["max_abs_parameter_delta"] = delta
    return result


def train_sft_arm(lock: dict, model_path: Path, train: list[dict], validation: list[dict], prompts: dict[str, str], groups: list[dict], seed: int, out: Path) -> dict:
    import torch
    import torch.nn.functional as F
    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer
    torch.manual_seed(seed)
    backend = HFLocalBackend(model_name=str(model_path), max_new_tokens=lock["training"]["max_new_tokens"], device="cpu", precision="fp32")
    model, tokenizer = backend.model, backend.tokenizer
    base_fp = fingerprint(model.state_dict())
    trainer = HFCausalLMGRPOTrainer(model, config=_metric_config(lock))
    positives = [rec for group in groups for rec in group["members"] if rec["reward"] == 1.0]
    if not positives:
        raise RuntimeError("SFT arm has no successful traces")
    trainer.optimizer.zero_grad(set_to_none=True); model.train()
    losses = []; target_tokens = 0
    for rec in positives:
        pids = rec["prompt_token_ids"]; rids = rec["response_token_ids"]
        ids = torch.tensor([pids + rids], dtype=torch.long, device="cpu")
        logits = model(input_ids=ids).logits[0]
        start = len(pids) - 1
        response_logits = logits[start:start + len(rids)].float()
        targets = torch.tensor(rids, dtype=torch.long)
        loss = F.cross_entropy(response_logits, targets, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("SFT sequence loss is non-finite")
        losses.append(float(loss.detach()))
        (loss / len(positives)).backward()
        target_tokens += len(rids)
        del ids, logits, response_logits, targets, loss
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_record = {"count": len(grads), "finite": sum(int(bool(torch.isfinite(g).all())) for g in grads), "nonzero": sum(int(bool(torch.count_nonzero(g))) for g in grads)}
    if not grads or grad_record["finite"] != len(grads) or grad_record["nonzero"] == 0:
        raise RuntimeError("SFT arm gradients are missing, non-finite, or zero")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), lock["training"]["max_grad_norm"], error_if_nonfinite=True)
    trainer.optimizer.step()
    expected_fp = fingerprint(model.state_dict())
    if expected_fp == base_fp:
        raise RuntimeError("SFT candidate left model unchanged")
    import tempfile
    with tempfile.TemporaryDirectory(prefix=f"vare-arc-sft-{seed}-") as td:
        ckpt = Path(td) / "candidate"; model.save_pretrained(ckpt, safe_serialization=True); tokenizer.save_pretrained(ckpt)
        hashes = {p.name: sha_file(p) for p in sorted(ckpt.iterdir()) if p.is_file()}
        del trainer, model, tokenizer, backend; gc.collect()
        result = _evaluate_reload(lock, ckpt, model_path, validation, prompts, seed, expected_fp)
        result.update({"optimizer_step_calls": 1, "positive_successful_traces": len(positives), "target_response_tokens": target_tokens,
            "mean_success_trace_loss": sum(losses)/len(losses), "grad_norm": float(norm),
            "gradient_tensor_count": grad_record["count"], "finite_gradient_tensor_count": grad_record["finite"],
            "nonzero_gradient_tensor_count": grad_record["nonzero"], "base_parameter_fingerprint": base_fp,
            "checkpoint_file_sha256": hashes, "candidate_parameter_fingerprint": expected_fp})
        return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rvl-source", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--train-parquet", type=Path, required=True)
    ap.add_argument("--validation-parquet", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8")); validate_schema(lock)
    if args.output.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output.resolve()}")
    signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("protocol wall cap reached")))
    signal.alarm(lock["resources"]["wall_seconds_cap"])
    summary = run(args.model_path.resolve(), args.rvl_source.resolve(), args.train_parquet.resolve(), args.validation_parquet.resolve(), args.output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
