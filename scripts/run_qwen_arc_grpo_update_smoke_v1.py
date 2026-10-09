#!/usr/bin/env python3
"""Run one frozen CPU-only VARE -> RVL GRPO update on ARC training data.

This is an update-cost/serialization smoke, not a task-improvement experiment.
"""
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
import re
import resource
import signal
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "protocols/qwen_arc_grpo_update_smoke_v1.lock.json"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_ID = "allenai/ai2_arc"
MODEL_REV = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_REV = "210d026faf9955653af8916fad021475a3f00453"
SYSTEM = "Answer the science question by choosing one of the listed options. Reply with exactly one option label: A, B, C, or D. Do not explain."


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(obj: dict) -> str:
    payload = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_fingerprint(state: dict[str, Any]) -> str:
    import torch
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().to("cpu").contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(json.dumps(list(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def rss_mib() -> float:
    # This protocol is pinned to macOS, where ru_maxrss is bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def choices(row: dict) -> list[tuple[str, str]]:
    value = row["choices"]
    pairs = zip(value["label"], value["text"], strict=True) if isinstance(value, dict) else ((x["label"], x["text"]) for x in value)
    return [(str(a), str(b)) for a, b in pairs]


def select(rows: list[dict], salt: str, n: int) -> list[dict]:
    eligible = [r for r in rows if [a for a, _ in choices(r)] == ["A", "B", "C", "D"]]
    return sorted(eligible, key=lambda r: hashlib.sha256(f"{salt}|{r['id']}".encode()).hexdigest())[:n]


def render(tokenizer, row: dict) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in choices(row))
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Question:\n{row['question']}\n\nOptions:\n{options}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_answer(raw: str) -> str | None:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]).strip()
    text = re.sub(r"(?i)^(?:answer:|the\s+answer\s+is:|the\s+correct\s+answer\s+is:?)\s*", "", text)
    match = re.fullmatch(r"\(?([A-Da-d])\)?[.)]?", text.strip())
    return match.group(1).upper() if match else None


def label_reward(response: str, answer: str) -> float:
    return float(parse_answer(response) == answer)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_sources(lock: dict, model_path: Path, rvl_path: Path) -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("protocol requires macOS arm64")
    if canonical_sha(lock) != lock.get("sha256"):
        raise RuntimeError("protocol canonical digest mismatch")
    if sha_file(Path(__file__)) != lock["source_hashes"]["runner"]:
        raise RuntimeError("runner hash mismatch")
    if sha_file(ROOT / "src/vare/integrations/rvl_grpo.py") != lock["source_hashes"]["vare_adapter"]:
        raise RuntimeError("VARE adapter hash mismatch")
    actual_rvl = {
        p.relative_to(rvl_path).as_posix(): sha_file(p)
        for p in sorted((rvl_path / "src/rvl_systems").rglob("*.py"))
    }
    if actual_rvl != lock["source_hashes"]["rvl_files"]:
        raise RuntimeError("pinned RVL source files mismatch")
    for name, digest in lock["model"]["files_sha256"].items():
        if sha_file(model_path / name) != digest:
            raise RuntimeError(f"model file mismatch: {name}")


def validate_lock_schema(lock: dict) -> None:
    if canonical_sha(lock) != lock.get("sha256"):
        raise RuntimeError("protocol canonical digest mismatch")
    if lock.get("protocol_id") != "qwen_arc_challenge_grpo_update_smoke_v1":
        raise RuntimeError("unexpected protocol ID")
    required = ("source_hashes", "model", "dataset", "selection", "runtime", "update", "rollout", "evaluation", "resources")
    if any(key not in lock for key in required):
        raise RuntimeError("protocol is missing a required section")
    if "torch_seed" not in lock["runtime"] or "seed_base" not in lock["rollout"] or "eval" not in lock["evaluation"].get("seeds", {}):
        raise RuntimeError("protocol is missing a frozen seed")
    if len(lock["selection"]["train_ids"]) != lock["selection"]["train_n"] or len(lock["selection"]["validation_ids"]) != lock["selection"]["validation_n"]:
        raise RuntimeError("protocol selection cardinality mismatch")
    if set(lock["selection"]["validation_ids"]) & set(lock["selection"]["excluded_validation_ids"]):
        raise RuntimeError("protocol validation sample overlaps consumed IDs")


def _equal_nested(a: Any, b: Any) -> bool:
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_equal_nested(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def run(model_path: Path, rvl_path: Path, train_parquet: Path, val_parquet: Path, out: Path) -> dict:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    validate_lock_schema(lock)
    validate_sources(lock, model_path, rvl_path)
    if sys.version.split()[0] != lock["runtime"]["python"]:
        raise RuntimeError("Python version mismatch")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"})
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(rvl_path / "src"))

    import pandas as pd
    import torch
    import transformers
    import pyarrow
    import huggingface_hub
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if torch.__version__ != lock["runtime"]["torch"] or transformers.__version__ != lock["runtime"]["transformers"]:
        raise RuntimeError("torch/transformers version mismatch")
    if pd.__version__ != lock["runtime"]["pandas"] or pyarrow.__version__ != lock["runtime"]["pyarrow"] or huggingface_hub.__version__ != lock["runtime"]["huggingface_hub"]:
        raise RuntimeError("dataset/runtime package version mismatch")
    if sha_file(train_parquet) != lock["dataset"]["train_sha256"] or sha_file(val_parquet) != lock["dataset"]["validation_sha256"]:
        raise RuntimeError("dataset parquet hash mismatch")
    torch.set_num_threads(lock["runtime"]["torch_threads"])
    torch.manual_seed(lock["runtime"]["torch_seed"])
    if torch.cuda.is_available():
        raise RuntimeError("protocol forbids CUDA")
    mps_available_but_disabled = bool(torch.backends.mps.is_available())

    from rvl_systems.hf_backend import HFLocalBackend
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig
    from rvl_systems.types import VerifiedGeneration
    from vare.integrations.rvl_grpo import RVLGRPOConfig, RVLGRPOHooks
    from vare.types import Attempt, Experience, Task, Verification

    train_rows = pd.read_parquet(train_parquet).to_dict(orient="records")
    val_rows = pd.read_parquet(val_parquet).to_dict(orient="records")
    train = select(train_rows, lock["selection"]["train_salt"], lock["selection"]["train_n"])
    excluded = set(lock["selection"]["excluded_validation_ids"])
    val_pool = [r for r in val_rows if str(r["id"]) not in excluded]
    val = select(val_pool, lock["selection"]["validation_salt"], lock["selection"]["validation_n"])
    if [str(x["id"]) for x in train] != lock["selection"]["train_ids"]:
        raise RuntimeError("train row selection does not match lock")
    if [str(x["id"]) for x in val] != lock["selection"]["validation_ids"]:
        raise RuntimeError("validation row selection does not match lock")

    started = time.monotonic()
    summary: dict[str, Any] = {
        "protocol_id": lock["protocol_id"], "protocol_sha256": sha_file(LOCK_PATH),
        "runner_sha256": sha_file(Path(__file__)), "status": "running", "device": "cpu",
        "rvl_revision": lock["sources"]["rvl_revision"], "vare_base_revision": lock["sources"]["vare_base_revision"],
        "dataset": {"train_sha256": lock["dataset"]["train_sha256"], "validation_sha256": lock["dataset"]["validation_sha256"]},
        "train_group_attempts": [], "base_eval": [], "post_reload_eval": [],
        "mps_available_but_disabled": mps_available_but_disabled,
    }
    write_json(out / "progress.json", summary)
    backend = trainer = hooks = tokenizer = model = None
    try:
        backend = HFLocalBackend(model_name=str(model_path), max_new_tokens=8, device="cpu", precision="fp32")
        model, tokenizer = backend.model, backend.tokenizer
        if next(model.parameters()).device.type != "cpu":
            raise RuntimeError("model loaded off CPU")
        summary["load_peak_rss_mib"] = rss_mib()
        if summary["load_peak_rss_mib"] > lock["resources"]["peak_rss_mib_cap"]:
            raise RuntimeError("RSS cap exceeded after model load")
        summary["base_model_fingerprint"] = tensor_fingerprint(model.state_dict())
        prompts = {str(row["id"]): render(tokenizer, row) for row in train + val}

        for row in val:
            task_id = str(row["id"])
            generated = asyncio.run(backend.generate(task_id, prompts[task_id], n=1, temperature=0.0, seed=lock["evaluation"]["seeds"]["eval"]))[0]
            parsed = parse_answer(generated.response)
            summary["base_eval"].append({
                "task_id": task_id, "prompt_sha256": hashlib.sha256(prompts[task_id].encode()).hexdigest(),
                "answer_key": str(row["answerKey"]).upper(), "raw_generation": generated.response,
                "generated_token_ids": generated.metadata["response_token_ids"], "parsed_label": parsed,
                "exact": parsed == str(row["answerKey"]).upper(),
            })
        write_json(out / "progress.json", summary)

        chosen = None
        for index, row in enumerate(train):
            task_id = str(row["id"])
            generations = asyncio.run(backend.generate(
                task_id, prompts[task_id], n=lock["update"]["group_size"],
                temperature=lock["update"]["sampling_temperature"],
                seed=lock["rollout"]["seed_base"] + index,
            ))
            answer = str(row["answerKey"]).upper()
            rewards = [label_reward(g.response, answer) for g in generations]
            group = {
                "train_index": index, "task_id": task_id, "prompt_sha256": hashlib.sha256(prompts[task_id].encode()).hexdigest(),
                "answer_key": answer, "raw_generations": [g.response for g in generations],
                "generated_token_ids": [g.metadata["response_token_ids"] for g in generations], "rewards": rewards,
            }
            summary["train_group_attempts"].append(group)
            write_json(out / "progress.json", summary)
            if len(set(rewards)) > 1:
                chosen = (row, generations, rewards, index)
                break
        if chosen is None:
            summary["status"] = "no_mixed_reward_group"
            summary["gate"] = False
            summary["decision"] = "no update attempted; retire this smoke protocol's group pool"
            summary["wall_seconds"] = time.monotonic() - started
            summary["peak_rss_mib"] = rss_mib()
            write_json(out / "summary.json", summary)
            return summary

        row, generations, rewards, group_index = chosen
        trainer = HFCausalLMGRPOTrainer(model, config=HFTTrainerConfig(
            learning_rate=lock["update"]["learning_rate"], clip_eps=lock["update"]["clip_eps"],
            advantage_eps=lock["update"]["advantage_epsilon"], clip_advantage=lock["update"]["advantage_clip"],
            objective_backend="torch",
        ))

        class RecordingTrainer(HFCausalLMGRPOTrainer):
            def __init__(self, inner):
                self.__dict__ = inner.__dict__
                self.calls = 0
                self.metrics = None
                self.gradient_count = self.finite_gradient_count = self.nonzero_gradient_count = 0
                self.advantages = None
            def train_step(self, samples, *, advantages=None):
                self.calls += 1
                self.advantages = list(advantages) if advantages is not None else None
                result = super().train_step(samples, advantages=advantages)
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                self.gradient_count = len(grads)
                self.finite_gradient_count = sum(int(bool(torch.isfinite(g).all())) for g in grads)
                self.nonzero_gradient_count = sum(int(bool(torch.count_nonzero(g))) for g in grads)
                self.metrics = result
                if not grads or self.finite_gradient_count != self.gradient_count or self.nonzero_gradient_count == 0:
                    raise RuntimeError("update did not produce finite nonzero gradients")
                return result

        trainer = RecordingTrainer(trainer)
        exp = []
        from rvl_systems.types import Generation
        task = Task(id=str(row["id"]), prompt=prompts[str(row["id"])], family="arc-challenge-update-smoke")
        for member, (g, reward) in enumerate(zip(generations, rewards, strict=True)):
            raw_generation = RVLGRPOHooks._generation_fields(g)
            attempt = Attempt(
                task=task, output=g.response, policy_id="policy-0", policy_version=0, created_step=0,
                logprob=g.logprob, latency_ms=g.latency_s * 1000,
                metadata={"rvl_generation": raw_generation, "vare_rollout_group": f"arc-smoke:{task.id}",
                          "vare_rollout_group_size": len(generations), "sampling_temperature": lock["update"]["sampling_temperature"],
                          "smoke_member": member},
            )
            verification = Verification(score=reward, passed=bool(reward), confidence=1.0,
                verifier_version=0, verifier_name="arc-answer-key-exact-match", trusted=True)
            exp.append(Experience(attempt=attempt, verification=verification, policy_lag=0, verifier_lag=0, shift_score=0.0))
        hooks = RVLGRPOHooks(backend=backend, trainer=trainer, eval_tasks=[], score_fn=lambda _t, _r: 0.0,
            config=RVLGRPOConfig(train_temperature=lock["update"]["sampling_temperature"], seed=lock["rollout"]["seed_base"]),
            generation_factory=Generation, verified_generation_factory=VerifiedGeneration)
        incumbent = tensor_fingerprint(trainer.model.state_dict())
        rng_before = torch.get_rng_state().clone()
        incumbent_optimizer = trainer.optimizer.state_dict()
        candidate_id = asyncio.run(hooks.train_candidate("policy-0", exp))
        state = trainer.model.state_dict()
        restored = tensor_fingerprint(state) == incumbent
        restored = restored and _equal_nested(trainer.optimizer.state_dict(), incumbent_optimizer)
        restored = restored and torch.equal(torch.get_rng_state(), rng_before)
        if not restored:
            raise RuntimeError("incumbent was not transactionally restored")
        cand = hooks._states[candidate_id]
        changed = sum(int(not torch.equal(cand["model"][k], hooks._states["policy-0"]["model"][k])) for k in state)
        delta = max(float((cand["model"][k] - hooks._states["policy-0"]["model"][k]).abs().max()) for k in state)
        if changed == 0 or delta == 0:
            raise RuntimeError("candidate update left model unchanged")
        summary.update({
            "selected_train_task_id": str(row["id"]), "selected_train_index": group_index,
            "group_rewards": rewards, "group_advantages": trainer.advantages,
            "optimizer_step_calls": trainer.calls, "trainer_metrics": trainer.metrics,
            "gradient_tensor_count": trainer.gradient_count, "finite_gradient_tensor_count": trainer.finite_gradient_count,
            "nonzero_gradient_tensor_count": trainer.nonzero_gradient_count,
            "incumbent_restored_exact": restored, "changed_parameter_tensor_count": changed,
            "max_abs_parameter_delta": delta,
        })
        if trainer.calls != 1:
            raise RuntimeError("expected exactly one optimizer call")
        if trainer.metrics is None or any(not math.isfinite(float(v)) for v in trainer.metrics.values()):
            raise RuntimeError("trainer metrics are missing or non-finite")

        trainer.restore_training_state(cand)
        expected_candidate = tensor_fingerprint(trainer.model.state_dict())
        import tempfile
        with tempfile.TemporaryDirectory(prefix="vare-arc-grpo-smoke-") as temp:
            saved = Path(temp) / "candidate"
            trainer.model.save_pretrained(saved, safe_serialization=True)
            tokenizer.save_pretrained(saved)
            summary["checkpoint_file_sha256"] = {p.name: sha_file(p) for p in sorted(saved.iterdir()) if p.is_file()}
            summary["candidate_parameter_fingerprint"] = expected_candidate
            hooks._states.clear()
            del cand, hooks, trainer, model, backend, generations, exp
            gc.collect()
            reloaded_tokenizer = AutoTokenizer.from_pretrained(saved, local_files_only=True, trust_remote_code=False)
            tokenizer_match = all(tokenizer(text, add_special_tokens=False)["input_ids"] == reloaded_tokenizer(text, add_special_tokens=False)["input_ids"] for text in prompts.values())
            if not tokenizer_match:
                raise RuntimeError("tokenizer differs after checkpoint round-trip")
            summary["tokenizer_roundtrip_exact"] = True
            del tokenizer, reloaded_tokenizer
            gc.collect()
            reload_backend = HFLocalBackend(model_name=str(saved), max_new_tokens=8, device="cpu", precision="fp32")
            reload_model = reload_backend.model
            got = tensor_fingerprint(reload_model.state_dict())
            if got != expected_candidate:
                raise RuntimeError("reloaded checkpoint fingerprint mismatch")
            summary["reloaded_parameter_fingerprint"] = got
            summary["checkpoint_roundtrip_exact"] = True
            for eval_row in val:
                task_id = str(eval_row["id"])
                generated = asyncio.run(reload_backend.generate(task_id, prompts[task_id], n=1, temperature=0.0, seed=lock["evaluation"]["seeds"]["eval"]))[0]
                parsed = parse_answer(generated.response)
                summary["post_reload_eval"].append({
                    "task_id": task_id, "prompt_sha256": hashlib.sha256(prompts[task_id].encode()).hexdigest(),
                    "answer_key": str(eval_row["answerKey"]).upper(), "raw_generation": generated.response,
                    "generated_token_ids": generated.metadata["response_token_ids"], "parsed_label": parsed,
                    "exact": parsed == str(eval_row["answerKey"]).upper(),
                })
            summary["post_reload_exact_successes"] = sum(bool(x["exact"]) for x in summary["post_reload_eval"])
            summary["post_reload_parse_rate"] = sum(x["parsed_label"] is not None for x in summary["post_reload_eval"]) / len(val)
            del reload_backend, reload_model
            gc.collect()
        summary["base_exact_successes"] = sum(bool(x["exact"]) for x in summary["base_eval"])
        summary["base_parse_rate"] = sum(x["parsed_label"] is not None for x in summary["base_eval"]) / len(val)
        summary["wall_seconds"] = time.monotonic() - started
        summary["peak_rss_mib"] = rss_mib()
        summary["gate"] = (len(summary["post_reload_eval"]) == lock["selection"]["validation_n"]
            and summary["finite_gradient_tensor_count"] == summary["gradient_tensor_count"]
            and summary["nonzero_gradient_tensor_count"] > 0 and trainer_calls(summary) == 1
            and restored and summary["checkpoint_roundtrip_exact"] and summary["tokenizer_roundtrip_exact"]
            and summary["wall_seconds"] <= lock["resources"]["wall_seconds_cap"]
            and summary["peak_rss_mib"] <= lock["resources"]["peak_rss_mib_cap"])
        if not summary["gate"]:
            raise RuntimeError("one or more frozen update-smoke gates failed")
        summary["status"] = "passed"
        summary["interpretation"] = "One CPU GRPO update and checkpoint round-trip ran; evaluation is smoke-only and cannot establish task improvement."
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        summary["wall_seconds"] = time.monotonic() - started
        summary["peak_rss_mib"] = rss_mib()
        summary["gate"] = False
    summary["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu", "platform": platform.platform()}
    write_json(out / "summary.json", summary)
    return summary


def trainer_calls(summary: dict) -> int:
    return int(summary.get("optimizer_step_calls", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rvl-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--validation-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("frozen wall-time limit exceeded")))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    validate_lock_schema(lock)
    if args.output.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output.resolve()}")
    signal.alarm(lock["resources"]["wall_seconds_cap"])
    try:
        summary = run(args.model_path.resolve(), args.rvl_source.resolve(), args.train_parquet.resolve(), args.validation_parquet.resolve(), args.output.resolve())
    except BaseException as exc:
        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=True)
        summary = {"protocol_id": lock["protocol_id"], "status": "failed_preflight", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("gate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
