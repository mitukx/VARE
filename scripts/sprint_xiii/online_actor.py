#!/usr/bin/env python3
"""Online Qwen GRPO under asynchronous policy and verifier transitions.

This is a synthetic systems experiment. The base Qwen checkpoint is frozen;
only a vocabulary-logit bias is trainable. Every accepted group is an actual
multi-token Qwen rollout scored by a versioned proxy and passed to RVL's HF
token-level GRPO trainer. VARE replay and use-time freshness checks gate data.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def write_bytes(path: Path, data: bytes, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, value: object, *, replace: bool = False) -> None:
    write_bytes(
        path,
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False).encode()
        + b"\n",
        replace=replace,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def append_journal(root: Path, event: dict) -> None:
    import fcntl

    path = root / "journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = [json.loads(line) for line in handle.read().splitlines()]
        row = {
            "seq": len(rows),
            "previous": rows[-1]["sha256"] if rows else "0" * 64,
            **event,
        }
        row["sha256"] = sha(canonical(row))
        handle.seek(0, os.SEEK_END)
        handle.write(canonical(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def protocol(path: Path) -> dict:
    value = read_json(path)
    expected = read_json(Path(str(path) + ".sha256"))["sha256"]
    if sha(path.read_bytes()) != expected:
        raise RuntimeError("protocol sidecar mismatch")
    return value


def task_by_id(p: dict, task_id: str) -> dict:
    for task in p["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)


def verifier_score(actions: str, task: dict, version: int) -> float:
    target = task["target_pair"]
    forbidden = task["forbidden_pair"]
    if version == 0:
        # v0 is a partial test: it notices the target's first label in its
        # first three decisions but does not check ordering or the forbidden pair.
        return min(1.0, actions[:3].count(target[0]) / 2.0)
    if version == 1:
        target_found = any(
            target == actions[i : i + 2] for i in range(max(0, len(actions) - 1))
        )
        checked = actions[:3]
        forbidden_found = any(
            forbidden == checked[i : i + 2] for i in range(max(0, len(checked) - 1))
        )
        if forbidden_found:
            return 0.0
        partial = min(2, actions.count(target[0])) / 2.0
        return 0.75 * float(target_found) + 0.25 * partial
    raise ValueError(f"unknown verifier version {version}")


def model_bundle(p: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(p["limits"]["torch_threads"])
    tok = AutoTokenizer.from_pretrained(
        p["model"]["snapshot_path"], local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        p["model"]["snapshot_path"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bias = torch.nn.Parameter(torch.zeros(model.config.vocab_size, dtype=torch.float32))
    model.register_parameter("vare_online_action_bias", bias)
    labels = p["action_labels"]
    ids = [tok.encode(label, add_special_tokens=False) for label in labels]
    if any(len(row) != 1 for row in ids):
        raise RuntimeError("frozen action label is not one tokenizer token")
    action_ids = [row[0] for row in ids]
    eos_id = tok.eos_token_id
    if eos_id is None or eos_id in action_ids:
        raise RuntimeError("EOS must be a distinct action token")
    support_ids = action_ids + [eos_id]
    mask = torch.full((model.config.vocab_size,), -1e9, dtype=torch.float32)
    mask[support_ids] = 0.0
    fixed_bias = torch.zeros_like(mask)
    fixed_bias[eos_id] = float(p["generation"]["termination_logit_offset"])
    model.vare_prompt_length = None
    model.vare_min_actions = int(p["generation"]["minimum_actions"])

    def constrain(_module, _inputs, logits):
        output = logits + model.vare_online_action_bias + mask + fixed_bias
        prompt_length = model.vare_prompt_length
        if prompt_length is not None:
            positions = torch.arange(output.shape[1], device=output.device)
            response_indices = positions - (int(prompt_length) - 1)
            early = (response_indices >= 0) & (
                response_indices < model.vare_min_actions
            )
            if bool(early.any()):
                output = output.clone()
                output[:, early, eos_id] = -1e9
        return output

    model.lm_head.register_forward_hook(constrain)
    model.eval()
    return torch, tok, model, action_ids, eos_id, support_ids


def prompt_text(tok, task: dict) -> tuple[str, list[int]]:
    messages = [
        {
            "role": "system",
            "content": "Produce a route as a short sequence of labels. Follow the stopping rule.",
        },
        {
            "role": "user",
            "content": (
                f"{task['instruction']} The desired consecutive pair is {task['target_pair']}. "
                f"Avoid the consecutive pair {task['forbidden_pair']}. Use only A, B, C, or D, "
                "with at least two and at most five route labels."
            ),
        },
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError("empty prompt")
    if sha(text.encode()) != task["prompt_sha256"] or ids != task["prompt_token_ids"]:
        raise RuntimeError(f"frozen prompt identity mismatch: {task['task_id']}")
    return text, ids


def checkpoint_state(path: Path, expected_sha: str | None = None) -> dict:
    meta = read_json(Path(str(path) + ".json"))
    raw = path.read_bytes()
    actual = sha(raw)
    if actual != meta["sha256"] or (
        expected_sha is not None and actual != expected_sha
    ):
        raise RuntimeError(f"checkpoint hash mismatch: {path}")
    import torch

    return meta, torch.load(
        io.BytesIO(gzip.decompress(raw)), map_location="cpu", weights_only=True
    )


def restore_actor(model, checkpoint: Path, expected_sha: str | None = None) -> dict:
    meta, state = checkpoint_state(checkpoint, expected_sha)
    with __import__("torch").no_grad():
        model.vare_online_action_bias.copy_(state["bias"])
    return meta


def save_checkpoint(path: Path, model, optimizer, torch, metadata: dict) -> dict:
    started = time.monotonic()
    state = {
        "bias": model.vare_online_action_bias.detach().cpu().clone(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state().cpu().clone(),
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    payload = gzip.compress(buffer.getvalue(), mtime=0)
    write_bytes(path, payload)
    manifest = {
        "sha256": sha(payload),
        "bytes": len(payload),
        "io_seconds": time.monotonic() - started,
        "metadata": metadata,
    }
    write_json(Path(str(path) + ".json"), manifest)
    return manifest


def active_record(root: Path) -> dict:
    record = read_json(root / "active.json")
    checkpoint = root / "models" / record["checkpoint"]
    manifest, _ = checkpoint_state(checkpoint, record["checkpoint_sha256"])
    record["checkpoint_path"] = str(checkpoint)
    record["checkpoint_metadata"] = manifest["metadata"]
    return record


def sample_batch(
    torch,
    tok,
    model,
    task: dict,
    prompt_ids: list[int],
    checkpoint_sha: str,
    policy_version: int,
    verifier_version: int,
    seed: int,
    count: int,
    p: dict,
) -> tuple[list[dict], list[float], float]:
    model.eval()
    model.vare_prompt_length = len(prompt_ids)
    device = next(model.parameters()).device
    sequences = torch.tensor(
        [prompt_ids for _ in range(count)], dtype=torch.long, device=device
    )
    response_ids: list[list[int]] = [[] for _ in range(count)]
    token_logps: list[list[float]] = [[] for _ in range(count)]
    finished = [False] * count
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    support = torch.tensor(model.vare_support_ids, dtype=torch.long, device=device)
    first_probs = None
    started = time.monotonic()
    for _step in range(int(p["generation"]["maximum_actions"])):
        with torch.inference_mode():
            logits = (
                model(input_ids=sequences)
                .logits[:, -1, :]
                .float()
                .index_select(-1, support)
            )
            scaled = logits / float(p["generation"]["temperature"])
            log_probs = torch.log_softmax(scaled, dim=-1)
            probs = log_probs.exp()
        if first_probs is None:
            first_probs = probs[0].tolist()
        selected = []
        for row in range(count):
            if finished[row]:
                selected.append(len(model.vare_action_ids))
                continue
            choice = int(torch.multinomial(probs[row], 1, generator=rng).item())
            token_id = int(model.vare_support_ids[choice])
            response_ids[row].append(token_id)
            token_logps[row].append(float(log_probs[row, choice].item()))
            selected.append(choice)
            if token_id == model.vare_eos_id:
                finished[row] = True
        append_ids = [
            int(model.vare_eos_id)
            if finished[row] and selected[row] == len(model.vare_action_ids)
            else int(model.vare_support_ids[selected[row]])
            for row in range(count)
        ]
        sequences = torch.cat(
            [
                sequences,
                torch.tensor(append_ids, dtype=torch.long, device=device).unsqueeze(1),
            ],
            dim=1,
        )
        if all(finished):
            break
    action_id_to_label = {
        int(token): label
        for token, label in zip(model.vare_action_ids, p["action_labels"], strict=True)
    }
    output = []
    for index in range(count):
        ids = response_ids[index]
        labels = [
            action_id_to_label[token] for token in ids if token != model.vare_eos_id
        ]
        terminal = "eos" if ids and ids[-1] == model.vare_eos_id else "maximum_actions"
        output.append(
            {
                "sample_index": index,
                "response_token_ids": ids,
                "response_token_logprobs": token_logps[index],
                "response": "".join(labels),
                "terminal_reason": terminal,
                "generated_token_count": len(ids),
                "action_count": len(labels),
                "behavior_logprob": sum(token_logps[index]),
                "policy_version": policy_version,
                "policy_checkpoint_sha256": checkpoint_sha,
                "verifier_version_at_rollout": verifier_version,
                "sampling_temperature": float(p["generation"]["temperature"]),
            }
        )
    if first_probs is None:
        raise RuntimeError("generation did not execute")
    model.eval()
    for key, value in (
        ("vare_action_ids", model.vare_action_ids),
        ("vare_support_ids", model.vare_support_ids),
        ("vare_eos_id", model.vare_eos_id),
    ):
        del key, value
    return output, first_probs, time.monotonic() - started


def attach_model_ids(model, action_ids, eos_id, support_ids) -> None:
    model.vare_action_ids = list(action_ids)
    model.vare_eos_id = int(eos_id)
    model.vare_support_ids = list(support_ids)


def child_init(p: dict, root: Path, seed: int) -> None:
    torch, _tok, model, actions, eos, support = model_bundle(p)
    attach_model_ids(model, actions, eos, support)
    torch.manual_seed(seed)
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig

    trainer = HFCausalLMGRPOTrainer(
        model, config=HFTTrainerConfig(learning_rate=p["training"]["learning_rate"])
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    filename = "policy-0.pt.gz"
    manifest = save_checkpoint(
        root / "models" / filename,
        model,
        trainer.optimizer,
        torch,
        {
            "policy_version": 0,
            "seed": seed,
            "transaction_id": "initial",
            "model_sha256": p["model"]["weight_sha256"],
        },
    )
    write_json(
        root / "active.json",
        {
            "policy_version": 0,
            "checkpoint": filename,
            "checkpoint_sha256": manifest["sha256"],
            "transaction_id": "initial",
        },
    )
    append_journal(
        root,
        {
            "kind": "initial_policy",
            "seed": seed,
            "policy_version": 0,
            "checkpoint_sha256": manifest["sha256"],
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
    )


def child_rollout(p: dict, root: Path, args) -> None:
    group_path = root / "groups" / f"{args.group_id}.json"
    ack_path = root / "groups" / f"{args.group_id}.ack.json"
    if group_path.exists():
        expected = read_json(Path(str(group_path) + ".sha256"))["sha256"]
        if file_sha(group_path) != expected:
            raise RuntimeError("existing rollout changed")
        if not ack_path.exists():
            write_json(
                ack_path, {"group_sha256": expected, "pid": os.getpid(), "reused": True}
            )
            append_journal(
                root,
                {
                    "kind": "rollout_ack_recovered",
                    "group_id": args.group_id,
                    "group_sha256": expected,
                    "pid": os.getpid(),
                    "time_ns": time.time_ns(),
                },
            )
        return
    rollout_started_unix_ns = time.time_ns()
    model_load_started = time.monotonic()
    task = task_by_id(p, args.task_id)
    checkpoint = root / "models" / args.checkpoint
    torch, tok, model, actions, eos, support = model_bundle(p)
    model_load_seconds = time.monotonic() - model_load_started
    attach_model_ids(model, actions, eos, support)
    checkpoint_meta = restore_actor(model, checkpoint, args.checkpoint_sha256)
    if checkpoint_meta["metadata"]["policy_version"] != args.policy_version:
        raise RuntimeError("worker policy version does not match checkpoint")
    text, prompt_ids = prompt_text(tok, task)
    generations, first_probs, generation_seconds = sample_batch(
        torch,
        tok,
        model,
        task,
        prompt_ids,
        args.checkpoint_sha256,
        args.policy_version,
        args.verifier_version,
        args.sample_seed,
        args.sample_count,
        p,
    )
    group = {
        "schema_version": 1,
        "run_id": p["run_id"],
        "case_id": args.case_id,
        "group_id": args.group_id,
        "task_id": task["task_id"],
        "task": task,
        "prompt": text,
        "prompt_ids": prompt_ids,
        "group_size": len(generations),
        "generations": generations,
        "first_step_support_probabilities": first_probs,
        "action_token_ids": actions,
        "eos_token_id": eos,
        "support_token_ids": support,
        "policy_version": args.policy_version,
        "policy_checkpoint": args.checkpoint,
        "policy_checkpoint_sha256": args.checkpoint_sha256,
        "verifier_version_at_rollout": args.verifier_version,
        "sampling_seed": args.sample_seed,
        "worker_pid": os.getpid(),
        "created_unix_ns": time.time_ns(),
        "rollout_started_unix_ns": rollout_started_unix_ns,
        "model_load_seconds": model_load_seconds,
        "generation_seconds": generation_seconds,
    }
    payload = (
        json.dumps(group, sort_keys=True, indent=2, ensure_ascii=False).encode() + b"\n"
    )
    write_bytes(group_path, payload)
    digest = sha(payload)
    write_json(Path(str(group_path) + ".sha256"), {"sha256": digest})
    append_journal(
        root,
        {
            "kind": "rollout_written",
            "group_id": args.group_id,
            "group_sha256": digest,
            "policy_version": args.policy_version,
            "verifier_version": args.verifier_version,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
    )
    if args.fault_after_write:
        append_journal(
            root,
            {
                "kind": "fault_injected",
                "fault": "rollout_write_before_ack",
                "group_id": args.group_id,
                "pid": os.getpid(),
                "signal": signal.SIGKILL,
                "time_ns": time.time_ns(),
            },
        )
        os.kill(os.getpid(), signal.SIGKILL)
    write_json(ack_path, {"group_sha256": digest, "pid": os.getpid(), "reused": False})
    append_journal(
        root,
        {
            "kind": "rollout_ack",
            "group_id": args.group_id,
            "group_sha256": digest,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
    )


def child_score(p: dict, root: Path, args) -> None:
    scoring_started = time.monotonic()
    group_path = root / "groups" / f"{args.group_id}.json"
    group_sha = file_sha(group_path)
    group = read_json(group_path)
    score_path = root / "scores" / f"{args.group_id}-v{args.verifier_version}.json"
    ack_path = root / "scores" / f"{args.group_id}-v{args.verifier_version}.ack.json"
    if score_path.exists():
        score = read_json(score_path)
        if score["group_sha256"] != group_sha:
            raise RuntimeError("score artifact binds another rollout")
        if not ack_path.exists():
            write_json(
                ack_path,
                {
                    "score_sha256": file_sha(score_path),
                    "pid": os.getpid(),
                    "reused": True,
                },
            )
        return
    version = int(args.verifier_version)
    rows = [
        verifier_score(row["response"], group["task"], version)
        for row in group["generations"]
    ]
    score = {
        "schema_version": 1,
        "group_id": args.group_id,
        "group_sha256": group_sha,
        "verifier_version": version,
        "verifier_config": p["verifiers"][str(version)],
        "verifier_config_sha256": sha(canonical(p["verifiers"][str(version)])),
        "regraded_from": args.regraded_from,
        "rewards": rows,
        "false_trusted_labels_not_disclosed": True,
        "scorer_pid": os.getpid(),
        "created_unix_ns": time.time_ns(),
        "scoring_seconds": time.monotonic() - scoring_started,
    }
    write_json(score_path, score)
    digest = file_sha(score_path)
    append_journal(
        root,
        {
            "kind": "verifier_result_written",
            "group_id": args.group_id,
            "verifier_version": version,
            "score_sha256": digest,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
    )
    if args.fault_after_write:
        append_journal(
            root,
            {
                "kind": "fault_injected",
                "fault": "verifier_write_before_replay_admission",
                "group_id": args.group_id,
                "pid": os.getpid(),
                "signal": signal.SIGKILL,
                "time_ns": time.time_ns(),
            },
        )
        os.kill(os.getpid(), signal.SIGKILL)
    write_json(ack_path, {"score_sha256": digest, "pid": os.getpid(), "reused": False})
    append_journal(
        root,
        {
            "kind": "verifier_result_ack",
            "group_id": args.group_id,
            "verifier_version": version,
            "score_sha256": digest,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
    )


def make_vare_samples(p: dict, group: dict, scores: dict):
    from vare.types import Attempt, Experience, Task, Verification
    from rvl_systems.types import Generation, VerifiedGeneration

    experiences = []
    verified = []
    for index, (row, reward) in enumerate(
        zip(group["generations"], scores["rewards"], strict=True)
    ):
        common = {
            "vare_rollout_group": group["group_id"],
            "vare_rollout_group_size": group["group_size"],
            "sample_index": index,
            "policy_checkpoint_sha256": group["policy_checkpoint_sha256"],
        }
        task = Task(
            id=group["task_id"],
            prompt=group["prompt"],
            family="synthetic-route-pair",
            metadata=dict(common),
        )
        attempt = Attempt(
            task=task,
            output=row["response"],
            policy_id=f"policy-{group['policy_version']}",
            policy_version=group["policy_version"],
            created_step=group["policy_version"],
            logprob=row["behavior_logprob"],
            metadata=dict(common),
        )
        verification = Verification(
            score=float(reward),
            passed=bool(reward),
            confidence=1.0,
            verifier_version=scores["verifier_version"],
            verifier_name="synthetic-pair-proxy",
            trusted=False,
            metadata={
                "verifier_config_sha256": scores["verifier_config_sha256"],
                "score_artifact_sha256": file_sha(Path(scores["_path"])),
            },
        )
        experience = Experience(attempt, verification, 0, 0, 0.0)
        experiences.append(experience)
        generation = Generation(
            prompt_id=group["group_id"],
            prompt=group["prompt"],
            response=row["response"],
            logprob=row["behavior_logprob"],
            token_count=row["generated_token_count"],
            latency_s=0.0,
            metadata={
                "prompt_token_ids": group["prompt_ids"],
                "response_token_ids": row["response_token_ids"],
                "response_token_logprobs": row["response_token_logprobs"],
                "sampling_temperature": row["sampling_temperature"],
            },
        )
        verified.append(
            VerifiedGeneration(
                generation, float(reward), 0.0, scores["verifier_version"]
            )
        )
    return experiences, verified


def child_train(p: dict, root: Path, args) -> None:
    txid = args.transaction_id
    active = read_json(root / "active.json")
    tx_dir = root / "transactions" / txid
    tx_dir.mkdir(parents=True, exist_ok=True)
    commit_path = tx_dir / "commit.json"
    if active.get("transaction_id") == txid:
        if not commit_path.exists():
            commit = {
                "transaction_id": txid,
                "attempt": active.get("attempt"),
                "group_id": active.get("group_id"),
                "effective_update": True,
                "metrics": active.get("metrics"),
                "verifier_version": active.get("verifier_version"),
                "recovered_after_publication": True,
                "active_pointer_sha256": file_sha(root / "active.json"),
                "checkpoint_sha256": active["checkpoint_sha256"],
                "policy_version": active["policy_version"],
                "pid": os.getpid(),
                "recovered_unix_ns": time.time_ns(),
            }
            write_json(commit_path, commit)
            append_journal(
                root, {"kind": "commit_recovered_after_publication", **commit}
            )
        return
    if commit_path.exists():
        raise RuntimeError("commit receipt exists but active pointer is behind")
    prior_attempts = sorted(tx_dir.glob("attempt-*.json"))
    attempt_number = len(prior_attempts) + 1
    injected_faults = [item for item in (args.fault or "").split(",") if item]
    attempt_path = tx_dir / f"attempt-{attempt_number}.json"
    group_path = root / "groups" / f"{args.group_id}.json"
    group = read_json(group_path)
    score_path = root / "scores" / f"{args.group_id}-v{args.score_version}.json"
    scores = read_json(score_path)
    scores["_path"] = str(score_path)
    if scores["group_sha256"] != file_sha(group_path):
        raise RuntimeError("training score does not bind current immutable rollout")
    if active["policy_version"] != args.expected_policy_version:
        raise RuntimeError("learner saw unexpected active policy version")
    if group["policy_version"] > active["policy_version"]:
        raise RuntimeError("rollout policy is ahead of learner")
    if len(set(scores["rewards"])) < 2:
        raise RuntimeError("NO_GO_constant_reward_group")

    p_src = Path(p["vare_source_root"]) / "src"
    rvl_src = Path(p["rvl_source_root"]) / "src"
    sys.path.insert(0, str(p_src))
    sys.path.insert(0, str(rvl_src))
    from vare.config import LagConfig, ReplayConfig
    from vare.lag import LagController, make_experience_freshness_assessor
    from vare.replay import PrioritizedReplay
    from rvl_systems.hf_trainer import HFCausalLMGRPOTrainer, HFTTrainerConfig

    experiences, verified = make_vare_samples(p, group, scores)
    replay = PrioritizedReplay(
        ReplayConfig(capacity=p["generation"]["group_size"] * 2),
        seed=group["sampling_seed"],
    )
    if not replay.add_group(experiences, [1.0] * len(experiences)):
        raise RuntimeError(
            "VARE durable-group replay rejected structurally valid group"
        )

    policy_lag = active["policy_version"] - group["policy_version"]
    verifier_lag = active["verifier_version"] - scores["verifier_version"]
    if policy_lag < 0 or verifier_lag < 0:
        raise RuntimeError("negative freshness lag")
    if args.arm == "C":
        lag_cfg = LagConfig(max_policy_lag=0, max_verifier_lag=0, max_shift=0.0)
    else:
        lag_cfg = LagConfig(max_policy_lag=1, max_verifier_lag=0, max_shift=0.0)
    assessor = make_experience_freshness_assessor(
        LagController(lag_cfg),
        active_policy_version=active["policy_version"],
        active_verifier_version=active["verifier_version"],
    )
    # Always record independently recomputed policy-only and verifier-only decisions.
    policy_only = LagController(
        LagConfig(max_policy_lag=0, max_verifier_lag=2, max_shift=0.0)
    ).assess(
        rollout_policy_version=group["policy_version"],
        active_policy_version=active["policy_version"],
        reward_verifier_version=scores["verifier_version"],
        active_verifier_version=active["verifier_version"],
        shift_score=0.0,
    )
    verifier_only = LagController(
        LagConfig(max_policy_lag=2, max_verifier_lag=0, max_shift=0.0)
    ).assess(
        rollout_policy_version=group["policy_version"],
        active_policy_version=active["policy_version"],
        reward_verifier_version=scores["verifier_version"],
        active_verifier_version=active["verifier_version"],
        shift_score=0.0,
    )
    decisions = [assessor(item) for item in experiences]
    if any(
        (d.policy_lag, d.verifier_lag) != (policy_lag, verifier_lag) for d in decisions
    ):
        raise RuntimeError("VARE use-time freshness did not match active versions")
    naive_bypass = args.arm == "B"
    sampled = replay.sample_current(
        len(experiences),
        grouped=True,
        freshness_fn=(lambda _item: 1.0)
        if naive_bypass
        else (lambda item: 1.0 if assessor(item).admitted else None),
    )
    admission = {
        "group_id": args.group_id,
        "transaction_id": txid,
        "active_policy_version": active["policy_version"],
        "rollout_policy_version": group["policy_version"],
        "policy_lag": policy_lag,
        "active_verifier_version": active["verifier_version"],
        "reward_verifier_version": scores["verifier_version"],
        "verifier_lag": verifier_lag,
        "policy_only_admitted": policy_only.admitted,
        "policy_only_reasons": list(policy_only.reasons),
        "verifier_only_admitted": verifier_only.admitted,
        "verifier_only_reasons": list(verifier_only.reasons),
        "combined_admitted": bool(sampled),
        "naive_staleness_bypass": naive_bypass,
        "replay_sample_count": len(sampled),
        "group_sha256": file_sha(group_path),
        "score_sha256": file_sha(score_path),
        "created_unix_ns": time.time_ns(),
    }
    admission_path = root / "admissions" / f"{args.group_id}.json"
    if not sampled:
        if admission_path.exists():
            if (
                read_json(admission_path).get("group_sha256")
                != admission["group_sha256"]
            ):
                raise RuntimeError("existing rejection receipt binds another rollout")
        else:
            write_json(admission_path, admission)
            append_journal(
                root, {"kind": "group_rejected", **admission, "pid": os.getpid()}
            )
        return
    if len(sampled) != len(experiences):
        raise RuntimeError("group-aware replay returned partial GRPO group")
    if admission_path.exists():
        existing_admission = read_json(admission_path)
        if existing_admission.get("group_sha256") != admission[
            "group_sha256"
        ] or not existing_admission.get("combined_admitted"):
            raise RuntimeError("existing admission receipt conflicts with this update")
    else:
        write_json(admission_path, admission)
        append_journal(
            root, {"kind": "group_admitted", **admission, "pid": os.getpid()}
        )

    torch, _tok, model, actions, eos, support = model_bundle(p)
    model.vare_prompt_length = len(group["prompt_ids"])
    attach_model_ids(model, actions, eos, support)
    active_path = root / "models" / active["checkpoint"]
    _meta, state = checkpoint_state(active_path, active["checkpoint_sha256"])
    with torch.no_grad():
        model.vare_online_action_bias.copy_(state["bias"])
    torch.set_rng_state(state["torch_rng"])
    trainer = HFCausalLMGRPOTrainer(
        model,
        config=HFTTrainerConfig(
            learning_rate=p["training"]["learning_rate"],
            clip_eps=p["training"]["clip_epsilon"],
            max_grad_norm=p["training"]["max_grad_norm"],
            advantage_eps=p["training"]["advantage_epsilon"],
            clip_advantage=p["training"]["clip_advantage"],
            objective_backend="torch",
        ),
    )
    trainer.optimizer.load_state_dict(state["optimizer"])
    attempt_started = {
        "transaction_id": txid,
        "attempt": attempt_number,
        "group_id": args.group_id,
        "pid": os.getpid(),
        "input_checkpoint_sha256": active["checkpoint_sha256"],
        "group_sha256": file_sha(group_path),
        "score_sha256": file_sha(score_path),
        "start_unix_ns": time.time_ns(),
    }
    write_json(attempt_path, attempt_started)
    before = model.vare_online_action_bias.detach().clone()
    metrics = trainer.train_step(verified)
    grad = model.vare_online_action_bias.grad.detach().clone()
    support_grad = grad[support].tolist()
    after = model.vare_online_action_bias.detach().clone()
    attempt_record = {
        **attempt_started,
        "status": "optimizer_applied_uncommitted",
        "support_ids": support,
        "gradient_support_after_global_clip": support_grad,
        "gradient_l2_after_global_clip": float(torch.linalg.vector_norm(grad).item()),
        "bias_before_support": before[support].tolist(),
        "bias_after_support": after[support].tolist(),
        "metrics": metrics,
        "completed_optimizer_computations": 1,
        "completed_unix_ns": time.time_ns(),
    }
    write_json(attempt_path, attempt_record, replace=True)
    append_journal(
        root,
        {
            "kind": "optimizer_applied",
            "transaction_id": txid,
            "attempt": attempt_number,
            "group_id": args.group_id,
            "input_checkpoint_sha256": active["checkpoint_sha256"],
            "gradient_support_after_global_clip": support_grad,
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
    )
    if (
        "after_optimizer_apply" in injected_faults
        and attempt_number == injected_faults.index("after_optimizer_apply") + 1
    ):
        append_journal(
            root,
            {
                "kind": "fault_injected",
                "fault": "optimizer_apply_before_commit",
                "transaction_id": txid,
                "pid": os.getpid(),
                "signal": signal.SIGKILL,
                "time_ns": time.time_ns(),
            },
        )
        os.kill(os.getpid(), signal.SIGKILL)
    new_version = active["policy_version"] + 1
    candidate_name = f"candidate-{txid}-attempt-{attempt_number}.pt.gz"
    candidate_manifest = save_checkpoint(
        root / "models" / candidate_name,
        model,
        trainer.optimizer,
        torch,
        {
            "policy_version": new_version,
            "seed": group["sampling_seed"],
            "transaction_id": txid,
            "group_id": args.group_id,
            "base_checkpoint_sha256": active["checkpoint_sha256"],
            "group_sha256": file_sha(group_path),
            "score_sha256": file_sha(score_path),
            "verifier_version": scores["verifier_version"],
            "model_sha256": p["model"]["weight_sha256"],
        },
    )
    attempt_record.update(
        {
            "candidate_checkpoint": candidate_name,
            "candidate_checkpoint_sha256": candidate_manifest["sha256"],
            "status": "candidate_durable_unpublished",
        }
    )
    write_json(attempt_path, attempt_record, replace=True)
    append_journal(
        root,
        {
            "kind": "candidate_checkpoint_written",
            "transaction_id": txid,
            "attempt": attempt_number,
            "checkpoint": candidate_name,
            "checkpoint_sha256": candidate_manifest["sha256"],
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
    )
    if args.wait_for_group_id:
        queued_path = root / "groups" / f"{args.wait_for_group_id}.json"
        deadline = time.monotonic() + p["limits"]["wall_seconds"]
        while not queued_path.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"queued rollout not durable before publication: {queued_path}"
                )
            time.sleep(0.02)
        append_journal(
            root,
            {
                "kind": "publication_wait_satisfied",
                "transaction_id": txid,
                "queued_group_id": args.wait_for_group_id,
                "queued_group_sha256": file_sha(queued_path),
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
            },
        )
    if (
        "before_publication" in injected_faults
        and attempt_number == injected_faults.index("before_publication") + 1
    ):
        append_journal(
            root,
            {
                "kind": "fault_injected",
                "fault": "checkpoint_before_publication",
                "transaction_id": txid,
                "pid": os.getpid(),
                "signal": signal.SIGKILL,
                "time_ns": time.time_ns(),
            },
        )
        os.kill(os.getpid(), signal.SIGKILL)
    pointer = {
        "policy_version": new_version,
        "verifier_version": active["verifier_version"],
        "checkpoint": candidate_name,
        "checkpoint_sha256": candidate_manifest["sha256"],
        "transaction_id": txid,
        "attempt": attempt_number,
        "group_id": args.group_id,
        "metrics": metrics,
        "parent_checkpoint_sha256": active["checkpoint_sha256"],
    }
    write_json(root / "active.json", pointer, replace=True)
    append_journal(
        root,
        {
            "kind": "checkpoint_published",
            "transaction_id": txid,
            "policy_version": new_version,
            "checkpoint_sha256": candidate_manifest["sha256"],
            "pid": os.getpid(),
            "time_ns": time.time_ns(),
        },
    )
    if (
        "after_publication" in injected_faults
        and attempt_number == injected_faults.index("after_publication") + 1
    ):
        append_journal(
            root,
            {
                "kind": "fault_injected",
                "fault": "publication_before_actor_refresh",
                "transaction_id": txid,
                "pid": os.getpid(),
                "signal": signal.SIGKILL,
                "time_ns": time.time_ns(),
            },
        )
        os.kill(os.getpid(), signal.SIGKILL)
    commit = {
        "transaction_id": txid,
        "group_id": args.group_id,
        "effective_update": True,
        "attempt": attempt_number,
        "active_pointer_sha256": file_sha(root / "active.json"),
        "checkpoint_sha256": candidate_manifest["sha256"],
        "policy_version": new_version,
        "verifier_version": scores["verifier_version"],
        "metrics": metrics,
        "pid": os.getpid(),
        "committed_unix_ns": time.time_ns(),
    }
    write_json(commit_path, commit)
    attempt_record["status"] = "committed"
    write_json(attempt_path, attempt_record, replace=True)
    append_journal(root, {"kind": "effective_update_commit", **commit})


def child_eval(p: dict, root: Path, args) -> None:
    active = active_record(root)
    checkpoint = root / "models" / active["checkpoint"]
    torch, tok, model, actions, eos, support = model_bundle(p)
    attach_model_ids(model, actions, eos, support)
    restore_actor(model, checkpoint, active["checkpoint_sha256"])
    rows = []
    task_ids = args.task_ids.split(",")
    for index, task_id in enumerate(task_ids):
        task = task_by_id(p, task_id)
        text, prompt_ids = prompt_text(tok, task)
        generations, first_probs, elapsed = sample_batch(
            torch,
            tok,
            model,
            task,
            prompt_ids,
            active["checkpoint_sha256"],
            active["policy_version"],
            active["verifier_version"],
            args.sample_seed + index * 100003,
            args.sample_count,
            p,
        )
        rows.append(
            {
                "task_id": task_id,
                "task": task,
                "prompt": text,
                "prompt_ids": prompt_ids,
                "policy_version": active["policy_version"],
                "policy_checkpoint": active["checkpoint"],
                "policy_checkpoint_sha256": active["checkpoint_sha256"],
                "verifier_version": active["verifier_version"],
                "sample_seed": args.sample_seed + index * 100003,
                "first_step_support_probabilities": first_probs,
                "generation_seconds": elapsed,
                "generations": generations,
            }
        )
    dest = root / args.output_name
    write_json(
        dest,
        {
            "schema_version": 1,
            "kind": args.kind,
            "case_id": args.case_id,
            "policy_version": active["policy_version"],
            "policy_checkpoint_sha256": active["checkpoint_sha256"],
            "verifier_version": active["verifier_version"],
            "worker_pid": os.getpid(),
            "tasks": rows,
        },
    )
    write_json(Path(str(dest) + ".sha256"), {"sha256": file_sha(dest)})


def child(p: dict, root: Path, args) -> None:
    if args.mode == "init":
        child_init(p, root, args.seed)
    elif args.mode == "rollout":
        child_rollout(p, root, args)
    elif args.mode == "score":
        child_score(p, root, args)
    elif args.mode == "train":
        child_train(p, root, args)
    elif args.mode == "eval":
        child_eval(p, root, args)
    else:
        raise ValueError(args.mode)


def args_for_child(protocol_path: Path, root: Path, **values) -> list[str]:
    args = [
        sys.executable,
        __file__,
        "child",
        "--protocol",
        str(protocol_path),
        "--output",
        str(root),
    ]
    for key, value in values.items():
        if value is True:
            args.append("--" + key.replace("_", "-"))
        elif value is not False and value is not None:
            args.extend(["--" + key.replace("_", "-"), str(value)])
    return args


def process_tree_rss(proc) -> int:
    import psutil

    total = 0
    try:
        parent = psutil.Process(proc.pid)
        for process in [parent, *parent.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    return total


def completed_generation_tokens(root: Path) -> int:
    total = 0
    for path in root.rglob("groups/*.json"):
        group = read_json(path)
        total += sum(
            int(row["generated_token_count"]) for row in group.get("generations", [])
        )
    for path in root.rglob("evaluation.json"):
        result = read_json(path)
        total += sum(
            int(generation["generated_token_count"])
            for task in result.get("tasks", [])
            for generation in task.get("generations", [])
        )
    return total


def completed_prompt_tokens(root: Path) -> int:
    total = 0
    for path in root.rglob("groups/*.json"):
        group = read_json(path)
        total += len(group.get("prompt_ids", [])) * len(group.get("generations", []))
    for path in root.rglob("evaluation.json"):
        result = read_json(path)
        total += sum(
            len(task.get("prompt_ids", [])) * len(task.get("generations", []))
            for task in result.get("tasks", [])
        )
    return total


def launch_job(protocol_path: Path, root: Path, job: dict, process_log: list) -> dict:
    attempt = job["attempt"] + 1
    job["attempt"] = attempt
    log_path = root / "process-logs" / f"{job['name']}-attempt-{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    command = args_for_child(protocol_path, job["root"], **job["values"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(job["repo_root"] / "src"),
            str(Path(job["rvl_root"]) / "src"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    proc = subprocess.Popen(
        command, stdout=handle, stderr=subprocess.STDOUT, env=environment
    )
    record = {
        "name": job["name"],
        "attempt": attempt,
        "pid": proc.pid,
        "command": command,
        "log": str(log_path.relative_to(root)),
        "started_unix_ns": time.time_ns(),
    }
    process_log.append(record)
    job["proc"] = proc
    job["handle"] = handle
    job["record"] = record
    return job


def run_jobs(
    protocol_path: Path,
    root: Path,
    jobs: list[dict],
    process_log: list,
    started: float,
    baseline_swap: int,
) -> dict:
    import psutil

    pending = {
        job["name"]: launch_job(protocol_path, root, job, process_log) for job in jobs
    }
    peak = 0
    minimum_ram = psutil.virtual_memory().available
    max_swap_growth = 0
    try:
        while pending:
            total_rss = 0
            for name, job in list(pending.items()):
                proc = job["proc"]
                if proc.poll() is not None:
                    code = proc.returncode
                    job["handle"].close()
                    job["record"]["returncode"] = code
                    job["record"]["ended_unix_ns"] = time.time_ns()
                    if code == -signal.SIGKILL and job["expected_kills"] > 0:
                        job["expected_kills"] -= 1
                        pending[name] = launch_job(protocol_path, root, job, process_log)
                        continue
                    if code != 0:
                        raise RuntimeError(
                            f"{name} exited with {code}; inspect {job['record']['log']}"
                        )
                    if job["expected_kills"]:
                        raise RuntimeError(
                            f"{name} omitted {job['expected_kills']} frozen SIGKILL fault(s)"
                        )
                    del pending[name]
                else:
                    total_rss += process_tree_rss(proc)
            peak = max(peak, total_rss)
            minimum_ram = min(minimum_ram, psutil.virtual_memory().available)
            swap_growth = max(0, psutil.swap_memory().used - baseline_swap)
            max_swap_growth = max(max_swap_growth, swap_growth)
            limits = read_json(root / "protocol-reference.json")["limits"]
            if (
                total_rss > limits["max_child_rss_bytes"]
                or minimum_ram < limits["hard_min_available_bytes"]
                or swap_growth > limits["max_swap_growth_bytes"]
            ):
                raise RuntimeError("RESOURCE_ABORT host/RSS/swap cap")
            generated_tokens = completed_generation_tokens(root)
            if generated_tokens > limits["max_total_generated_tokens"]:
                raise RuntimeError("RESOURCE_ABORT generated-token cap")
            model_tokens = generated_tokens + completed_prompt_tokens(root)
            if model_tokens > limits["max_total_model_tokens"]:
                raise RuntimeError("RESOURCE_ABORT total model-token cap")
            if time.monotonic() - started > limits["wall_seconds"]:
                raise RuntimeError("RESOURCE_ABORT wall-clock cap")
            time.sleep(0.1)
    except BaseException:
        for job in pending.values():
            proc = job["proc"]
            if proc.poll() is None:
                proc.kill()
                proc.wait()
                job["record"]["returncode"] = proc.returncode
                job["record"]["ended_unix_ns"] = time.time_ns()
            if not job["handle"].closed:
                job["handle"].close()
        raise
    return {
        "peak_child_rss_bytes": peak,
        "minimum_host_available_bytes": minimum_ram,
        "swap_growth_bytes": max_swap_growth,
    }


def worker_job(
    repo_root: Path,
    rvl_root: str,
    root: Path,
    name: str,
    case_id: str,
    group_id: str,
    task_id: str,
    policy: dict,
    verifier_version: int,
    seed: int,
    count: int,
    expected_kills: int = 0,
    fault_after_write: bool = False,
) -> dict:
    return {
        "name": name,
        "attempt": 0,
        "expected_kills": expected_kills,
        "repo_root": repo_root,
        "rvl_root": rvl_root,
        "root": root,
        "values": {
            "mode": "rollout",
            "case_id": case_id,
            "group_id": group_id,
            "task_id": task_id,
            "checkpoint": policy["checkpoint"],
            "checkpoint_sha256": policy["checkpoint_sha256"],
            "policy_version": policy["policy_version"],
            "verifier_version": verifier_version,
            "sample_seed": seed,
            "sample_count": count,
            "fault_after_write": fault_after_write,
        },
    }


def scorer_job(
    repo_root: Path,
    rvl_root: str,
    root: Path,
    name: str,
    case_id: str,
    group_id: str,
    version: int,
    expected_kills: int = 0,
    fault_after_write: bool = False,
    regraded_from=None,
) -> dict:
    return {
        "name": name,
        "attempt": 0,
        "expected_kills": expected_kills,
        "repo_root": repo_root,
        "rvl_root": rvl_root,
        "root": root,
        "values": {
            "mode": "score",
            "case_id": case_id,
            "group_id": group_id,
            "verifier_version": version,
            "fault_after_write": fault_after_write,
            "regraded_from": regraded_from,
        },
    }


def learner_job(
    repo_root: Path,
    rvl_root: str,
    root: Path,
    name: str,
    case_id: str,
    arm: str,
    group_id: str,
    txid: str,
    expected_policy: int,
    score_version: int,
    expected_kills: int = 0,
    fault: str | None = None,
    wait_for_group_id: str | None = None,
) -> dict:
    return {
        "name": name,
        "attempt": 0,
        "expected_kills": expected_kills,
        "repo_root": repo_root,
        "rvl_root": rvl_root,
        "root": root,
        "values": {
            "mode": "train",
            "case_id": case_id,
            "arm": arm,
            "group_id": group_id,
            "transaction_id": txid,
            "expected_policy_version": expected_policy,
            "score_version": score_version,
            "fault": fault,
            "wait_for_group_id": wait_for_group_id,
        },
    }


def copy_initial(template: Path, root: Path, seed: int) -> dict:
    (root / "models").mkdir(parents=True)
    source = template / "models" / "policy-0.pt.gz"
    destination = root / "models" / source.name
    shutil.copyfile(source, destination)
    shutil.copyfile(Path(str(source) + ".json"), Path(str(destination) + ".json"))
    meta = read_json(Path(str(destination) + ".json"))
    pointer = {
        "policy_version": 0,
        "verifier_version": 0,
        "checkpoint": destination.name,
        "checkpoint_sha256": meta["sha256"],
        "transaction_id": "initial",
    }
    write_json(root / "active.json", pointer)
    return pointer


def append_case_result(root: Path, result: dict) -> None:
    write_json(root / "case-result.json", result)


def check_preflight(p: dict, output: Path) -> dict:
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    disk = psutil.disk_usage(str(output.parent))
    if vm.available < p["limits"]["min_available_bytes"]:
        raise RuntimeError("PREFLIGHT_ABORT available RAM below frozen minimum")
    if sw.free < p["limits"]["min_free_swap_bytes"]:
        raise RuntimeError("PREFLIGHT_ABORT free swap below frozen minimum")
    if disk.free < p["limits"]["min_disk_bytes"]:
        raise RuntimeError("PREFLIGHT_ABORT disk space below frozen minimum")
    return {
        "available_ram_bytes": vm.available,
        "free_swap_bytes": sw.free,
        "disk_free_bytes": disk.free,
        "swap_used_bytes": sw.used,
        "captured_unix_ns": time.time_ns(),
    }


def campaign(p_path: Path, output: Path) -> None:
    import psutil

    if output.exists():
        raise FileExistsError(f"append-only output already exists: {output}")
    p = protocol(p_path)
    for rel, digest in p["source_hashes"].items():
        if file_sha(Path(p["repository_root"]) / rel) != digest:
            raise RuntimeError(f"frozen experiment source mismatch: {rel}")
    for rel, digest_value in p["model"]["snapshot_files"].items():
        if file_sha(Path(p["model"]["snapshot_path"]) / rel) != digest_value:
            raise RuntimeError(f"cached model snapshot changed: {rel}")
    for rel, digest_value in p["rvl_source_hashes"].items():
        if file_sha(Path(p["rvl_source_root"]) / rel) != digest_value:
            raise RuntimeError(f"frozen RVL source mismatch: {rel}")
    if subprocess.run(
        [
            "git",
            "-C",
            p["repository_root"],
            "merge-base",
            "--is-ancestor",
            p["source_commit"],
            "HEAD",
        ],
        check=False,
    ).returncode != 0:
        raise RuntimeError(
            "frozen source commit is not an ancestor of the experiment checkout"
        )
    if (
        subprocess.check_output(
            ["git", "-C", p["rvl_source_root"], "rev-parse", "HEAD"], text=True
        ).strip()
        != p["rvl_commit"]
    ):
        raise RuntimeError("RVL checkout does not match frozen revision")
    if subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import torch,transformers;print(torch.__version__, transformers.__version__)",
        ],
        text=True,
    ).strip().split() != [p["runtime"]["torch"], p["runtime"]["transformers"]]:
        raise RuntimeError("runtime package versions do not match frozen protocol")
    preflight = check_preflight(p, output)
    output.mkdir(parents=True)
    shutil.copyfile(p_path, output / "protocol-v1.json")
    write_json(output / "protocol-v1.json.sha256", {"sha256": sha(p_path.read_bytes())})
    write_json(output / "preflight.json", preflight)
    write_json(output / "resource-limit.json", p["limits"])
    protocol_copy = output / "protocol-reference.json"
    write_json(protocol_copy, p)
    calibration_path = output / "verifier-calibration.json"
    calibration = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("audit_online_actor.py")),
            "--protocol",
            str(p_path),
            "--output",
            str(calibration_path),
            "--calibration-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (output / "calibration.stdout.log").write_text(
        calibration.stdout + calibration.stderr
    )
    started = time.monotonic()
    baseline_swap = psutil.swap_memory().used
    process_log: list[dict] = []
    case_metrics = []
    repo_root = Path(p["repository_root"])
    rvl_root = p["rvl_source_root"]

    templates = {}
    baseline_roots = {}
    eval_task_ids = [
        task["task_id"] for task in p["tasks"] if task["split"] == "evaluation"
    ]
    for seed in p["seeds"]:
        template = output / "_templates" / f"seed-{seed}"
        template.mkdir(parents=True)
        init = {
            "name": f"init-seed-{seed}",
            "attempt": 0,
            "expected_kills": 0,
            "repo_root": repo_root,
            "rvl_root": rvl_root,
            "root": template,
            "values": {"mode": "init", "seed": seed},
        }
        resources = run_jobs(
            p_path, output, [init], process_log, started, baseline_swap
        )
        case_metrics.append({"case": f"init-{seed}", **resources})
        templates[seed] = template
        baseline_root = output / "baseline" / f"seed-{seed}"
        baseline_root.mkdir(parents=True)
        copy_initial(template, baseline_root, seed)
        baseline_job = {
            "name": f"baseline-eval-{seed}",
            "attempt": 0,
            "expected_kills": 0,
            "repo_root": repo_root,
            "rvl_root": rvl_root,
            "root": baseline_root,
            "values": {
                "mode": "eval",
                "case_id": f"baseline-{seed}",
                "kind": "baseline",
                "task_ids": ",".join(eval_task_ids),
                    "sample_seed": seed * 10000 + 79,
                "sample_count": p["evaluation"]["samples_per_task"],
                "output_name": "evaluation.json",
            },
        }
        resources = run_jobs(
            p_path, output, [baseline_job], process_log, started, baseline_swap
        )
        baseline_roots[seed] = baseline_root
        case_metrics.append({"case": f"baseline-{seed}", **resources})

    for seed, arm in p["execution_order"]:
        case_id = f"seed-{seed}-{arm}"
        root = output / "cases" / case_id
        root.mkdir(parents=True)
        case_started = time.monotonic()
        case_resource = {
            "peak_child_rss_bytes": 0,
            "minimum_host_available_bytes": preflight["available_ram_bytes"],
            "swap_growth_bytes": 0,
        }

        def execute(jobs):
            measured = run_jobs(
                p_path, output, jobs, process_log_local, started, baseline_swap
            )
            case_resource["peak_child_rss_bytes"] = max(
                case_resource["peak_child_rss_bytes"], measured["peak_child_rss_bytes"]
            )
            case_resource["minimum_host_available_bytes"] = min(
                case_resource["minimum_host_available_bytes"],
                measured["minimum_host_available_bytes"],
            )
            case_resource["swap_growth_bytes"] = max(
                case_resource["swap_growth_bytes"], measured["swap_growth_bytes"]
            )
            return measured

        process_log_local: list[dict] = []
        active = copy_initial(templates[seed], root, seed)
        task0 = p["training_tasks"][0]
        task1 = p["training_tasks"][1]
        async_arm = arm in ("B", "C", "D", "E")
        seed_key = p["fault_plan"].get(case_id, {})

        group0 = f"{case_id}-group-0"
        job0 = worker_job(
            repo_root,
            rvl_root,
            root,
            f"{case_id}-rollout-0",
            case_id,
            group0,
            task0,
            active,
            0,
            seed * 1000 + 11,
            p["generation"]["group_size"],
        )
        resources = execute([job0])
        score0 = scorer_job(
            repo_root, rvl_root, root, f"{case_id}-score-0-v0", case_id, group0, 0
        )
        execute([score0])

        stale_worker = None
        if async_arm:
            stale_worker = worker_job(
                repo_root,
                rvl_root,
                root,
                f"{case_id}-async-rollout-1",
                case_id,
                f"{case_id}-group-1",
                task1,
                active,
                0,
                seed * 1000 + 29,
                p["generation"]["group_size"],
                expected_kills=int(seed_key.get("rollout_write_before_ack", False)),
                fault_after_write=bool(seed_key.get("rollout_write_before_ack", False)),
            )
            learner0 = learner_job(
                repo_root,
                rvl_root,
                root,
                f"{case_id}-learn-0",
                case_id,
                arm,
                group0,
                f"{case_id}-update-0",
                0,
                0,
                wait_for_group_id=f"{case_id}-group-1",
            )
            resources = execute([stale_worker, learner0])
        else:
            learner0 = learner_job(
                repo_root,
                rvl_root,
                root,
                f"{case_id}-learn-0",
                case_id,
                arm,
                group0,
                f"{case_id}-update-0",
                0,
                0,
            )
            resources = execute([learner0])
        active = active_record(root)
        if active["policy_version"] != 1:
            raise RuntimeError(
                "first online GRPO update did not publish policy version 1"
            )
        active["verifier_version"] = 1
        write_json(
            root / "active.json",
            {
                key: value
                for key, value in active.items()
                if key not in ("checkpoint_path", "checkpoint_metadata")
            },
            replace=True,
        )
        append_journal(
            root,
            {
                "kind": "verifier_transition",
                "from_version": 0,
                "to_version": 1,
                "after_policy_version": 1,
                "pid": os.getpid(),
                "time_ns": time.time_ns(),
            },
        )

        group1 = f"{case_id}-group-1"
        if not async_arm:
            active = active_record(root)
            job1 = worker_job(
                repo_root,
                rvl_root,
                root,
                f"{case_id}-rollout-1",
                case_id,
                group1,
                task1,
                active,
                1,
                seed * 1000 + 29,
                p["generation"]["group_size"],
            )
            resources = execute([job1])
        if arm == "A":
            score1_v1 = scorer_job(
                repo_root, rvl_root, root, f"{case_id}-score-1-v1", case_id, group1, 1
            )
            execute([score1_v1])
            score_version = 1
        else:
            score1_v0 = scorer_job(
                repo_root, rvl_root, root, f"{case_id}-score-1-v0", case_id, group1, 0
            )
            execute([score1_v0])
        if arm in ("D", "E"):
            score1_v1 = scorer_job(
                repo_root,
                rvl_root,
                root,
                f"{case_id}-score-1-v1",
                case_id,
                group1,
                1,
                expected_kills=int(
                    seed_key.get("verifier_write_before_admission", False)
                ),
                fault_after_write=bool(
                    seed_key.get("verifier_write_before_admission", False)
                ),
                regraded_from=file_sha(root / "scores" / f"{group1}-v0.json"),
            )
            execute([score1_v1])
            score_version = 1
        elif arm != "A":
            score_version = 0
        learner1 = learner_job(
            repo_root,
            rvl_root,
            root,
            f"{case_id}-learn-1",
            case_id,
            arm,
            group1,
            f"{case_id}-update-1",
            1,
            score_version,
            expected_kills=sum(
                bool(seed_key.get(key))
                for key in (
                    "optimizer_apply_before_commit",
                    "checkpoint_before_publication",
                    "publication_before_actor_refresh",
                )
            ),
            fault=",".join(
                name
                for key, name in (
                    ("optimizer_apply_before_commit", "after_optimizer_apply"),
                    ("checkpoint_before_publication", "before_publication"),
                    ("publication_before_actor_refresh", "after_publication"),
                )
                if seed_key.get(key)
            )
            or None,
        )
        execute([learner1])

        if arm == "C":
            group2 = f"{case_id}-group-2-replacement"
            active = active_record(root)
            replacement = worker_job(
                repo_root,
                rvl_root,
                root,
                f"{case_id}-replacement-rollout",
                case_id,
                group2,
                task1,
                active,
                1,
                seed * 1000 + 31,
                p["generation"]["group_size"],
            )
            execute([replacement])
            score2 = scorer_job(
                repo_root,
                rvl_root,
                root,
                f"{case_id}-replacement-score",
                case_id,
                group2,
                1,
            )
            execute([score2])
            learner2 = learner_job(
                repo_root,
                rvl_root,
                root,
                f"{case_id}-replacement-learn",
                case_id,
                arm,
                group2,
                f"{case_id}-update-1-replacement",
                1,
                1,
            )
            execute([learner2])

        final_active = active_record(root)
        if final_active["policy_version"] != 2:
            raise RuntimeError(
                "second effective GRPO update did not publish policy version 2"
            )
        probe_group = f"{case_id}-fresh-policy-probe"
        probe = worker_job(
            repo_root,
            rvl_root,
            root,
            f"{case_id}-probe",
            case_id,
            probe_group,
            task1,
            final_active,
            1,
            seed * 1000 + 47,
            p["evaluation"]["samples_per_task"],
        )
        execute([probe])
        evaluation = {
            "name": f"{case_id}-eval",
            "attempt": 0,
            "expected_kills": 0,
            "repo_root": repo_root,
            "rvl_root": rvl_root,
            "root": root,
            "values": {
                "mode": "eval",
                "case_id": case_id,
                "kind": "post_training",
                "task_ids": ",".join(eval_task_ids),
                "sample_seed": seed * 10000 + 79,
                "sample_count": p["evaluation"]["samples_per_task"],
                "output_name": "evaluation.json",
            },
        }
        execute([evaluation])
        write_json(root / "processes.json", process_log_local)
        case_resource["wall_seconds"] = time.monotonic() - case_started
        append_case_result(
            root,
            {
                "case_id": case_id,
                "seed": seed,
                "arm": arm,
                "updates": 2,
                "final_policy_version": final_active["policy_version"],
                "final_checkpoint_sha256": final_active["checkpoint_sha256"],
                "process_attempts": process_log_local,
                "resource_peak": case_resource,
                "async_group_id": group1 if async_arm else None,
                "fresh_probe_group_id": probe_group,
            },
        )
        case_metrics.append({"case": case_id, **case_resource})

    total = {
        "run_id": p["run_id"],
        "wall_seconds": time.monotonic() - started,
        "peak_child_rss_bytes": max(
            row.get("peak_child_rss_bytes", 0) for row in case_metrics
        ),
        "minimum_host_available_bytes": min(
            row.get("minimum_host_available_bytes", preflight["available_ram_bytes"])
            for row in case_metrics
        ),
        "maximum_swap_growth_bytes": max(
            row.get("swap_growth_bytes", 0) for row in case_metrics
        ),
        "total_generated_tokens": completed_generation_tokens(output),
        "total_prompt_tokens": completed_prompt_tokens(output),
        "total_model_tokens": completed_generation_tokens(output)
        + completed_prompt_tokens(output),
        "cases": case_metrics,
    }
    if total["wall_seconds"] > p["limits"]["wall_seconds"]:
        raise RuntimeError("RESOURCE_ABORT: campaign wall-clock cap")
    write_json(output / "processes.json", process_log)
    write_json(output / "resource-summary.json", total)
    write_json(
        output / "run-status.json",
        {
            "status": "MODEL_RUNS_COMPLETE_AUDIT_PENDING",
            "completed_unix_ns": time.time_ns(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    campaign_parser = sub.add_parser("campaign")
    campaign_parser.add_argument("--protocol", required=True, type=Path)
    campaign_parser.add_argument("--output", required=True, type=Path)
    child_parser = sub.add_parser("child")
    child_parser.add_argument("--protocol", required=True, type=Path)
    child_parser.add_argument("--output", required=True, type=Path)
    child_parser.add_argument(
        "--mode", required=True, choices=("init", "rollout", "score", "train", "eval")
    )
    child_parser.add_argument("--seed", type=int, default=0)
    child_parser.add_argument("--case-id", default="")
    child_parser.add_argument("--group-id", default="")
    child_parser.add_argument("--task-id", default="")
    child_parser.add_argument("--checkpoint", default="")
    child_parser.add_argument("--checkpoint-sha256", default="")
    child_parser.add_argument("--policy-version", type=int, default=0)
    child_parser.add_argument("--verifier-version", type=int, default=0)
    child_parser.add_argument("--sample-seed", type=int, default=0)
    child_parser.add_argument("--sample-count", type=int, default=0)
    child_parser.add_argument("--fault-after-write", action="store_true")
    child_parser.add_argument("--regraded-from")
    child_parser.add_argument("--arm", default="A")
    child_parser.add_argument("--transaction-id", default="")
    child_parser.add_argument("--expected-policy-version", type=int, default=0)
    child_parser.add_argument("--score-version", type=int, default=0)
    child_parser.add_argument("--fault")
    child_parser.add_argument("--wait-for-group-id")
    child_parser.add_argument("--task-ids", default="")
    child_parser.add_argument("--kind", default="")
    child_parser.add_argument("--output-name", default="output.json")
    args = parser.parse_args()
    if args.command == "campaign":
        output = args.output.resolve()
        existed = output.exists()
        try:
            campaign(args.protocol.resolve(), output)
        except BaseException as exc:
            if not existed:
                output.mkdir(parents=True, exist_ok=True)
            if not existed and not (output / "failure.json").exists():
                message = str(exc)
                status = (
                    "PREFLIGHT_ABORT"
                    if "PREFLIGHT_ABORT" in message
                    else "RESOURCE_ABORT"
                    if "RESOURCE_ABORT" in message
                    else "MODEL_RUN_FAILURE"
                )
                write_json(
                    output / "failure.json",
                    {
                        "status": status,
                        "error_type": type(exc).__name__,
                        "error": message,
                        "protocol_sha256": file_sha(args.protocol.resolve()),
                        "failed_unix_ns": time.time_ns(),
                    },
                )
            raise
    else:
        p = protocol(args.protocol.resolve())
        root = args.output.resolve()
        child(p, root, args)


if __name__ == "__main__":
    main()
