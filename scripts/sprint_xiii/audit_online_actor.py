#!/usr/bin/env python3
"""Independent audit for the Sprint XIII online-actor experiment.

This module intentionally does not import online_actor.py or RVL's trainer.
It reloads Qwen, replays behavior probabilities and GRPO/Adam updates, checks
the durable version/provenance chain, and computes the trusted synthetic
evaluation from retained responses.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def check(condition: bool, message: str, findings: list[str]) -> None:
    if not condition:
        findings.append(message)


def active_matches_latest_commit(active: dict, commit: dict) -> bool:
    return (
        active.get("transaction_id") == commit.get("transaction_id")
        and active.get("checkpoint_sha256") == commit.get("checkpoint_sha256")
        and active.get("policy_version") == commit.get("policy_version")
    )


def gradient_origin_matches(
    input_bias, attempt: dict, support_ids: list[int], tolerance: float
) -> bool:
    expected = [float(value) for value in attempt["bias_before_support"]]
    actual = input_bias[support_ids].tolist()
    return len(actual) == len(expected) and max(
        abs(float(left) - right) for left, right in zip(actual, expected, strict=True)
    ) <= tolerance


def evaluation_aggregation_matches(task_results: dict, claimed_mean: float) -> bool:
    if not task_results:
        return False
    recomputed = statistics.mean(
        float(item["mean_trusted_reward"]) for item in task_results.values()
    )
    return math.isfinite(float(claimed_mean)) and abs(recomputed - float(claimed_mean)) <= 1e-12


def distribution_metrics(before: list[float], after: list[float]) -> dict:
    if len(before) != len(after) or not before:
        raise ValueError("policy distributions must have equal, nonzero support")
    floor = 1e-30
    tv = 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(before, after, strict=True))
    kl = sum(
        float(a) * math.log(max(float(a), floor) / max(float(b), floor))
        for a, b in zip(before, after, strict=True)
    )
    return {"total_variation": tv, "kl_before_to_after": kl}


def validate_protocol_contract(p: dict) -> None:
    if not isinstance(p.get("run_id"), str) or not p["run_id"].strip():
        raise ValueError("protocol requires a non-empty run_id")
    tasks = p.get("tasks")
    manifest_tasks = p.get("task_manifest", {}).get("tasks")
    if not isinstance(tasks, list) or tasks != manifest_tasks:
        raise ValueError("protocol tasks must exactly match task_manifest.tasks")


def trusted(actions: str, task: dict) -> float:
    target = task["target_pair"]
    forbidden = task["forbidden_pair"]
    target_found = any(
        actions[i : i + 2] == target for i in range(max(0, len(actions) - 1))
    )
    forbidden_found = any(
        actions[i : i + 2] == forbidden for i in range(max(0, len(actions) - 1))
    )
    return float(target_found and not forbidden_found)


def proxy(actions: str, task: dict, version: int) -> float:
    target = task["target_pair"]
    forbidden = task["forbidden_pair"]
    if version == 0:
        return min(1.0, actions[:3].count(target[0]) / 2.0)
    if version == 1:
        has_target = any(
            actions[i : i + 2] == target for i in range(max(0, len(actions) - 1))
        )
        prefix = actions[:3]
        has_forbidden_in_checked_prefix = any(
            prefix[i : i + 2] == forbidden for i in range(max(0, len(prefix) - 1))
        )
        if has_forbidden_in_checked_prefix:
            return 0.0
        partial = min(2, actions.count(target[0])) / 2.0
        return 0.75 * float(has_target) + 0.25 * partial
    raise ValueError(version)


def render_prompt(tok, task: dict) -> tuple[str, list[int]]:
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
    if digest(text.encode()) != task["prompt_sha256"] or ids != task["prompt_token_ids"]:
        raise RuntimeError(f"frozen prompt identity mismatch: {task['task_id']}")
    return text, ids


def load_snapshot(p: dict):
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
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(getattr(module, "attention_dropout", None), (int, float)):
            module.attention_dropout = 0.0
    encoded = [
        tok.encode(label, add_special_tokens=False) for label in p["action_labels"]
    ]
    if any(len(item) != 1 for item in encoded):
        raise ValueError("action label tokenizer contract failed")
    action_ids = [item[0] for item in encoded]
    eos_id = tok.eos_token_id
    support_ids = action_ids + [eos_id]
    mask = torch.full((model.config.vocab_size,), -1e9, dtype=torch.float32)
    mask[support_ids] = 0.0
    fixed = torch.zeros_like(mask)
    fixed[eos_id] = float(p["generation"]["termination_logit_offset"])
    return torch, tok, model, action_ids, eos_id, support_ids, mask, fixed


def checkpoint(root: Path, filename: str, expected_sha: str, findings: list[str]):
    import torch

    path = root / "models" / filename
    if not path.exists():
        findings.append(f"missing checkpoint {path}")
        return None, None
    raw = path.read_bytes()
    manifest_path = Path(str(path) + ".json")
    if not manifest_path.exists():
        findings.append(f"missing checkpoint sidecar {manifest_path}")
        return None, None
    manifest = load_json(manifest_path)
    actual = digest(raw)
    check(
        actual == expected_sha == manifest.get("sha256"),
        f"checkpoint hash mismatch {filename}",
        findings,
    )
    state = torch.load(
        io.BytesIO(gzip.decompress(raw)), map_location="cpu", weights_only=True
    )
    return manifest, state


def token_probabilities(
    torch,
    model,
    group: dict,
    state: dict,
    tok,
    p: dict,
    action_ids,
    eos_id,
    support_ids,
    support_mask,
    fixed_bias,
    findings: list[str],
    label: str,
):
    generations = group["generations"]
    prompt_ids = list(group["prompt_ids"])
    expected_text, expected_ids = render_prompt(tok, group["task"])
    check(
        expected_text == group["prompt"], f"prompt text mismatch in {label}", findings
    )
    check(expected_ids == prompt_ids, f"prompt token IDs mismatch in {label}", findings)
    check(
        group["policy_checkpoint_sha256"] == generations[0]["policy_checkpoint_sha256"],
        f"group/sample checkpoint mismatch in {label}",
        findings,
    )
    check(
        group["policy_version"] == generations[0]["policy_version"],
        f"group/sample policy version mismatch in {label}",
        findings,
    )
    check(
        group["verifier_version_at_rollout"]
        == generations[0]["verifier_version_at_rollout"],
        f"rollout verifier version mismatch in {label}",
        findings,
    )
    support = set(support_ids)
    rows = []
    maximum_length = 0
    for row in generations:
        response = [int(token) for token in row["response_token_ids"]]
        check(
            bool(response), f"empty response in {label}/{row['sample_index']}", findings
        )
        check(
            all(token in support for token in response),
            f"unsupported generated token in {label}/{row['sample_index']}",
            findings,
        )
        check(
            len(response) == len(row["response_token_logprobs"]),
            f"token/logprob length mismatch in {label}",
            findings,
        )
        action_tokens = response[:-1] if response[-1] == eos_id else response
        labels = {
            token: label
            for token, label in zip(action_ids, p["action_labels"], strict=True)
        }
        action_string = "".join(labels[token] for token in action_tokens)
        check(
            action_string == row["response"],
            f"decoded response mismatch in {label}/{row['sample_index']}",
            findings,
        )
        check(
            len(action_string) >= p["generation"]["minimum_actions"],
            f"minimum action count violated in {label}/{row['sample_index']}",
            findings,
        )
        check(
            len(action_string) <= p["generation"]["maximum_actions"],
            f"maximum action count violated in {label}/{row['sample_index']}",
            findings,
        )
        check(
            row["terminal_reason"]
            == ("eos" if response[-1] == eos_id else "maximum_actions"),
            f"terminal reason mismatch in {label}/{row['sample_index']}",
            findings,
        )
        rows.append((row, response))
        maximum_length = max(maximum_length, len(prompt_ids) + len(response))
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    input_ids = torch.full((len(rows), maximum_length), int(pad), dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for index, (_row, response) in enumerate(rows):
        seq = prompt_ids + response
        input_ids[index, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention[index, : len(seq)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    bias = state["bias"].float()
    recomputed = []
    first_probs = None
    temperature = float(p["generation"]["temperature"])
    for index, (row, response) in enumerate(rows):
        token_logps = []
        for token_index, token in enumerate(response):
            position = len(prompt_ids) - 1 + token_index
            z = logits[index, position] + bias + support_mask + fixed_bias
            if token_index < p["generation"]["minimum_actions"]:
                z = z.clone()
                z[eos_id] = -1e9
            log_probs = torch.log_softmax(z / temperature, dim=-1)
            token_logps.append(float(log_probs[int(token)].item()))
            if token_index == 0 and first_probs is None:
                first_probs = log_probs[support_ids].exp().tolist()
        expected_logps = [float(x) for x in row["response_token_logprobs"]]
        max_error = max(
            abs(a - b) for a, b in zip(token_logps, expected_logps, strict=True)
        )
        check(
            max_error <= p["audit"]["behavior_logprob_abs_tolerance"],
            f"behavior logprob mismatch {label}/{row['sample_index']} error={max_error}",
            findings,
        )
        check(
            abs(sum(token_logps) - float(row["behavior_logprob"]))
            <= p["audit"]["behavior_logprob_abs_tolerance"],
            f"behavior total logprob mismatch {label}/{row['sample_index']}",
            findings,
        )
        recomputed.append(
            {
                "sample_index": row["sample_index"],
                "response": row["response"],
                "token_count": len(response),
                "behavior_logprob_max_abs_error": max_error,
            }
        )
    if first_probs is not None and "first_step_support_probabilities" in group:
        errors = [
            abs(a - float(b))
            for a, b in zip(
                first_probs, group["first_step_support_probabilities"], strict=True
            )
        ]
        check(
            max(errors) <= p["audit"]["behavior_logprob_abs_tolerance"],
            f"initial behavior distribution mismatch {label}",
            findings,
        )
    return recomputed


def journal_audit(root: Path, findings: list[str]) -> list[dict]:
    path = root / "journal.jsonl"
    if not path.exists():
        findings.append(f"missing journal {root}")
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    previous = "0" * 64
    for index, row in enumerate(rows):
        payload = {key: value for key, value in row.items() if key != "sha256"}
        check(row.get("seq") == index, f"journal sequence mismatch in {root}", findings)
        check(
            row.get("previous") == previous,
            f"journal predecessor mismatch in {root}/{index}",
            findings,
        )
        check(
            row.get("sha256") == digest(canonical(payload)),
            f"journal hash mismatch in {root}/{index}",
            findings,
        )
        previous = row.get("sha256", "")
    return rows


def independent_update(
    torch,
    tok,
    model,
    p: dict,
    root: Path,
    group: dict,
    scores: dict,
    input_state: dict,
    actions,
    eos_id,
    support_ids,
    support_mask,
    fixed_bias,
    attempt: dict,
    findings: list[str],
    label: str,
) -> dict:
    import copy

    if hasattr(model, "vare_online_action_bias"):
        with torch.no_grad():
            model.vare_online_action_bias.copy_(input_state["bias"].detach().clone())
    else:
        model.register_parameter(
            "vare_online_action_bias",
            torch.nn.Parameter(input_state["bias"].detach().clone()),
        )
    for parameter in model.parameters():
        if parameter is not model.vare_online_action_bias:
            parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(getattr(module, "attention_dropout", None), (int, float)):
            module.attention_dropout = 0.0
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["training"]["learning_rate"])
    optimizer.load_state_dict(copy.deepcopy(input_state["optimizer"]))
    torch.set_rng_state(input_state["torch_rng"].cpu())
    optimizer.zero_grad(set_to_none=True)
    rewards = [float(value) for value in scores["rewards"]]
    mean = sum(rewards) / len(rewards)
    variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
    scale = math.sqrt(variance + p["training"]["advantage_epsilon"])
    advantages = [
        max(
            -p["training"]["clip_advantage"],
            min(p["training"]["clip_advantage"], (reward - mean) / scale),
        )
        for reward in rewards
    ]
    task = group["task"]
    _text, prompt_ids = render_prompt(tok, task)
    if len(set(rewards)) < 2:
        findings.append(f"constant reward group admitted for update {label}")
    for row, reward, advantage in zip(
        group["generations"], rewards, advantages, strict=True
    ):
        response = [int(token) for token in row["response_token_ids"]]
        sequence = torch.tensor([prompt_ids + response], dtype=torch.long)
        raw = model(input_ids=sequence).logits[0].float()
        start = len(prompt_ids) - 1
        z = (
            raw[start : start + len(response)]
            + model.vare_online_action_bias
            + support_mask
            + fixed_bias
        )
        z = z.clone()
        for token_index in range(
            min(p["generation"]["minimum_actions"], len(response))
        ):
            z[token_index, eos_id] = -1e9
        targets = torch.tensor(response, dtype=torch.long)
        current = (
            torch.log_softmax(z / float(p["generation"]["temperature"]), dim=-1)
            .gather(1, targets.unsqueeze(1))
            .squeeze(1)
        )
        old = torch.tensor(row["response_token_logprobs"], dtype=current.dtype)
        advantage_tensor = torch.full_like(current, float(advantage))
        raw_logratio = current - old
        logratio = torch.clamp(raw_logratio, min=-20.0, max=20.0)
        ratio = torch.exp(logratio)
        clipped = torch.clamp(
            ratio,
            1.0 - p["training"]["clip_epsilon"],
            1.0 + p["training"]["clip_epsilon"],
        )
        surrogate = torch.minimum(ratio * advantage_tensor, clipped * advantage_tensor)
        loss = -surrogate.mean() / len(rewards)
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), p["training"]["max_grad_norm"], error_if_nonfinite=True
    )
    gradient = model.vare_online_action_bias.grad.detach().clone()
    support_gradient = gradient[support_ids].tolist()
    expected_gradient = [
        float(value) for value in attempt["gradient_support_after_global_clip"]
    ]
    gradient_error = max(
        abs(a - b) for a, b in zip(support_gradient, expected_gradient, strict=True)
    )
    check(
        gradient_error <= p["audit"]["gradient_abs_tolerance"],
        f"gradient replay mismatch {label} error={gradient_error}",
        findings,
    )
    check(
        abs(float(grad_norm.item()) - float(attempt["gradient_l2_after_global_clip"]))
        <= p["audit"]["gradient_abs_tolerance"],
        f"gradient norm replay mismatch {label}",
        findings,
    )
    before = model.vare_online_action_bias.detach().clone()
    bias_before = before[support_ids].tolist()
    check(
        gradient_origin_matches(
            before,
            attempt,
            support_ids,
            p["audit"]["state_abs_tolerance"],
        ),
        f"pre-update bias mismatch {label}",
        findings,
    )
    optimizer.step()
    after = model.vare_online_action_bias.detach().clone()
    check(
        max(
            abs(float(a) - float(b))
            for a, b in zip(
                after[support_ids].tolist(), attempt["bias_after_support"], strict=True
            )
        )
        <= p["audit"]["state_abs_tolerance"],
        f"post-update bias mismatch {label}",
        findings,
    )
    if attempt.get("candidate_checkpoint"):
        _manifest, candidate_state = checkpoint(
            root,
            attempt["candidate_checkpoint"],
            attempt["candidate_checkpoint_sha256"],
            findings,
        )
        if candidate_state is not None:
            check(
                torch.allclose(
                    after,
                    candidate_state["bias"],
                    atol=p["audit"]["state_abs_tolerance"],
                    rtol=p["audit"]["state_relative_tolerance"],
                ),
                f"candidate policy replay mismatch {label}",
                findings,
            )
            compare_optimizer(
                torch,
                optimizer.state_dict(),
                candidate_state["optimizer"],
                p,
                findings,
                label,
            )
            check(
                torch.equal(torch.get_rng_state(), candidate_state["torch_rng"]),
                f"candidate RNG mismatch {label}",
                findings,
            )
    return {
        "gradient_max_abs_error": gradient_error,
        "gradient_norm": float(grad_norm.item()),
        "bias_support_before": bias_before,
        "bias_support_after": after[support_ids].tolist(),
    }


def compare_optimizer(
    torch, expected: dict, actual: dict, p: dict, findings: list[str], label: str
) -> None:
    def visit(left, right, path="optimizer"):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if left.dtype.is_floating_point:
                if not torch.allclose(
                    left.cpu(),
                    right.cpu(),
                    atol=p["audit"]["state_abs_tolerance"],
                    rtol=p["audit"]["state_relative_tolerance"],
                ):
                    findings.append(f"{label}: {path} tensor mismatch")
            elif not torch.equal(left.cpu(), right.cpu()):
                findings.append(f"{label}: {path} integer tensor mismatch")
        elif isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                findings.append(f"{label}: {path} key mismatch")
                return
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                findings.append(f"{label}: {path} length mismatch")
                return
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, f"{path}[{index}]")
        elif left != right:
            findings.append(f"{label}: {path} value mismatch")

    visit(expected, actual)


def audit_case(
    p,
    root: Path,
    torch,
    tok,
    model,
    action_ids,
    eos_id,
    support_ids,
    support_mask,
    fixed_bias,
    findings: list[str],
) -> dict:
    case = root.name
    case_record = load_json(root / "case-result.json")
    rows = journal_audit(root, findings)
    active = load_json(root / "active.json")
    active_manifest, final_state = checkpoint(
        root, active["checkpoint"], active["checkpoint_sha256"], findings
    )
    check(
        active["policy_version"] == 2,
        f"{case}: final policy version is not 2",
        findings,
    )
    check(
        final_state is not None
        and active_manifest["metadata"]["policy_version"] == active["policy_version"],
        f"{case}: active pointer does not bind checkpoint version",
        findings,
    )
    model_files = sorted((root / "models").glob("*.pt.gz"))
    checkpoint_by_sha = {}
    for model_path in model_files:
        manifest_path = Path(str(model_path) + ".json")
        if manifest_path.exists():
            metadata = load_json(manifest_path)
            checkpoint_by_sha[metadata["sha256"]] = model_path

    groups = []
    trusted_rows = []
    token_counts = []
    for group_path in sorted((root / "groups").glob("*.json")):
        group_sha = file_digest(group_path)
        sidecar = load_json(Path(str(group_path) + ".sha256"))
        check(
            group_sha == sidecar["sha256"],
            f"{case}: rollout sha sidecar mismatch {group_path.name}",
            findings,
        )
        ack_path = root / "groups" / f"{group_path.stem}.ack.json"
        check(ack_path.exists(), f"{case}: missing rollout acknowledgement {ack_path.name}", findings)
        if ack_path.exists():
            check(
                load_json(ack_path).get("group_sha256") == group_sha,
                f"{case}: rollout acknowledgement hash mismatch {ack_path.name}",
                findings,
            )
        group = load_json(group_path)
        manifest, state = checkpoint(
            root,
            group["policy_checkpoint"],
            group["policy_checkpoint_sha256"],
            findings,
        )
        if state is not None:
            check(
                manifest["metadata"]["policy_version"] == group["policy_version"],
                f"{case}/{group['group_id']}: worker policy version mismatch",
                findings,
            )
            group_checks = token_probabilities(
                torch,
                model,
                group,
                state,
                tok,
                p,
                action_ids,
                eos_id,
                support_ids,
                support_mask,
                fixed_bias,
                findings,
                f"{case}/{group['group_id']}",
            )
        else:
            group_checks = []
        for row in group["generations"]:
            truth = trusted(row["response"], group["task"])
            token_counts.append(row["generated_token_count"])
            trusted_rows.append(
                {
                    "task_id": group["task_id"],
                    "group_id": group["group_id"],
                    "response": row["response"],
                    "trusted_reward": truth,
                }
            )
        groups.append(
            {
                "group_id": group["group_id"],
                "task_id": group["task_id"],
                "policy_version": group["policy_version"],
                "policy_checkpoint_sha256": group["policy_checkpoint_sha256"],
                "verifier_version_at_rollout": group["verifier_version_at_rollout"],
                "sample_count": len(group["generations"]),
                "token_checks": group_checks,
            }
        )
        for score_path in sorted(
            (root / "scores").glob(f"{group['group_id']}-v*.json")
        ):
            score = load_json(score_path)
            version = int(score["verifier_version"])
            recomputed = [
                proxy(row["response"], group["task"], version)
                for row in group["generations"]
            ]
            check(
                score["group_sha256"] == group_sha,
                f"{case}: score references wrong rollout {score_path.name}",
                findings,
            )
            check(
                score["verifier_config_sha256"]
                == digest(canonical(p["verifiers"][str(version)])),
                f"{case}: verifier config hash mismatch {score_path.name}",
                findings,
            )
            check(
                recomputed == score["rewards"],
                f"{case}: proxy reward replay mismatch {score_path.name}",
                findings,
            )
            ack_path = root / "scores" / f"{group['group_id']}-v{version}.ack.json"
            check(
                ack_path.exists(),
                f"{case}: missing score acknowledgement {ack_path.name}",
                findings,
            )
            if ack_path.exists():
                check(
                    load_json(ack_path).get("score_sha256") == file_digest(score_path),
                    f"{case}: score acknowledgement hash mismatch {ack_path.name}",
                    findings,
                )
            if score.get("regraded_from"):
                source = root / "scores" / f"{group['group_id']}-v0.json"
                check(
                    source.exists() and file_digest(source) == score["regraded_from"],
                    f"{case}: regrade does not bind original v0 receipt",
                    findings,
                )

    admissions = {
        path.stem: load_json(path) for path in (root / "admissions").glob("*.json")
    }
    for group_id, admission in admissions.items():
        group = load_json(root / "groups" / f"{group_id}.json")
        score_path = (
            root / "scores" / f"{group_id}-v{admission['reward_verifier_version']}.json"
        )
        score = load_json(score_path)
        policy_lag = admission["active_policy_version"] - group["policy_version"]
        verifier_lag = admission["active_verifier_version"] - score["verifier_version"]
        check(
            admission["policy_lag"] == policy_lag,
            f"{case}/{group_id}: policy lag was not recomputed",
            findings,
        )
        check(
            admission["verifier_lag"] == verifier_lag,
            f"{case}/{group_id}: verifier lag was not recomputed",
            findings,
        )
        check(
            admission["group_sha256"]
            == file_digest(root / "groups" / f"{group_id}.json"),
            f"{case}/{group_id}: admission hash mismatch",
            findings,
        )
        check(
            admission["score_sha256"] == file_digest(score_path),
            f"{case}/{group_id}: admission reward hash mismatch",
            findings,
        )
        if case_record["arm"] == "B" and group_id.endswith("group-1"):
            check(
                admission["naive_staleness_bypass"]
                and not admission["verifier_only_admitted"],
                f"{case}: naive arm did not retain the stale-verifier counterexample",
                findings,
            )
        if case_record["arm"] == "C" and group_id.endswith("group-1"):
            check(
                not admission["combined_admitted"]
                and not admission["policy_only_admitted"]
                and not admission["verifier_only_admitted"],
                f"{case}: strict rejection did not independently detect both lags",
                findings,
            )
        if case_record["arm"] in ("D", "E") and group_id.endswith("group-1"):
            check(
                admission["combined_admitted"]
                and admission["policy_lag"] == 1
                and admission["verifier_lag"] == 0,
                f"{case}: regraded admission did not isolate policy lag from verifier freshness",
                findings,
            )

    commits = sorted(
        (root / "transactions").glob("*/commit.json"),
        key=lambda path: load_json(path).get("policy_version", 0),
    )
    check(
        len(commits) == 2,
        f"{case}: expected two effective update commits, found {len(commits)}",
        findings,
    )
    if commits:
        latest_commit = load_json(commits[-1])
        check(
            active_matches_latest_commit(active, latest_commit),
            f"{case}: active pointer differs from latest commit receipt",
            findings,
        )
    update_checks = []
    optimizer_computations = 0
    queue_latencies = []
    for commit_path in commits:
        commit = load_json(commit_path)
        tx_dir = commit_path.parent
        attempt_paths = sorted(tx_dir.glob("attempt-*.json"))
        check(bool(attempt_paths), f"{case}: commit has no optimizer attempt", findings)
        tx_attempts = []
        for attempt_path in attempt_paths:
            attempt = load_json(attempt_path)
            optimizer_computations += int(
                attempt.get("completed_optimizer_computations", 0)
            )
            group = load_json(root / "groups" / f"{attempt['group_id']}.json")
            group_sha = file_digest(root / "groups" / f"{attempt['group_id']}.json")
            score_path = next(
                (
                    path
                    for path in (root / "scores").glob(f"{attempt['group_id']}-v*.json")
                    if file_digest(path) == attempt["score_sha256"]
                ),
                None,
            )
            check(
                score_path is not None,
                f"{case}: attempt reward file missing by hash",
                findings,
            )
            input_path = checkpoint_by_sha.get(attempt["input_checkpoint_sha256"])
            check(
                input_path is not None,
                f"{case}: update input checkpoint missing",
                findings,
            )
            if score_path is not None and input_path is not None:
                _manifest, input_state = checkpoint(
                    root, input_path.name, attempt["input_checkpoint_sha256"], findings
                )
                score = load_json(score_path)
                if input_state is not None:
                    independent = independent_update(
                        torch,
                        tok,
                        model,
                        p,
                        root,
                        group,
                        score,
                        input_state,
                        action_ids,
                        eos_id,
                        support_ids,
                        support_mask,
                        fixed_bias,
                        attempt,
                        findings,
                        f"{case}/{tx_dir.name}/{attempt_path.name}",
                    )
                    tx_attempts.append(independent)
            check(
                attempt["group_sha256"] == group_sha,
                f"{case}: attempt group hash mismatch",
                findings,
            )
        selected_attempt = next(
            (
                load_json(path)
                for path in attempt_paths
                if load_json(path).get("attempt") == commit.get("attempt")
            ),
            None,
        )
        if selected_attempt and selected_attempt.get("candidate_checkpoint"):
            check(
                selected_attempt["candidate_checkpoint_sha256"]
                == commit["checkpoint_sha256"],
                f"{case}: commit does not bind its optimizer candidate",
                findings,
            )
            candidate_path = root / "models" / selected_attempt["candidate_checkpoint"]
            if candidate_path.exists():
                candidate_manifest = load_json(Path(str(candidate_path) + ".json"))
                check(
                    candidate_manifest["metadata"]["transaction_id"]
                    == commit["transaction_id"],
                    f"{case}: checkpoint transaction identity mismatch",
                    findings,
                )
                check(
                    candidate_manifest["metadata"]["policy_version"]
                    == commit["policy_version"],
                    f"{case}: checkpoint version mismatch",
                    findings,
                )
        commit_time = commit.get("committed_unix_ns", commit.get("recovered_unix_ns"))
        group_id = commit.get("group_id") or (selected_attempt or {}).get("group_id")
        if commit_time and group_id:
            rollout = load_json(root / "groups" / f"{group_id}.json")
            queue_latencies.append(
                (int(commit_time) - int(rollout["rollout_started_unix_ns"])) / 1e9
            )
        check(
            commit["effective_update"] is True,
            f"{case}: non-effective commit counted",
            findings,
        )
        update_checks.append(
            {
                "transaction_id": commit["transaction_id"],
                "policy_version": commit["policy_version"],
                "attempts": len(attempt_paths),
                "replay_checks": tx_attempts,
            }
        )

    process_records = load_json(root / "processes.json")
    killed = {
        int(row["pid"]): row for row in process_records if row.get("returncode") == -9
    }
    recovery_latencies = []
    for event in rows:
        if event.get("kind") != "fault_injected":
            continue
        pid = int(event["pid"])
        check(
            pid in killed,
            f"{case}: SIGKILL event has no matching child exit status",
            findings,
        )
        killed_row = killed.get(pid)
        if killed_row:
            retried = [
                item
                for item in process_records
                if item["name"] == killed_row["name"]
                and item["attempt"] > killed_row["attempt"]
            ]
            check(bool(retried), f"{case}: faulted child was not restarted", findings)
            if retried:
                check(
                    int(retried[0]["pid"]) != pid,
                    f"{case}: recovery reused the killed process identity",
                    findings,
                )
                recovery_latencies.append(
                    (
                        int(retried[0]["started_unix_ns"])
                        - int(killed_row["ended_unix_ns"])
                    )
                    / 1e9
                )

    result = load_json(root / "evaluation.json")
    check(
        file_digest(root / "evaluation.json")
        == load_json(root / "evaluation.json.sha256")["sha256"],
        f"{case}: evaluation file hash mismatch",
        findings,
    )
    check(
        result["policy_version"] == active["policy_version"]
        and result["policy_checkpoint_sha256"] == active["checkpoint_sha256"],
        f"{case}: evaluation did not use active final policy",
        findings,
    )
    evaluation_by_task = {}
    for eval_task in result["tasks"]:
        eval_group = {
            "task": eval_task["task"],
            "task_id": eval_task["task_id"],
            "prompt": eval_task["prompt"],
            "prompt_ids": eval_task["prompt_ids"],
            "policy_version": result["policy_version"],
            "policy_checkpoint_sha256": result["policy_checkpoint_sha256"],
            "verifier_version_at_rollout": result["verifier_version"],
            "generations": eval_task["generations"],
            "first_step_support_probabilities": eval_task[
                "first_step_support_probabilities"
            ],
        }
        _manifest, eval_state = checkpoint(
            root,
            eval_task["policy_checkpoint"],
            eval_task["policy_checkpoint_sha256"],
            findings,
        )
        if eval_state is not None:
            token_probabilities(
                torch,
                model,
                eval_group,
                eval_state,
                tok,
                p,
                action_ids,
                eos_id,
                support_ids,
                support_mask,
                fixed_bias,
                findings,
                f"{case}/eval/{eval_task['task_id']}",
            )
        values = [
            trusted(row["response"], eval_task["task"])
            for row in eval_task["generations"]
        ]
        evaluation_by_task[eval_task["task_id"]] = {
            "mean_trusted_reward": sum(values) / len(values),
            "n": len(values),
            "positive": sum(values),
        }
        for row, value in zip(eval_task["generations"], values, strict=True):
            trusted_rows.append(
                {
                    "task_id": eval_task["task_id"],
                    "group_id": "independent_eval",
                    "response": row["response"],
                    "trusted_reward": value,
                }
            )
            token_counts.append(row["generated_token_count"])
    probe_path = next((root / "groups").glob("*-fresh-policy-probe.json"), None)
    refresh_metrics = {"total_variation": None, "kl_before_to_after": None}
    if probe_path is None:
        findings.append(f"{case}: missing post-update actor probe")
    else:
        probe = load_json(probe_path)
        check(
            probe["policy_version"] == 2
            and probe["policy_checkpoint_sha256"] == active["checkpoint_sha256"],
            f"{case}: fresh actor probe not produced by final policy",
            findings,
        )
        prior_group_path = root / "groups" / f"{case}-group-1.json"
        if prior_group_path.exists():
            prior_group = load_json(prior_group_path)
            refresh_metrics = distribution_metrics(
                prior_group["first_step_support_probabilities"],
                probe["first_step_support_probabilities"],
            )
            check(
                refresh_metrics["total_variation"]
                > p["audit"]["minimum_refresh_tv"],
                f"{case}: updated actor distribution did not measurably change",
                findings,
            )
        else:
            refresh_metrics = {
                "total_variation": None,
                "kl_before_to_after": None,
            }
    if final_state is not None:
        init_path = root / "models" / "policy-0.pt.gz"
        _init_manifest, init_state = checkpoint(
            root, init_path.name, file_digest(init_path), findings
        )
        if init_state is not None:
            bias_delta = float(
                torch.linalg.vector_norm(
                    final_state["bias"] - init_state["bias"]
                ).item()
            )
            check(
                bias_delta > 0.0,
                f"{case}: final policy adapter did not change",
                findings,
            )
        else:
            bias_delta = math.nan
    else:
        bias_delta = math.nan
    event_kinds = [
        row.get("fault") for row in rows if row.get("kind") == "fault_injected"
    ]
    if "checkpoint_before_publication" in event_kinds:
        fault_event = next(
            row
            for row in rows
            if row.get("kind") == "fault_injected"
            and row.get("fault") == "checkpoint_before_publication"
        )
        fault_tx_dir = root / "transactions" / fault_event["transaction_id"]
        fault_attempts = [
            load_json(path) for path in fault_tx_dir.glob("attempt-*.json")
        ]
        committed_hashes = {load_json(path)["checkpoint_sha256"] for path in commits}
        orphan_candidates = [
            attempt["candidate_checkpoint_sha256"]
            for attempt in fault_attempts
            if attempt.get("candidate_checkpoint_sha256")
            and attempt["candidate_checkpoint_sha256"] not in committed_hashes
        ]
        check(
            bool(orphan_candidates),
            f"{case}: pre-publication fault failed to retain its orphan candidate",
            findings,
        )
    trusted_eval_mean = statistics.mean(
        item["mean_trusted_reward"] for item in evaluation_by_task.values()
    )
    check(
        evaluation_aggregation_matches(evaluation_by_task, trusted_eval_mean),
        f"{case}: evaluation aggregate differs from task-level results",
        findings,
    )
    case_summary = {
        "case_id": case,
        "seed": case_record["seed"],
        "arm": case_record["arm"],
        "effective_updates": len(commits),
        "optimizer_computations_including_retries": optimizer_computations,
        "final_policy_version": active["policy_version"],
        "adapter_l2_from_initial": bias_delta,
        "groups": groups,
        "trusted_eval_by_task": evaluation_by_task,
        "trusted_eval_mean_over_task_means": trusted_eval_mean,
        "actor_refresh_distribution_change": refresh_metrics,
        "update_replay": update_checks,
        "generated_token_count_histogram": {
            str(count): token_counts.count(count) for count in sorted(set(token_counts))
        },
        "fresh_actor_probe": str(probe_path.name) if probe_path else None,
        "faults": event_kinds,
        "queue_inclusive_update_seconds": queue_latencies,
        "sigkill_recovery_seconds": recovery_latencies,
        "proxy_vs_trusted_training": {
            f"v{version}": {
                "n": sum(
                    len(load_json(path)["rewards"])
                    for path in (root / "scores").glob(f"*-v{version}.json")
                ),
                "false_positive": sum(
                    score_value >= 0.5
                    and trusted(
                        row["response"],
                        load_json(
                            root
                            / "groups"
                            / f"{load_json(score_path)['group_id']}.json"
                        )["task"],
                    )
                    == 0
                    for score_path in (root / "scores").glob(f"*-v{version}.json")
                    for score_value, row in zip(
                        load_json(score_path)["rewards"],
                        load_json(
                            root
                            / "groups"
                            / f"{load_json(score_path)['group_id']}.json"
                        )["generations"],
                        strict=True,
                    )
                ),
                "false_negative": sum(
                    score_value < 0.5
                    and trusted(
                        row["response"],
                        load_json(
                            root
                            / "groups"
                            / f"{load_json(score_path)['group_id']}.json"
                        )["task"],
                    )
                    == 1
                    for score_path in (root / "scores").glob(f"*-v{version}.json")
                    for score_value, row in zip(
                        load_json(score_path)["rewards"],
                        load_json(
                            root
                            / "groups"
                            / f"{load_json(score_path)['group_id']}.json"
                        )["generations"],
                        strict=True,
                    )
                ),
            }
            for version in (0, 1)
        },
        "trusted_training_labels": trusted_rows,
    }
    return case_summary


def audit_eval_file(
    p: dict,
    root: Path,
    torch,
    tok,
    model,
    action_ids,
    eos_id,
    support_ids,
    support_mask,
    fixed_bias,
    findings: list[str],
) -> dict:
    path = root / "evaluation.json"
    if not path.exists():
        findings.append(f"missing independent evaluation file {root}")
        return {}
    sidecar_path = Path(str(path) + ".sha256")
    check(
        sidecar_path.exists()
        and file_digest(path) == load_json(sidecar_path).get("sha256"),
        f"evaluation hash mismatch {root}",
        findings,
    )
    result = load_json(path)
    active = load_json(root / "active.json")
    check(
        result["policy_version"] == active["policy_version"],
        f"evaluation active version mismatch {root}",
        findings,
    )
    check(
        result["policy_checkpoint_sha256"] == active["checkpoint_sha256"],
        f"evaluation active checkpoint mismatch {root}",
        findings,
    )
    output = {}
    for row in result["tasks"]:
        group = {
            "task": row["task"],
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "prompt_ids": row["prompt_ids"],
            "policy_version": result["policy_version"],
            "policy_checkpoint_sha256": result["policy_checkpoint_sha256"],
            "verifier_version_at_rollout": result["verifier_version"],
            "generations": row["generations"],
            "first_step_support_probabilities": row["first_step_support_probabilities"],
        }
        _manifest, state = checkpoint(
            root, row["policy_checkpoint"], row["policy_checkpoint_sha256"], findings
        )
        if state is not None:
            token_probabilities(
                torch,
                model,
                group,
                state,
                tok,
                p,
                action_ids,
                eos_id,
                support_ids,
                support_mask,
                fixed_bias,
                findings,
                f"{root.name}/eval/{row['task_id']}",
            )
        rewards = [
            trusted(item["response"], row["task"]) for item in row["generations"]
        ]
        output[row["task_id"]] = {
            "n": len(rewards),
            "positive": sum(rewards),
            "mean_trusted_reward": sum(rewards) / len(rewards),
        }
    return output


def calibrate(p: dict) -> dict:
    from itertools import product

    validate_protocol_contract(p)
    rows = []
    for task in p["tasks"]:
        if task["split"] not in ("train", "evaluation"):
            continue
        for length in range(2, p["generation"]["maximum_actions"] + 1):
            for labels in product(p["action_labels"], repeat=length):
                actions = "".join(labels)
                truth = trusted(actions, task)
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "actions": actions,
                        "truth": truth,
                        "v0": proxy(actions, task, 0),
                        "v1": proxy(actions, task, 1),
                    }
                )
    result = {}
    for version in (0, 1):
        fp = sum(row[f"v{version}"] >= 0.5 and row["truth"] == 0 for row in rows)
        fn = sum(row[f"v{version}"] < 0.5 and row["truth"] == 1 for row in rows)
        mae = sum(abs(row[f"v{version}"] - row["truth"]) for row in rows) / len(rows)
        result[f"v{version}"] = {
            "n": len(rows),
            "false_positive": fp,
            "false_negative": fn,
            "false_positive_rate": fp / len(rows),
            "false_negative_rate": fn / len(rows),
            "disagreement_rate": (fp + fn) / len(rows),
            "mean_absolute_reward_error": mae,
        }
    result["v0_v1_disagreement"] = sum(
        (row["v0"] >= 0.5) != (row["v1"] >= 0.5) for row in rows
    ) / len(rows)
    result["enumerated_world_sha256"] = digest(canonical(rows))
    return result


def audit(p_path: Path, root: Path) -> dict:
    p = load_json(p_path)
    validate_protocol_contract(p)
    sidecar = load_json(Path(str(p_path) + ".sha256"))
    findings: list[str] = []
    check(
        file_digest(p_path) == sidecar["sha256"], "protocol sidecar mismatch", findings
    )
    weight_path = Path(p["model"]["weight_file"])
    check(
        file_digest(weight_path) == p["model"]["weight_sha256"],
        "base model weight hash mismatch",
        findings,
    )
    for rel, expected in p["source_hashes"].items():
        check(
            file_digest(Path(p["repository_root"]) / rel) == expected,
            f"frozen source changed: {rel}",
            findings,
        )
    for rel, expected in p["model"]["snapshot_files"].items():
        check(
            file_digest(Path(p["model"]["snapshot_path"]) / rel) == expected,
            f"cached model snapshot changed: {rel}",
            findings,
        )
    for rel, expected in p["rvl_source_hashes"].items():
        check(
            file_digest(Path(p["rvl_source_root"]) / rel) == expected,
            f"frozen RVL source changed: {rel}",
            findings,
        )
    if findings:
        return {
            "schema_version": 1,
            "audit_version": "online-actor-independent-audit-v1",
            "status": "FAIL",
            "findings": findings,
            "model": p["model"],
            "completed_unix_ns": time.time_ns(),
        }
    torch, tok, model, action_ids, eos_id, support_ids, support_mask, fixed_bias = (
        load_snapshot(p)
    )
    actual_ids = [
        tok.encode(label, add_special_tokens=False)[0] for label in p["action_labels"]
    ]
    check(actual_ids == action_ids, "action IDs changed", findings)
    check(eos_id == tok.eos_token_id, "EOS token changed", findings)
    run_status_path = root / "run-status.json"
    check(run_status_path.exists(), "missing run status receipt", findings)
    if run_status_path.exists():
        check(
            load_json(run_status_path).get("status") == "MODEL_RUNS_COMPLETE_AUDIT_PENDING",
            "campaign did not reach complete model-run status",
            findings,
        )
    resource_path = root / "resource-summary.json"
    check(resource_path.exists(), "missing resource summary", findings)
    if resource_path.exists():
        resource = load_json(resource_path)
        check(
            resource["peak_child_rss_bytes"] <= p["limits"]["max_child_rss_bytes"]
            and resource["minimum_host_available_bytes"]
            >= p["limits"]["hard_min_available_bytes"]
            and resource["maximum_swap_growth_bytes"]
            <= p["limits"]["max_swap_growth_bytes"]
            and resource["wall_seconds"] <= p["limits"]["wall_seconds"]
            and resource["total_generated_tokens"]
            <= p["limits"]["max_total_generated_tokens"]
            and resource["total_model_tokens"]
            <= p["limits"]["max_total_model_tokens"],
            "campaign resource summary exceeds a frozen cap",
            findings,
        )
    for jobs in ("rollout", "learner"):
        check(
            p["limits"][f"max_{jobs}_concurrency"] == 1,
            f"protocol relaxed {jobs} process concurrency",
            findings,
        )
    if p.get("rvl_source_root") and p.get("rvl_commit"):
        actual_rvl = subprocess.run(
            ["git", "-C", p["rvl_source_root"], "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        check(
            actual_rvl.returncode == 0
            and actual_rvl.stdout.strip() == p["rvl_commit"],
            "RVL source revision changed",
            findings,
        )
    if (root / "protocol-v1.json").exists():
        check(
            file_digest(root / "protocol-v1.json")
            == load_json(root / "protocol-v1.json.sha256")["sha256"],
            "run copy protocol mismatch",
            findings,
        )
    expected_calibration = calibrate(p)
    actual_calibration = load_json(root / "verifier-calibration.json")
    check(
        expected_calibration
        == {key: actual_calibration.get(key) for key in expected_calibration},
        "verifier calibration mismatch",
        findings,
    )
    baseline_eval = {}
    for seed in p["seeds"]:
        baseline_eval[str(seed)] = audit_eval_file(
            p,
            root / "baseline" / f"seed-{seed}",
            torch,
            tok,
            model,
            action_ids,
            eos_id,
            support_ids,
            support_mask,
            fixed_bias,
            findings,
        )
    cases = []
    for case_root in sorted((root / "cases").iterdir()):
        if case_root.is_dir() and (case_root / "case-result.json").exists():
            cases.append(
                audit_case(
                    p,
                    case_root,
                    torch,
                    tok,
                    model,
                    action_ids,
                    eos_id,
                    support_ids,
                    support_mask,
                    fixed_bias,
                    findings,
                )
            )
    paired_deltas = []
    for row in cases:
        baseline = baseline_eval.get(str(row["seed"]), {})
        by_task = {
            task_id: row["trusted_eval_by_task"][task_id]["mean_trusted_reward"]
            - baseline[task_id]["mean_trusted_reward"]
            for task_id in p["evaluation"]["task_ids"]
            if task_id in row["trusted_eval_by_task"] and task_id in baseline
        }
        paired_deltas.append(
            {
                "case_id": row["case_id"],
                "per_task": by_task,
                "mean_over_task_units": statistics.mean(by_task.values())
                if by_task
                else None,
            }
        )
    all_faults = {fault for item in cases for fault in item["faults"]}
    required_faults = set(p["fault_plan"]["required_faults"])
    check(
        required_faults.issubset(all_faults),
        f"missing fault cutpoints: {sorted(required_faults - all_faults)}",
        findings,
    )
    for seed in p["seeds"]:
        d = next(
            (row for row in cases if row["seed"] == seed and row["arm"] == "D"), None
        )
        e = next(
            (row for row in cases if row["seed"] == seed and row["arm"] == "E"), None
        )
        if d is None or e is None:
            findings.append(f"missing D/E paired case for seed {seed}")
            continue
        d_root = root / "cases" / d["case_id"]
        e_root = root / "cases" / e["case_id"]
        d_active = load_json(d_root / "active.json")
        e_active = load_json(e_root / "active.json")
        _dm, dstate = checkpoint(
            d_root, d_active["checkpoint"], d_active["checkpoint_sha256"], findings
        )
        _em, estate = checkpoint(
            e_root, e_active["checkpoint"], e_active["checkpoint_sha256"], findings
        )
        if dstate is not None and estate is not None:
            check(
                torch.equal(dstate["bias"], estate["bias"]),
                f"seed {seed}: crash recovery adapter differs from no-crash regrade arm",
                findings,
            )
            compare_optimizer(
                torch,
                dstate["optimizer"],
                estate["optimizer"],
                p,
                findings,
                f"seed-{seed}-D-vs-E",
            )
            check(
                torch.equal(dstate["torch_rng"], estate["torch_rng"]),
                f"seed {seed}: crash recovery RNG differs from D arm",
                findings,
            )
    status = "PASS" if not findings else "FAIL"
    return {
        "schema_version": 1,
        "audit_version": "online-actor-independent-audit-v1",
        "status": status,
        "findings": findings,
        "model": p["model"],
        "verifier_calibration": expected_calibration,
        "baseline_trusted_eval": baseline_eval,
        "paired_task_deltas": paired_deltas,
        "cases": cases,
        "fault_cutpoints_seen": sorted(all_faults),
        "effective_update_count": sum(row["effective_updates"] for row in cases),
        "optimizer_computation_count_including_retries": sum(
            row["optimizer_computations_including_retries"] for row in cases
        ),
        "independent_task_units": len(p["evaluation"]["task_ids"]),
        "completed_unix_ns": time.time_ns(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    args = parser.parse_args()
    if args.calibration_only:
        p = load_json(args.protocol.resolve())
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(calibrate(p), sort_keys=True, indent=2, ensure_ascii=False)
            + "\n"
        )
        print(json.dumps({"status": "CALIBRATION_COMPLETE", "path": str(args.output)}))
        return
    if args.receipt is None:
        parser.error("--receipt is required unless --calibration-only is set")
    report = audit(args.protocol.resolve(), args.output.resolve())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    args.receipt.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "findings": report["findings"],
                "cases": len(report["cases"]),
                "effective_updates": report["effective_update_count"],
                "optimizer_computations": report[
                    "optimizer_computation_count_including_retries"
                ],
            },
            ensure_ascii=False,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
