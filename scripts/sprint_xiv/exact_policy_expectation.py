#!/usr/bin/env python3
"""Retrospectively enumerate the finite Sprint XIII synthetic route policy.

This program performs deterministic model forward passes only. It never samples
or generates a token, modifies an optimizer, or writes into historical runs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "protocols/sprint_xiv_exact_policy_expectation_v1.json"
DEFAULT_RUN = ROOT / "results/sprint_xiii_20261029/run-003"
DEFAULT_OUTPUT = ROOT / "results/sprint_xiv_exact_policy_expectation_v1"
sys.path.insert(0, str(ROOT))
from scripts.sprint_xiii.online_actor import prompt_text, verifier_score  # noqa: E402


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def frozen_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol = json.loads(raw)
    sidecar = read_json(Path(str(path) + ".sha256"))
    sha = digest_bytes(raw)
    if protocol.get("status") != "frozen_retrospective_analysis":
        raise ValueError("protocol is not frozen as a retrospective analysis")
    if sidecar.get("sha256") != sha:
        raise ValueError("analysis protocol sidecar mismatch")
    if protocol.get("retrospective") is not True:
        raise ValueError("analysis must remain explicitly retrospective")
    return protocol, sha


def verify_inputs(protocol: dict[str, Any], run_root: Path) -> dict[str, Any]:
    campaign_path = ROOT / protocol["source"]["campaign_protocol_path"]
    campaign_raw = campaign_path.read_bytes()
    if digest_bytes(campaign_raw) != protocol["source"]["campaign_protocol_sha256"]:
        raise ValueError("Sprint XIII campaign protocol changed")
    campaign = json.loads(campaign_raw)
    archive = ROOT / protocol["source"]["run_archive_path"]
    archive_sha = digest_file(archive)
    if archive_sha != protocol["source"]["run_archive_sha256"]:
        raise ValueError("Sprint XIII evidence archive hash mismatch")
    archive_sidecar = read_json(Path(str(archive) + ".sha256"))
    if archive_sidecar.get("sha256") != archive_sha:
        raise ValueError("Sprint XIII evidence archive sidecar mismatch")
    manifest_path = run_root / "sha256-manifest-v1.json"
    manifest_raw = manifest_path.read_bytes()
    manifest_sha = digest_bytes(manifest_raw)
    if manifest_sha != protocol["source"]["run_manifest_sha256"]:
        raise ValueError("Sprint XIII run manifest hash mismatch")
    manifest = json.loads(manifest_raw)
    if manifest.get("campaign_protocol_sha256") != protocol["source"]["campaign_protocol_sha256"]:
        raise ValueError("run manifest points to a different campaign protocol")
    manifest_sidecar = read_json(Path(str(manifest_path) + ".sha256"))
    if manifest_sidecar.get("sha256") != manifest_sha:
        raise ValueError("run manifest sidecar mismatch")

    expected_paths: set[str] = set()
    for row in manifest["files"]:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe path in historical manifest: {rel}")
        expected_paths.add(rel.as_posix())
        path = run_root / rel
        if not path.is_file():
            raise ValueError(f"missing historical input {rel}")
        if path.stat().st_size != row["bytes"] or digest_file(path) != row["sha256"]:
            raise ValueError(f"historical input hash mismatch: {rel}")
    actual_paths = {
        p.relative_to(run_root).as_posix()
        for p in run_root.rglob("*")
        if p.is_file() and p.name != "sha256-manifest-v1.json.sha256"
    }
    if actual_paths != expected_paths | {"sha256-manifest-v1.json"}:
        raise ValueError("historical run has missing or unexpected files")

    receipt = run_root / "independent-audit-v5.json"
    if digest_file(receipt) != protocol["source"]["independent_audit_v5_sha256"]:
        raise ValueError("independent audit v5 receipt hash mismatch")
    audit = read_json(receipt)
    if audit.get("status") != "PASS_COMPLETED_SUBSET" or audit.get("findings"):
        raise ValueError("historical completed-subset audit was not a clean PASS")
    if len(audit.get("completed_case_ids", [])) != 13 or audit.get("effective_update_count") != 26:
        raise ValueError("historical audit population differs from frozen analysis scope")
    if set(audit.get("incomplete_case_ids", [])) != {"seed-909-B", "seed-909-C"}:
        raise ValueError("historical incomplete cases differ from the frozen analysis scope")
    if digest_bytes(campaign_raw) != audit.get("protocol_sha256"):
        raise ValueError("audit receipt does not bind the frozen campaign protocol")
    return {
        "archive_sha256": archive_sha,
        "run_manifest_sha256": manifest_sha,
        "audit_v5_sha256": digest_file(receipt),
        "campaign_protocol_sha256": digest_bytes(campaign_raw),
        "manifest_file_count": len(manifest["files"]),
        "completed_cases": list(audit["completed_case_ids"]),
        "incomplete_cases": list(audit["incomplete_case_ids"]),
        "effective_updates": audit["effective_update_count"],
        "model_revision": campaign["model"]["revision"],
        "model_weight_sha256": campaign["model"]["weight_sha256"],
    }


def system_snapshot() -> dict[str, Any]:
    memory = subprocess.run(
        ["memory_pressure", "-Q"], capture_output=True, text=True, check=True
    ).stdout
    free_match = re.search(r"free percentage:\s*(\d+)%", memory)
    swap = subprocess.run(
        ["sysctl", "vm.swapusage"], capture_output=True, text=True, check=True
    ).stdout
    used_match = re.search(r"used =\s*([0-9.]+)M", swap)
    free_match_swap = re.search(r"free =\s*([0-9.]+)M", swap)
    if not free_match or not used_match or not free_match_swap:
        raise RuntimeError("could not parse macOS memory/swap preflight")
    return {
        "system_free_memory_percent": int(free_match.group(1)),
        "swap_used_bytes": int(float(used_match.group(1)) * 1024 * 1024),
        "swap_free_bytes": int(float(free_match_swap.group(1)) * 1024 * 1024),
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "memory_pressure_output": memory.strip(),
        "swap_output": swap.strip(),
    }


def current_rss_bytes() -> int:
    raw = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # macOS ps reports RSS in KiB.
    return int(raw) * 1024


def support_softmax(logits: list[float], bias: list[float], eos_offset: float,
                    temperature: float, eos_index: int, eos_allowed: bool) -> list[float]:
    adjusted = [float(x) + float(b) for x, b in zip(logits, bias, strict=True)]
    adjusted[eos_index] += eos_offset
    if not eos_allowed:
        adjusted[eos_index] = -1e30
    scaled = [x / temperature for x in adjusted]
    peak = max(scaled)
    exps = [math.exp(x - peak) for x in scaled]
    total = sum(exps)
    return [x / total for x in exps]


def terminal_distribution(
    node_logits: dict[str, list[float]],
    action_labels: list[str],
    bias_support: list[float],
    eos_offset: float,
    temperature: float,
    minimum_actions: int,
    maximum_actions: int,
    task: dict[str, Any],
) -> dict[str, float]:
    """Return exact probabilities keyed by action string and terminal type."""
    eos_index = len(action_labels)
    outcomes: dict[str, float] = {}

    def visit(prefix: str, mass: float) -> None:
        if len(prefix) >= maximum_actions:
            raise AssertionError("terminal action prefix should have been recorded")
        probabilities = support_softmax(
            node_logits[prefix], bias_support, eos_offset, temperature,
            eos_index, len(prefix) >= minimum_actions,
        )
        for idx, action in enumerate(action_labels):
            next_prefix = prefix + action
            next_mass = mass * probabilities[idx]
            if len(next_prefix) == maximum_actions:
                outcomes[f"{next_prefix}|maximum_actions"] = next_mass
            else:
                visit(next_prefix, next_mass)
        if len(prefix) >= minimum_actions:
            outcomes[f"{prefix}|eos"] = mass * probabilities[eos_index]

    visit("", 1.0)
    if len(outcomes) != 1360:
        raise AssertionError(f"unexpected terminal outcome count: {len(outcomes)}")
    total = math.fsum(outcomes.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise AssertionError(f"terminal mass is {total}, not one")
    return outcomes


def summarize_distribution(outcomes: dict[str, float], task: dict[str, Any]) -> dict[str, Any]:
    scores_v0: list[tuple[float, float]] = []
    scores_v1: list[tuple[float, float]] = []
    success_mass = 0.0
    reward_one_mass = 0.0
    entropy = 0.0
    expected_generated_tokens = 0.0
    action_mass: dict[str, float] = {}
    for key, prob in outcomes.items():
        actions, terminal = key.rsplit("|", 1)
        r0 = verifier_score(actions, task, 0)
        r1 = verifier_score(actions, task, 1)
        scores_v0.append((prob, r0))
        scores_v1.append((prob, r1))
        if r1 >= 1.0 - 1e-12:
            reward_one_mass += prob
        if (task["target_pair"] in actions and
                task["forbidden_pair"] not in actions[:3]):
            success_mass += prob
        entropy -= prob * math.log(prob) if prob > 0 else 0.0
        expected_generated_tokens += prob * (len(actions) + int(terminal == "eos"))
        action_mass[actions] = action_mass.get(actions, 0.0) + prob

    def moments(rows: list[tuple[float, float]]) -> tuple[float, float]:
        mean = math.fsum(p * r for p, r in rows)
        var = math.fsum(p * (r - mean) ** 2 for p, r in rows)
        return mean, var

    mean0, var0 = moments(scores_v0)
    mean1, var1 = moments(scores_v1)
    return {
        "outcome_count": len(outcomes),
        "probability_mass": math.fsum(outcomes.values()),
        "expected_proxy_reward_v0": mean0,
        "proxy_reward_variance_v0": var0,
        "expected_trusted_reward_v1": mean1,
        "trusted_reward_variance_v1": var1,
        "probability_trusted_reward_equals_1": reward_one_mass,
        "probability_exact_task_success": success_mass,
        "trajectory_entropy_nats": entropy,
        "expected_generated_tokens": expected_generated_tokens,
        "action_string_distribution": action_mass,
    }


def compare_distributions(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    if reference.keys() != candidate.keys():
        raise ValueError("policy outcome spaces differ")
    tv = 0.5 * math.fsum(abs(reference[k] - candidate[k]) for k in reference)
    kl_ref_candidate = math.fsum(
        p * math.log(p / candidate[k])
        for k, p in reference.items() if p > 0 and candidate[k] > 0
    )
    kl_candidate_ref = math.fsum(
        p * math.log(p / reference[k])
        for k, p in candidate.items() if p > 0 and reference[k] > 0
    )
    return {
        "trajectory_total_variation": tv,
        "trajectory_kl_reference_to_candidate_nats": kl_ref_candidate,
        "trajectory_kl_candidate_to_reference_nats": kl_candidate_ref,
    }


def load_state(path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    if digest_bytes(raw) != expected_sha:
        raise ValueError(f"checkpoint digest mismatch: {path}")
    meta = read_json(Path(str(path) + ".json"))
    if meta.get("sha256") != expected_sha:
        raise ValueError(f"checkpoint sidecar mismatch: {path}")
    import torch

    state = torch.load(
        __import__("io").BytesIO(gzip.decompress(raw)),
        map_location="cpu",
        weights_only=True,
    )
    bias = state.get("bias")
    if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
        raise ValueError(f"checkpoint has no valid action bias: {path}")
    return meta, state


def collect_policies(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    policies: list[dict[str, Any]] = []
    baseline_by_seed: dict[str, str] = {}
    for seed in (707, 808, 909):
        base = run_root / "baseline" / f"seed-{seed}" / "models" / "policy-0.pt.gz"
        expected = read_json(Path(str(base) + ".json"))["sha256"]
        meta, state = load_state(base, expected)
        if not torch.equal(state["bias"], torch.zeros_like(state["bias"])):
            raise ValueError(f"baseline seed {seed} does not have zero action bias")
        policy_id = f"baseline-seed-{seed}"
        baseline_by_seed[str(seed)] = policy_id
        policies.append({
            "policy_id": policy_id,
            "kind": "baseline",
            "seed": seed,
            "checkpoint_sha256": expected,
            "bias": state["bias"],
            "policy_version": meta.get("metadata", {}).get("policy_version"),
        })

    audit = read_json(run_root / "independent-audit-v5.json")
    completed = set(audit["completed_case_ids"])
    actual = {p.name for p in (run_root / "cases").iterdir() if p.is_dir() and (p / "case-result.json").is_file()}
    if actual != completed:
        raise ValueError("on-disk completed cases differ from independent audit")
    for case_id in sorted(completed):
        case_root = run_root / "cases" / case_id
        case_result = read_json(case_root / "case-result.json")
        active = read_json(case_root / "active.json")
        if active.get("checkpoint_sha256") != case_result.get("final_checkpoint_sha256"):
            raise ValueError(f"active pointer and case result disagree for {case_id}")
        if active.get("policy_version") != case_result.get("final_policy_version"):
            raise ValueError(f"active policy version mismatch for {case_id}")
        path = case_root / "models" / active["checkpoint"]
        sha = case_result["final_checkpoint_sha256"]
        meta, state = load_state(path, sha)
        if meta.get("metadata", {}).get("policy_version") != case_result["final_policy_version"]:
            raise ValueError(f"checkpoint policy version mismatch for {case_id}")
        seed = int(case_id.split("-")[1])
        policies.append({
            "policy_id": case_id,
            "kind": "post_training",
            "seed": seed,
            "arm": case_result["arm"],
            "checkpoint_sha256": sha,
            "bias": state["bias"],
            "policy_version": case_result["final_policy_version"],
            "update_count": case_result["updates"],
            "final_active_pointer": active["checkpoint"],
        })
    return policies, baseline_by_seed


def enumerate_logits(torch, model, task: dict[str, Any], tokenizer, action_ids: list[int], eos_id: int,
                     batch_size: int, started: float, protocol: dict[str, Any], base_swap: int,
                     output: Path, run_root: Path) -> dict[str, list[float]]:
    _text, prompt_ids = prompt_text(tokenizer, task)
    labels = protocol["enumeration"]["actions"]
    all_prefixes = ["".join(x) for depth in range(5) for x in itertools.product(labels, repeat=depth)]
    node_logits: dict[str, list[float]] = {}
    for depth in range(5):
        level = ["".join(x) for x in itertools.product(labels, repeat=depth)]
        sequences = [
            prompt_ids + [action_ids[labels.index(char)] for char in prefix]
            for prefix in level
        ]
        for i in range(0, len(sequences), batch_size):
            if time.monotonic() - started > protocol["resource_budget"]["maximum_wall_seconds"]:
                raise TimeoutError("frozen 900-second wall limit exceeded")
            batch = torch.tensor(sequences[i:i + batch_size], dtype=torch.long)
            with torch.inference_mode():
                outputs = model.model(input_ids=batch)
                hidden = outputs.last_hidden_state[:, -1, :]
                logits = model.lm_head(hidden).float().index_select(
                    -1, torch.tensor(action_ids + [eos_id], dtype=torch.long)
                )
            for prefix, row in zip(level[i:i + batch_size], logits.tolist(), strict=True):
                node_logits[prefix] = [float(x) for x in row]
            if (i // batch_size) % 4 == 0:
                limits = protocol["resource_budget"]
                rss = current_rss_bytes()
                system_now = system_snapshot()
                swap_now = system_now["swap_used_bytes"]
                if rss > limits["maximum_process_rss_bytes"]:
                    raise MemoryError(f"RSS cap exceeded: {rss}")
                if swap_now - base_swap > limits["maximum_swap_growth_bytes"]:
                    raise MemoryError(f"swap-growth cap exceeded: {swap_now - base_swap}")
                if system_now["system_free_memory_percent"] < limits["minimum_system_free_memory_percent"]:
                    raise MemoryError("system free-memory floor crossed during exact enumeration")
                if time.monotonic() - started > limits["maximum_wall_seconds"]:
                    raise TimeoutError("frozen 900-second wall limit exceeded")
        print(f"forward task={task['task_id']} depth={depth} nodes={len(level)}", flush=True)
    if len(node_logits) != len(all_prefixes) or set(node_logits) != set(all_prefixes):
        raise ValueError("exact policy prefix set is incomplete")
    return node_logits


def retained_mc(run_root: Path, seed: int, task_id: str) -> dict[str, Any]:
    rows = []
    sources = [run_root / "baseline" / f"seed-{seed}" / "evaluation.json"]
    sources.extend(sorted((run_root / "cases").glob(f"seed-{seed}-*/evaluation.json")))
    for path in sources:
        sidecar = read_json(Path(str(path) + ".sha256"))
        if sidecar.get("sha256") != digest_file(path):
            raise ValueError(f"retained evaluation sidecar mismatch: {path}")
        evaluation = read_json(path)
        for task in evaluation["tasks"]:
            if task["task_id"] != task_id:
                continue
            scores = [verifier_score(g["response"], task["task"], 1) for g in task["generations"]]
            rows.append({
                "case_id": evaluation["case_id"],
                "kind": evaluation["kind"],
                "sample_count": len(scores),
                "trusted_reward_sum": math.fsum(scores),
                "trusted_reward_mean": math.fsum(scores) / len(scores),
                "successes_reward_1": sum(x >= 1.0 - 1e-12 for x in scores),
                "responses": [g["response"] for g in task["generations"]],
            })
    return {"records": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {args.output}")
    protocol, protocol_sha = frozen_protocol(args.protocol)
    preflight = system_snapshot()
    limits = protocol["resource_budget"]
    free_disk = preflight["disk_free_bytes"]
    if preflight["system_free_memory_percent"] < limits["minimum_system_free_memory_percent"]:
        raise MemoryError("system free-memory preflight below frozen minimum")
    if preflight["swap_free_bytes"] < limits["minimum_free_swap_bytes"]:
        raise MemoryError("free-swap preflight below frozen minimum")
    if free_disk < limits["minimum_disk_free_bytes"]:
        raise OSError("disk-free preflight below frozen minimum")
    input_ids = verify_inputs(protocol, args.run_root)
    campaign = read_json(ROOT / protocol["source"]["campaign_protocol_path"])
    snapshot = Path(protocol["model"]["snapshot_path"])
    if campaign["model"]["revision"] != protocol["model"]["revision"]:
        raise ValueError("model revision does not match the historical protocol")
    if campaign["model"]["weight_sha256"] != protocol["model"]["weight_sha256"]:
        raise ValueError("weight identity does not match the historical protocol")
    for rel, expected in campaign["model"]["snapshot_files"].items():
        if digest_file(snapshot / rel) != expected:
            raise ValueError(f"model snapshot identity mismatch: {rel}")
    if digest_file(Path(campaign["model"]["weight_file"])) != protocol["model"]["weight_sha256"]:
        raise ValueError("cached model weight hash mismatch")

    args.output.mkdir(parents=True)
    started = time.monotonic()
    write_json(args.output / "preflight.json", {
        "protocol_sha256": protocol_sha,
        "input_identities": input_ids,
        "system": preflight,
        "python": sys.version,
        "platform": platform.platform(),
        "started_unix_time": time.time(),
        "analysis_script_sha256": digest_file(Path(__file__)),
        "generated_tokens": 0,
        "optimizer_updates": 0,
    })
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(limits["cpu_threads"])
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        labels = protocol["enumeration"]["actions"]
        action_ids_raw = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
        if any(len(row) != 1 for row in action_ids_raw):
            raise ValueError("frozen action tokens are no longer single-token IDs")
        action_ids = [row[0] for row in action_ids_raw]
        eos_id = tokenizer.eos_token_id
        if eos_id is None or eos_id in action_ids:
            raise ValueError("EOS token conflicts with frozen action IDs")
        policies, baseline_by_seed = collect_policies(args.run_root)
        exact: dict[str, dict[str, Any]] = {}
        outcomes_by_policy_task: dict[tuple[str, str], dict[str, float]] = {}
        calibration = []
        for task in campaign["tasks"]:
            if task["task_id"] not in protocol["source"]["analysis_tasks"]:
                continue
            node_logits = enumerate_logits(
                torch, model, task, tokenizer, action_ids, eos_id,
                limits["batch_size"], started, protocol,
                preflight["swap_used_bytes"], args.output, args.run_root,
            )
            first_probs = {}
            for policy in policies:
                bias_support = policy["bias"][action_ids + [eos_id]].tolist()
                distribution = terminal_distribution(
                    node_logits, labels, [float(x) for x in bias_support],
                    float(campaign["generation"]["termination_logit_offset"]),
                    float(campaign["generation"]["temperature"]),
                    int(campaign["generation"]["minimum_actions"]),
                    int(campaign["generation"]["maximum_actions"]),
                    task,
                )
                metrics = summarize_distribution(distribution, task)
                exact.setdefault(policy["policy_id"], {})[task["task_id"]] = metrics
                outcomes_by_policy_task[(policy["policy_id"], task["task_id"])] = distribution
                if policy["kind"] == "baseline":
                    first_probs[policy["policy_id"]] = support_softmax(
                        node_logits[""], [float(x) for x in bias_support],
                        float(campaign["generation"]["termination_logit_offset"]),
                        float(campaign["generation"]["temperature"]), len(labels), False,
                    )

            if task["task_id"].startswith("eval-"):
                for seed in (707, 808, 909):
                    path = args.run_root / "baseline" / f"seed-{seed}" / "evaluation.json"
                    ev = read_json(path)
                    record = next(t for t in ev["tasks"] if t["task_id"] == task["task_id"])
                    expected = record["first_step_support_probabilities"]
                    actual = first_probs[baseline_by_seed[str(seed)]]
                    error = max(abs(float(a) - float(b)) for a, b in zip(expected, actual, strict=True))
                    calibration.append({"seed": seed, "task_id": task["task_id"], "max_abs_error": error})
                    if error > 1e-5:
                        raise ValueError(f"baseline first-step calibration failed: {task['task_id']} seed={seed} err={error}")
            print(f"enumerated task={task['task_id']} policies={len(policies)}", flush=True)

        comparisons = []
        for policy in policies:
            if policy["kind"] != "post_training":
                continue
            base_id = baseline_by_seed[str(policy["seed"])]
            for task_id in protocol["source"]["analysis_tasks"]:
                base_metrics = exact[base_id][task_id]
                trained_metrics = exact[policy["policy_id"]][task_id]
                divergence = compare_distributions(
                    outcomes_by_policy_task[(base_id, task_id)],
                    outcomes_by_policy_task[(policy["policy_id"], task_id)],
                )
                comparisons.append({
                    "policy_id": policy["policy_id"],
                    "seed": policy["seed"],
                    "arm": policy["arm"],
                    "task_id": task_id,
                    "delta_expected_proxy_reward_v0": trained_metrics["expected_proxy_reward_v0"] - base_metrics["expected_proxy_reward_v0"],
                    "delta_expected_trusted_reward_v1": trained_metrics["expected_trusted_reward_v1"] - base_metrics["expected_trusted_reward_v1"],
                    "delta_probability_trusted_reward_equals_1": trained_metrics["probability_trusted_reward_equals_1"] - base_metrics["probability_trusted_reward_equals_1"],
                    "delta_probability_exact_task_success": trained_metrics["probability_exact_task_success"] - base_metrics["probability_exact_task_success"],
                    **divergence,
                })

        mc = {}
        for seed in (707, 808, 909):
            for task_id in protocol["source"]["analysis_tasks"]:
                if not task_id.startswith("eval-"):
                    continue
                mc[f"{seed}:{task_id}"] = retained_mc(args.run_root, seed, task_id)
        final_system = system_snapshot()
        final_rss = current_rss_bytes()
        if final_rss > limits["maximum_process_rss_bytes"]:
            raise MemoryError(f"RSS cap exceeded at completion: {final_rss}")
        if final_system["swap_used_bytes"] - preflight["swap_used_bytes"] > limits["maximum_swap_growth_bytes"]:
            raise MemoryError("swap-growth cap exceeded at completion")
        if final_system["system_free_memory_percent"] < limits["minimum_system_free_memory_percent"]:
            raise MemoryError("system free-memory floor crossed at completion")
        if time.monotonic() - started > limits["maximum_wall_seconds"]:
            raise TimeoutError("frozen wall limit exceeded before publication")
        result = {
            "schema_version": 1,
            "analysis_kind": "retrospective_exact_finite_policy_enumeration",
            "protocol_sha256": protocol_sha,
            "input_identities": input_ids,
            "model": {
                "repository": protocol["model"]["repository"],
                "revision": protocol["model"]["revision"],
                "weights_sha256": protocol["model"]["weight_sha256"],
                "backend": protocol["model"]["backend"],
                "generated_tokens": 0,
                "optimizer_updates": 0,
            },
            "scope": {
                "policy_count": len(policies),
                "baseline_count": sum(p["kind"] == "baseline" for p in policies),
                "final_checkpoint_count": sum(p["kind"] == "post_training" for p in policies),
                "task_count": len(exact[policies[0]["policy_id"]]),
                "terminal_sequences_per_task": 1360,
                "task_unit": "one already-consumed synthetic prompt; not independent coding tasks",
            },
            "policies": [
                {k: v for k, v in p.items() if k != "bias"}
                | {"bias_support": [float(x) for x in p["bias"][action_ids + [eos_id]].tolist()]}
                for p in policies
            ],
            "exact_metrics": exact,
            "paired_policy_deltas_vs_seed_baseline": comparisons,
            "first_step_calibration": calibration,
            "retained_sampled_evaluation": mc,
            "resource": {
                "system_preflight": preflight,
                "final_system": final_system,
                "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "elapsed_seconds": time.monotonic() - started,
            },
            "interpretation_limits": protocol["interpretation"],
        }
        # The complete action-string distributions are useful for reward integration,
        # but are derivable from terminal sequence mass and would bloat the main report.
        for row in result["exact_metrics"].values():
            for metrics in row.values():
                metrics.pop("action_string_distribution", None)
        write_json(args.output / "exact-policy-analysis.json", result)
        report_lines = [
            "# Sprint XIV Retrospective Exact Policy Analysis",
            "",
            "This is an exact analysis of consumed synthetic Sprint XIII prompts; it does not contain new generated tokens, optimizer updates, coding tasks, or fresh success evidence.",
            "",
            f"- Model: `{protocol['model']['repository']}` @ `{protocol['model']['revision']}` (CPU fp32; weights SHA-256 `{protocol['model']['weight_sha256']}`).",
            f"- Inputs: archive SHA-256 `{input_ids['archive_sha256']}`; audit v5 `{input_ids['audit_v5_sha256']}`; {len(policies)} policies; {len(exact[policies[0]['policy_id']])} prompts; 1,360 terminal sequences per prompt.",
            f"- First-step calibration max error: `{max(x['max_abs_error'] for x in calibration):.3g}` across {len(calibration)} retained baseline rows.",
            "",
            "| Policy | Task | Exact trusted reward v1 | P(reward=1) | P(task success) | Δ trusted vs baseline | Trajectory TV | KL(candidate‖baseline) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for policy in policies:
            if policy["kind"] != "post_training":
                continue
            for task_id in protocol["source"]["analysis_tasks"]:
                metrics = exact[policy["policy_id"]][task_id]
                delta = next(x for x in comparisons if x["policy_id"] == policy["policy_id"] and x["task_id"] == task_id)
                report_lines.append(
                    f"| {policy['policy_id']} | {task_id} | {metrics['expected_trusted_reward_v1']:.6f} | {metrics['probability_trusted_reward_equals_1']:.6f} | {metrics['probability_exact_task_success']:.6f} | {delta['delta_expected_trusted_reward_v1']:+.6f} | {delta['trajectory_total_variation']:.6f} | {delta['trajectory_kl_candidate_to_reference_nats']:.6g} |"
                )
        report_lines += [
            "",
            "Interpretation: task-family/prompt observations remain clustered within the original synthetic cohort; these rows are not independent replication units. Retained Monte Carlo samples are shown in the JSON next to exact probabilities so sampling fluctuation is not confused with an objective change.",
            "",
            f"Generated tokens: **0**. Optimizer updates: **0**. Wall time: **{result['resource']['elapsed_seconds']:.1f}s**. Peak process RSS: **{result['resource']['process_max_rss_bytes'] / (1024**3):.2f} GiB**.",
        ]
        (args.output / "report.md").write_text("\n".join(report_lines) + "\n")
        output_manifest = {}
        for path in sorted(args.output.iterdir()):
            if path.is_file() and path.name != "sha256-manifest.json":
                output_manifest[path.name] = {"bytes": path.stat().st_size, "sha256": digest_file(path)}
        write_json(args.output / "sha256-manifest.json", output_manifest)
        print(f"complete output={args.output} wall={result['resource']['elapsed_seconds']:.1f}s", flush=True)
        return 0
    except BaseException as exc:
        write_json(args.output / "abort.json", {
            "protocol_sha256": protocol_sha,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_seconds": time.monotonic() - started,
            "system": system_snapshot(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
