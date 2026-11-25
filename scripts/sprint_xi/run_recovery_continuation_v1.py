#!/usr/bin/env python3
"""Continue Sprint XI from its committed v1 checkpoint without rerunning inputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


def canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{__import__('os').getpid()}")
    with temp.open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2).encode() + b"\n")
        stream.flush()
        __import__("os").fsync(stream.fileno())
    __import__("os").replace(temp, path)
    fd = __import__("os").open(path.parent, __import__("os").O_RDONLY)
    try:
        __import__("os").fsync(fd)
    finally:
        __import__("os").close(fd)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(root: Path, name: str = "sha256-manifest.json") -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == name:
            continue
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    manifest = {"schema_version": 1, "files": files}
    write_json(root / name, manifest)
    return manifest


def run(protocol_path: Path, original_protocol_path: Path, base_run: Path, output: Path, vare_root: Path, rvl_root: Path) -> dict[str, Any]:
    lock = json.loads(protocol_path.read_text())
    if lock.get("frozen_before_run") is not True:
        raise RuntimeError("continuation protocol is not frozen")
    if sha(Path(__file__).resolve()) != lock["runner_sha256"]:
        raise RuntimeError("continuation runner hash mismatch")
    if sha(original_protocol_path) != lock["original_protocol_sha256"]:
        raise RuntimeError("original protocol hash changed")
    original_script = Path(__file__).with_name("run_model_coupled_v1.py")
    original = load_module(original_script, "sprint_xi_original_runner")
    protocol = original._read_protocol(original_protocol_path)
    original._validate_sources(protocol, vare_root, rvl_root)
    if protocol["protocol_id"] != lock["parent_protocol_id"]:
        raise RuntimeError("parent protocol identity mismatch")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite continuation output {output}")

    cp1 = base_run / "checkpoints/candidate-v1"
    stale_path = base_run / "rollouts/queued-v0-stale.json"
    fresh_v1_path = base_run / "rollouts/fresh-v1.json"
    required = {
        "committed_v1_checkpoint": cp1 / "manifest.json",
        "queued_stale_trajectory": stale_path,
        "queued_stale_sidecar": stale_path.with_suffix(stale_path.suffix + ".manifest.json"),
        "worker_ack": base_run / "worker-recovery-ack.json",
        "partial_failure_record": base_run / "attempt-failure.json",
    }
    for name, path in required.items():
        expected = lock["parent_evidence_sha256"][name]
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"parent evidence changed: {name}")
    if fresh_v1_path.exists():
        raise RuntimeError("declared fresh_v1 cohort was already generated; refusing reuse")
    cp1_manifest = original._checkpoint_manifest(cp1)
    if cp1_manifest["checkpoint_sha256"] != lock["initial_policy_checkpoint_sha256"]:
        raise RuntimeError("committed v1 checkpoint identity mismatch")

    import psutil
    preflight = original._system_preflight(protocol, Path(protocol["model"]["local_snapshot"]))
    limits = protocol["resource_limits"]
    output.mkdir(parents=True)
    write_json(output / "preflight.json", preflight)
    initial_swap = psutil.swap_memory().used
    started = time.monotonic()
    env = original._child_environment(vare_root, rvl_root)
    script = original_script.resolve()
    process_rows: list[dict[str, Any]] = []

    def child(name: str, mode: str, arguments: list[str], expected: int = 0) -> dict[str, Any]:
        command = original._command_base(script, mode, original_protocol_path, vare_root, rvl_root) + arguments
        rows = original._run_processes([(name, command)], output_root=output, env=env, limits=limits, baseline_swap=initial_swap)
        process_rows.extend(rows)
        if rows[0]["returncode"] != expected:
            raise RuntimeError(f"child {name} exited {rows[0]['returncode']} (expected {expected}); see {rows[0]['log']}")
        return rows[0]

    # Freshness admission uses the first complete group from the prior queued
    # worker, even though its reward vector is constant. No optimizer call is
    # made on that constant group.
    stale_payload = json.loads(stale_path.read_text())
    complete = next((row for row in stale_payload["task_records"] if len(row["generations"]) == int(protocol["sampling"]["group_size"])), None)
    if complete is None:
        raise RuntimeError("queued stale artifact has no complete group")
    stale_payload["selected_group_id"] = complete["group_id"]
    stale_payload["_vare_root"] = str(vare_root)
    stale_experiences = original._build_experiences(protocol, stale_payload)
    sys.path.insert(0, str(vare_root / "src"))
    from vare.config import LagConfig, ReplayConfig
    from vare.lag import LagController, make_experience_freshness_assessor
    from vare.replay import PrioritizedReplay

    replay = PrioritizedReplay(ReplayConfig(capacity=32), seed=int(protocol["optimizer"]["torch_seed"]))
    group_added = replay.add_group(stale_experiences, [1.0] * len(stale_experiences))
    assessor = make_experience_freshness_assessor(
        LagController(LagConfig(max_policy_lag=0, max_verifier_lag=0, max_shift=0.0)),
        active_policy_version=1,
        active_verifier_version=0,
    )
    stale_policy_sha = complete["policy_checkpoint_sha256"]
    vaware = replay.sample_current(
        4,
        freshness_fn=lambda exp: assessor(exp).freshness if assessor(exp).admitted and exp.attempt.metadata.get("policy_checkpoint_sha256") == stale_policy_sha else None,
        grouped=True,
    )
    naive = replay.sample_current(4, freshness_fn=lambda _exp: 1.0, grouped=True)
    stale_result = {
        "protocol_id": lock["protocol_id"],
        "group_id": complete["group_id"],
        "prompt_id": complete["prompt_id"],
        "policy_version": complete["policy_version"],
        "active_policy_version": 1,
        "rewards": complete["rewards"],
        "replay_group_added": group_added,
        "vare_admitted_members": len(vaware),
        "naive_admitted_members": len(naive),
        "vare_rejected_stale_group": group_added and not vaware,
        "naive_accepted_same_stale_group": len(naive) == 4,
        "optimizer_update_on_constant_group": False,
        "queue_age_seconds": max(0.0, time.time() - json.loads(required["queued_stale_sidecar"].read_text())["persisted_unix"]),
        "source_artifact_sha256": sha(stale_path),
    }
    write_json(output / "staleness-admission.json", stale_result)
    if not stale_result["vare_rejected_stale_group"] or not stale_result["naive_accepted_same_stale_group"]:
        raise RuntimeError("stale freshness contrast failed")

    rollouts = output / "rollouts"
    checkpoints = output / "checkpoints"
    runs = output / "learner-runs"
    fresh_v1_path = rollouts / "fresh-v1.json"
    child("worker-fresh-v1", "worker", [
        "--cohort", "fresh_v1", "--policy-version", "1", "--seed-offset", "31", "--checkpoint", str(cp1), "--artifact", str(fresh_v1_path)
    ])
    fresh = json.loads(fresh_v1_path.read_text())
    if fresh["status"] != "mixed_reward_group_found":
        result = {
            "protocol_id": lock["protocol_id"],
            "status": "partial_no_mixed_fresh_v1_group",
            "staleness_admission": stale_result,
            "fresh_v1_rollout_status": fresh["status"],
            "processes": process_rows,
            "scope": "partial continuation; no dependent optimizer update was attempted",
            "campaign_wall_seconds": time.monotonic() - started,
            "host_swap_growth_bytes": psutil.swap_memory().used - initial_swap,
        }
        write_json(output / "continuation-summary.json", result)
        write_manifest(output)
        return result

    ref_cp = checkpoints / "reference-v2"
    child("synchronous-reference-v2", "learn", [
        "--rollout", str(fresh_v1_path), "--checkpoint", str(cp1), "--checkpoint-out", str(ref_cp),
        "--run-dir", str(runs / "reference-v2"), "--update-id", "update-v2", "--admission", "vaware"
    ])
    fault_dir = runs / "faulted-v2-attempt"
    fault_started = time.monotonic()
    child("learner-sigkill-after-optimizer", "learn", [
        "--rollout", str(fresh_v1_path), "--checkpoint", str(cp1), "--checkpoint-out", str(checkpoints / "uncommitted-v2"),
        "--run-dir", str(fault_dir), "--update-id", "update-v2", "--admission", "vaware", "--fault-after-optimizer"
    ], expected=-signal.SIGKILL)
    killed_at = time.monotonic()
    recovered_cp = checkpoints / "recovered-v2"
    child("learner-recovery-v2", "learn", [
        "--rollout", str(fresh_v1_path), "--checkpoint", str(cp1), "--checkpoint-out", str(recovered_cp),
        "--run-dir", str(runs / "recovery-v2"), "--update-id", "update-v2", "--admission", "vaware"
    ])
    recovery_seconds = time.monotonic() - killed_at

    evaluations = output / "evaluations"
    evaluations.mkdir()
    for name, checkpoint in (("v1", cp1), ("reference-v2", ref_cp), ("recovered-v2", recovered_cp)):
        child(f"evaluate-{name}", "evaluate", ["--checkpoint", str(checkpoint), "--output", str(evaluations / f"{name}.json")])
    v1_score = json.loads((evaluations / "v1.json").read_text())["mean_correct_choice_probability"]
    recovered_score = json.loads((evaluations / "recovered-v2.json").read_text())["mean_correct_choice_probability"]
    incumbent = cp1 if recovered_score < v1_score else recovered_cp
    incumbent_score = v1_score if incumbent == cp1 else recovered_score
    incumbent_sha = original._checkpoint_manifest(incumbent)["checkpoint_sha256"]
    pointer = output / "promoted-policy.json"
    write_json(pointer, {
        "checkpoint_path": str(incumbent),
        "checkpoint_sha256": incumbent_sha,
        "policy_version": int(original._checkpoint_manifest(incumbent)["metadata"]["policy_version"]),
        "incumbent_score": incumbent_score,
        "promotion_rule": "nondecreasing synthetic choice probability",
    })
    child("evaluate-adverse-candidate", "evaluate", ["--checkpoint", str(incumbent), "--bias-override", "-20.0", "--override-token", "B", "--output", str(evaluations / "adverse.json")])
    adverse_score = json.loads((evaluations / "adverse.json").read_text())["mean_correct_choice_probability"]
    rollback = {
        "adverse_candidate_score": adverse_score,
        "incumbent_score": incumbent_score,
        "rejected": adverse_score < incumbent_score,
        "pointer_sha256_after_rejection": sha(pointer),
        "active_checkpoint_sha256_after_rejection": original._checkpoint_manifest(incumbent)["checkpoint_sha256"],
        "incumbent_identity_preserved": original._checkpoint_manifest(incumbent)["checkpoint_sha256"] == incumbent_sha,
    }
    write_json(output / "rollback-negative-control.json", rollback)

    summary = {
        "protocol_id": lock["protocol_id"],
        "protocol_sha256": sha(protocol_path),
        "runner_sha256": sha(Path(__file__).resolve()),
        "parent_protocol_id": protocol["protocol_id"],
        "parent_partial_run": str(base_run),
        "parent_partial_failure_sha256": sha(base_run / "attempt-failure.json"),
        "status": "completed",
        "model_revision": protocol["model"]["revision"],
        "fresh_v1_group_id": fresh["selected_group_id"],
        "v1_checkpoint_sha256": cp1_manifest["checkpoint_sha256"],
        "reference_v2_checkpoint_sha256": original._checkpoint_manifest(ref_cp)["checkpoint_sha256"],
        "recovered_v2_checkpoint_sha256": original._checkpoint_manifest(recovered_cp)["checkpoint_sha256"],
        "staleness_admission": stale_result,
        "recovery_latency_seconds": recovery_seconds,
        "optimizer_attempts_for_v2_recovery": 2,
        "effective_v2_recovery_commits": 1,
        "synchronous_reference_commit_count": 1,
        "evaluation": {
            "v1": v1_score,
            "reference_v2": json.loads((evaluations / "reference-v2.json").read_text())["mean_correct_choice_probability"],
            "recovered_v2": recovered_score,
        },
        "promotion": json.loads(pointer.read_text()),
        "rollback_negative_control": rollback,
        "naive_stale_optimizer_update": "NOT_RUN: queued cohort had constant zero reward across all eight prompts; admission-only contrast passed without using the group for training",
        "worker_sigkill_recovery_reused_from_parent_partial_run": True,
        "learner_fault_returncode": -signal.SIGKILL,
        "processes": process_rows,
        "campaign_wall_seconds": time.monotonic() - started,
        "host_swap_growth_bytes": psutil.swap_memory().used - initial_swap,
        "host_ram_available_bytes_after": psutil.virtual_memory().available,
        "scope": "real Qwen inference plus two VARE/RVL GRPO commits on an output-bias adapter; synthetic rewards; systems/recovery feasibility only",
        "capability_claim": False,
    }
    write_json(output / "continuation-summary.json", summary)
    write_manifest(output)
    return summary


def main() -> int:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--original-protocol", type=Path, required=True)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vare-root", type=Path, required=True)
    parser.add_argument("--rvl-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol.resolve(), args.original_protocol.resolve(), args.base_run.resolve(), args.output.resolve(), args.vare_root.resolve(), args.rvl_root.resolve())
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
