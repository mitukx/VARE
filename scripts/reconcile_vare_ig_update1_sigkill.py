#!/usr/bin/env python3
"""Reconcile a durable update-1 checkpoint after its locked pre-ack SIGKILL."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402
from scripts.resume_vare_ig_preupdate_batch import _clear_mlx_cache_then_guard  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_once(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite recovery evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = experiment.validate_protocol(protocol_path)
    experiment._install_existing_runner_contract(protocol_path)
    seed = int(protocol["sampling"]["seed"])
    runner.protocol_payload(protocol_path, seed=seed)
    recovery = protocol.get("recovery", {})
    if recovery.get("initial_update_sigkill_boundary") != "after_checkpoint_before_completion_ack":
        raise RuntimeError("run protocol does not lock the initial checkpoint-before-ack boundary")
    if recovery.get("initial_update_sigkill_arm") != "informative_per_cost" \
            or recovery.get("initial_update_sigkill_seed") != seed:
        raise RuntimeError("this run is not assigned the initial pre-ack recovery boundary")

    result_path = run / "training-result.json"
    state_path = run / "checkpoint-state.json"
    event_path = run / "learner-events.json"
    reconciled_path = run / "learner-events-reconciled.json"
    if not result_path.is_file() or not state_path.is_file():
        raise RuntimeError("training result and update-1 checkpoint state must be durable before recovery")
    if event_path.exists() or reconciled_path.exists():
        raise RuntimeError("learner acknowledgment already exists; refusing duplicate reconciliation")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if result.get("update_applied") is not True or result.get("params_changed") is not True \
            or state.get("update_step") != 1 or state.get("optimizer_step") != 1:
        raise RuntimeError("durable files do not contain exactly one applied update-1")

    # Verify the generated sample provenance before trusting the checkpoint boundary.
    rollouts = run / "rollouts-train-step-0.jsonl"
    streamed = run / "training-samples-streamed-train-step-0.jsonl"
    samples_before = run / "training-samples-before-logprob.json"
    artifacts = (rollouts, streamed, samples_before, run / "scheduler-selection.json",
                 run / "trajectory-store.sqlite3")
    if any(not path.is_file() for path in artifacts):
        raise RuntimeError("pre-update rollout or trajectory provenance is incomplete")
    raw_rows = [json.loads(line) for line in rollouts.read_text(encoding="utf-8").splitlines()]
    streamed_rows = [json.loads(line) for line in streamed.read_text(encoding="utf-8").splitlines()]
    samples = json.loads(samples_before.read_text(encoding="utf-8"))
    if len(raw_rows) != 24 or len(streamed_rows) != 24 or len(samples) != 24:
        raise RuntimeError("persisted initial sample counts differ from the frozen 24-completion budget")
    def key(row):
        return (str(row.get("task_id", row.get("instance_id"))), int(row["sample_index"]))
    raw_by = {key(row): row for row in raw_rows}
    streamed_by = {key(row): row for row in streamed_rows}
    sample_by = {key(row): row for row in samples}
    if len(raw_by) != 24 or set(raw_by) != set(streamed_by) or set(raw_by) != set(sample_by):
        raise RuntimeError("persisted sample identities are incomplete or duplicated")
    for identity, raw in raw_by.items():
        captured, learner = streamed_by[identity], sample_by[identity]
        learner_text = learner.get("response_text")
        if hashlib.sha256(raw["text"].encode("utf-8")).hexdigest() != raw["response_sha256"] \
                or captured.get("response_sha256") != raw["response_sha256"] \
                or not isinstance(learner_text, str) \
                or hashlib.sha256(learner_text.encode("utf-8")).hexdigest() != raw["response_sha256"]:
            raise RuntimeError("raw, streamed, and pre-update response identities do not match")
        if learner_text != raw["text"]:
            raise RuntimeError("pre-update response text differs from its graded rollout")
        if captured.get("prompt_ids") != learner.get("prompt_ids") \
                or captured.get("completion_ids") != learner.get("completion_ids") \
                or captured.get("candidate_source_sha256") != raw.get("candidate_source_sha256"):
            raise RuntimeError("captured token IDs or candidate provenance differ from the learner batch")

    resource_before = runner.resource_snapshot(run)
    if resource_before["available_memory_gib"] < runner.AVAILABLE_MEMORY_MIN_GIB:
        raise runner.ResourceStop("available memory is below the locked floor before checkpoint reconciliation")
    started = time.monotonic()
    import mlx.core as mx

    original_guard = runner.resource_guard
    runner.resource_guard = lambda path, phase_started: _clear_mlx_cache_then_guard(
        mx, lambda: original_guard(path, phase_started))
    # _install_existing_runner_contract wraps this restore to reconstruct the
    # missing durable acknowledgment from the already-published step-1 state.
    _meta, model, optimizer, _saved, tokenizer = runner.restore_and_verify(
        run, seed=seed, phase="resume", protocol_path=protocol_path)
    parameter_sha = __import__("vare.grpo_mlx", fromlist=["trainable_digest"]).trainable_digest(model)
    optimizer_sha = runner.optimizer_digest(optimizer)
    step = int(optimizer.state["step"].item())
    del model, tokenizer, optimizer
    gc.collect()
    mx.clear_cache()

    if step != 1 or parameter_sha != state["trainable_parameters_sha256"] \
            or optimizer_sha != state["optimizer_state_sha256"]:
        raise RuntimeError("restored update-1 state differs from its durable checkpoint")
    if not reconciled_path.is_file() or event_path.exists():
        raise RuntimeError("missing or unexpected learner acknowledgment after reconciliation")
    reconciled = json.loads(reconciled_path.read_text(encoding="utf-8"))
    if reconciled.get("duplicate_effective_update") is not False \
            or reconciled.get("effective_update_count") != 1 \
            or reconciled.get("checkpoint_state_sha256") != sha256_file(state_path):
        raise RuntimeError("reconciled acknowledgment does not identify exactly one durable update")
    consumption_path = run / "trajectory-consumption-0001.json"
    if not consumption_path.is_file():
        raise RuntimeError("verified trajectory groups were not durably consumed with the learner decision")
    consumption = json.loads(consumption_path.read_text(encoding="utf-8"))
    if consumption.get("acknowledgment_sha256") != sha256_file(reconciled_path):
        raise RuntimeError("trajectory-consumption record is not bound to the reconciled acknowledgment")
    summary = {
        "schema": "vare_ig_initial_update_sigkill_recovery_v1",
        "run_protocol_sha256": sha256_file(protocol_path),
        "training_result_sha256": sha256_file(result_path),
        "checkpoint_state_sha256": sha256_file(state_path),
        "reconciled_acknowledgment_sha256": sha256_file(reconciled_path),
        "trajectory_consumption_sha256": sha256_file(consumption_path),
        "parameter_sha256": parameter_sha,
        "optimizer_sha256": optimizer_sha,
        "optimizer_step": step,
        "effective_update_count": reconciled["effective_update_count"],
        "duplicate_effective_update": reconciled["duplicate_effective_update"],
        "sample_count": len(raw_rows),
        "sample_provenance_verified": True,
        "resource": {"before": resource_before, "after": runner.resource_snapshot(run),
                     "recovery_wall_seconds": time.monotonic() - started},
    }
    atomic_write_once(run / "initial-update-sigkill-recovery.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
