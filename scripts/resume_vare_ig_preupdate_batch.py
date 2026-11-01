#!/usr/bin/env python3
"""Resume the existing GRPO update from a frozen, persisted pre-update batch.

This recovery path does not resample or regrade candidates. It verifies the
original run protocol, rollout JSONL, trajectory selection, sample payload,
base policy, and a recovery-extension lock before using the existing MLX GRPO
functions to compute behavior log-probabilities and apply at most one update.
"""
from __future__ import annotations

import argparse
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

from vare import grpo_mlx  # noqa: E402
from vare.learning_pipeline.verified_rollout_replay import acknowledge_durable_decision  # noqa: E402
from vare.learning_pipeline import vare_ig_tasks as tasks  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _assert_saved_response_matches_rollout(sample_row: dict[str, Any], rollout: dict[str, Any]) -> None:
    """Bind the learner's preserved response text to the graded rollout digest.

    The pre-update sample schema stores ``response_text`` and omits the digest
    that ``serialize_sample`` adds after an update. Validate both the raw
    rollout's own digest and the learner sample's text against that digest.
    """
    rollout_text = rollout.get("text")
    sample_text = sample_row.get("response_text")
    if not isinstance(rollout_text, str) or not isinstance(sample_text, str):
        raise RuntimeError("persisted sample or rollout is missing its response text")
    rollout_digest = hashlib.sha256(rollout_text.encode("utf-8")).hexdigest()
    if rollout.get("response_sha256") != rollout_digest:
        raise RuntimeError("persisted rollout response digest does not match its graded text")
    if hashlib.sha256(sample_text.encode("utf-8")).hexdigest() != rollout_digest:
        raise RuntimeError("serialized sample response differs from its graded rollout")


def _clear_mlx_cache_then_guard(mx: Any, guard: Any) -> None:
    """Release evaluated per-sample buffers before checking the locked floor."""
    mx.clear_cache()
    guard()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite recovery evidence: {path}")
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def _verify_recovery_lock(lock_path: Path, protocol_path: Path, run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    protocol = experiment.validate_protocol(protocol_path, run_seed=int(lock["seed"]))
    experiment._install_existing_runner_contract(protocol_path)
    runner.protocol_payload(protocol_path, seed=int(lock["seed"]))
    if lock.get("schema") != "vare_ig_preupdate_batch_recovery_v1":
        raise RuntimeError("wrong pre-update recovery-lock schema")
    if lock.get("target_run_protocol_sha256") != sha256_file(protocol_path):
        raise RuntimeError("pre-update recovery lock is bound to another run protocol")
    if lock.get("target_run_id") != protocol.get("planned_run_id"):
        raise RuntimeError("pre-update recovery lock targets another run identifier")
    if lock.get("target_seed") != int(protocol["sampling"]["seed"]):
        raise RuntimeError("pre-update recovery lock targets another seed")
    if lock.get("recovery_helper_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("pre-update recovery helper changed after its lock was frozen")
    if lock.get("preregistered_before_optimizer_update") is not True:
        raise RuntimeError("recovery action was not locked before applying an optimizer update")
    return lock, protocol


def verify_saved_batch(run: Path, lock: dict[str, Any], protocol: dict[str, Any]) -> tuple[
        list[dict[str, Any]], list[grpo_mlx.Sample], dict[str, Any]]:
    run_meta_path = run / "run_meta.json"
    base_policy_path = run / "base-policy.json"
    rollout_path = run / "rollouts-train-step-0.jsonl"
    samples_path = run / "training-samples-before-logprob.json"
    selection_path = run / "scheduler-selection.json"
    artifacts = {
        "run_meta.json": run_meta_path,
        "base-policy.json": base_policy_path,
        "rollouts-train-step-0.jsonl": rollout_path,
        "training-samples-before-logprob.json": samples_path,
        "scheduler-selection.json": selection_path,
        "task_manifest.json": run / "task_manifest.json",
        "trajectory-store.sqlite3": run / "trajectory-store.sqlite3",
    }
    for relative, path in artifacts.items():
        expected = lock["input_artifact_sha256"].get(relative)
        if not expected or sha256_file(path) != expected:
            raise RuntimeError(f"saved input artifact changed or is missing: {relative}")
    for forbidden in (
        run / "training-result.json",
        run / "checkpoint-state.json",
        runner.checkpoint_path(run, "update-0001") / "checkpoint.json",
        run / "learner-events.json",
    ):
        if forbidden.exists():
            raise RuntimeError(f"run already has an effective or acknowledged update: {forbidden.name}")

    meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    base = json.loads(base_policy_path.read_text(encoding="utf-8"))
    if meta.get("protocol_sha256") != lock["target_run_protocol_sha256"]:
        raise RuntimeError("saved run metadata protocol hash differs from the recovery lock")
    if meta.get("source_hashes") != runner.source_hashes():
        raise RuntimeError("base runner sources differ from the saved run")
    if meta.get("task_hashes") != runner.task_hashes():
        raise RuntimeError("saved run task hashes differ from the frozen VARE-IG contracts")
    if meta.get("experiment_id") != protocol.get("planned_run_id"):
        raise RuntimeError("saved run ID differs from the frozen protocol")
    if base.get("model_revision") != runner.MODEL_REVISION:
        raise RuntimeError("base policy model revision differs from the recovery protocol")
    if base.get("policy_checkpoint_id") != meta.get("vare_ig", {}).get("base_policy_checkpoint_id",
                                                                         base.get("policy_checkpoint_id")):
        raise RuntimeError("base policy identity differs from run metadata")

    tasks_by_id = {task.task_id: task for task in tasks.TRAIN_TASKS}
    rollouts = [json.loads(line) for line in rollout_path.read_text(encoding="utf-8").splitlines()]
    serialized = json.loads(samples_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    budget = int(protocol["selection"]["fixed_budget"]["candidate_generations_per_run"])
    if len(rollouts) != budget or len(serialized) != budget:
        raise RuntimeError("persisted rollout count differs from the frozen candidate budget")
    if selection.get("rollouts_sha256") != sha256_file(rollout_path):
        raise RuntimeError("scheduler selection does not bind the persisted rollout file")
    if selection.get("trajectory_replay_sample_count") != budget:
        raise RuntimeError("verified trajectory replay count differs from the rollout budget")
    if selection.get("group_provenance_homogeneous") is not True or selection.get("freshness_policy_lag") != 0:
        raise RuntimeError("persisted trajectory group provenance is not homogeneous and fresh")
    if selection.get("arm") != protocol["selection"]["arm"] or selection.get("seed") != lock["seed"]:
        raise RuntimeError("scheduler arm or seed differs from the frozen run")

    rollout_by_key = {}
    for row in rollouts:
        task = tasks_by_id.get(row.get("task_id"))
        if task is None or task.sha256() != row.get("task_sha256"):
            raise RuntimeError("persisted rollout includes an ineligible or changed Train task")
        key = (row["task_id"], int(row["sample_index"]))
        if key in rollout_by_key:
            raise RuntimeError("duplicate persisted rollout identity")
        if row.get("trusted_contract") is not True or \
                row.get("grader", {}).get("verifier_id") != runner.verifier_id(task):
            raise RuntimeError("persisted reward does not carry the locked parent-oracle identity")
        if len(row.get("grader", {}).get("case_results", [])) != 5:
            raise RuntimeError("persisted rollout lacks the five parent-owned cases")
        failures = {case.get("failure") for case in row["grader"]["case_results"]}
        if failures & {"protocol_mismatch", "container_start_failure", "container_start_timeout"}:
            raise RuntimeError("persisted rollout contains an invalid grader-runtime failure")
        if row.get("reward") not in (0.0, 1.0):
            raise RuntimeError("persisted reward is outside the locked binary reward contract")
        rollout_by_key[key] = row

    samples: list[grpo_mlx.Sample] = []
    seen = set()
    for row in serialized:
        key = (str(row["instance_id"]), int(row["sample_index"]))
        rollout = rollout_by_key.get(key)
        task = tasks_by_id.get(key[0])
        if rollout is None or task is None or key in seen:
            raise RuntimeError("serialized sample identities do not match the raw rollout set")
        seen.add(key)
        if row.get("task_sha256") != task.sha256() or row.get("phase") != "train-step-0":
            raise RuntimeError("serialized sample task provenance is invalid")
        if float(row["reward"]) != float(rollout["reward"]):
            raise RuntimeError("serialized sample reward differs from its parent-graded rollout")
        if row.get("verifier") != rollout.get("grader", {}).get("verifier_id"):
            raise RuntimeError("serialized sample verifier differs from its graded rollout")
        _assert_saved_response_matches_rollout(row, rollout)
        if row.get("candidate_source_sha256") != rollout.get("candidate_source_sha256"):
            raise RuntimeError("serialized sample source identity differs from its graded rollout")
        if row.get("behavior_checkpoint_id") != base["policy_checkpoint_id"] or int(row["behavior_step"]) != 0:
            raise RuntimeError("serialized sample does not belong to the base behavior policy")
        if row.get("behavior_logprobs") is not None:
            raise RuntimeError("pre-update source samples must not already contain learner log-probabilities")
        samples.append(runner.deserialize_sample(row))
    if seen != set(rollout_by_key):
        raise RuntimeError("one or more persisted rollouts lack a serialized learner sample")
    groups = runner.informative_group_stats(samples)
    if groups["n_groups"] != int(protocol["selection"]["group_budget"]):
        raise RuntimeError("persisted sample groups differ from the frozen six-group budget")
    if groups["n_informative_groups"] == 0:
        # The no-update case is valid, but the update command will record and stop.
        pass
    return serialized, samples, {"base_policy": base, "run_meta": meta, "groups": groups}


def _existing_behavior_logprobs(run: Path, seed: int, protocol_sha: str,
                                samples: list[grpo_mlx.Sample]) -> bool:
    path = run / "preupdate-behavior-logprobs.json"
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("payload_sha256", None)
    if recorded != sha256_json(payload) or payload.get("run_protocol_sha256") != protocol_sha:
        raise RuntimeError("persisted behavior-logprob continuation is corrupt or belongs to another lock")
    rows = payload.get("samples")
    if int(payload.get("seed", -1)) != seed or not isinstance(rows, list) or len(rows) != len(samples):
        raise RuntimeError("persisted behavior-logprob sample count or seed changed")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("instance_id"), str):
            raise RuntimeError("persisted behavior-logprob sample identity is invalid")
        try:
            key = (row["instance_id"], int(row["sample_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("persisted behavior-logprob sample identity is invalid") from exc
        if key in by_key:
            raise RuntimeError("persisted behavior-logprob samples contain duplicate identities")
        by_key[key] = row

    informative_keys = {
        (sample.instance_id, sample.sample_index)
        for sample in grpo_mlx.informative_samples(samples, runner.GRPO_CONFIG)
    }
    for sample in samples:
        key = (sample.instance_id, sample.sample_index)
        row = by_key.get(key)
        if row is None or row.get("behavior_checkpoint_id") != sample.behavior_checkpoint_id:
            raise RuntimeError("persisted behavior-logprob identities do not cover the exact saved samples")
        logprobs = row.get("behavior_logprobs")
        if logprobs is None:
            if key in informative_keys:
                raise RuntimeError("persisted behavior log-probabilities are missing for an informative sample")
            continue
        if not isinstance(logprobs, list) or len(logprobs) != len(sample.completion_ids):
            raise RuntimeError("persisted behavior log-probabilities do not align with the saved completion")
        try:
            sample.behavior_logprobs = [float(value) for value in logprobs]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("persisted behavior log-probabilities are invalid") from exc
    if set(by_key) != {(sample.instance_id, sample.sample_index) for sample in samples}:
        raise RuntimeError("persisted behavior-logprob identities differ from the complete saved batch")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-protocol", type=Path, required=True)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    run_protocol_path = args.run_protocol.expanduser().resolve()
    recovery_protocol_path = args.recovery_protocol.expanduser().resolve()
    lock, protocol = _verify_recovery_lock(recovery_protocol_path, run_protocol_path, run)

    artifacts = {
        "run_meta.json": run / "run_meta.json",
        "base-policy.json": run / "base-policy.json",
        "rollouts-train-step-0.jsonl": run / "rollouts-train-step-0.jsonl",
        "training-samples-before-logprob.json": run / "training-samples-before-logprob.json",
        "scheduler-selection.json": run / "scheduler-selection.json",
        "task_manifest.json": run / "task_manifest.json",
        "trajectory-store.sqlite3": run / "trajectory-store.sqlite3",
    }
    input_hashes = {name: sha256_file(path) for name, path in artifacts.items()}
    if input_hashes != lock.get("input_artifact_sha256"):
        raise RuntimeError("pre-update source artifacts differ from the recovery lock")
    serialized, samples, details = verify_saved_batch(run, lock, protocol)
    if args.validate_only:
        print(json.dumps({
            "validated": True,
            "run_protocol_sha256": sha256_file(run_protocol_path),
            "recovery_protocol_sha256": sha256_file(recovery_protocol_path),
            "sample_count": len(samples),
            "group_count": details["groups"]["n_groups"],
            "resource": runner.resource_snapshot(run),
        }, sort_keys=True), flush=True)
        return 0

    try:
        preflight = runner.preflight(run, "preupdate-resume")
        started = time.monotonic()
        cpu_started = runner.process_cpu_seconds()
        import mlx.core as mx
        import mlx.optimizers as optim

        mx.reset_peak_memory()
        model, tokenizer = runner.load_policy(seed=int(lock["seed"]))
        base = details["base_policy"]
        if grpo_mlx.trainable_digest(model) != base["trainable_parameters_sha256"]:
            raise RuntimeError("loaded base policy parameters differ from the persisted rollout policy")
        policy_id = runner.make_policy_id(model, 0)
        if policy_id != base["policy_checkpoint_id"]:
            raise RuntimeError("loaded base policy checkpoint identity differs from saved samples")
        optimizer = optim.Adam(learning_rate=runner.OPTIMIZER_CONFIG["learning_rate"])
        groups = details["groups"]
        if groups["n_informative_groups"] == 0:
            result = {
                "update_applied": False,
                "reason": "no_informative_group",
                "update_step": 0,
                "groups": groups,
                "params_changed": False,
                "recovered_from_preupdate_batch": True,
                "recovery_protocol_sha256": sha256_file(recovery_protocol_path),
                "resource": {"preflight": preflight, "end_snapshot": runner.resource_snapshot(run)},
            }
            runner.atomic_json(run / "training-result.json", result)
            return 0

        informative = grpo_mlx.informative_samples(samples, runner.GRPO_CONFIG)
        if not _existing_behavior_logprobs(run, int(lock["seed"]), sha256_file(run_protocol_path), samples):
            mx.clear_cache()
            with runner.gradient_checkpointing(model):
                grpo_mlx.compute_behavior_logprobs(
                    model, informative, runner.GRPO_CONFIG,
                    after_sample=lambda: _clear_mlx_cache_then_guard(
                        mx, lambda: runner.resource_guard(run, started)),
                )
            payload = {
                "schema": "vare_ig_preupdate_behavior_logprobs_v1",
                "run_protocol_sha256": sha256_file(run_protocol_path),
                "recovery_protocol_sha256": sha256_file(recovery_protocol_path),
                "seed": int(lock["seed"]),
                "base_policy_checkpoint_id": policy_id,
                "base_policy_parameters_sha256": grpo_mlx.trainable_digest(model),
                "samples": [runner.serialize_sample(
                    sample,
                    response={"text": row["response_text"],
                              "candidate_source_sha256": row.get("candidate_source_sha256"),
                              "finish": row.get("finish")},
                    task_sha256=row["task_sha256"], phase=row["phase"],
                ) for sample, row in zip(samples, serialized, strict=True)],
            }
            payload["payload_sha256"] = sha256_json(payload)
            atomic_json(run / "preupdate-behavior-logprobs.json", payload)
        else:
            # Recompute gradients only from the exact committed sample payload.
            pass

        mx.clear_cache()
        mx.reset_peak_memory()
        update_cpu_started = runner.process_cpu_seconds()
        with runner.gradient_checkpointing(model):
            update = grpo_mlx.train_step(
                model, optimizer, samples, runner.GRPO_CONFIG,
                policy_checkpoint_id=policy_id, policy_step=0,
                after_sample=lambda: _clear_mlx_cache_then_guard(
                    mx, lambda: runner.resource_guard(run, started)),
            )
        mx.eval(model.parameters(), optimizer.state)
        update_done = time.monotonic()
        if not update.get("update_applied") or not update.get("params_changed"):
            raise RuntimeError("verified informative batch did not produce a changed LoRA update")
        if int(optimizer.state["step"].item()) != 1:
            raise RuntimeError("initial optimizer step count is not exactly one")

        runner.atomic_json(run / "training-samples.json", [
            runner.serialize_sample(
                sample,
                response={"text": row["response_text"],
                          "candidate_source_sha256": row.get("candidate_source_sha256"),
                          "finish": row.get("finish")},
                task_sha256=row["task_sha256"], phase=row["phase"],
            ) for sample, row in zip(samples, serialized, strict=True)
        ])
        update["resource"] = {
            "preflight": preflight,
            "peak_mlx_gib": mx.get_peak_memory() / runner.GIB,
            "peak_rss_gib": runner.resource_snapshot(run)["peak_rss_gib"],
            "cpu_seconds": runner.process_cpu_seconds() - update_cpu_started,
            "phase_cpu_seconds": runner.process_cpu_seconds() - cpu_started,
            "wall_seconds": update_done - started,
            "end_snapshot": runner.resource_snapshot(run),
        }
        update["groups"] = groups
        update["training_task_count"] = len(tasks.TRAIN_TASKS)
        update["generation_count"] = len(samples)
        update["recovered_from_preupdate_batch"] = True
        update["recovery_protocol_sha256"] = sha256_file(recovery_protocol_path)
        update["persisted_rollouts_sha256"] = sha256_file(run / "rollouts-train-step-0.jsonl")
        checkpoint = runner.checkpoint_path(run, "update-0001")
        meta = details["run_meta"]
        manifest = grpo_mlx.save_checkpoint(
            checkpoint, model, optimizer, step=1, base_seed=int(lock["seed"]),
            extra={"experiment_id": meta["experiment_id"], "model_revision": runner.MODEL_REVISION,
                   "policy_checkpoint_id_before": policy_id, "task_hashes": meta["task_hashes"],
                   "optimizer": runner.OPTIMIZER_CONFIG, "lora": runner.LORA_CONFIG,
                   "attempt_index": 1, "informative_groups": groups["n_informative_groups"],
                   "recovered_from_preupdate_batch": True,
                   "recovery_protocol_sha256": sha256_file(recovery_protocol_path)},
        )
        sentinel = runner.rng_sentinel(int(lock["seed"]), 1)
        state = runner.write_checkpoint_state(run, manifest=manifest, optimizer=optimizer,
                                              model=model, seed=int(lock["seed"]),
                                              applied_step=1, sentinel=sentinel)
        update["checkpoint"] = state
        update["params_digest_after"] = grpo_mlx.trainable_digest(model)
        update["optimizer_step_after"] = int(optimizer.state["step"].item())
        runner.atomic_json(run / "training-result.json", update)
        event = [{
            "event": "optimizer_update_committed",
            "update_step": 1,
            "checkpoint_state_sha256": sha256_file(run / "checkpoint-state.json"),
            "parameter_sha256": update["params_digest_after"],
            "optimizer_sha256": state["optimizer_state_sha256"],
            "recovery_protocol_sha256": sha256_file(recovery_protocol_path),
            "time": time.time(),
        }]
        runner.atomic_json(run / "learner-events.json", event)
        acknowledge_durable_decision(
            run, update_step=1, acknowledgment_path=run / "learner-events.json",
            decision={"update_applied": True, "effective_update_count": 1,
                      "event": "optimizer_update_commit_acknowledged_after_preupdate_recovery"},
        )
        print(json.dumps({
            "update_applied": True,
            "update_step": 1,
            "informative_groups": groups["n_informative_groups"],
            "parameter_sha256": update["params_digest_after"],
            "optimizer_sha256": state["optimizer_state_sha256"],
            "checkpoint": str(checkpoint),
            "recovery_protocol_sha256": sha256_file(recovery_protocol_path),
        }, sort_keys=True), flush=True)
        return 0
    except runner.ResourceStop as exc:
        detail = {"phase": "preupdate-resume", "reason": str(exc),
                  "resource": runner.resource_snapshot(run), "time": time.time(),
                  "recovery_protocol_sha256": sha256_file(recovery_protocol_path)}
        path = run / "resource-stop-preupdate-resume.json"
        if not path.exists():
            runner.atomic_json(path, detail)
        print(json.dumps({"resource_stop": detail}, sort_keys=True), flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
