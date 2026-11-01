#!/usr/bin/env python3
"""Run a locked paired Development evaluation for a recovered VARE-IG update 1."""
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

from vare import grpo_mlx  # noqa: E402
from vare.learning_pipeline import vare_ig_tasks as task_pack  # noqa: E402
from scripts import qwen3_function_repair_grpo as runner  # noqa: E402
from scripts import qwen3_vare_ig_experiment as experiment  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_paths(run: Path) -> dict[str, Path]:
    return {
        "run_meta.json": run / "run_meta.json",
        "base-policy.json": run / "base-policy.json",
        "training-result.json": run / "training-result.json",
        "checkpoint-state.json": run / "checkpoint-state.json",
        "learner-events.json": run / "learner-events.json",
        "development-pre.jsonl": run / "development-pre.jsonl",
        "checkpoint.json": run / "checkpoints/update-0001/checkpoint.json",
        "adapters.safetensors": run / "checkpoints/update-0001/adapters.safetensors",
        "optimizer.safetensors": run / "checkpoints/update-0001/optimizer.safetensors",
    }


def validate_lock(lock_path: Path, run_protocol_path: Path, run: Path) -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema") != "vare_ig_paired_development_evaluation_v1":
        raise RuntimeError("wrong paired-evaluation lock schema")
    if lock.get("evaluation_helper_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("paired-evaluation helper changed after its lock was frozen")
    if lock.get("run_protocol_sha256") != sha256_file(run_protocol_path):
        raise RuntimeError("paired-evaluation lock targets another run protocol")
    recovery_path = Path(lock["recovery_protocol_path"]).expanduser().resolve()
    if lock.get("recovery_protocol_sha256") != sha256_file(recovery_path):
        raise RuntimeError("paired-evaluation lock targets another recovery protocol")
    protocol = experiment.validate_protocol(run_protocol_path, run_seed=int(lock["seed"]))
    experiment._install_existing_runner_contract(run_protocol_path)
    runner.protocol_payload(run_protocol_path, seed=int(lock["seed"]))

    paths = _input_paths(run)
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if actual_hashes != lock.get("input_artifact_sha256"):
        raise RuntimeError("update-1 checkpoint or baseline artifacts differ from the evaluation lock")
    meta = json.loads(paths["run_meta.json"].read_text(encoding="utf-8"))
    base = json.loads(paths["base-policy.json"].read_text(encoding="utf-8"))
    result = json.loads(paths["training-result.json"].read_text(encoding="utf-8"))
    state = json.loads(paths["checkpoint-state.json"].read_text(encoding="utf-8"))
    events = json.loads(paths["learner-events.json"].read_text(encoding="utf-8"))
    if meta.get("protocol_sha256") != lock["run_protocol_sha256"]:
        raise RuntimeError("saved run metadata differs from the frozen training protocol")
    if meta.get("source_hashes") != runner.source_hashes() or meta.get("task_hashes") != runner.task_hashes():
        raise RuntimeError("training sources or task contracts differ from the saved run")
    if base.get("model_revision") != runner.MODEL_REVISION:
        raise RuntimeError("saved base policy uses another model revision")
    if not (result.get("update_applied") is True and result.get("params_changed") is True
            and result.get("optimizer_step_after") == 1
            and result.get("recovered_from_preupdate_batch") is True):
        raise RuntimeError("the saved run does not contain the required real update-1")
    if state.get("update_step") != 1 or state.get("optimizer_step") != 1:
        raise RuntimeError("checkpoint state is not exactly optimizer step 1")
    if state.get("trainable_parameters_sha256") != result.get("params_digest_after"):
        raise RuntimeError("checkpoint parameters differ from the training result")
    if base.get("trainable_parameters_sha256") == state.get("trainable_parameters_sha256"):
        raise RuntimeError("saved update-1 checkpoint did not change the base policy parameters")
    if len(events) != 1 or events[0].get("event") != "optimizer_update_committed" \
            or events[0].get("update_step") != 1 \
            or events[0].get("checkpoint_state_sha256") != sha256_file(paths["checkpoint-state.json"]):
        raise RuntimeError("update-1 has no matching durable learner acknowledgment")
    dev_hashes = {task.task_id: task.sha256() for task in task_pack.DEVELOPMENT_TASKS}
    if lock.get("development_task_hashes") != dev_hashes:
        raise RuntimeError("Development cohort differs from the prospective evaluation lock")
    if lock.get("evaluation_settings") != {
            "temperature": 0.0,
            "max_new_tokens": runner.DEVELOPMENT_MAX_NEW_TOKENS,
            "decode": "greedy",
            "paired_seed_schedule": "seed_for(run_seed, 'paired-step1', task_id, task_index)",
            "metric": "task-level binary pass@1, four fixed Development tasks; exploratory only",
    }:
        raise RuntimeError("paired-evaluation settings differ from the frozen contract")
    return lock, protocol, base, state


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "passed": sum(row["reward"] == 1 for row in rows),
        "task_results": [{"task_id": row["task_id"], "reward": row["reward"],
                          "response_sha256": row["response_sha256"]} for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-protocol", type=Path, required=True)
    parser.add_argument("--evaluation-protocol", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    run_protocol_path = args.run_protocol.expanduser().resolve()
    eval_lock_path = args.evaluation_protocol.expanduser().resolve()
    lock, _protocol, base, state = validate_lock(eval_lock_path, run_protocol_path, run)
    root = run / lock["output_directory"]
    base_dir, trained_dir = root / "base", root / "update-0001"
    adapter_dir = root / "adapter-update-0001"
    summary_path = root / "paired-summary.json"
    if args.validate_only:
        print(json.dumps({
            "validated": True,
            "evaluation_protocol_sha256": sha256_file(eval_lock_path),
            "run_protocol_sha256": sha256_file(run_protocol_path),
            "development_task_count": len(task_pack.DEVELOPMENT_TASKS),
            "resource": runner.resource_snapshot(run),
            "outputs_absent": not root.exists(),
        }, sort_keys=True), flush=True)
        return 0
    if root.exists():
        raise FileExistsError(f"refusing to overwrite paired-evaluation evidence: {root}")
    root.mkdir(parents=True)
    base_dir.mkdir()
    trained_dir.mkdir()

    import mlx.core as mx
    import mlx.optimizers as optim

    original_guard = runner.resource_guard

    def release_cache_then_guard(path: Path, phase_started: float) -> dict[str, Any]:
        mx.clear_cache()
        return original_guard(path, phase_started)

    runner.resource_guard = release_cache_then_guard
    seed = int(lock["seed"])
    try:
        base_preflight = runner.preflight(base_dir, "paired-development-base")
        started = time.monotonic()
        base_model, base_tokenizer = runner.load_policy(seed=seed)
        if grpo_mlx.trainable_digest(base_model) != base["trainable_parameters_sha256"]:
            raise RuntimeError("reloaded base model parameters differ from the training baseline")
        base_rows = runner.development_evaluation(
            base_model, base_tokenizer, phase="paired-step1", seed=seed,
            run=base_dir, phase_started=started,
        )
        del base_model, base_tokenizer
        gc.collect()
        mx.clear_cache()

        trained_preflight = runner.preflight(trained_dir, "paired-development-update-0001")
        started = time.monotonic()
        trained_model, trained_tokenizer = runner.load_policy(seed=seed)
        optimizer = optim.Adam(learning_rate=runner.OPTIMIZER_CONFIG["learning_rate"])
        manifest = grpo_mlx.load_checkpoint(
            run / "checkpoints/update-0001", trained_model, optimizer)
        if int(manifest["step"]) != 1 or int(optimizer.state["step"].item()) != 1:
            raise RuntimeError("loaded update-1 checkpoint is not at optimizer step 1")
        if grpo_mlx.trainable_digest(trained_model) != state["trainable_parameters_sha256"]:
            raise RuntimeError("loaded update-1 checkpoint parameters differ from its state manifest")
        trained_rows = runner.development_evaluation(
            trained_model, trained_tokenizer, phase="paired-step1", seed=seed,
            run=trained_dir, phase_started=started,
        )
        adapter_digest = runner.export_adapter(run / "checkpoints/update-0001", adapter_dir)
        adapter_task = task_pack.DEVELOPMENT_TASKS[0]
        adapter_reference = next(row for row in trained_rows if row["task_id"] == adapter_task.task_id)
        del trained_model, trained_tokenizer, optimizer
        gc.collect()
        mx.clear_cache()

        adapter_preflight = runner.preflight(trained_dir, "paired-development-adapter-reload")
        reload_started = time.monotonic()
        adapter_model, adapter_tokenizer = runner.load_policy(seed=seed, adapter_path=adapter_dir)
        release_cache_then_guard(trained_dir, reload_started)
        generation = runner.generate(
            adapter_model, adapter_tokenizer, runner.task_prompt(adapter_task),
            seed=runner.seed_for(seed, "paired-step1", adapter_task.task_id, 0),
            temperature=0.0, max_tokens=runner.DEVELOPMENT_MAX_NEW_TOKENS,
        )
        adapter_score = runner.score_source(adapter_task, generation["text"])
        adapter_response_sha = hashlib.sha256(generation["text"].encode("utf-8")).hexdigest()
        adapter_check = {
            "task_id": adapter_task.task_id,
            "adapter_directory_sha256": adapter_digest,
            "checkpoint_adapter_sha256": sha256_file(run / "checkpoints/update-0001/adapters.safetensors"),
            "checkpoint_loaded_response_sha256": adapter_reference["response_sha256"],
            "adapter_loaded_response_sha256": adapter_response_sha,
            "response_exact_match": adapter_response_sha == adapter_reference["response_sha256"],
            "checkpoint_reward": adapter_reference["reward"],
            "adapter_reward": adapter_score["reward"],
            "reward_exact_match": adapter_score["reward"] == adapter_reference["reward"],
            "generation_seconds": generation["seconds"],
        }
        if not adapter_check["response_exact_match"] or not adapter_check["reward_exact_match"]:
            raise RuntimeError("exported adapter does not reproduce the checkpoint-loaded Development output")
        del adapter_model, adapter_tokenizer
        gc.collect()
        mx.clear_cache()

        summary = {
            "schema": "vare_ig_paired_development_evaluation_v1",
            "run_protocol_sha256": sha256_file(run_protocol_path),
            "evaluation_protocol_sha256": sha256_file(eval_lock_path),
            "model_revision": runner.MODEL_REVISION,
            "development_split": "Development; exploratory, not held-out",
            "development_task_hashes": lock["development_task_hashes"],
            "base": _summarize(base_rows),
            "update_0001": _summarize(trained_rows),
            "adapter_reload_check": adapter_check,
            "parameter_sha256_before": base["trainable_parameters_sha256"],
            "parameter_sha256_after": state["trainable_parameters_sha256"],
            "optimizer_step": state["optimizer_step"],
            "adapter_path": str(adapter_dir),
            "adapter_sha256": adapter_digest,
            "resource": {
                "base_preflight": base_preflight,
                "update_preflight": trained_preflight,
                "adapter_reload_preflight": adapter_preflight,
                "end_snapshot": runner.resource_snapshot(run),
            },
        }
        runner.atomic_json(summary_path, summary)
        print(json.dumps({"base": summary["base"], "update_0001": summary["update_0001"],
                          "adapter_reload_check": adapter_check,
                          "adapter_sha256": adapter_digest}, sort_keys=True), flush=True)
        return 0
    except runner.ResourceStop as exc:
        diagnostic = {
            "phase": "paired-update-1-evaluation",
            "reason": str(exc),
            "resource": runner.resource_snapshot(run),
            "evaluation_protocol_sha256": sha256_file(eval_lock_path),
            "time": time.time(),
        }
        runner.atomic_json(root / "resource-stop.json", diagnostic)
        print(json.dumps({"resource_stop": diagnostic}, sort_keys=True), flush=True)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
