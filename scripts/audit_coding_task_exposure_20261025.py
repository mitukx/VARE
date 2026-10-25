"""Reconstruct the coding-task readiness stop from immutable retained records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CATALOG_REVISION = "acca07004eb9a44cf23d31ff0d68929b7ea3fc1d"
EXPOSURE_REVISION = "02684dcce4c0a5474256b1d97a2adbda879388d2"
INPUTS = {
    "results/mlsys-task-deep-qualification-v1/qualification.json": "2b2a1a7c6e64a866dd3732fcd9723999969a1ac69d86f8a4d9f35edf67322c9d",
    "results/mlsys-task-deep-qualification-v1/protocol.json": "7ef9a28cedf2e1214a74456bb0c5209afc3e3e9f1681a430398e84710bcd26a4",
    "protocols/mlsys_actual_model_update_no_go_v1.lock.json": "12db5f0a7200ce64de3801781e45b41d3667bd80a405ec8c4dc23fea8f550081",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def blob(revision: str, path: str) -> bytes:
    env = {**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"}
    return subprocess.check_output(
        ["git", "-c", "core.useReplaceRefs=false", "show", f"{revision}:{path}"],
        cwd=ROOT, env=env,
    )


def audit() -> dict:
    inputs = {}
    for path, expected in INPUTS.items():
        raw = blob(CATALOG_REVISION, path)
        if digest(raw) != expected:
            raise ValueError(f"frozen audit input mismatch: {path}")
        inputs[path] = json.loads(raw)
    catalog = inputs["results/mlsys-task-deep-qualification-v1/qualification.json"]
    split = inputs["results/mlsys-task-deep-qualification-v1/protocol.json"]["split"]
    gate = inputs["protocols/mlsys_actual_model_update_no_go_v1.lock.json"]["readiness_gate"]
    tasks = catalog["tasks"]
    if len(tasks) != 15 or len({t["task_id"] for t in tasks}) != 15:
        raise ValueError("source inventory cardinality changed")
    for task in tasks:
        if task["task_id"] not in split[task["split"]]:
            raise ValueError("source split and qualification disagree")
        for path, expected in task["raw_log_sha256"].items():
            if digest(blob(CATALOG_REVISION, "results/mlsys-task-deep-qualification-v1/" + path)) != expected:
                raise ValueError(f"source calibration log mismatch: {path}")

    consumed = next(t for t in tasks if t["task_id"] == "transformers-49412-fusion-idempotence")
    task_path = "benchmarks/mlsys_repo_eval_v4/transformers_49412_fusion_idempotence_v2/task.json"
    task_raw = blob(EXPOSURE_REVISION, task_path)
    exposed_task = json.loads(task_raw)
    if (exposed_task["source"]["repository"] != consumed["repository"]
            or exposed_task["source"]["base_revision"] != consumed["baseline_revision"]):
        raise ValueError("exposed alias is not bound to the original source task baseline")
    receipts = []
    for prefix, seeds in (("", (201, 202, 203)), ("v3-", (401, 402, 403)), ("v4-", (601, 602, 603))):
        for seed in seeds:
            path = f"results/transformers-phase-a-structured-v1/rollouts/49412-{prefix}seed-{seed}.json"
            raw = blob(EXPOSURE_REVISION, path)
            row = json.loads(raw)
            if (row["task_id"] != exposed_task["task_id"]
                    or row["task_revisions"]["base"] != consumed["baseline_revision"]
                    or row["seed"] != seed or row["new_tokens"] <= 0
                    or not row["model_repository"] or not row["model_snapshot"]
                    or not row["initial_messages"] or not row["transcript"]):
                raise ValueError(f"model exposure receipt is incomplete: {path}")
            receipts.append({
                "path": path, "sha256": digest(raw), "seed": seed,
                "model_repository": row["model_repository"],
                "model_snapshot": row["model_snapshot"], "new_tokens": row["new_tokens"],
                "task_id": row["task_id"], "baseline_revision": row["task_revisions"]["base"],
            })
    heldout = split["heldout"]
    if len(heldout) != 9 or consumed["task_id"] not in heldout:
        raise ValueError("unexpected original held-out allocation")
    remaining = [task for task in heldout if task != consumed["task_id"]]
    return {
        "audit_id": "coding-task-exposure-20261025", "catalog_revision": CATALOG_REVISION,
        "exposure_revision": EXPOSURE_REVISION, "input_sha256": INPUTS,
        "source_calibration_log_hashes_verified": sum(len(t["raw_log_sha256"]) for t in tasks),
        "source_task_count": len(tasks), "original_split": split,
        "exposed_reserved_task": consumed["task_id"],
        "exposed_task_alias": exposed_task["task_id"],
        "exposed_task_definition": {"path": task_path, "sha256": digest(task_raw)},
        "model_exposure_receipts": receipts,
        "remaining_reserved_task_ids": remaining,
        "remaining_reserved_ceiling_before_further_qualification": len(remaining),
        "eligible_heldout_tasks_admitted_by_this_audit": 0,
        "remaining_task_status": "UNKNOWN: source-level reproduction is not a complete current isolated-grader, exposure, and environment certificate",
        "required_eligible_heldout_tasks": gate["required_unique_eligible_heldout_tasks"],
        "decision": "NO_GO", "model_inference": "NOT_RUN", "training": "NOT_RUN",
        "scope": "Finite retained-record audit. No absence-of-exposure claim outside inspected artifacts; no hidden label access or new task outcome.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in (
        "source_calibration_log_hashes_verified", "remaining_reserved_ceiling_before_further_qualification",
        "eligible_heldout_tasks_admitted_by_this_audit", "required_eligible_heldout_tasks", "decision",
    )}))


if __name__ == "__main__":
    main()
