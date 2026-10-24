"""Independently audit a v7 run stopped by its preregistered pilot gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def word(text: str, target: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(target)}(?![A-Za-z0-9_])",
                          text, flags=re.IGNORECASE))


def score(rule: str, text: str) -> float:
    if rule == "ground_word":
        return float(word(text, "ground"))
    if rule == "punctuation_or_location":
        return float(bool(re.search(r"[.!?]", text)) or any(
            word(text, value) for value in ("ground", "grass", "floor", "sky", "branch")))
    if rule == "exact_ground_period":
        return float(bool(re.fullmatch(r"\s*ground\.\s*", text, flags=re.IGNORECASE)))
    if rule == "not_ground_word":
        return float(not word(text, "ground"))
    raise ValueError(f"unknown verifier rule: {rule}")


def verify_inventory(root: Path) -> dict:
    manifest_path = root / "SHA256.json"
    manifest = json.loads(manifest_path.read_text())
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256.json"
              and not path.name.endswith(("-wal", "-shm"))}
    require(actual == set(manifest), "partial-run inventory is incomplete or has extra files")
    for relative, digest in manifest.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                f"partial-run file hash mismatch: {relative}")
    return manifest


def audit(run: Path, protocol_path: Path, runner_path: Path) -> dict:
    protocol = json.loads(protocol_path.read_text())
    copied_protocol = run / "protocol.json"
    root_manifest = verify_inventory(run)
    require(sha(copied_protocol) == sha(protocol_path),
            "partial run used a different frozen protocol")
    require(protocol["schema"] == "vare.verifier-drift-scenarios.v7.3"
            and sha(runner_path) == protocol["runner_sha256"],
            "runner or amended protocol identity differs")

    failure = json.loads((run / "failure.json").read_text())
    root_progress = json.loads((run / "progress.json").read_text())
    arm_dir = run / "false_accept/sync"
    arm_progress = json.loads((arm_dir / "progress.json").read_text())
    raw = json.loads((arm_dir / "rollouts.json").read_text())
    require(failure == {
        "type": "RuntimeError",
        "message": "preregistered stop: degenerate GRPO group false_accept/sync/2",
        "time": failure["time"],
    }, "run did not stop at the exact preregistered degeneracy condition")
    require(root_progress["status"] == "RUNNING"
            and root_progress["completed_scenarios"] == []
            and arm_progress["scenario"] == "false_accept"
            and arm_progress["arm"] == "sync"
            and arm_progress["generated_groups"] == 2
            and arm_progress["optimizer_updates"] == 2,
            "partial progress record differs from the stopped sync pilot")
    require(raw["scenario"] == "false_accept" and raw["arm"] == "sync"
            and len(raw["groups"]) == 3,
            "expected three generated groups in the stopped pilot")

    group_reports = []
    for expected_index, group in enumerate(raw["groups"]):
        group_id = f"g-{expected_index:02d}-false_accept"
        require(group["group_id"] == group_id and len(group["members"]) == 8,
                f"wrong group identity/cardinality: {group_id}")
        measured, authoritative = [], []
        for member in group["members"]:
            text = member["attempt"]["output"]
            version = member["verifier_version"]
            require(version == 1, "synchronous arm unexpectedly used verifier v0")
            label = score(protocol["scenarios"][0]["verifiers"]["v1"]["rule"], text)
            require(float(member["score"]) == label,
                    f"training reward cannot be independently reproduced: {group_id}")
            measured.append(label)
            authoritative.append(label)
        require(group["scores"] == measured,
                f"group label vector mismatch: {group_id}")
        require(group["behavior_policy"]["policy_version"] == expected_index,
                f"policy version did not advance synchronously for {group_id}")
        group_reports.append({
            "group_id": group_id,
            "members": len(measured),
            "verifier_calls": len(measured),
            "reward_vector": measured,
            "authoritative_accepts": int(sum(authoritative)),
            "disagreement_members": 0,
            "reward_degenerate": len(set(measured)) == 1,
            "optimizer_trained": expected_index < 2,
            "outputs": [member["attempt"]["output"] for member in group["members"]],
        })
    require([row["reward_degenerate"] for row in group_reports] == [False, False, True]
            and group_reports[2]["reward_vector"] == [0.0] * 8,
            "observed degeneracy does not match the recorded stop reason")

    events = [json.loads(line) for line in (arm_dir / "events.jsonl").read_text().splitlines()]
    promoted = [event for event in events if event.get("event") == "candidate_promoted"]
    completed = [event for event in events if event.get("event") == "round_complete"]
    require(len(promoted) == len(completed) == 2
            and [row["candidate_id"] for row in promoted] == ["policy-1", "policy-2"],
            "promotion/event evidence does not show exactly two completed updates")

    import site
    for entry in site.getsitepackages():
        if entry not in sys.path:
            sys.path.append(entry)
    import torch
    checkpoint_steps = {}
    for policy, expected_step in (("policy-0", 0), ("policy-1", 1), ("policy-2", 2)):
        path = arm_dir / f"checkpoints/{policy}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(set(state) == {"model", "optimizer", "cpu_rng", "module_modes"},
                f"checkpoint omits full trainer state: {policy}")
        steps = sorted({int(value["step"].item()) for value in state["optimizer"]["state"].values()
                        if "step" in value})
        require((not steps and expected_step == 0)
                or steps == [expected_step], f"Adam step count differs in {policy}: {steps}")
        checkpoint_steps[policy] = expected_step

    with sqlite3.connect(arm_dir / "trajectories.sqlite") as db:
        verifier_rows = db.execute(
            "SELECT name,version,config FROM verifiers ORDER BY name,version").fetchall()
        learner_verifier = json.loads(db.execute(
            "SELECT value FROM meta WHERE key='learner_verifier'").fetchone()[0])
        ledger_rows = db.execute(
            "SELECT group_id,state,count(*) FROM trajectories GROUP BY group_id,state"
        ).fetchall()
    expected_v1 = sha_json({"scenario": "false_accept", "version": 1,
                            "rule": "ground_word"})
    require(("landing", 1, expected_v1) in verifier_rows
            and learner_verifier == ["landing", 1, expected_v1]
            and ledger_rows == [("g-00-false_accept", "consumed", 8),
                                ("g-01-false_accept", "consumed", 8)],
            "trajectory ledger does not bind the two trained groups to verifier v1")

    return {
        "status": "PASS_PARTIAL_STOP_AUDIT",
        "decision": "NO_GO_FOR_ASYNC_COMPARISON",
        "protocol_schema": protocol["schema"],
        "protocol_sha256": sha(copied_protocol),
        "runner_sha256": sha(runner_path),
        "root_inventory_sha256": sha(run / "SHA256.json"),
        "root_inventory_files": len(root_manifest),
        "partial_run_status": root_progress["status"],
        "stop_reason": failure["message"],
        "scenario_arms_started": ["false_accept/sync"],
        "groups_generated": 3,
        "model_completions": 24,
        "verifier_calls": 24,
        "actual_optimizer_commits": 2,
        "effective_trajectories_consumed": 16,
        "checkpoint_adam_steps": checkpoint_steps,
        "groups": group_reports,
        "resource_observations": {
            "last_arm_progress_elapsed_seconds": arm_progress["elapsed_s"],
            "peak_rss_bytes": arm_progress["peak_rss_bytes"],
        },
        "not_measured": [
            "naive_async or VARE async arms",
            "stale policy/verifier queue admission",
            "held-out quality",
            "SIGKILL recovery",
            "coding-task performance",
        ],
        "claim_limit": (
            "The frozen synthetic pilot produced two genuine optimizer commits, then stopped on an all-zero third group. "
            "The asynchronous verifier-drift comparison was not reached; this does not support a VARE improvement claim."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run, args.protocol, args.runner)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
