"""Independently audit sustained VARE learner SIGKILL recovery artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inventory(root: Path) -> dict:
    manifest = json.loads((root / "SHA256.json").read_text())
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256.json"
              and not path.name.endswith(("-wal", "-shm"))}
    require(actual == set(manifest), "artifact inventory is incomplete or has extra files")
    for relative, digest in manifest.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                f"artifact hash mismatch: {relative}")
    return manifest


def equal_state(actual, expected, torch, path="state") -> None:
    if isinstance(actual, torch.Tensor):
        require(isinstance(expected, torch.Tensor)
                and actual.dtype == expected.dtype
                and actual.shape == expected.shape
                and torch.equal(actual.cpu(), expected.cpu()),
                f"recovered tensor differs: {path}")
    elif isinstance(actual, dict):
        require(isinstance(expected, dict) and actual.keys() == expected.keys(),
                f"recovered state keys differ: {path}")
        for key in actual:
            equal_state(actual[key], expected[key], torch, f"{path}.{key}")
    elif isinstance(actual, (tuple, list)):
        require(isinstance(expected, type(actual)) and len(actual) == len(expected),
                f"recovered state sequence differs: {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            equal_state(left, right, torch, f"{path}[{index}]")
    else:
        require(type(actual) is type(expected) and actual == expected,
                f"recovered scalar differs: {path}")


def ledger_state(path: Path) -> dict:
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT state,count(*) FROM trajectories GROUP BY state"
        ).fetchall()
        group = db.execute(
            "SELECT group_id,state,count(*) FROM trajectories GROUP BY group_id,state"
        ).fetchall()
        verifiers = db.execute(
            "SELECT name,version,config FROM verifiers ORDER BY name,version"
        ).fetchall()
        verification_rows = db.execute(
            "SELECT verification FROM trajectories ORDER BY group_id,member"
        ).fetchall()
    return {"states": {state: int(count) for state, count in rows},
            "groups": group,
            "verifiers": verifiers,
            "verifications": [json.loads(row[0]) for row in verification_rows]}


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def audit(crash_output: Path, fixture_dir: Path, source_run: Path,
          source_audit: Path, protocol_path: Path) -> dict:
    import site
    for entry in site.getsitepackages():
        if entry not in sys.path:
            sys.path.append(entry)
    import torch

    protocol = json.loads(protocol_path.read_text())
    fixture = json.loads((fixture_dir / "fixture.json").read_text())
    summary = json.loads((crash_output / "summary.json").read_text())
    source_summary = json.loads((source_run / "summary.json").read_text())
    source_audit_report = json.loads(source_audit.read_text())
    source_inventory = inventory(source_run)
    fixture_inventory = inventory(fixture_dir)
    crash_inventory = inventory(crash_output)
    del fixture_inventory, crash_inventory

    require(source_audit_report.get("status") == "PASS"
            and fixture["source_run_protocol_sha256"] == sha(source_run / "protocol.json")
            and fixture["source_run_inventory_sha256"] == sha(source_run / "SHA256.json")
            and fixture["source_audit_report_sha256"] == sha(source_audit),
            "crash fixture does not bind to the independently audited v7 run")
    require(fixture["source_commit"] == source_summary["source_commit"]
            and source_summary["status"] == "COMPLETE"
            and len(source_inventory) > 0,
            "crash fixture source run is not complete")
    require(fixture["committed_prefix_updates"] == 2
            and fixture["target_update_number"] == 3,
            "fixture does not start after two committed updates")

    arm = source_run / "false_accept/vare_async"
    prefix_source = arm / "checkpoints" / f"{fixture['prefix_policy_id']}.pt"
    target_source = arm / "checkpoints" / f"{fixture['expected_candidate_policy_id']}.pt"
    require(sha(prefix_source) == fixture["prefix_checkpoint_sha256"]
            and sha(target_source) == fixture["expected_candidate_sha256"],
            "source prefix or reference candidate checkpoint hash differs")
    expected_prefix = torch.load(prefix_source, map_location="cpu", weights_only=True)
    fixture_prefix = torch.load(fixture_dir / "sync/checkpoints/policy-0.pt",
                                map_location="cpu", weights_only=True)
    equal_state(fixture_prefix, expected_prefix, torch, "two-update-prefix")
    expected_candidate = torch.load(target_source, map_location="cpu", weights_only=True)
    target_rollouts = json.loads((arm / "rollouts.json").read_text())
    target_group = next(group for group in target_rollouts["groups"]
                        if group["group_id"] == fixture["target_group_id"])
    target_verifier = fixture["target_verifier_identity"]
    expected_v7_verifier = {
        "name": "landing", "version": 1, "rule": "ground_word",
        "config_sha256": sha_json({
            "scenario": "false_accept", "version": 1, "rule": "ground_word"}),
    }
    require(target_verifier == expected_v7_verifier
            and fixture["target_verifier_versions"] == [1] * 8,
            "retained third-update group is not bound to the v7 authoritative verifier")
    expected_rewards = sorted(
        target_group["scores"], key=lambda value: 1.5 if value == 0.0 else 1.0,
        reverse=True)

    cutpoints = protocol["crash_recovery"]["cutpoints"]
    require([row["cutpoint"] for row in summary["cutpoints"]] == cutpoints
            and summary["process_kills"] == len(cutpoints)
            and summary["all_sigkill_verified"] is True,
            "not all preregistered SIGKILL cutpoints completed")

    attempts_expected = protocol["crash_recovery"]["optimizer_attempts_by_cutpoint"]
    cases = []
    for row in summary["cutpoints"]:
        cutpoint = row["cutpoint"]
        proof = row["marker"]
        worker = row["worker_summary"]
        case_dir = crash_output / cutpoint
        require(row["signal_returncode"] == -9 and row["restart_returncode"] == 0
                and row["old_process_alive"] is False
                and proof["cutpoint"] == cutpoint
                and proof["pid"] == row["pid"]
                and proof["pgid"] == row["process_group_id"],
                f"process-level SIGKILL proof failed at {cutpoint}")
        require(worker["committed_updates"] == 1
                and worker["effective_consumed_trajectories"] == 8
                and worker["optimizer_attempts"] == attempts_expected[cutpoint]
                and worker["store_audit"].get("pass") is True,
                f"transaction recovery count/audit failed at {cutpoint}")
        require(worker["promotion_journal"]["active_policy_id"] == "policy-1"
                and worker["promotion_journal"]["pending_tx_id"] is None
                and worker["store_counts"].get("consumed", 0) == 8,
                f"promotion or trajectory ledger state is unresolved at {cutpoint}")

        case_ledger = ledger_state(case_dir / "trajectory-snapshot.sqlite")
        require(case_ledger["states"] == {"consumed": 8}
                and case_ledger["groups"] == [("crash-group", "consumed", 8)],
                f"offline ledger replay did not show one effective consumption at {cutpoint}")
        legacy_v1_sha = sha_json({
            "name": "landing", "version": 1,
            "rule": "canonical ground semantics"})
        require(case_ledger["verifiers"] == [
            ("landing", 0, sha_json({
                "name": "landing", "version": 0,
                "rule": "punctuation completeness"})),
            ("landing", 1, legacy_v1_sha),
        ] and len(case_ledger["verifications"]) == 8
                and all(item["verifier_name"] == "landing"
                        and item["verifier_version"] == 1
                        and item["metadata"]["verifier_config_sha256"] == legacy_v1_sha
                        for item in case_ledger["verifications"]),
                f"recovered trajectory ledger changed verifier identity at {cutpoint}")
        case_prefix = torch.load(case_dir / "checkpoints/policy-0.pt",
                                 map_location="cpu", weights_only=True)
        equal_state(case_prefix, expected_prefix, torch, f"prefix-{cutpoint}")
        actual_candidate = torch.load(case_dir / "checkpoints/policy-1.pt",
                                     map_location="cpu", weights_only=True)
        require(set(expected_candidate) >= {"model", "optimizer", "cpu_rng", "module_modes"},
                "reference checkpoint omits a required recovery component")
        equal_state(actual_candidate, expected_candidate, torch, f"recovered-{cutpoint}")
        require(worker["training_reward_order"] == expected_rewards,
                f"recovered target group reward order differs at {cutpoint}")
        require(worker["main_sync_candidate_model_matches"] is True
                and worker["main_sync_candidate_sha256"] == fixture["expected_candidate_sha256"]
                and worker["candidate_model_tensor_sha256"]
                == fixture["expected_candidate_policy_tensor_sha256"]
                and worker["promotion_journal"]["active_checkpoint_sha256"]
                == worker["candidate_sha256"],
                f"promoted state identity differs from reference at {cutpoint}")
        component_checks = {
            name: True for name in ("model", "optimizer", "cpu_rng", "module_modes")
        }
        cases.append({
            "cutpoint": cutpoint,
            "signal_returncode": row["signal_returncode"],
            "optimizer_attempts": worker["optimizer_attempts"],
            "effective_optimizer_updates": worker["committed_updates"],
            "effective_consumed_trajectories": worker["effective_consumed_trajectories"],
            "exact_state_components": component_checks,
            "recovery_wall_s": row["recovery_wall_s"],
            "case_wall_s": row["case_wall_s"],
            "peak_rss_bytes": worker["peak_rss_bytes"],
            "ledger_states": case_ledger["states"],
        })

    return {
        "status": "PASS",
        "protocol_sha256": sha(protocol_path),
        "source_run_inventory_sha256": sha(source_run / "SHA256.json"),
        "fixture_inventory_sha256": sha(fixture_dir / "SHA256.json"),
        "crash_inventory_sha256": sha(crash_output / "SHA256.json"),
        "continued_updates_before_target": 2,
        "third_update_reference_checkpoint_sha256": fixture["expected_candidate_sha256"],
        "source_target_verifier_identity": target_verifier,
        "sigkill_cases": cases,
        "recovery_cost_seconds": {
            "total": summary["harness_wall_s"],
            "case_p50": sorted(row["case_wall_s"] for row in cases)[len(cases) // 2],
            "restart_p50": sorted(row["recovery_wall_s"] for row in cases)[len(cases) // 2],
        },
        "claim_limit": (
            "Five cutpoint recoveries from a shared two-update prefix and one"
            " retained third-update group; this is a controlled systems result,"
            " not independent model-quality replication."),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crash-output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.crash_output, args.fixture, args.source_run,
                   args.source_audit, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
