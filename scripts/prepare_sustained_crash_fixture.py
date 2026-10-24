"""Prepare a hash-bound third-update crash fixture from the frozen v7 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_inventory(root: Path) -> dict:
    manifest = json.loads((root / "SHA256.json").read_text())
    for relative, digest in manifest.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                f"source-run artifact failed hash verification: {relative}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads((args.run / "protocol.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    audit_report = json.loads(args.audit_report.read_text())
    require(summary["status"] == "COMPLETE", "crash extension requires a complete v7 run")
    require(audit_report.get("status") == "PASS"
            and audit_report.get("protocol_sha256") == sha(args.run / "protocol.json")
            and audit_report.get("source_commit") == summary["source_commit"],
            "independent v7 audit must pass before crash fixture creation")
    source_manifest = verify_inventory(args.run)
    arm = summary["scenarios"]["false_accept"]["vare_async"]
    updates = arm["updates"]
    require(len(updates) == 4, "sustained crash fixture requires four VARE updates")
    require([row["training_group"] for row in updates]
            == ["g-01-false_accept", "g-02-false_accept",
                "g-03-false_accept", "g-04-false_accept"],
            "VARE update sequence differs from the preregistered false-accept schedule")
    prefix_update, target_update = updates[1], updates[2]
    require(prefix_update["training_group"] == "g-02-false_accept"
            and target_update["training_group"] == "g-03-false_accept",
            "fixture does not continue after exactly two committed updates")

    arm_root = args.run / "false_accept/vare_async"
    raw = json.loads((arm_root / "rollouts.json").read_text())
    target = next((group for group in raw["groups"]
                   if group["group_id"] == target_update["training_group"]), None)
    require(target is not None, "third-update rollout group missing from retained trace")
    prefix_policy = prefix_update["promoted_id"]
    target_policy = target_update["promoted_id"]
    require(target["behavior_policy"]["policy_id"] == prefix_policy,
            "target rollouts were not sampled from the two-update prefix policy")
    require(len(target["members"]) == 8 and set(target["versions"]) == {1},
            "target group is not one complete current-verifier group")
    checkpoint_root = arm_root / "checkpoints"
    prefix_checkpoint = checkpoint_root / f"{prefix_policy}.pt"
    target_checkpoint = checkpoint_root / f"{target_policy}.pt"
    require(prefix_checkpoint.is_file() and target_checkpoint.is_file(),
            "two-update prefix or third-update reference checkpoint is missing")

    require(not args.output.exists(), "crash fixture output already exists")
    args.output.mkdir(parents=True)
    target_dir = args.output / "sync"
    (target_dir / "checkpoints").mkdir(parents=True)
    shutil.copyfile(prefix_checkpoint, target_dir / "checkpoints/policy-0.pt")
    shutil.copyfile(target_checkpoint, target_dir / "checkpoints/policy-1.pt")

    members = []
    for row in target["members"]:
        member = json.loads(json.dumps(row))
        attempt = member["attempt"]
        # Give the continued process a local policy-0 alias for the exact
        # checkpoint produced by the two completed VARE updates. Token-level
        # behavior data and source provenance remain unchanged and are bound
        # in fixture.json and the artifact inventory.
        attempt["policy_id"] = "policy-0"
        attempt["policy_version"] = 0
        members.append(member)
    (target_dir / "rollouts.json").write_text(json.dumps({
        "fresh": {"members": members}}, indent=2, sort_keys=True) + "\n")

    fixture = {
        "schema": "vare.sustained-crash-fixture.v1",
        "source_run_protocol_sha256": sha(args.run / "protocol.json"),
        "source_run_inventory_sha256": sha(args.run / "SHA256.json"),
        "source_audit_report_sha256": sha(args.audit_report),
        "source_commit": summary["source_commit"],
        "source_seed": summary["seed"],
        "scenario": "false_accept",
        "arm": "vare_async",
        "committed_prefix_updates": 2,
        "prefix_update_records": [
            {"training_group": row["training_group"],
             "promoted_id": row["promoted_id"],
             "policy_tensor_sha256": row["policy_tensor_sha256"]}
            for row in updates[:2]],
        "prefix_policy_id": prefix_policy,
        "prefix_checkpoint_source_relative": str(prefix_checkpoint.relative_to(args.run)),
        "prefix_checkpoint_sha256": sha(prefix_checkpoint),
        "prefix_checkpoint_alias": {"policy_id": "policy-0", "policy_version": 0},
        "target_update_number": 3,
        "target_group_id": target["group_id"],
        "target_behavior_policy": target["behavior_policy"],
        "target_group_sha256": hashlib.sha256(json.dumps(
            target, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "expected_candidate_source": str(target_checkpoint),
        "expected_candidate_policy_id": target_policy,
        "expected_candidate_source_relative": str(target_checkpoint.relative_to(args.run)),
        "expected_candidate_sha256": sha(target_checkpoint),
        "expected_candidate_policy_tensor_sha256": target_update["policy_tensor_sha256"],
        "target_group_members": 8,
        "target_verifier_versions": target["versions"],
        "target_verifier_identity": {
            "name": "landing",
            "version": 1,
            "rule": protocol["scenarios"][0]["verifiers"]["v1"]["rule"],
            "config_sha256": sha_json({
                "scenario": "false_accept", "version": 1,
                "rule": protocol["scenarios"][0]["verifiers"]["v1"]["rule"],
            }),
        },
        "crash_harness_seed": 82106,
        "crash_harness_seed_role": (
            "fixed legacy replay-order/run identity only; the target samples and"
            " training checkpoint are retained from v7 source_seed and no model"
            " sampling occurs in the crash workers"),
        "historical_source_file_manifest": {
            "source_inventory_entries": len(source_manifest),
        },
    }
    (args.output / "fixture.json").write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    inventory = {str(path.relative_to(args.output)): sha(path)
                 for path in args.output.rglob("*") if path.is_file()
                 and path.name != "SHA256.json"}
    (args.output / "SHA256.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fixture": str(args.output),
                      "prefix_checkpoint_sha256": fixture["prefix_checkpoint_sha256"],
                      "expected_candidate_sha256": fixture["expected_candidate_sha256"],
                      "target_group": fixture["target_group_id"]}, indent=2))


if __name__ == "__main__":
    main()
