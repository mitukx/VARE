"""Offline audit for the frozen verifier-drift scenario artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_output_safety import (  # noqa: E402
    preflight_new_output,
    write_new_output,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def word(text: str, target: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(target)}(?![A-Za-z0-9_])",
                          text, flags=re.IGNORECASE))


def grade(rule: str, text: str) -> float:
    if rule == "ground_word":
        return float(word(text, "ground"))
    if rule == "punctuation_or_location":
        return float(bool(re.search(r"[.!?]", text)) or any(
            word(text, target) for target in ("ground", "grass", "floor", "sky", "branch")))
    if rule == "exact_ground_period":
        return float(bool(re.fullmatch(r"\s*ground\.\s*", text, flags=re.IGNORECASE)))
    if rule == "not_ground_word":
        return float(not word(text, "ground"))
    raise ValueError(f"unknown rule {rule!r}")


def ranking(observed: list[float], expected: list[float]) -> dict[str, int]:
    inversions = comparable = 0
    for left in range(len(observed)):
        for right in range(left + 1, len(observed)):
            a, b = observed[left] - observed[right], expected[left] - expected[right]
            if a != 0 and b != 0:
                comparable += 1
                inversions += int(a * b < 0)
    return {"pairwise_inversions": inversions, "comparable_pairs": comparable}


def verify_manifest(root: Path, manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    require(isinstance(manifest, dict), f"invalid manifest: {manifest_path}")
    for relative, expected in manifest.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"missing/unsafe artifact: {path}")
        require(sha(path) == expected, f"artifact hash mismatch: {relative}")
    actual = {str(path.relative_to(root)) for path in root.rglob("*")
              if path.is_file() and path.name != "SHA256.json"
              and not path.name.endswith(("-wal", "-shm"))}
    require(actual == set(manifest), "artifact manifest has missing or unlisted files")
    return manifest


def ledger_group_states(path: Path) -> dict[str, tuple[str, int]]:
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT group_id,state,count(*) FROM trajectories GROUP BY group_id,state"
        ).fetchall()
    result = {}
    for group_id, state, count in rows:
        require(group_id not in result, f"trajectory group has mixed terminal states: {group_id}")
        result[group_id] = (state, int(count))
    return result


def audit(output: Path) -> dict:
    protocol_path = output / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads((output / "summary.json").read_text())
    require(summary["protocol_sha256"] == sha(protocol_path), "protocol hash mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", summary["source_commit"]) is not None,
            "run does not identify a full source commit")
    require(re.fullmatch(r"[0-9a-f]{40}", protocol["base_commit"]) is not None,
            "protocol does not identify a full frozen base commit")
    root_manifest = verify_manifest(output, output / "SHA256.json")
    del root_manifest

    reports = []
    primary_differences = []
    for scenario in protocol["scenarios"]:
        scenario_id = scenario["id"]
        if scenario_id not in summary["scenarios"]:
            require(summary["status"] == "STOPPED_PILOT_NO_SIGNAL",
                    f"missing preregistered scenario {scenario_id}")
            break
        arm_rows = summary["scenarios"][scenario_id]
        require(set(arm_rows) == {"sync", "naive_async", "vare_async"},
                f"incomplete arm set: {scenario_id}")
        initial_hashes = {row["initial_model_tensor_sha256"] for row in arm_rows.values()}
        first_group_hashes = {row["groups"][0]["response_text_sha256"]
                              for row in arm_rows.values()}
        require(len(initial_hashes) == 1 and len(first_group_hashes) == 1,
                f"matched checkpoint/rollout control failed: {scenario_id}")

        for arm, arm_summary in arm_rows.items():
            arm_dir = output / scenario_id / arm
            verify_manifest(arm_dir, arm_dir / "SHA256.json")
            raw = json.loads((arm_dir / "rollouts.json").read_text())
            require(raw["scenario"] == scenario_id and raw["arm"] == arm,
                    "raw rollout identity mismatch")
            require(len(raw["groups"]) == 5, "wrong number of rollout groups")
            group_by_id = {}
            for index, group in enumerate(raw["groups"]):
                members = group["members"]
                require(len(members) == 8, f"wrong group size: {scenario_id}/{arm}/{index}")
                observed = []
                expected = []
                responses = []
                policy_versions = set()
                policy_ids = set()
                for member in members:
                    text = member["attempt"]["output"]
                    policy_versions.add(member["attempt"]["policy_version"])
                    policy_ids.add(member["attempt"]["policy_id"])
                    version = member["verifier_version"]
                    rule = scenario["verifiers"]["v0" if version == 0 else "v1"]["rule"]
                    measured = grade(rule, text)
                    truth = grade(scenario["verifiers"]["v1"]["rule"], text)
                    require(float(member["score"]) == measured,
                            f"reward label not reproducible: {scenario_id}/{arm}/{index}")
                    observed.append(measured)
                    expected.append(truth)
                    responses.append(text)
                require(group["scores"] == observed, "raw score vector mismatch")
                require(len(policy_versions) == 1 and len(policy_ids) == 1,
                        "a GRPO group mixes behavior policies")
                require(group["policy_versions"] == sorted(policy_versions),
                        "policy-version provenance mismatch")
                require(group["policy_ids"] == sorted(policy_ids),
                        "policy-id provenance mismatch")
                require(group["behavior_policy"] == {
                    "policy_id": next(iter(policy_ids)),
                    "policy_version": next(iter(policy_versions)),
                }, "behavior-policy identity differs from its rollout members")
                require(group["independent_scores"] == expected,
                        "independent oracle vector mismatch")
                false_accept = sum(a == 1 and b == 0 for a, b in zip(observed, expected, strict=True))
                false_reject = sum(a == 0 and b == 1 for a, b in zip(observed, expected, strict=True))
                require(group["disagreements"] == {
                    "false_accepts": false_accept,
                    "false_rejects": false_reject,
                    "total_disagreements": false_accept + false_reject,
                }, "false accept/reject accounting mismatch")
                require(group["ranking"] == ranking(observed, expected),
                        "pairwise ranking mismatch")
                response_digest = hashlib.sha256(json.dumps(
                    responses, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
                require(group["response_text_sha256"] == response_digest,
                        "response trace digest mismatch")
                require(isinstance(group.get("queue_residence_s"), (int, float))
                        and group["queue_residence_s"] >= 0,
                        "group queue residence is missing or invalid")
                require(isinstance(group.get("delivery_lag"), dict),
                        "group delivery-version lag is missing")
                versions = group["versions"]
                if scenario_id == "no_drift" or arm == "sync" or index > 0:
                    require(set(versions) == {1}, "unexpected verifier version after refresh/control")
                else:
                    require(versions == [0] * 4 + [1] * 4,
                            "mixed-version transition does not match frozen cutover")
                    transition = group["transition"]
                    require(transition and transition["verified_members"] == 4
                            and transition["rollouts_still_active"] is True,
                            "verifier transition was not overlapped with rollout work")
                group_by_id[group["group_id"]] = group

            expected_groups = [f"g-{i:02d}-{scenario_id}"
                               for i in protocol["training_groups"][arm]]
            actual_groups = [row["training_group"] for row in arm_summary["updates"]]
            require(actual_groups == expected_groups,
                    f"training groups differ from fixed workload for {scenario_id}/{arm}")
            require(arm_summary["committed_updates"] == 4
                    and arm_summary["optimizer_calls"] == 4
                    and len(arm_summary["updates"]) == 4,
                    f"wrong committed optimizer update count: {scenario_id}/{arm}")
            require(arm_summary["rollouts"] == 40
                    and arm_summary["verifier_calls"] == 40,
                    f"matched rollout/verifier budget mismatch: {scenario_id}/{arm}")
            for update in arm_summary["updates"]:
                replay = update["update_replay"]
                errors = replay.get("errors", {})
                require(errors and max(errors.values()) < 2e-6,
                        f"independent optimizer replay tolerance failed: {scenario_id}/{arm}")
                require(replay.get("behavior_alignment_error", 1.0) < 1e-4,
                        "behavior probability alignment failed")
                require(update["accepted"] is True,
                        f"update not independently promoted: {scenario_id}/{arm}")
            invalid_commits = sum(
                group_by_id[group_id]["disagreements"]["total_disagreements"] > 0
                for group_id in actual_groups)
            require(arm_summary["invalid_committed_updates"] == invalid_commits,
                    f"invalid commit count mismatch: {scenario_id}/{arm}")
            require(arm_summary["valid_committed_updates"] == 4 - invalid_commits,
                    "valid commit count mismatch")
            if arm != "naive_async":
                require(arm_summary["store_audit"].get("pass") is True,
                        f"trajectory-store audit failed: {scenario_id}/{arm}")
                states = ledger_group_states(arm_dir / "snapshot.sqlite")
                expected_states = {
                    f"g-{index:02d}-{scenario_id}": (
                        "quarantined" if arm == "vare_async" and index == 0
                        else "verified" if arm == "sync" and index == 4
                        else "consumed", 8)
                    for index in range(5)
                }
                require(states == expected_states,
                        f"ledger terminal state differs from schedule: {scenario_id}/{arm}")
            audit_records = json.loads((arm_dir / "audit_records.json").read_text())
            queue_events = [row for row in audit_records
                            if row.get("kind") == "buffered_multi_version_queue"]
            if arm == "sync":
                require(not queue_events, "synchronous arm unexpectedly buffered stale groups")
            else:
                require(len(queue_events) == 1, "async arm lacks its fixed two-group buffer event")
                queue = queue_events[0]
                require(queue["delivered_next"] == f"g-00-{scenario_id}"
                        and queue["also_buffered"] == f"g-02-{scenario_id}"
                        and queue["queue_depth_groups"] == 2
                        and queue["policy_versions"] == [0, 1],
                        "two-policy queue provenance differs from preregistration")
                expected_verifiers = [1] if scenario_id == "no_drift" else [0, 1]
                require(queue["verifier_versions"] == expected_verifiers,
                        "queue verifier versions differ from scenario")
                raw_by_id = {row["group_id"]: row for row in raw["groups"]}
                require(raw_by_id[f"g-00-{scenario_id}"]["behavior_policy"]["policy_version"] == 0
                        and raw_by_id[f"g-02-{scenario_id}"]["behavior_policy"]["policy_version"] == 1,
                        "queue does not span the expected behavior-policy versions")
                require(raw_by_id[f"g-00-{scenario_id}"]["delivery_lag"]["policy_lag"] == 1,
                        "stale queued group did not cross a policy promotion")
            if arm == "vare_async":
                counts = arm_summary["store_counts"]
                require(counts.get("quarantined", 0) >= 8,
                        "VARE did not quarantine all members of stale group")
                stale = group_by_id[f"g-00-{scenario_id}"]
                valid_rejected = sum(
                    observed == expected for observed, expected in zip(
                        stale["scores"], stale["independent_scores"], strict=True))
                require(arm_summary["quarantined_valid_trajectories"] == valid_rejected,
                        "valid-but-quarantined trajectory count mismatch")
                store_queues = [row for row in audit_records
                                if row.get("kind") == "trajectory_store_multi_version_queue"]
                require(len(store_queues) == 1
                        and store_queues[0]["group_ids"] == [
                            f"g-00-{scenario_id}", f"g-02-{scenario_id}"]
                        and store_queues[0]["policy_versions"] == [0, 1],
                        "VARE ledger never held both policy versions together")
                consumed_from_queue = [row for row in audit_records
                                       if row.get("kind") == "group_consumed_from_buffered_queue"]
                require(consumed_from_queue == [{
                    "kind": "group_consumed_from_buffered_queue",
                    "group_id": f"g-02-{scenario_id}",
                    "consumer_delivery": f"g-00-{scenario_id}",
                }], "fresh group was not consumed with the stale-group scan")
            if arm == "naive_async" and scenario_id != "no_drift":
                reports.append({"scenario": scenario_id,
                                "false_accepts": group_by_id[f"g-00-{scenario_id}"]["disagreements"]["false_accepts"],
                                "false_rejects": group_by_id[f"g-00-{scenario_id}"]["disagreements"]["false_rejects"],
                                "pairwise_inversions": group_by_id[f"g-00-{scenario_id}"]["ranking"]["pairwise_inversions"],
                                "invalid_committed_updates": invalid_commits})

        naive = arm_rows["naive_async"]["invalid_committed_updates"]
        vare = arm_rows["vare_async"]["invalid_committed_updates"]
        primary_differences.append({"scenario": scenario_id,
                                    "naive_minus_vare": naive - vare})

    if summary["status"] == "COMPLETE":
        require(len(summary["scenarios"]) == len(protocol["scenarios"]),
                "complete run omits a preregistered scenario")
    drift_differences = [row["naive_minus_vare"] for row in primary_differences
                         if row["scenario"] != "no_drift"]
    signs = [value for value in drift_differences if value != 0]
    sign_wins = sum(value > 0 for value in signs)
    sign_p = (sum(math.comb(len(signs), count)
                  for count in range(sign_wins, len(signs) + 1)) / (2 ** len(signs))
              if signs else None)
    return {"status": "PASS", "protocol_sha256": sha(protocol_path),
            "source_commit": summary["source_commit"],
            "scenarios_audited": list(summary["scenarios"]),
            "naive_async_drift_scenarios": reports,
            "primary_differences": primary_differences,
            "descriptive_sign_test": {
                "positive_drift_scenarios": sign_wins,
                "non_tied_drift_scenarios": len(signs),
                "one_sided_exact_p_if_independent_signs": sign_p,
                "interpretation": "purposively selected verifier semantics; descriptive only",
            },
            "claim_limit": "synthetic reward semantics and one fixed seed per scenario; no coding-capability or population-wide reliability claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (preflight_new_output(args.output, [args.run])
              if args.output is not None else None)
    report = audit(args.run)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        write_new_output(output, rendered)
    print(rendered, end="")
