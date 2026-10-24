"""Independently reconstruct the frozen dense-reward calibration records."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/verifier_consistent_async_grpo_v7/reward-calibration-v1.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def independent_reward(response: str, target: str, spec: dict) -> tuple[str, float, float]:
    # Construct a single alternation in the independent auditor rather than
    # reusing the sampler's per-alias first-match routine.
    alias_owner = {alias.casefold(): category
                   for category, aliases in spec["surface_classes"].items()
                   for alias in aliases}
    alternatives = sorted(alias_owner, key=lambda alias: (-len(alias), alias))
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(word) for word in alternatives)
        + r")(?![A-Za-z0-9_])", re.IGNORECASE)
    found = pattern.search(response)
    category = "unknown" if found is None else alias_owner[found.group(1).casefold()]
    if category == "unknown":
        return category, float(spec["v0"]["unknown_score"]), float(spec["v1"]["unknown_score"])
    generic = float(spec["v0"]["class_scores"][category])
    if category == "air":
        target_score = 0.0
    elif category == target:
        target_score = float(spec["v1"]["exact_score"])
    elif category in spec["v1"]["related_classes"][target]:
        target_score = float(spec["v1"]["related_score"])
    else:
        target_score = float(spec["v1"]["other_support_score"])
    return category, generic, target_score


def audit(run: Path) -> dict:
    run = run.resolve()
    check(run.is_dir() and not run.is_symlink(), "unsafe calibration run path")
    protocol = json.loads((run / "protocol.json").read_text())
    frozen = json.loads(PROTOCOL.read_text())
    check(digest(PROTOCOL) == digest(run / "protocol.json"),
          "run protocol differs from the frozen calibration protocol")
    check(protocol == frozen, "protocol content differs from the frozen calibration")
    check(protocol["source_identity"]["sampler_sha256"] == digest(
        ROOT / "scripts/calibrate_verifier_reward.py"), "sampler identity mismatch")
    check(protocol["source_identity"]["auditor_sha256"] == digest(Path(__file__).resolve()),
          "auditor identity mismatch")

    members = []
    group_rows = []
    for task in protocol["sampling"]["tasks"]:
        path = run / "groups" / f"{task['id']}.json"
        group = json.loads(path.read_text())
        check(group["task_id"] == task["id"]
              and group["target_class"] == task["target_class"]
              and group["prompt"] == task["prompt"], "task identity differs")
        check(len(group["members"]) == protocol["sampling"]["group_size"],
              "incomplete calibration group")
        scores = {"v0": [], "v1": []}
        for row in group["members"]:
            category, v0, v1 = independent_reward(
                row["output_text"], task["target_class"], protocol["reward_definition"])
            check(row["surface_class"] == category, "surface parse does not reconstruct")
            check(math.isclose(float(row["v0"]), v0, rel_tol=0, abs_tol=1e-12),
                  "v0 reward does not reconstruct")
            check(math.isclose(float(row["v1"]), v1, rel_tol=0, abs_tol=1e-12),
                  "v1 reward does not reconstruct")
            check(math.isfinite(float(row["behavior_logprob"])), "invalid behavior logprob")
            scores["v0"].append(v0)
            scores["v1"].append(v1)
            members.append({"v0": v0, "v1": v1, "output_text": row["output_text"]})
        for version in ("v0", "v1"):
            expected = group["reward_summary"][version]
            check(expected["unique_values"] == sorted(set(scores[version])),
                  "group reward support differs")
            check(expected["positive_count"] == sum(value > 0 for value in scores[version]),
                  "group reward positive count differs")
            check(math.isclose(expected["mean"], sum(scores[version]) / len(scores[version]),
                               rel_tol=0, abs_tol=1e-12), "group reward mean differs")
        group_rows.append({
            "task_id": task["id"],
            "v0_unique_values": len(set(scores["v0"])),
            "v1_unique_values": len(set(scores["v1"])),
            "v1_positive": sum(value > 0 for value in scores["v1"]),
        })

    shifts = [abs(row["v0"] - row["v1"]) for row in members]
    mean_shift = sum(shifts) / len(shifts)
    fraction_shifted = sum(value >= 0.25 for value in shifts) / len(shifts)
    nonconstant = sum(row["v1_unique_values"] > 1 for row in group_rows)
    gates = {
        "all_groups_nonconstant_v1": nonconstant == len(group_rows),
        "mean_absolute_shift": mean_shift >= protocol["decision_rules"][
            "minimum_mean_absolute_reward_shift_v0_to_v1"],
        "fraction_shift_at_least_0_25": fraction_shifted >= protocol["decision_rules"][
            "minimum_fraction_with_absolute_reward_shift_at_least_0_25"],
    }
    summary = json.loads((run / "summary.json").read_text())
    check(summary["gates"] == gates, "runner summary gate decisions differ")
    check(math.isclose(summary["verifier_version_mean_absolute_reward_shift"],
                       mean_shift, rel_tol=0, abs_tol=1e-12), "runner mean shift differs")
    expected_status = "CALIBRATION_PASS" if all(gates.values()) else "NO_GO_REWARD_CALIBRATION"
    check(summary["status"] == expected_status, "runner status differs from reconstructed gate")

    inventory = json.loads((run / "SHA256.json").read_text())
    actual_files = {str(path.relative_to(run)): digest(path)
                    for path in sorted(run.rglob("*")) if path.is_file()
                    and path.name != "SHA256.json" and not path.name.endswith(("-wal", "-shm"))}
    check(inventory == actual_files, "run SHA-256 inventory differs")
    return {
        "status": "PASS_INDEPENDENT_RECONSTRUCTION",
        "calibration_gate": expected_status,
        "groups": len(group_rows), "members": len(members),
        "v1_nonconstant_groups": nonconstant,
        "mean_absolute_reward_shift": mean_shift,
        "fraction_with_shift_at_least_0_25": fraction_shifted,
        "gates": gates,
        "claim_limit": "Independent reward reconstruction only; no optimizer update or coding capability result.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2))
