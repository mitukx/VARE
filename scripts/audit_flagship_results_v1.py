"""Offline byte/provenance audit; imports no model, grader, or experiment runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit_teacher(read, protocol, calibration):
    """Reconstruct the observed negative outcomes from independent records."""
    summary = json.loads(read("summary.json"))
    require(summary["status"] == "COMPLETED", "teacher run incomplete")
    tokens, seconds, rows = 0, 0.0, []
    for task in protocol["tasks"]:
        baseline = json.loads((ROOT / calibration / task["id"] / "baseline.json").read_bytes())
        require(not baseline["passed"], "baseline must fail calibration")
        baseline_cases = [(case["case_id"], case["passed"]) for case in baseline["cases"]]
        for attempt in protocol["attempts"]:
            prefix = task["id"] + "/" + attempt["id"] + "/"
            agent = json.loads(read(prefix + "agent.json"))
            reward = json.loads(read(prefix + "reward.json"))
            grade = json.loads(read(prefix + "independent-grade.json"))
            monitor = json.loads(read(prefix + "monitor/resource-monitor.json"))
            source = read(prefix + "workspace/target.py")
            require(sha(source) == task["baseline_sha256"], "observed source was not unchanged baseline")
            require(agent["model_snapshot"] == protocol["model"]["revision"], "policy revision mismatch")
            require(agent["model_files"] == protocol["model"]["files_sha256"], "policy file identity mismatch")
            require(agent["seed"] == attempt["seed"], "seed mismatch")
            require(agent["source_hashes_before"]["target.py"] == sha(source), "initial source mismatch")
            require(agent["source_hashes_after"]["target.py"] == sha(source), "final source mismatch")
            require(not agent["edit_applied"] and not agent["candidate_diff"], "unexpected edit")
            require(not grade["passed"] and reward["reward"] == 0, "negative outcome mismatch")
            require([(c["case_id"], c["passed"]) for c in grade["cases"]] == baseline_cases,
                    "unchanged baseline grading disagrees with calibration")
            require(agent["source_hashes_before"]["test_public.py"] == sha(read(prefix+"workspace/test_public.py")),
                    "protected public test changed")
            observations = monitor["observations"]
            require(monitor["status"] == "COMPLETED" and monitor["returncode"] == 0, "resource run incomplete")
            require(observations, "resource telemetry missing")
            require(all(row["memory_free_percent"] >= protocol["resources"]["minimum_free_memory_percent"] for row in observations),
                    "free-memory gate violated")
            require(all(row["swap_used_mib"] < monitor["swap_abort_mib"] for row in observations), "swap gate violated")
            require(monitor["max_child_rss_mib"] < protocol["resources"]["max_process_rss_mib"], "RSS gate violated")
            events = [json.loads(line) for line in read(prefix+"generation-trace.jsonl").splitlines()]
            generated = sum(event["generated_tokens_including_terminal"] for event in events if event["event"] == "model_output")
            token_events = [event for event in events if event["event"] == "model_token"]
            require(generated == agent["new_tokens"] == len(token_events), "exact generation token count mismatch")
            require(not any(event["event"] == "public_test" for event in events), "unexpected public-test execution")
            require(not any(call.get("name") != "recovered_edit" for turn in agent["transcript"] for call in turn["tool_calls"]),
                    "unexpected executed model tool")
            tokens += generated
            seconds += monitor["elapsed_seconds"]
            rows.append({"task": task["id"], "attempt": attempt["id"], "reward": 0,
                         "negative_source_independently_bound_to_calibrated_baseline": True})
    require(len(rows) == protocol["budgets"]["total_attempts"], "attempt/group cardinality mismatch")
    require(len(summary["attempts"]) == len(rows), "summary attempt count mismatch")
    require(math.isclose(summary["model_run_seconds"], seconds, abs_tol=1e-9), "compute accounting mismatch")
    require(summary["successes"] == 0 and summary["optimizer_steps"] == 0, "unsupported learning outcome")
    return {"attempts": len(rows), "generated_tokens": tokens, "independent_successes": 0,
            "valid_edits": 0, "executed_model_tools": 0, "public_test_calls": 0,
            "groups": len(protocol["tasks"]), "all_groups_constant_zero_reward": True,
            "eligibility": "exploratory consumed Development; original grading-image qualification revoked",
            "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args()
    data = args.manifest.read_bytes()
    require(sha(data) == args.manifest_sha256, "external result manifest anchor mismatch")
    manifest = json.loads(data)
    for name, expected in manifest["files_sha256"].items():
        require(sha((ROOT / name).read_bytes()) == expected, "artifact hash mismatch: "+name)
    archive_path = ROOT / manifest["teacher_archive"]
    with tarfile.open(archive_path, "r:gz") as archive:
        total = 0
        inventory = {}
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe archive path")
            require(member.isdir() or member.isfile(), "archive links/devices are forbidden")
            require(member.size < 2_000_000, "oversized raw artifact")
            total += member.size
            require(total < 50_000_000, "archive expansion budget exceeded")
            if member.isfile():
                require(member.name not in inventory, "duplicate archive member")
                inventory[member.name] = sha(archive.extractfile(member).read())
        require(inventory == manifest["teacher_archive_members_sha256"], "raw teacher inventory mismatch")
        protocol = json.loads((ROOT / manifest["teacher_protocol"]).read_bytes())
        require(json.loads(archive.extractfile("run-1/summary.json").read())["protocol_sha256"]
                == sha((ROOT / manifest["teacher_protocol"]).read_bytes()), "teacher protocol identity mismatch")
        teacher = audit_teacher(lambda name: archive.extractfile("run-1/"+name).read(), protocol,
                                protocol["qualification_run"])
    raw_results = {}
    with tarfile.open(ROOT / manifest["leakage_archive"], "r:gz") as archive:
        total = 0
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe leakage archive path")
            require(member.isdir() or member.isfile(), "leakage archive contains links/devices")
            total += member.size
            require(member.size < 2_000_000 and total < 50_000_000, "leakage archive budget exceeded")
            if member.isfile():
                require(member.name not in raw_results, "duplicate leakage archive member")
                raw_results[member.name] = archive.extractfile(member).read()
    require({name: sha(data) for name, data in raw_results.items()} == manifest["leakage_archive_members_sha256"],
            "raw leakage artifact inventory mismatch")

    def artifact(name):
        return raw_results[name] if name in raw_results else (ROOT / name).read_bytes()

    result = json.loads(artifact(manifest["leakage_summary"]))
    leakage_protocol = json.loads((ROOT / manifest["leakage_protocol"]).read_bytes())
    require(result["protocol_sha256"] == sha((ROOT / manifest["leakage_protocol"]).read_bytes()), "leakage protocol identity mismatch")
    lookup = {}
    for row in result["results"]:
        require(row["passed"] == all(case["passed"] for case in row["cases"]), "inconsistent grader aggregate")
        path = str(PurePosixPath(manifest["leakage_run"]) / row["image_arm"] / row["task"] / (row["variant"]+".json"))
        require(json.loads(artifact(path)) == row, "grader detail/summary mismatch")
        lookup[(row["image_arm"], row["task"], row["variant"])] = row["passed"]
        if row["image_arm"] == "baseline-only":
            require(row["passed"] == (row["variant"] == "known-fix"), "clean-image calibration failed")
    require(lookup[("original", "gptq-groups", "installed-delegation")], "installed-fix leakage not reproduced")
    require(not lookup[("baseline-only", "gptq-groups", "installed-delegation")], "installed-fix route remains open")
    require(set(leakage_protocol["images"]) == {"original", "baseline-only"}, "image-arm identity mismatch")
    compiled = json.loads(artifact(manifest["compiled_summary"]))
    require(compiled["protocol_sha256"] == result["protocol_sha256"], "compiled-probe protocol differs")
    for arm in ("original", "source-only", "portable"):
        row = compiled["results"][arm]
        require(row["passed"] == (arm != "portable"), "compiled repair outcome differs")
        require(len(row["cases"]) == 4, "compiled-repair verifier budget differs")
        require(sum(case["passed"] for case in row["cases"]) == (0 if arm == "portable" else 4),
                "compiled-repair case count differs")
    assets = PurePosixPath(manifest["image_asset_probe"])
    control = json.loads(json.loads(artifact(str(assets / "source-control.json")))["stdout"])
    clean = json.loads(json.loads(artifact(str(assets / "final.json")))["stdout"])
    require(control["python_source_inventory"] == clean["python_source_inventory"], "image Python sources differ")
    require(not clean["bytecode_files"], "compiled upstream source remains installed")
    for asset in clean["assets"].values():
        require(asset["actual_sha256"] == asset["record_sha256"], "stale installed repair hash")
        require(asset["actual_bytes"] == asset["record_bytes"], "stale installed repair size")
    for task in protocol["tasks"]:
        name = task["target"].removeprefix("mlx_lm/")
        require(clean["assets"][name]["actual_sha256"] == task["baseline_sha256"], "installed source is not pinned baseline")
    print(json.dumps({"status": "PASS", "teacher": teacher,
                      "installed_source_and_compiled_repair_leakage_reproduced_and_closed": True,
                      "image_record_integrity_and_source_inventory_checked": True,
                      "training": "NO_GO", "heldout_evaluation": "NOT_RUN"}, indent=2))


if __name__ == "__main__":
    main()
