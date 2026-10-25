#!/usr/bin/env python3
"""Independently reconstruct the frozen MLX Development qualification result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlx_core_development_qualification_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def git(source: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(source), *args], check=True,
        capture_output=True, text=not binary,
    )
    return result.stdout


def expected_checks(observed: dict) -> dict[str, bool]:
    take = observed["take_negative_index"]
    random = observed["random_normal_shape"]
    return {
        "default_device_is_cpu": observed.get("default_device_is_cpu") is True,
        "take_negative_axis0": take["negative_axis0"].get("value") == [3, 4, 5],
        "take_negative_flat": take["negative_flat"].get("value") == 5,
        "take_negative_axis1": take["negative_axis1"].get("value") == [1, 4],
        "take_positive_control": take["positive_control"].get("value") == [1, 4],
        "normal_invalid_loc_rejected": (
            random["invalid_loc"].get("error_type") == "ValueError"
            and "Cannot generate random values" in random["invalid_loc"].get("error", "")
        ),
        "normal_invalid_scale_rejected": (
            random["invalid_scale"].get("error_type") == "ValueError"
            and "Cannot generate random values" in random["invalid_scale"].get("error", "")
        ),
        "normal_valid_broadcast": (
            random["valid_broadcast"].get("value", {}).get("shape") == [2, 3]
            and random["valid_broadcast"].get("value", {}).get("all_finite") is True
        ),
    }


def check_manifest(run: Path) -> list[dict]:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256":
        raise RuntimeError("unsupported evidence manifest")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise RuntimeError("evidence manifest lacks a file list")
    seen = set()
    for row in rows:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in seen:
            raise RuntimeError("unsafe or duplicate manifest path")
        seen.add(row["path"])
        path = run / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"manifest file missing or unsafe: {row['path']}")
        if path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"evidence hash mismatch: {row['path']}")
    actual = {
        path.relative_to(run).as_posix()
        for path in run.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != seen:
        raise RuntimeError("run contains missing or unexpected files relative to manifest")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--build-python", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    source = args.source_repo.resolve()
    dependencies = args.dependencies.resolve()
    build_python = args.build_python.resolve()
    worktree = args.worktree.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("refusing to overwrite independent audit output")
    if sha256(PROTOCOL) != args.protocol_sha256:
        raise RuntimeError("protocol hash mismatch")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for path_text, expected in protocol["implementation_sha256"].items():
        path = ROOT / path_text
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"pinned implementation differs: {path_text}")
    for path_text, expected in protocol["preregistration_inputs"].items():
        path = ROOT / path_text
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"pinned preregistration input differs: {path_text}")
    installed_freeze = subprocess.check_output(
        [str(build_python), "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    if installed_freeze != protocol["build_python_freeze"]:
        raise RuntimeError("build Python package inventory differs")
    for name, spec in protocol["dependency_sources"].items():
        path = dependencies / spec["path"]
        if not path.is_dir() or tree_sha256(path) != spec["tree_sha256"]:
            raise RuntimeError(f"build dependency source hash differs: {name}")
        if "archive_filename" in spec:
            archive = dependencies / spec["archive_filename"]
            if not archive.is_file() or sha256(archive) != spec["archive_sha256"]:
                raise RuntimeError(f"build dependency archive hash differs: {name}")
        if spec["kind"] == "git" and git(path, "rev-parse", "HEAD").strip() != spec["commit"]:
            raise RuntimeError(f"build dependency commit differs: {name}")
    if git(source, "rev-parse", f"{protocol['base_revision']}^{{commit}}").strip() != protocol["base_revision"]:
        raise RuntimeError("pinned base commit does not resolve")
    for task in protocol["tasks"]:
        for revision, path_key, hash_key in (
            (protocol["base_revision"], "target_path", "baseline_sha256"),
            (task["fixed_revision"], "target_path", "fixed_sha256"),
        ):
            blob = git(source, "show", f"{revision}:{task[path_key]}", binary=True)
            if sha256_bytes(blob) != task[hash_key]:
                raise RuntimeError(f"pinned task source differs: {task['task_id']}")
        fixed = git(source, "show", f"{task['fixed_revision']}:{task['target_path']}", binary=True)
        for mutation in task["mutations"]:
            old = mutation["old"].encode()
            if fixed.count(old) != 1:
                raise RuntimeError(f"mutant anchor does not uniquely match: {task['task_id']}")
            fixed = fixed.replace(old, mutation["new"].encode(), 1)
        if sha256_bytes(fixed) != task["mutant_sha256"]:
            raise RuntimeError(f"mutant hash does not reconstruct: {task['task_id']}")
    for spec in protocol["grader_files"]:
        blob = git(source, "show", f"{spec['revision']}:{spec['source_path']}", binary=True)
        if sha256_bytes(blob) != spec["sha256"]:
            raise RuntimeError(f"pinned public test source differs: {spec['path']}")
        if sha256(run / "grader" / spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"isolated grader copy differs: {spec['path']}")

    def check_execution(record: dict, log: Path, samples: Path, label: str) -> None:
        if sha256(log) != record["log_sha256"] or sha256(samples) != record["samples_sha256"]:
            raise RuntimeError(f"{label}: execution artifact hash mismatch")
        sample_lines = samples.read_text(encoding="utf-8").splitlines()
        if not sample_lines:
            raise RuntimeError(f"{label}: missing resource telemetry samples")
        for line in sample_lines:
            json.loads(line)

    rows = check_manifest(run)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    if summary and summary["protocol_sha256"] != args.protocol_sha256:
        raise RuntimeError("summary points to a different protocol")
    if summary and (summary["model_inference_calls"] != 0 or summary["training_updates"] != 0):
        raise RuntimeError("pilot unexpectedly contains model calls or updates")
    if summary:
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                        protocol["controller_parent_revision"], summary["controller_revision"]], check=True)
        if summary["controller_tracked_worktree_clean"] is not True:
            raise RuntimeError("controller tracked worktree was not clean at execution")
    expected_phase_status = {
        "baseline": "BASELINE_REJECTED",
        "fixed": "KNOWN_FIX_ACCEPTED",
        "mutant": "MUTANT_REJECTED",
    }
    reconstructed = {}
    observed_failure = None
    preflight_failure_path = run / "preflight-failure.json"
    if preflight_failure_path.is_file():
        preflight_failure = json.loads(preflight_failure_path.read_text(encoding="utf-8"))
        observed_failure = f"preflight:{preflight_failure['reason']}"
    for phase_name, status in expected_phase_status.items():
        if preflight_failure_path.is_file():
            break
        phase_dir = run / phase_name
        phase_path = phase_dir / "phase.json"
        if not phase_path.is_file():
            observed_failure = f"phase_not_completed:{phase_name}"
            break
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
        expected_hash_key = {
            "baseline": "baseline_sha256",
            "fixed": "fixed_sha256",
            "mutant": "mutant_sha256",
        }[phase_name]
        expected_source_hashes = {
            task["task_id"]: task[expected_hash_key] for task in protocol["tasks"]
        }
        if phase["source_hashes"] != expected_source_hashes:
            raise RuntimeError(f"{phase_name}: pinned task source hashes do not match")
        check_execution(
            phase["build"], phase_dir / "build.log",
            phase_dir / "build.resources.jsonl", f"{phase_name}/build",
        )
        if phase["status"] == "BUILD_FAILED":
            observed_failure = f"build_failed:{phase_name}"
            reconstructed[phase_name] = {"status": phase["status"]}
            break
        check_execution(
            phase["runtime"], phase_dir / "observations.stdout.jsonl",
            phase_dir / "observations.resources.jsonl", f"{phase_name}/observations",
        )
        raw = (phase_dir / "observations.stdout.jsonl").read_text(encoding="utf-8").splitlines()
        if len(raw) != 1:
            raise RuntimeError(f"{phase_name}: observation output is not one JSON line")
        observed = json.loads(raw[0])
        if observed != phase["observed"]:
            raise RuntimeError(f"{phase_name}: structured output differs from raw output")
        checks = expected_checks(observed)
        if checks != phase["checks"]:
            raise RuntimeError(f"{phase_name}: oracle checks do not reconstruct")
        module_path = Path(observed["mlx_core_module_path"]).resolve()
        try:
            module_path.relative_to((worktree / "python").resolve())
        except ValueError as exc:
            raise RuntimeError(f"{phase_name}: imported MLX outside the pinned source tree") from exc
        task_results = {}
        for task in protocol["tasks"]:
            test = phase["upstream_tests"][task["task_id"]]
            log = phase_dir / f"{task['task_id']}.upstream-test.log"
            if sha256(log) != test["log_sha256"]:
                raise RuntimeError(f"{phase_name}/{task['task_id']}: test log hash mismatch")
            samples = phase_dir / f"{task['task_id']}.upstream-test.resources.jsonl"
            if sha256(samples) != test["samples_sha256"]:
                raise RuntimeError(f"{phase_name}/{task['task_id']}: resource telemetry hash mismatch")
            sample_lines = samples.read_text(encoding="utf-8").splitlines()
            if not sample_lines:
                raise RuntimeError(f"{phase_name}/{task['task_id']}: missing resource telemetry")
            for line in sample_lines:
                json.loads(line)
            log_text = log.read_text(encoding="utf-8", errors="replace")
            log_passed = (
                test["exit_code"] == 0
                and test["abort_reason"] is None
                and "Ran 1 test" in log_text
                and "\nOK\n" in f"\n{log_text}\n"
                and "FAILED" not in log_text
            )
            if test["passed"] != log_passed:
                raise RuntimeError(f"{phase_name}/{task['task_id']}: test decision mismatch")
            passed = all(checks[key] for key in task["oracle_checks"]) and log_passed
            task_results[task["task_id"]] = passed
        expected_tasks = {task["task_id"] for task in protocol["tasks"]}
        if set(phase["task_decisions"]) != expected_tasks:
            raise RuntimeError(f"{phase_name}: task result inventory differs")
        for task_id, passed in task_results.items():
            if phase["task_decisions"][task_id]["passed"] != passed:
                raise RuntimeError(f"{phase_name}/{task_id}: task decision does not reconstruct")
        actual_status = {
            "baseline": "BASELINE_FALSE_ACCEPTANCE",
            "fixed": "KNOWN_FIX_REJECTED",
            "mutant": "MUTANT_FALSE_ACCEPTANCE",
        }[phase_name]
        if phase_name == "baseline" and not any(task_results.values()):
            actual_status = "BASELINE_REJECTED"
        elif phase_name == "fixed" and all(task_results.values()):
            actual_status = "KNOWN_FIX_ACCEPTED"
        elif phase_name == "mutant" and not any(task_results.values()):
            actual_status = "MUTANT_REJECTED"
        if phase["status"] != actual_status:
            raise RuntimeError(f"{phase_name}: phase decision does not reconstruct")
        reconstructed[phase_name] = {
            "status": phase["status"],
            "task_decisions": task_results,
            "observation_log_sha256": sha256(phase_dir / "observations.stdout.jsonl"),
        }
        if phase["status"] != status:
            observed_failure = f"phase_gate_failed:{phase_name}"
            break
    completed_all = set(reconstructed) == set(expected_phase_status)
    if summary:
        expected_summary_status = (
            "PASS_TASK_CERTIFICATION_PILOT"
            if completed_all and all(
                reconstructed[name]["status"] == status
                for name, status in expected_phase_status.items()
            ) else "NO_GO"
        )
        if summary["status"] != expected_summary_status:
            raise RuntimeError("summary status does not reconstruct from phase records")
        if summary["phases"] != {
            name: json.loads((run / name / "phase.json").read_text(encoding="utf-8"))
            for name in reconstructed
        }:
            raise RuntimeError("summary phase records differ from their durable phase files")
    elif not (run / "partial-failure.json").is_file() and not preflight_failure_path.is_file():
        raise RuntimeError("incomplete run has no durable failure record")
    passed = (
        completed_all and summary is not None
        and summary["status"] == "PASS_TASK_CERTIFICATION_PILOT"
    )
    if not passed and observed_failure is None:
        observed_failure = "one_or_more_preregistered_controls_failed"
    report = {
        "schema_version": 1,
        "protocol_sha256": args.protocol_sha256,
        "manifest_files_verified": len(rows),
        "source_base_revision": protocol["base_revision"],
        "task_ids": [task["task_id"] for task in protocol["tasks"]],
        "reconstructed_phases": reconstructed,
        "decision": (
            "PASS_TWO_DEVELOPMENT_SOURCE_GRADER_CERTIFICATES"
            if passed else "NO_GO_PILOT_DID_NOT_QUALIFY_BOTH_TASKS"
        ),
        "failure_reason": observed_failure,
        "claim_limit": protocol["claim_limit"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps({"decision": report["decision"], "manifest_files_verified": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
