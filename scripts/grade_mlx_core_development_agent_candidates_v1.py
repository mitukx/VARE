#!/usr/bin/env python3
"""Independently grade a frozen MLX Development agent cohort with V5's oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_mlx_core_development_qualification_v5 as v5  # noqa: E402


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True,
        input=input_text, capture_output=True, text=True,
    )
    return result.stdout.strip()


def write_json_new(path: Path, payload: dict) -> None:
    raw = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def verify_agent_record(task: dict, protocol: dict, v5_tasks: dict[str, dict]) -> tuple[dict, Path]:
    task_id = task["task_id"]
    frozen = v5_tasks.get(task_id)
    if frozen is None or task["target_path"] != frozen["target_path"]:
        raise RuntimeError(f"task is absent from the independent V5 grader: {task_id}")
    if task["baseline_sha256"] != frozen["baseline_sha256"]:
        raise RuntimeError(f"task baseline differs from the V5 certificate: {task_id}")
    record_path = ROOT / task["agent_record_path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record["task_id"] != task_id or record["seed"] != protocol["inference"]["seed"]:
        raise RuntimeError(f"agent record identity mismatch: {task_id}")
    if sha256_file(ROOT / protocol["inference"]["runner"]) != protocol["inference"]["runner_sha256"]:
        raise RuntimeError("local agent runner source differs from preregistration")
    task_json_path = ROOT / task["task_json_path"]
    prompt_path = ROOT / task["prompt_path"]
    if sha256_file(task_json_path) != task["task_json_sha256"]:
        raise RuntimeError(f"task JSON changed after preregistration: {task_id}")
    if sha256_file(prompt_path) != task["prompt_sha256"]:
        raise RuntimeError(f"task prompt changed after preregistration: {task_id}")
    model = protocol["model"]
    expected_model_files = {item["name"]: item["sha256"] for item in model["snapshot_files"]}
    if (record["model_repository"] != model["repository"]
            or record["model_snapshot"] != model["revision"]
            or record["model_files"] != expected_model_files):
        raise RuntimeError(f"model snapshot identity mismatch: {task_id}")
    expected_instruction = protocol["inference"]["instruction_template"].format(
        target_path=task["target_path"]
    )
    if not record["initial_messages"] or record["initial_messages"][-1] != {
        "role": "user", "content": expected_instruction,
    }:
        raise RuntimeError(f"agent instruction differs from preregistration: {task_id}")
    if task["prompt"].rstrip() not in record["initial_messages"][0]["content"]:
        raise RuntimeError(f"task prompt absent from frozen model system message: {task_id}")
    if record["task_input_hashes"].get("task.json") != task["task_json_sha256"]:
        raise RuntimeError(f"task JSON provenance mismatch: {task_id}")
    if record["task_input_hashes"].get("prompt_sha256") != task["prompt_sha256"]:
        raise RuntimeError(f"task prompt provenance mismatch: {task_id}")
    if record["source_hashes_before"].get(task["target_path"]) != task["baseline_sha256"]:
        raise RuntimeError(f"agent baseline source hash mismatch: {task_id}")
    if record["budgets"] != protocol["inference"]["runner_budgets"]:
        raise RuntimeError(f"agent budget differs from preregistration: {task_id}")
    diff = record["candidate_diff"]
    if sha256_bytes(diff.encode()) != record["candidate_diff_sha256"]:
        raise RuntimeError(f"agent diff digest mismatch: {task_id}")
    return record, record_path


def grade_one(task: dict, record: dict, record_path: Path, protocol: dict,
              v5_protocol: dict, source_repo: Path, dependencies: Path,
              build_python: Path, task_output: Path, task_worktree: Path) -> dict:
    task_output.mkdir(parents=True)
    result = {
        "task_id": task["task_id"],
        "agent_record_path": record_path.relative_to(ROOT).as_posix(),
        "agent_record_sha256": sha256_file(record_path),
        "candidate_diff_sha256": record["candidate_diff_sha256"],
        "agent_stop_reason": record["stop_reason"],
        "agent_elapsed_seconds": record["elapsed_seconds"],
        "agent_new_tokens": record["new_tokens"],
        "patch_applied": False,
        "verified_success": False,
    }
    try:
        git(source_repo, "worktree", "add", "--detach", str(task_worktree), protocol["source"]["base_revision"])
        if not record["candidate_diff"].strip():
            result["patch_status"] = "NO_DIFF"
            result["status"] = "NO_CANDIDATE_REPAIR"
            return result
        try:
            git(task_worktree, "apply", "--check", "--binary", "-", input_text=record["candidate_diff"])
            git(task_worktree, "apply", "--binary", "-", input_text=record["candidate_diff"])
        except subprocess.CalledProcessError as exc:
            result["patch_status"] = "PATCH_REJECTED"
            result["patch_error"] = (exc.stderr or "")[-4000:]
            result["status"] = "NO_CANDIDATE_REPAIR"
            return result

        changed = git(task_worktree, "diff", "--name-only", protocol["source"]["base_revision"]).splitlines()
        if changed != [task["target_path"]]:
            raise RuntimeError(f"candidate changed paths outside the task allowlist: {changed}")
        candidate_hash = sha256_file(task_worktree / task["target_path"])
        if candidate_hash != record["source_hashes_after"].get(task["target_path"]):
            raise RuntimeError("applied candidate source differs from the agent workspace hash")
        if candidate_hash == task["baseline_sha256"]:
            result["patch_status"] = "NO_SOURCE_CHANGE"
            result["status"] = "NO_CANDIDATE_REPAIR"
            return result
        result.update({"patch_status": "APPLIED", "patch_applied": True,
                       "candidate_source_sha256": candidate_hash})

        grader_root = task_output / "grader"
        grader_root.mkdir()
        required_names = {task["upstream_test_file"], "mlx_tests.py"}
        for spec in v5_protocol["grader_files"]:
            if spec["path"] not in required_names:
                continue
            content = subprocess.run(
                ["git", "-C", str(source_repo), "show", f"{spec['revision']}:{spec['source_path']}"],
                check=True, capture_output=True,
            ).stdout
            if sha256_bytes(content) != spec["sha256"]:
                raise RuntimeError(f"pinned grader source hash mismatch: {spec['path']}")
            with (grader_root / spec["path"]).open("xb") as stream:
                stream.write(content)

        limits = v5_protocol["resource_limits"]
        build_env = os.environ.copy()
        build_env.update({
            "CMAKE_BUILD_PARALLEL_LEVEL": str(limits["build_parallelism"]),
            "PATH": f"{build_python.parent}:{os.environ.get('PATH', '')}",
            "PYTHONPATH": f"{dependencies / 'nanobind' / 'src'}:{task_worktree / 'python'}",
            "CMAKE_ARGS": " ".join([
                "-DMLX_BUILD_METAL=OFF", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
                *[f"-DFETCHCONTENT_SOURCE_DIR_{spec['fetchcontent_name'].upper()}={dependencies / spec['path']}"
                  for _, spec in sorted(v5_protocol["dependency_sources"].items())],
            ]),
        })
        preflight = v5.preflight(limits, task_output)
        v5.write_json_new(task_output / "preflight.json", preflight)
        build_start = v5.preflight(limits, task_output)
        build = v5.monitored(
            ["/usr/bin/sandbox-exec", "-p", v5.NETWORK_PROFILE,
             str(build_python), "setup.py", "build_ext", "--inplace"],
            cwd=task_worktree, env=build_env,
            log_path=task_output / "build.log", samples_path=task_output / "build.resources.jsonl",
            limits=limits, start_state=build_start,
        )
        v5.write_json_new(task_output / "build.json", build)
        result["build"] = build
        if build["exit_code"] != 0 or build["abort_reason"]:
            result["status"] = "CANDIDATE_BUILD_FAILED"
            return result

        observed_start = v5.preflight(limits, task_output)
        observations = v5.run_observations(task_worktree, build_python, task_output, limits, observed_start)
        grader_task = next(item for item in v5_protocol["tasks"] if item["task_id"] == task["task_id"])
        upstream_tests = v5.run_upstream_tests(
            task_worktree, build_python, grader_root, task_output, [grader_task], limits,
        )
        decision = v5.task_decisions(observations["checks"], [grader_task], upstream_tests)[task["task_id"]]
        result.update({
            "independent_checks": decision["checks"],
            "upstream_test": upstream_tests[task["task_id"]],
            "verified_success": decision["passed"],
            "status": "VERIFIED_SUCCESS" if decision["passed"] else "CANDIDATE_REJECTED",
            "native_extensions": v5.native_extensions(task_worktree),
            "observations": observations,
        })
        v5.write_json_new(task_output / "decision.json", result)
        return result
    except Exception as exc:
        result.update({"status": "GRADER_ERROR", "error_type": type(exc).__name__, "error": str(exc)})
        v5.write_json_new(task_output / "failure.json", result)
        raise
    finally:
        if not (task_output / "manifest.json").exists():
            v5.write_manifest(task_output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--v5-protocol-sha256", required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--build-python", type=Path, required=True)
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_path = args.protocol.resolve()
    if sha256_file(protocol_path) != args.protocol_sha256:
        raise SystemExit("candidate grading protocol hash mismatch")
    v5_protocol_path = ROOT / "protocols/mlx_core_development_qualification_v5.json"
    if sha256_file(v5_protocol_path) != args.v5_protocol_sha256:
        raise SystemExit("independent V5 grader protocol hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    v5_protocol = json.loads(v5_protocol_path.read_text(encoding="utf-8"))
    source_repo = args.source_repo.resolve()
    dependencies = args.dependencies.resolve()
    build_python = args.build_python.expanduser().absolute()
    worktree_root = args.worktree_root.resolve()
    output = args.output.resolve()
    if output.exists() or worktree_root.exists():
        raise SystemExit("refusing to overwrite candidate grader output or worktree")
    if git(source_repo, "rev-parse", f"{protocol['source']['base_revision']}^{{commit}}") != protocol["source"]["base_revision"]:
        raise SystemExit("pinned MLX base revision is unavailable")
    if source_repo != (ROOT / protocol["build_environment"]["source_repo_path"]).resolve():
        raise SystemExit("source repository path differs from preregistration")
    if dependencies != (ROOT / protocol["build_environment"]["dependency_root"]).resolve():
        raise SystemExit("dependency root differs from preregistration")
    build_python_expected = (ROOT / protocol["build_environment"]["build_python_path"]).absolute()
    if build_python != build_python_expected:
        raise SystemExit("build Python path differs from preregistration")
    if worktree_root != (ROOT / protocol["build_environment"]["candidate_grader_worktree_root"]).resolve():
        raise SystemExit("candidate grader worktree path differs from preregistration")
    if output != (ROOT / protocol["output"]["grader_records"]).resolve():
        raise SystemExit("candidate grader output path differs from preregistration")
    for path, expected in protocol["implementation_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise SystemExit(f"pinned implementation changed: {path}")
    for name, spec in v5_protocol["dependency_sources"].items():
        path = dependencies / spec["path"]
        if not path.is_dir() or v5.tree_sha256(path) != spec["tree_sha256"]:
            raise SystemExit(f"pinned build dependency differs: {name}")
        if spec.get("kind") == "git" and git(path, "rev-parse", "HEAD") != spec["commit"]:
            raise SystemExit(f"pinned dependency commit differs: {name}")
        for relative, expected in spec.get("submodule_commits", {}).items():
            if git(path / relative, "rev-parse", "HEAD") != expected:
                raise SystemExit(f"pinned dependency submodule differs: {name}/{relative}")

    output.mkdir(parents=True)
    worktree_root.mkdir(parents=True)
    task_map = {task["task_id"]: task for task in v5_protocol["tasks"]}
    results = {}
    for task in protocol["tasks"]:
        record, record_path = verify_agent_record(task, protocol, task_map)
        task_output = output / task["task_id"]
        task_worktree = worktree_root / task["task_id"]
        try:
            results[task["task_id"]] = grade_one(
                task, record, record_path, protocol, v5_protocol, source_repo,
                dependencies, build_python, task_output, task_worktree,
            )
        except Exception as exc:
            results[task["task_id"]] = {
                "status": "GRADER_ERROR", "error_type": type(exc).__name__, "error": str(exc),
            }
            write_json_new(output / f"{task['task_id']}-failure.json", results[task["task_id"]])
            break
    successes = sum(item.get("verified_success") is True for item in results.values())
    summary = {
        "schema_version": 1,
        "protocol_sha256": args.protocol_sha256,
        "v5_protocol_sha256": args.v5_protocol_sha256,
        "status": "PASS_ONE_OR_MORE_CANDIDATE_REPAIRS" if successes else "NO_CANDIDATE_REPAIR_VERIFIED",
        "verified_successes": successes,
        "tasks_attempted": len(results),
        "tasks": results,
        "claim_limit": "Two fresh Development attempts; descriptive feasibility only, no training, held-out evaluation, or capability improvement claim.",
    }
    write_json_new(output / "cohort-summary.json", summary)
    v5.write_manifest(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
