#!/usr/bin/env python3
"""Build and qualify two prospective CPU/Apple-Silicon MLX repair families."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlx_core_development_qualification_v1.json"
OBSERVATIONS = ROOT / "scripts/mlx_core_task_observations_v1.py"
TEST_LAUNCHER = ROOT / "scripts/mlx_core_upstream_test_launcher_v1.py"
NETWORK_PROFILE = "(version 1) (allow default) (deny network*)"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value: dict) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)


def write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "manifest.json":
            continue
        rows.append({
            "path": path.relative_to(output).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    write_json_new(output / "manifest.json", {
        "schema_version": 1,
        "algorithm": "sha256",
        "files": rows,
    })


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*")
        if item.is_file() and ".git" not in item.relative_to(root).parts
    ):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git(source: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(source), *args], check=True,
        capture_output=True, text=not binary,
    )
    return result.stdout


def host_state(path: Path) -> dict[str, int | float | str]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "memory_available_bytes": memory.available,
        "memory_total_bytes": memory.total,
        "swap_free_bytes": swap.free,
        "swap_used_bytes": swap.used,
        "disk_free_bytes": shutil.disk_usage(path).free,
    }


def preflight(limits: dict, path: Path) -> dict:
    state = host_state(path)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("qualification requires the pinned native Apple Silicon path")
    checks = {
        "memory_available_bytes": limits["minimum_available_memory_bytes"],
        "swap_free_bytes": limits["minimum_free_swap_bytes"],
        "disk_free_bytes": limits["minimum_disk_free_bytes"],
    }
    failed = [key for key, minimum in checks.items() if state[key] < minimum]
    if failed:
        raise RuntimeError("resource preflight failed: " + ", ".join(failed))
    return state


def monitored(command: list[str], *, cwd: Path, env: dict[str, str],
              log_path: Path, samples_path: Path, limits: dict,
              start_state: dict) -> dict:
    started = time.monotonic()
    max_rss = 0
    growth_limit = limits["max_swap_growth_bytes"]
    timeout = limits["max_phase_seconds"]
    with log_path.open("wb") as log, samples_path.open("w", encoding="utf-8") as samples:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(proc.pid)
        abort = None
        while proc.poll() is None:
            state = host_state(cwd)
            try:
                rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            max_rss = max(max_rss, rss)
            samples.write(json.dumps({**state, "process_tree_rss_bytes": rss}, sort_keys=True) + "\n")
            samples.flush()
            if state["memory_available_bytes"] < limits["minimum_available_memory_bytes"]:
                abort = "available_memory_below_floor"
            elif state["swap_free_bytes"] < limits["minimum_free_swap_bytes"]:
                abort = "free_swap_below_floor"
            elif state["swap_used_bytes"] - start_state["swap_used_bytes"] > growth_limit:
                abort = "swap_growth_above_limit"
            elif state["disk_free_bytes"] < limits["minimum_disk_free_bytes"]:
                abort = "disk_free_below_floor"
            elif time.monotonic() - started > timeout:
                abort = "phase_wall_limit"
            if abort:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                break
            time.sleep(limits["sample_interval_seconds"])
        code = proc.wait()
    return {
        "command": command,
        "exit_code": code,
        "abort_reason": abort,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_process_tree_rss_bytes": max_rss,
        "log_sha256": sha256_file(log_path),
        "samples_sha256": sha256_file(samples_path),
    }


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


def task_decisions(checks: dict[str, bool], tasks: list[dict],
                   upstream_tests: dict[str, dict]) -> dict[str, dict]:
    groups = {task["task_id"]: task["oracle_checks"] for task in tasks}
    if any(not set(keys).issubset(checks) for keys in groups.values()):
        raise RuntimeError("protocol refers to an unknown independent oracle check")
    return {
        task_id: {"checks": {key: checks[key] for key in keys},
                  "upstream_test_passed": upstream_tests[task_id]["passed"],
                  "passed": all(checks[key] for key in keys)
                  and upstream_tests[task_id]["passed"]}
        for task_id, keys in groups.items()
    }


def native_extensions(worktree: Path) -> list[dict[str, str | int]]:
    files = sorted((worktree / "python" / "mlx").rglob("*.so"))
    if not files:
        raise RuntimeError("native MLX Python extension was not produced")
    return [{
        "path": path.relative_to(worktree).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    } for path in files]


def run_observations(worktree: Path, python: Path, output: Path, limits: dict, start_state: dict) -> dict:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(worktree / "python"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "DEVICE": "cpu",
    })
    command = [
        "/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE,
        str(python), str(OBSERVATIONS), "--json",
    ]
    result = monitored(
        command, cwd=worktree, env=env,
        log_path=output / "observations.stdout.jsonl",
        samples_path=output / "observations.resources.jsonl",
        limits=limits, start_state=start_state,
    )
    if result["exit_code"] != 0 or result["abort_reason"]:
        raise RuntimeError(f"observation process failed: {result}")
    raw_lines = (output / "observations.stdout.jsonl").read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != 1:
        raise RuntimeError("observation process must emit exactly one JSON line")
    observed = json.loads(raw_lines[0])
    module_path = Path(observed.get("mlx_core_module_path", "")).resolve()
    try:
        module_path.relative_to((worktree / "python").resolve())
    except ValueError as exc:
        raise RuntimeError("observation imported mlx.core outside the pinned source worktree") from exc
    return {"runtime": result, "observed": observed, "checks": expected_checks(observed)}


def run_upstream_tests(worktree: Path, python: Path, grader_root: Path, output: Path,
                       tasks: list[dict], limits: dict) -> dict[str, dict]:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": f"{grader_root}:{worktree / 'python'}",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "DEVICE": "cpu",
    })
    records = {}
    for task in tasks:
        task_id = task["task_id"]
        start_state = preflight(limits, output)
        run = monitored(
            ["/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE,
             str(python), str(TEST_LAUNCHER), task["upstream_test_selector"]],
            cwd=grader_root, env=env,
            log_path=output / f"{task_id}.upstream-test.log",
            samples_path=output / f"{task_id}.upstream-test.resources.jsonl",
            limits=limits, start_state=start_state,
        )
        records[task_id] = {**run, "passed": run["exit_code"] == 0 and run["abort_reason"] is None}
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--build-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    source_repo = args.source_repo.resolve()
    dependencies = args.dependencies.resolve()
    build_python = args.build_python.resolve()
    output = args.output.resolve()
    worktree = args.worktree.resolve()
    if sha256_file(PROTOCOL) != args.protocol_sha256:
        raise RuntimeError("frozen protocol hash mismatch")
    if output.exists() or worktree.exists():
        raise RuntimeError("refusing to overwrite output or source worktree")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    controller_revision = git(ROOT, "rev-parse", "HEAD").strip()
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                    protocol["controller_parent_revision"], controller_revision], check=True)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--"], check=True)
    subprocess.run(["git", "-C", str(ROOT), "diff", "--cached", "--quiet"], check=True)
    for path_text, expected in protocol["implementation_sha256"].items():
        path = ROOT / path_text
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"pinned implementation source differs: {path_text}")
    for path_text, expected in protocol["preregistration_inputs"].items():
        path = ROOT / path_text
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"pinned preregistration input differs: {path_text}")
    limits = protocol["resource_limits"]
    installed_freeze = subprocess.check_output(
        [str(build_python), "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    if installed_freeze != protocol["build_python_freeze"]:
        raise RuntimeError("build Python package inventory differs from frozen protocol")
    output.mkdir(parents=True)
    try:
        start = preflight(limits, ROOT)
    except RuntimeError as exc:
        write_json_new(output / "preflight-failure.json", {
            "status": "RESOURCE_OR_PLATFORM_PREFLIGHT_FAILED",
            "reason": str(exc),
            "state": host_state(ROOT),
        })
        write_manifest(output)
        return 2
    write_json_new(output / "preflight.json", start)
    base = protocol["base_revision"]
    if git(source_repo, "rev-parse", f"{base}^{{commit}}").strip() != base:
        raise RuntimeError("source base revision is unavailable")
    for task in protocol["tasks"]:
        baseline = git(source_repo, "show", f"{base}:{task['target_path']}", binary=True)
        fixed = git(source_repo, "show", f"{task['fixed_revision']}:{task['target_path']}", binary=True)
        upstream_test = git(source_repo, "show", f"{task['fixed_revision']}:{task['upstream_test_path']}", binary=True)
        if sha256_bytes(baseline) != task["baseline_sha256"]:
            raise RuntimeError(f"protocol baseline hash mismatch: {task['task_id']}")
        if sha256_bytes(fixed) != task["fixed_sha256"]:
            raise RuntimeError(f"protocol known-fix hash mismatch: {task['task_id']}")
        if sha256_bytes(upstream_test) != task["upstream_test_sha256"]:
            raise RuntimeError(f"protocol upstream-test hash mismatch: {task['task_id']}")
    git(source_repo, "worktree", "add", "--detach", str(worktree), base)
    for name, spec in protocol["dependency_sources"].items():
        path = dependencies / spec["path"]
        if not path.is_dir() or tree_sha256(path) != spec["tree_sha256"]:
            raise RuntimeError(f"pinned build dependency differs: {name}")
        if "archive_filename" in spec:
            archive = dependencies / spec["archive_filename"]
            if not archive.is_file() or sha256_file(archive) != spec["archive_sha256"]:
                raise RuntimeError(f"pinned dependency archive differs: {name}")
        if spec["kind"] == "git":
            commit = git(path, "rev-parse", "HEAD").strip()
            if commit != spec["commit"]:
                raise RuntimeError(f"build dependency revision differs: {name}")
    grader_root = output / "grader"
    grader_root.mkdir()
    grader_files = {}
    for spec in protocol["grader_files"]:
        relative = Path(spec["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("grader file path escapes the isolated grader directory")
        content = git(
            source_repo, "show", f"{spec['revision']}:{spec['source_path']}", binary=True
        )
        observed_hash = sha256_bytes(content)
        if observed_hash != spec["sha256"]:
            raise RuntimeError(f"pinned grader source differs: {spec['path']}")
        destination = grader_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(content)
        grader_files[relative.as_posix()] = observed_hash
    before_files = {}
    for task in protocol["tasks"]:
        target = task["target_path"]
        if sha256_file(worktree / target) != task["baseline_sha256"]:
            raise RuntimeError(f"baseline source mismatch: {task['task_id']}")
        before_files[task["task_id"]] = {
            "target_path": target,
            "baseline_sha256": sha256_file(worktree / target),
            "fixed_sha256": task["fixed_sha256"],
        }

    phases: dict[str, dict] = {}
    build_env = os.environ.copy()
    build_env.update({
        "CMAKE_BUILD_PARALLEL_LEVEL": str(limits["build_parallelism"]),
        "PATH": f"{build_python.parent}:{os.environ.get('PATH', '')}",
        "PYTHONPATH": f"{dependencies / 'nanobind' / 'src'}:{worktree / 'python'}",
        "CMAKE_ARGS": " ".join([
            "-DMLX_BUILD_METAL=OFF",
            "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            f"-DFETCHCONTENT_SOURCE_DIR_JSON={dependencies / 'json'}",
            f"-DFETCHCONTENT_SOURCE_DIR_FMT={dependencies / 'fmt'}",
            f"-DFETCHCONTENT_SOURCE_DIR_NANOBIND={dependencies / 'nanobind'}",
        ]),
    })
    for phase in ("baseline", "fixed", "mutant"):
        phase_dir = output / phase
        phase_dir.mkdir()
        if phase != "baseline":
            if phase == "fixed":
                for task in protocol["tasks"]:
                    content = git(source_repo, "show", f"{task['fixed_revision']}:{task['target_path']}", binary=True)
                    target = worktree / task["target_path"]
                    target.write_bytes(content)
                    if sha256_file(target) != task["fixed_sha256"]:
                        raise RuntimeError(f"fixed source mismatch: {task['task_id']}")
            else:
                for task in protocol["tasks"]:
                    target = worktree / task["target_path"]
                    content = target.read_bytes()
                    for mutation in task["mutations"]:
                        old = mutation["old"].encode()
                        if content.count(old) != 1:
                            raise RuntimeError(f"mutant anchor is not unique: {task['task_id']}")
                        content = content.replace(old, mutation["new"].encode(), 1)
                    target.write_bytes(content)
                    if sha256_file(target) != task["mutant_sha256"]:
                        raise RuntimeError(f"mutant source mismatch: {task['task_id']}")
        phase_source_hashes = {
            task["task_id"]: sha256_file(worktree / task["target_path"])
            for task in protocol["tasks"]
        }
        expected_hash_key = {
            "baseline": "baseline_sha256",
            "fixed": "fixed_sha256",
            "mutant": "mutant_sha256",
        }[phase]
        if any(
            phase_source_hashes[task["task_id"]] != task[expected_hash_key]
            for task in protocol["tasks"]
        ):
            raise RuntimeError(f"{phase}: task source hash inventory differs before build")
        build_start = preflight(limits, output)
        build = monitored(
            ["/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE,
             str(build_python), "setup.py", "build_ext", "--inplace"],
            cwd=worktree, env=build_env,
            log_path=phase_dir / "build.log",
            samples_path=phase_dir / "build.resources.jsonl",
            limits=limits, start_state=build_start,
        )
        if build["exit_code"] != 0 or build["abort_reason"]:
            phases[phase] = {
                "source_hashes": phase_source_hashes,
                "build": build,
                "status": "BUILD_FAILED",
            }
            write_json_new(phase_dir / "phase.json", phases[phase])
            break
        observed_start = preflight(limits, output)
        built_extensions = native_extensions(worktree)
        checks = run_observations(worktree, build_python, phase_dir, limits, observed_start)
        upstream_tests = run_upstream_tests(
            worktree, build_python, grader_root, phase_dir, protocol["tasks"], limits
        )
        task_results = task_decisions(checks["checks"], protocol["tasks"], upstream_tests)
        if phase == "baseline":
            controls_ok = all(not task["passed"] for task in task_results.values())
        elif phase == "fixed":
            controls_ok = all(task["passed"] for task in task_results.values())
        else:
            controls_ok = all(not task["passed"] for task in task_results.values())
        phase_status = {
            "baseline": "BASELINE_REJECTED" if controls_ok else "BASELINE_FALSE_ACCEPTANCE",
            "fixed": "KNOWN_FIX_ACCEPTED" if controls_ok else "KNOWN_FIX_REJECTED",
            "mutant": "MUTANT_REJECTED" if controls_ok else "MUTANT_FALSE_ACCEPTANCE",
        }[phase]
        phases[phase] = {"source_hashes": phase_source_hashes,
                         "build": build, "native_extensions": built_extensions,
                         **checks, "upstream_tests": upstream_tests,
                         "task_decisions": task_results,
                         "status": phase_status}
        write_json_new(phase_dir / "phase.json", phases[phase])
        if phase_status in {"BUILD_FAILED", "BASELINE_FALSE_ACCEPTANCE", "KNOWN_FIX_REJECTED", "MUTANT_FALSE_ACCEPTANCE"}:
            break
    summary = {
        "protocol_sha256": args.protocol_sha256,
        "source_base_revision": base,
        "source_repo_url": protocol["source_repo_url"],
        "controller_revision": controller_revision,
        "controller_tracked_worktree_clean": True,
        "host": {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version},
        "toolchain": {
            name: subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
            for name, command in {
                "compiler": ["clang", "--version"],
                "cmake": [str(build_python.parent / "cmake"), "--version"],
                "git": ["git", "--version"],
                "build_python": [str(build_python), "--version"],
            }.items()
        },
        "grader_files": grader_files,
        "model_inference_calls": 0,
        "training_updates": 0,
        "networked_grader_calls": 0,
        "inputs": before_files,
        "phases": phases,
        "status": "PASS_TASK_CERTIFICATION_PILOT" if set(phases) == {"baseline", "fixed", "mutant"}
        and phases["baseline"]["status"] == "BASELINE_REJECTED"
        and phases["fixed"]["status"] == "KNOWN_FIX_ACCEPTED"
        and phases["mutant"]["status"] == "MUTANT_REJECTED" else "NO_GO",
        "claim_limit": protocol["claim_limit"],
    }
    write_json_new(output / "summary.json", summary)
    write_manifest(output)
    return 0 if summary["status"] == "PASS_TASK_CERTIFICATION_PILOT" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        try:
            output_index = sys.argv.index("--output") + 1
            failure_dir = Path(sys.argv[output_index]).resolve()
        except (ValueError, IndexError):
            failure_dir = None
        if failure_dir is not None and failure_dir.is_dir() and not (failure_dir / "manifest.json").exists():
            failure = {
                "schema_version": 1,
                "status": "INCOMPLETE_FAILURE",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                write_json_new(failure_dir / "partial-failure.json", failure)
                write_manifest(failure_dir)
            except FileExistsError:
                pass
        raise
