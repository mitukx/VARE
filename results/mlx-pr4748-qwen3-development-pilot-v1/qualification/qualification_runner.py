#!/usr/bin/env python3
"""Execute frozen MLX #4748 baseline, incorrect-mutant, and known-fix phases."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import psutil

NETWORK_PROFILE = "(version 1) (allow default) (deny network*)"


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(x for x in root.rglob("*") if x.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha_file(path)))
    return digest.hexdigest()


def write_new(path: Path, value) -> None:
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def manifest(root: Path) -> None:
    rows = []
    for path in sorted(x for x in root.rglob("*") if x.is_file() and x.name != "manifest.json"):
        rows.append({"path": path.relative_to(root).as_posix(),
                     "size_bytes": path.stat().st_size, "sha256": sha_file(path)})
    write_new(root / "manifest.json", {"schema_version": 1, "algorithm": "sha256", "files": rows})


def git(source: Path, *args: str, binary=False):
    result = subprocess.run(["git", "-C", str(source), *args], check=True,
                            capture_output=True, text=not binary)
    return result.stdout


def host_state(path: Path) -> dict:
    memory, swap = psutil.virtual_memory(), psutil.swap_memory()
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "available_memory_bytes": memory.available, "total_memory_bytes": memory.total,
            "free_swap_bytes": swap.free, "swap_used_bytes": swap.used,
            "disk_free_bytes": shutil.disk_usage(path).free}


def check_preflight(limits: dict, path: Path) -> dict:
    state = host_state(path)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("requires native Darwin arm64 host")
    thresholds = {"available_memory_bytes": limits["minimum_available_memory_bytes"],
                  "free_swap_bytes": limits["minimum_free_swap_bytes"],
                  "disk_free_bytes": limits["minimum_disk_free_bytes"]}
    bad = [key for key, minimum in thresholds.items() if state[key] < minimum]
    if bad:
        raise RuntimeError("resource preflight failed: " + ", ".join(bad))
    return state


def run_monitored(command: list[str], cwd: Path, env: dict, output: Path,
                  label: str, limits: dict, initial: dict) -> dict:
    log_path = output / f"{label}.stdout.log"
    sample_path = output / f"{label}.resources.jsonl"
    started = time.monotonic()
    peak_rss, abort, code = 0, None, None
    with log_path.open("xb") as log, sample_path.open("x", encoding="utf-8") as samples:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        root = psutil.Process(proc.pid)
        while proc.poll() is None:
            state = host_state(cwd)
            try:
                members = [root, *root.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in members if p.is_running())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            peak_rss = max(peak_rss, rss)
            row = {**state, "process_tree_rss_bytes": rss}
            samples.write(json.dumps(row, sort_keys=True) + "\n")
            samples.flush()
            if state["available_memory_bytes"] < limits["minimum_available_memory_bytes"]:
                abort = "available_memory_below_floor"
            elif state["free_swap_bytes"] < limits["minimum_free_swap_bytes"]:
                abort = "free_swap_below_floor"
            elif state["swap_used_bytes"] - initial["swap_used_bytes"] > limits["max_swap_growth_bytes"]:
                abort = "swap_growth_above_limit"
            elif state["disk_free_bytes"] < limits["minimum_disk_free_bytes"]:
                abort = "disk_free_below_floor"
            elif time.monotonic() - started > limits["max_phase_seconds"]:
                abort = "phase_wall_limit"
            if abort:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                break
            time.sleep(limits["sample_interval_seconds"])
        code = proc.wait()
        log.flush(); os.fsync(log.fileno())
        samples.flush(); os.fsync(samples.fileno())
    return {"argv": command, "cwd": str(cwd), "network_profile": NETWORK_PROFILE,
            "exit_code": code, "abort_reason": abort,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_process_tree_rss_bytes": peak_rss,
            "stdout_sha256": sha_file(log_path), "samples_sha256": sha_file(sample_path)}


def read_one_json(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"expected one JSON record in {path.name}, got {len(lines)}")
    return json.loads(lines[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--build-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    protocol_bytes = protocol_path.read_bytes()
    if sha_bytes(protocol_bytes) != args.protocol_sha256:
        raise RuntimeError("protocol hash mismatch")
    protocol = json.loads(protocol_bytes)
    task = protocol["task"]
    limits = protocol["build_environment"]["resource_limits"]
    repo, deps = args.source_repo.resolve(), args.dependencies.resolve()
    output, source = args.output.resolve(), args.source_worktree.resolve()
    py = args.build_python.expanduser().absolute()
    if output.exists() or source.exists():
        raise RuntimeError("output or source workspace already exists")
    output.mkdir(parents=True)
    run_status = "NO_GO"
    phases = {}
    try:
        if sha_file(protocol_path) != args.protocol_sha256:
            raise RuntimeError("protocol changed after launch")
        if git(repo, "rev-parse", f"{task['baseline_commit']}^{{commit}}").strip() != task["baseline_commit"]:
            raise RuntimeError("baseline commit unavailable")
        if git(repo, "rev-parse", f"{task['fixed_commit']}^{{commit}}").strip() != task["fixed_commit"]:
            raise RuntimeError("fixed commit unavailable")
        fixed = git(repo, "show", f"{task['fixed_commit']}:{task['target_path']}", binary=True)
        if sha_bytes(fixed) != task["fixed_sha256"]:
            raise RuntimeError("fixed source hash mismatch")
        old = b"linalg::tri_inv(a, upper_, stream())"
        new = b"linalg::tri_inv(a, !upper_, stream())"
        if fixed.count(old) != 1 or sha_bytes(fixed.replace(old, new, 1)) != task["mutant_sha256"]:
            raise RuntimeError("predeclared mutant anchor/hash mismatch")
        lock = protocol["build_environment"]
        for name, spec in lock["dependency_sources"].items():
            dep = deps / spec["path"]
            if not dep.is_dir() or tree_sha(dep) != spec["tree_sha256"]:
                raise RuntimeError(f"dependency tree hash mismatch: {name}")
            if spec.get("kind") == "git" and git(dep, "rev-parse", "HEAD").strip() != spec["commit"]:
                raise RuntimeError(f"dependency commit mismatch: {name}")
        freeze = subprocess.check_output([str(py), "-m", "pip", "freeze", "--all"], text=True).splitlines()
        expected_freeze = lock["python_build_environment"]["build_python_freeze"]
        if freeze != expected_freeze:
            raise RuntimeError("build Python package inventory mismatch")
        start = check_preflight(limits, output)
        write_new(output / "preflight.json", start)
        archive = output / "baseline-source.tar"
        with archive.open("xb") as stream:
            subprocess.run(["git", "-C", str(repo), "archive", "--format=tar",
                            task["baseline_commit"]], check=True, stdout=stream)
            stream.flush(); os.fsync(stream.fileno())
        source.mkdir(parents=True)
        with tarfile.open(archive, "r:") as tar:
            tar.extractall(source, filter="data")
        baseline_file = source / task["target_path"]
        if sha_file(baseline_file) != task["baseline_sha256"]:
            raise RuntimeError("extracted baseline source hash mismatch")
        grader = output / "grader"
        grader.mkdir()
        test_specs = protocol["grader_files"]
        for spec in test_specs:
            body = git(repo, "show", f"{spec['revision']}:{spec['source_path']}", binary=True)
            if sha_bytes(body) != spec["sha256"]:
                raise RuntimeError("pinned grader hash mismatch: " + spec["path"])
            path = grader / spec["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(body)
                stream.flush(); os.fsync(stream.fileno())
        venv_freeze = subprocess.check_output([str(py), "-m", "pip", "freeze", "--all"], text=True)
        (output / "build-python-freeze.txt").write_text(venv_freeze, encoding="utf-8")
        env = os.environ.copy()
        env.update({"PYPI_RELEASE": "1",
                    "CMAKE_BUILD_PARALLEL_LEVEL": str(limits["build_parallelism"]),
                    "PATH": f"{py.parent}:{os.environ.get('PATH', '')}",
                    "PYTHONPATH": f"{deps / 'nanobind' / 'src'}:{source / 'python'}"})
        args_cmake = ["-DMLX_BUILD_METAL=OFF", "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"]
        for _, spec in sorted(lock["dependency_sources"].items()):
            args_cmake.append(f"-DFETCHCONTENT_SOURCE_DIR_{spec['fetchcontent_name'].upper()}={deps / spec['path']}")
        env["CMAKE_ARGS"] = " ".join(args_cmake)
        baseline_archive_hash = sha_file(archive)
        for phase in task["phase_order"]:
            phase_dir = output / phase
            phase_dir.mkdir()
            if phase == "baseline":
                target_content = git(repo, "show", f"{task['baseline_commit']}:{task['target_path']}", binary=True)
            elif phase == "fixed":
                target_content = fixed
            else:
                target_content = fixed.replace(old, new, 1)
            baseline_file.write_bytes(target_content)
            phase_hash = sha_file(baseline_file)
            expected_hash = {"baseline": task["baseline_sha256"], "mutant": task["mutant_sha256"],
                             "fixed": task["fixed_sha256"]}[phase]
            if phase_hash != expected_hash:
                raise RuntimeError(f"{phase} target hash mismatch")
            build_start = check_preflight(limits, output)
            build = run_monitored(["/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE,
                                   str(py), "setup.py", "build_ext", "--inplace"],
                                  source, env, phase_dir, "build", limits, build_start)
            record = {"phase": phase, "target_sha256": phase_hash,
                      "build": build, "baseline_archive_sha256": baseline_archive_hash}
            if build["exit_code"] != 0 or build["abort_reason"]:
                record["status"] = "BUILD_FAILED"
                write_new(phase_dir / "phase.json", record)
                phases[phase] = record
                break
            runtime_env = os.environ.copy()
            runtime_env.update({"PYTHONPATH": f"{grader}:{source / 'python'}",
                                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                                "DEVICE": "cpu"})
            oracle_start = check_preflight(limits, output)
            oracle = run_monitored(["/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE,
                                    str(py), str(protocol_path.parent / "grader/independent_oracle.py")],
                                   source, runtime_env, phase_dir, "oracle", limits, oracle_start)
            oracle_record = read_one_json(phase_dir / "oracle.stdout.log") if oracle["exit_code"] in (0, 1) and not oracle["abort_reason"] else None
            test_start = check_preflight(limits, output)
            selector = task["upstream_test_selector"]
            test = run_monitored(["/usr/bin/sandbox-exec", "-p", NETWORK_PROFILE,
                                  str(py), "-m", "unittest", "-v", selector],
                                 grader, runtime_env, phase_dir, "upstream-test", limits, test_start)
            cases = (oracle_record or {}).get("cases", {})
            direct = [v["passed"] for k, v in cases.items() if k.endswith("_direct")]
            vmapped = [v["passed"] for k, v in cases.items() if "vmap" in k]
            semantic = {"direct_all_pass": len(direct) == 2 and all(direct),
                        "any_vmapped_fail": len(vmapped) == 4 and not all(vmapped),
                        "all_cases_pass": len(cases) == 6 and all(v["passed"] for v in cases.values())}
            if phase in ("baseline", "mutant"):
                accepted = semantic["direct_all_pass"] and semantic["any_vmapped_fail"] and test["exit_code"] != 0
            else:
                accepted = semantic["all_cases_pass"] and test["exit_code"] == 0
            record.update({"oracle_runtime": oracle, "oracle": oracle_record,
                           "upstream_test": test, "semantic_checks": semantic,
                           "phase_accepted": bool(accepted),
                           "status": ("CONTROL_EXPECTED" if accepted and phase != "fixed" else
                                      "KNOWN_FIX_ACCEPTED" if accepted else "CONTROL_MISMATCH")})
            write_new(phase_dir / "phase.json", record)
            phases[phase] = record
            if not accepted or oracle["abort_reason"] or test["abort_reason"]:
                break
        run_status = ("PASS_TASK_CERTIFICATION_PILOT"
                      if set(phases) == {"baseline", "mutant", "fixed"}
                      and phases["baseline"]["phase_accepted"]
                      and phases["mutant"]["phase_accepted"]
                      and phases["fixed"]["phase_accepted"] else "NO_GO")
        result = {"protocol_id": protocol["protocol_id"], "protocol_sha256": args.protocol_sha256,
                  "runner_sha256": sha_file(Path(__file__).resolve()),
                  "task_id": task["task_id"], "role": protocol["task_role"],
                  "source_repo": str(repo), "baseline_commit": task["baseline_commit"],
                  "fixed_commit": task["fixed_commit"], "controller_host": platform.platform(),
                  "cgroup_memory": "unavailable on macOS; not claimed",
                  "model_inference_calls": 0, "training_updates": 0,
                  "phases": phases, "status": run_status,
                  "claim_limit": protocol["claim_limit"]}
        write_new(output / "summary.json", result)
        return 0 if run_status == "PASS_TASK_CERTIFICATION_PILOT" else 2
    except BaseException as exc:
        if not (output / "partial-failure.json").exists():
            write_new(output / "partial-failure.json", {
                "status": "EXECUTION_ABORTED", "exception_type": type(exc).__name__,
                "message": str(exc), "time_utc": datetime.now(timezone.utc).isoformat(),
                "phase_records_so_far": sorted(phases)})
        raise
    finally:
        if not (output / "manifest.json").exists():
            manifest(output)


if __name__ == "__main__":
    raise SystemExit(main())
