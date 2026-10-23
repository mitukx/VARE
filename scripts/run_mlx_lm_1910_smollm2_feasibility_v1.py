#!/usr/bin/env python3
"""Run one locked, matched SmolLM2 MLX feasibility pair on Development task #1910."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK_REL = "benchmarks/mlsys_repo_eval_v3/mlxlm_1910_agent_task"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observe() -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import monitor_local_agent_resources_v1 as monitor
    return monitor.observe()


def verify(protocol_path: Path, model_path: Path, python_exe: Path,
           git_source: Path) -> dict[str, Any]:
    lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    sidecar_path = protocol_path.with_name("protocol.sha256")
    if sha256(protocol_path) != sidecar_path.read_text(encoding="utf-8").split()[0]:
        raise ValueError("protocol sidecar SHA-256 mismatch")
    if lock.get("status") != "FROZEN_BEFORE_RUN":
        raise ValueError("protocol is not frozen before inference")
    if lock.get("protocol_id") != "mlsys-repo-task-mlx-lm-1910-smollm2-feasibility-v1":
        raise ValueError("unexpected protocol identity")
    if lock["task"].get("exposure") != "DEVELOPMENT_ONLY":
        raise ValueError("task exposure must remain Development-only")
    ancestor = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
         lock["implementation"]["commit"], "HEAD"], check=False,
    )
    if ancestor.returncode:
        raise ValueError("implementation commit is not an ancestor of this checkout")
    for relative, expected in lock["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked code SHA-256 mismatch: {relative}")
    if (lock["grader"]["grader_sha256"] != lock["code_sha256"]["scripts/grade_mlx_lm_1910_candidate_v1.py"]
            or lock["grader"]["calibration_runner_sha256"]
            != lock["code_sha256"]["scripts/calibrate_mlx_lm_1910_candidate_grader_v1.py"]):
        raise ValueError("grader protocol does not attest the calibrated grader sources")
    for relative, expected in lock["task"]["bundle_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked task SHA-256 mismatch: {relative}")
    task_workspace = ROOT / TASK_REL / "workspace"
    git_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp",
               "GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_CONFIG_GLOBAL": os.devnull}
    task_head = subprocess.run(
        ["git", "-C", str(task_workspace), "rev-parse", "HEAD"],
        env=git_env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    task_status = subprocess.run(
        ["git", "-C", str(task_workspace), "status", "--porcelain"],
        env=git_env, capture_output=True, text=True, check=True,
    ).stdout
    if task_head != lock["task"]["vare_workspace_base_commit"] or task_status:
        raise ValueError("candidate workspace Git baseline or cleanliness mismatch")
    fixed_commit_probe = subprocess.run(
        ["git", "-C", str(task_workspace), "cat-file", "-e",
         f"{lock['task']['upstream_fixed_commit']}^{{commit}}"],
        env=git_env, capture_output=True, check=False,
    )
    if fixed_commit_probe.returncode == 0:
        raise ValueError("known-fix commit is present in the candidate workspace object database")
    license_path = ROOT / lock["task"]["license_provenance_path"]
    if sha256(license_path) != lock["task"]["license_provenance_sha256"]:
        raise ValueError("task license provenance identity mismatch")
    prior_exposure_path = ROOT / lock["task"]["prior_exposure_manifest_path"]
    if sha256(prior_exposure_path) != lock["task"]["prior_exposure_manifest_sha256"]:
        raise ValueError("prior Development exposure manifest identity mismatch")
    prior_exposure_sidecar = prior_exposure_path.with_name("evidence-manifest.sha256")
    if (sha256(prior_exposure_sidecar) != lock["task"]["prior_exposure_manifest_sidecar_sha256"]
            or prior_exposure_sidecar.read_text(encoding="utf-8").split()[0] != sha256(prior_exposure_path)):
        raise ValueError("prior Development exposure manifest sidecar mismatch")
    prior_exposure = json.loads(prior_exposure_path.read_text(encoding="utf-8"))
    for item in prior_exposure["files"]:
        path = ROOT / item["path"]
        if (not path.is_file() or path.stat().st_size != item["size_bytes"]
                or sha256(path) != item["sha256"]):
            raise ValueError(f"prior exposure evidence changed: {item['path']}")

    calibration_path = ROOT / lock["grader"]["calibration_path"]
    if sha256(calibration_path) != lock["grader"]["calibration_sha256"]:
        raise ValueError("grader calibration evidence SHA-256 mismatch")
    for relative, expected_hash in lock["grader"]["calibration_files_sha256"].items():
        if sha256(ROOT / relative) != expected_hash:
            raise ValueError(f"grader calibration raw evidence mismatch: {relative}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "CALIBRATION_PASS_DEVELOPMENT_ONLY":
        raise ValueError("grader calibration is not passing")
    expected = {"baseline": "FAIL", "known_fix": "PASS",
                "mutant_restore_default_string": "FAIL",
                "mutant_parse_any_json_scalar": "FAIL", "mutant_object_only": "FAIL"}
    records = {item["arm"]: item["grader_status"] for item in calibration["records"]}
    if records != expected:
        raise ValueError("grader calibration did not retain the locked baseline/fix/mutant pattern")

    prior_protocol = ROOT / lock["model"]["resource_smoke_protocol_path"]
    if sha256(prior_protocol) != lock["model"]["resource_smoke_protocol_sha256"]:
        raise ValueError("prior load-smoke protocol identity mismatch")
    prior_protocol_sidecar = prior_protocol.with_name("protocol.sha256")
    if sha256(prior_protocol) != prior_protocol_sidecar.read_text(encoding="utf-8").split()[0]:
        raise ValueError("prior load-smoke protocol sidecar mismatch")
    prior = json.loads(prior_protocol.read_text(encoding="utf-8"))
    if prior.get("protocol_id") != "vare.local-mlx-model-load-smoke.v3":
        raise ValueError("model has no matching completed load smoke")
    if lock["model"]["snapshot_revision"] != prior["model"]["revision"]:
        raise ValueError("model revision differs from the completed load smoke")
    smoke_result_path = ROOT / lock["model"]["resource_smoke_result_path"]
    if sha256(smoke_result_path) != lock["model"]["resource_smoke_result_sha256"]:
        raise ValueError("completed load-smoke result identity mismatch")
    smoke_result = json.loads(smoke_result_path.read_text(encoding="utf-8"))
    if (smoke_result.get("status") != "COMPLETED" or smoke_result.get("generated_tokens") != 0
            or smoke_result.get("optimizer_steps") != 0 or not smoke_result.get("load_result", {}).get("loaded")):
        raise ValueError("referenced model-load smoke is not a zero-generation resource pass")
    smoke_manifest_path = ROOT / lock["model"]["resource_smoke_manifest_path"]
    if sha256(smoke_manifest_path) != lock["model"]["resource_smoke_manifest_sha256"]:
        raise ValueError("completed load-smoke evidence manifest identity mismatch")
    smoke_manifest = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
    for relative, identity in smoke_manifest["raw_files"].items():
        path = ROOT / relative
        if not path.is_file() or path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"completed load-smoke raw evidence mismatch: {relative}")
    model_path = model_path.resolve(strict=True)
    if model_path.name != lock["model"]["snapshot_revision"]:
        raise ValueError("local model snapshot revision mismatch")
    for name, identity in prior["model"]["files"].items():
        file_path = model_path / name
        if (not file_path.is_file() or file_path.stat().st_size != identity["bytes"]
                or sha256(file_path) != identity["sha256"]):
            raise ValueError(f"model checkpoint file mismatch: {name}")

    runtime_lock = ROOT / lock["runtime"]["freeze_path"]
    if sha256(runtime_lock) != lock["runtime"]["freeze_sha256"]:
        raise ValueError("runtime freeze identity mismatch")
    probe = subprocess.run(
        [str(python_exe.absolute()), "-c",
         "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in "
         "('mlx-lm','mlx','mlx-metal','transformers')}))"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode or json.loads(probe.stdout) != lock["runtime"]["packages"]:
        raise ValueError("installed MLX runtime does not match the frozen package set")
    docker_image = subprocess.run(
        ["docker", "--context", "colima", "image", "inspect", lock["runtime"]["public_test_image"],
         "--format", "{{.Id}}"], capture_output=True, text=True, check=False,
    )
    if docker_image.returncode or docker_image.stdout.strip() != lock["runtime"]["public_test_image"]:
        raise ValueError("public-test container image identity mismatch")
    docker_info = subprocess.run(
        ["docker", "--context", "colima", "info", "--format", "{{.MemTotal}} {{.NCPU}}"],
        capture_output=True, text=True, check=False,
    )
    try:
        docker_memory, docker_cpus = map(int, docker_info.stdout.split())
    except (ValueError, TypeError):
        raise ValueError("Colima memory/CPU preflight is unavailable") from None
    if (docker_info.returncode or docker_memory > lock["resources"]["max_colima_memory_bytes"]
            or docker_cpus > lock["resources"]["max_colima_cpus"]):
        raise ValueError("Colima runtime exceeds frozen CPU or memory cap")
    if not git_source.is_dir() or git_source.is_symlink():
        raise ValueError("pinned upstream Git source must be a real directory")
    for revision in (lock["task"]["upstream_base_commit"],
                     lock["task"]["upstream_fixed_commit"]):
        source_check = subprocess.run(
            ["git", "-C", str(git_source), "-c", "core.hooksPath=/dev/null",
             "cat-file", "-e", f"{revision}^{{commit}}"],
            env=git_env, capture_output=True, check=False,
        )
        if source_check.returncode:
            raise ValueError(f"pinned upstream commit is unavailable: {revision}")
    return lock


def process_rss_mib(pid: int) -> float:
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                            capture_output=True, text=True, check=True, timeout=5)
    value = result.stdout.strip()
    if not value:
        raise RuntimeError("agent RSS telemetry unavailable")
    return float(value) / 1024.0


def run_monitored(command: list[str], output: Path, limits: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "DOCKER_CONTEXT": "colima"})
    try:
        before = observe()
    except Exception as exc:
        record = {"status": "ABORTED_RESOURCE", "returncode": None,
                  "abort_reason": "resource_telemetry_unavailable_before_process",
                  "telemetry_error": f"{type(exc).__name__}: {exc}", "resource_before": None,
                  "resource_after": None, "swap_abort_threshold_mib": None,
                  "max_agent_rss_mib": 0.0, "max_wall_seconds": limits["max_wall_seconds_per_attempt"],
                  "observations": [], "elapsed_seconds": 0.0, "process_started": False}
        (output / "resource-monitor.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record
    swap_limit = min(limits["max_absolute_swap_mib"],
                     before["swap_used_mib"] + limits["max_swap_growth_mib"])
    if (before["swap_used_mib"] >= limits["max_absolute_swap_mib"]
            or before["memory_free_percent"] < limits["min_runtime_free_memory_percent"]):
        record = {"status": "ABORTED_RESOURCE", "returncode": None,
                  "abort_reason": "resource_preflight_before_model_process",
                  "telemetry_error": None, "resource_before": before,
                  "resource_after": before, "swap_abort_threshold_mib": swap_limit,
                  "max_agent_rss_mib": 0.0, "max_wall_seconds": limits["max_wall_seconds_per_attempt"],
                  "observations": [], "elapsed_seconds": 0.0, "process_started": False}
        (output / "resource-monitor.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    max_rss = 0.0
    reason = None
    telemetry_error = None
    process = None
    with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=True)
        except OSError as exc:
            record = {"status": "FAILED", "returncode": None,
                      "abort_reason": None, "launch_error": f"{type(exc).__name__}: {exc}",
                      "resource_before": before, "resource_after": observe(),
                      "swap_abort_threshold_mib": swap_limit, "max_agent_rss_mib": 0.0,
                      "max_wall_seconds": limits["max_wall_seconds_per_attempt"],
                      "observations": [], "elapsed_seconds": round(time.monotonic() - started, 3),
                      "process_started": False}
            (output / "resource-monitor.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return record
        while process.poll() is None:
            try:
                host = observe()
                rss = process_rss_mib(process.pid)
            except Exception as exc:
                reason = "resource_telemetry_unavailable"
                telemetry_error = f"{type(exc).__name__}: {exc}"
                break
            max_rss = max(max_rss, rss)
            samples.append({"elapsed_seconds": round(time.monotonic() - started, 3),
                            **host, "agent_rss_mib": round(rss, 3)})
            if host["swap_used_mib"] >= swap_limit:
                reason = "swap_growth_or_absolute_abort"
                break
            if host["memory_free_percent"] < limits["min_runtime_free_memory_percent"]:
                reason = "memory_free_abort"
                break
            if rss >= limits["max_agent_rss_mib"]:
                reason = "agent_rss_abort"
                break
            if time.monotonic() - started >= limits["max_wall_seconds_per_attempt"]:
                reason = "wall_time_abort"
                break
            remaining = limits["max_wall_seconds_per_attempt"] - (time.monotonic() - started)
            time.sleep(min(limits["poll_interval_seconds"], max(remaining, 0.0)))
        if reason:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    try:
        after = observe()
    except Exception as exc:
        after = None
        if reason is None:
            reason = "resource_telemetry_unavailable_after_process"
        telemetry_error = telemetry_error or f"{type(exc).__name__}: {exc}"
    status = "ABORTED_RESOURCE" if reason else "COMPLETED" if process.returncode == 0 else "FAILED"
    record = {"status": status, "returncode": process.returncode,
              "process_started": True,
              "abort_reason": reason, "telemetry_error": telemetry_error,
              "resource_before": before, "resource_after": after,
              "swap_abort_threshold_mib": swap_limit, "max_agent_rss_mib": round(max_rss, 3),
              "max_wall_seconds": limits["max_wall_seconds_per_attempt"],
              "observations": samples, "elapsed_seconds": round(time.monotonic() - started, 3)}
    (output / "resource-monitor.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.repo_root.resolve(strict=True) != ROOT:
        parser.error("repo root must be the checkout containing this runner")
    if args.check_only:
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout
        if dirty:
            parser.error("working tree must be clean before the frozen run")
        try:
            lock = verify(args.protocol.resolve(strict=True), args.model_path,
                          args.python, args.git_source)
            preflight = observe()
            limits = lock["resources"]
            disk_free_gib = round(shutil.disk_usage(ROOT).free / (1024 ** 3), 3)
            if (preflight["swap_used_mib"] >= limits["max_start_swap_mib"]
                    or preflight["memory_free_percent"] < limits["min_start_free_memory_percent"]
                    or disk_free_gib < limits["min_disk_free_gib"]):
                raise ValueError("locked resource preflight failed")
        except Exception as exc:
            parser.error(f"preflight failed: {type(exc).__name__}: {exc}")
        print(json.dumps({"status": "READY_NO_INFERENCE", "protocol_id": lock["protocol_id"],
                          "resources": {**preflight, "disk_free_gib": disk_free_gib},
                          "model_loaded": False}))
        return 0
    if args.output is None:
        parser.error("--output is required unless --check-only is set")
    output = args.output.absolute()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           capture_output=True, text=True, check=True).stdout
    if dirty:
        parser.error("working tree must be clean before the frozen run")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    experiment_started = time.monotonic()
    (output / "protocol.json").write_bytes(args.protocol.read_bytes())
    (output / "protocol.sha256").write_text(sha256(args.protocol) + "\n")
    try:
        lock = verify(args.protocol.resolve(strict=True), args.model_path,
                      args.python, args.git_source)
        preflight = observe()
        limits = lock["resources"]
        disk_free_gib = round(shutil.disk_usage(ROOT).free / (1024 ** 3), 3)
        if (preflight["swap_used_mib"] >= limits["max_start_swap_mib"]
                or preflight["memory_free_percent"] < limits["min_start_free_memory_percent"]
                or disk_free_gib < limits["min_disk_free_gib"]):
            raise ValueError("locked resource preflight failed")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        resource_abort = "resource" in str(exc).lower() or "memory" in str(exc).lower() or "swap" in str(exc).lower()
        preflight_result = {"status": "ABORTED_RESOURCE" if resource_abort else "BLOCKED",
                            "reason": error, "model_loaded": False, "inference_started": False}
        (output / "run-preflight.json").write_text(
            json.dumps(preflight_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output / "summary.json").write_text(
            json.dumps(preflight_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {path.relative_to(output).as_posix(): sha256(path)
                    for path in sorted(output.rglob("*")) if path.is_file()}
        (output / "evidence-manifest.json").write_text(
            json.dumps({"schema_version": 1, "files": manifest}, indent=2, sort_keys=True) + "\n")
        (output / "evidence-manifest.sha256").write_text(sha256(output / "evidence-manifest.json") + "\n")
        print(json.dumps(preflight_result))
        return 3
    (output / "preflight.json").write_text(
        json.dumps({"status": "PASSED", "resources": {**preflight, "disk_free_gib": disk_free_gib},
                    "model_loaded": False, "inference_started": False}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    task_root = ROOT / TASK_REL
    arms: list[dict[str, Any]] = []
    abort_pair = False
    for arm in lock["arms"]:
        if abort_pair:
            arms.append({"arm": arm["arm"], "status": "NOT_RUN",
                         "reason": "earlier arm reached a resource abort"})
            continue
        attempt = output / arm["arm"]
        attempt.mkdir(mode=0o700)
        workspace = attempt / "workspace"
        shutil.copytree(task_root / "workspace", workspace, symlinks=False)
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        command = [str(args.python.absolute()), str(ROOT / "scripts/run_local_agent_smollm_tool_compat_v1.py"),
                   "--workspace", str(workspace), "--model-path", str(args.model_path.resolve(strict=True)),
                   "--model-repository", lock["model"]["repository"], "--backend", "mlx",
                   "--public-test-image", lock["runtime"]["public_test_image"],
                   "--task-root", str(task_root), "--output", str(attempt / "agent.json"),
                   "--instruction", lock["agent"]["instruction"], "--seed", str(arm["seed"]),
                   "--max-turns", str(lock["agent"]["turn_budget"]),
                   "--max-new-tokens-per-turn", str(lock["agent"]["max_new_tokens_per_turn"]),
                   "--max-total-new-tokens", str(lock["agent"]["max_total_new_tokens"]),
                   "--max-context-tokens", str(lock["agent"]["max_context_tokens"]),
                   "--wall-seconds", str(lock["agent"]["wall_seconds_per_attempt"]),
                   "--temperature", str(arm["temperature"]), "--top-p", str(arm["top_p"]),
                   "--top-k", str(arm["top_k"]), "--edit-format", lock["agent"]["edit_format"]]
        command.append("--do-sample" if arm["do_sample"] else "--no-do-sample")
        monitor = run_monitored(command, attempt / "process", limits)
        for log_name in ("stdout.log", "stderr.log"):
            (attempt / f"agent-{log_name}").write_bytes((attempt / "process" / log_name).read_bytes())

        test_path = workspace / "test_public.py"
        test_expected = lock["task"]["bundle_sha256"][f"{TASK_REL}/workspace/test_public.py"]
        grader_status = None
        grader_error = None
        candidate_path = workspace / "qwen3_coder.py"
        if not test_path.is_file() or sha256(test_path) != test_expected:
            grader_error = "public test hash changed in candidate workspace"
            abort_pair = True
        grader_elapsed = None
        if grader_error is None and monitor["status"] != "ABORTED_RESOURCE":
            grade_dir = attempt / "independent-grade"
            grader_command = [str(args.python.absolute()), str(ROOT / "scripts/grade_mlx_lm_1910_candidate_v1.py"),
                              "--git-source", str(args.git_source.resolve(strict=True)),
                              "--candidate-source", str(candidate_path), "--output", str(grade_dir)]
            try:
                grade_started = time.monotonic()
                graded = subprocess.run(grader_command, cwd=ROOT, capture_output=True,
                                        timeout=lock["resources"]["max_grader_seconds_per_attempt"],
                                        check=False)
                grader_elapsed = round(time.monotonic() - grade_started, 3)
                (attempt / "grader.stdout.log").write_bytes(graded.stdout)
                (attempt / "grader.stderr.log").write_bytes(graded.stderr)
                grade_file = grade_dir / "grade.json"
                if grade_file.is_file():
                    grader_status = json.loads(grade_file.read_text(encoding="utf-8"))["status"]
                    expected_exit = 0 if grader_status == "PASS" else 2 if grader_status == "FAIL" else None
                    if expected_exit is None or graded.returncode != expected_exit:
                        grader_error = "grader status and process return code disagree"
                        abort_pair = True
                else:
                    grader_error = f"grader exited {graded.returncode} without grade.json"
                    abort_pair = True
            except subprocess.TimeoutExpired:
                grader_error = "grader timed out at frozen per-attempt limit"
                abort_pair = True
        elif monitor["status"] == "ABORTED_RESOURCE":
            grader_error = "grading skipped after resource abort"

        agent_record = None
        if (attempt / "agent.json").is_file():
            agent_record = json.loads((attempt / "agent.json").read_text(encoding="utf-8"))
        entry = {"arm": arm["arm"], "status": monitor["status"], "seed": arm["seed"],
                 "started_at_utc": started_at, "monitor": monitor, "grader_status": grader_status,
                 "grader_error": grader_error, "grader_elapsed_seconds": grader_elapsed,
                 "candidate_source_sha256": sha256(candidate_path) if candidate_path.is_file() else None,
                 "agent_metrics": None if agent_record is None else {
                     key: agent_record.get(key) for key in ["stop_reason", "elapsed_seconds", "new_tokens",
                         "tool_turns", "edit_attempted", "edit_applied", "format_parse_failure_count",
                         "public_test_result", "max_rss_bytes", "runtime", "candidate_diff_sha256"]}}
        (attempt / "attempt-summary.json").write_text(
            json.dumps(entry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        arms.append(entry)
        abort_pair = abort_pair or monitor["status"] == "ABORTED_RESOURCE"

    graded = [entry for entry in arms if entry.get("grader_status") in {"PASS", "FAIL"}]
    successes = sum(entry["grader_status"] == "PASS" for entry in graded)
    rewards = [int(entry["grader_status"] == "PASS") for entry in graded]
    recorded_agents = [entry["agent_metrics"] for entry in arms if entry.get("agent_metrics")]
    failure_taxonomy: dict[str, int] = {}
    for record in recorded_agents:
        reason = record.get("stop_reason") or "unknown"
        failure_taxonomy[reason] = failure_taxonomy.get(reason, 0) + 1
    test_results: list[dict[str, Any]] = []
    tool_call_count = 0
    for arm in arms:
        agent_file = output / arm["arm"] / "agent.json"
        if not agent_file.is_file():
            continue
        record = json.loads(agent_file.read_text(encoding="utf-8"))
        for turn in record.get("transcript", []):
            automatic = turn.get("automatic_public_test")
            if isinstance(automatic, dict):
                test_results.append(automatic)
            calls = turn.get("tool_calls", [])
            tool_call_count += len(calls)
            for call in calls:
                if call.get("name") == "run_public_tests" and isinstance(call.get("result"), dict):
                    test_results.append(call["result"])
    reward_mean = (sum(rewards) / len(rewards)) if rewards else None
    reward_variance = (sum((value - reward_mean) ** 2 for value in rewards) / len(rewards)
                       if rewards else None)
    completed = all(entry.get("status") == "COMPLETED" for entry in arms)
    status = "COMPLETED" if completed and len(graded) == 2 else (
        "ABORTED_RESOURCE" if any(entry.get("status") == "ABORTED_RESOURCE" for entry in arms)
        else "FAILED")
    summary = {"schema_version": 1, "protocol_id": lock["protocol_id"], "status": status,
               "arms": arms, "attempt_successes": successes, "attempt_denominator": len(graded),
               "agent_attempts_started": sum(bool(entry.get("monitor", {}).get("process_started"))
                                              for entry in arms),
               "unique_development_task_successes": int(successes > 0),
               "unique_development_task_denominator": 1,
               "binary_reward_mean": reward_mean, "binary_reward_population_variance": reward_variance,
               "agent_records": len(recorded_agents),
               "model_loaded_attempts": sum(bool(record.get("runtime")) for record in recorded_agents),
               "valid_edit_attempts": sum(bool(record.get("edit_applied")) for record in recorded_agents),
               "format_parse_failures": sum(record.get("format_parse_failure_count", 0) or 0
                                             for record in recorded_agents),
               "failure_taxonomy_by_stop_reason": failure_taxonomy,
               "public_test_invocations": len(test_results),
               "public_test_passes": sum(bool(result.get("passed")) for result in test_results),
               "tool_call_count": tool_call_count,
               "generated_tokens": sum((entry.get("agent_metrics") or {}).get("new_tokens", 0) or 0
                                        for entry in arms),
               "optimizer_steps": 0,
               "total_wall_seconds": round(time.monotonic() - experiment_started, 3),
               "claim_limit": "Two matched attempts on one Development task; not held-out success or learning evidence."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = {path.relative_to(output).as_posix(): sha256(path)
                for path in sorted(output.rglob("*")) if path.is_file()
                and path.name not in {"evidence-manifest.json", "evidence-manifest.sha256"}}
    (output / "evidence-manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": manifest}, indent=2, sort_keys=True) + "\n")
    (output / "evidence-manifest.sha256").write_text(sha256(output / "evidence-manifest.json") + "\n")
    print(json.dumps({key: summary[key] for key in
                      ["status", "attempt_successes", "attempt_denominator", "generated_tokens"]}))
    return 0 if status == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
