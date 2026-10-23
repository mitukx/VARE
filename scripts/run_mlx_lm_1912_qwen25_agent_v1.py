#!/usr/bin/env python3
"""Run matched greedy/sampled Qwen2.5-3B attempts on Dev-only MLX-LM #1912."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
BUNDLE = ROOT / "results/mlsys-repo-task-mlx-lm-1912-qwen25-3b-agent-v1"
TASK_ROOT = BUNDLE / "task"
PROTOCOL_PATH = BUNDLE / "protocol.json"
PROTOCOL_SHA_PATH = BUNDLE / "protocol.sha256"
TASK_ID = "mlxlm-1912-top-p-min-token"
BASELINE = "47e43a4526d42a6dcee758f07eb04b78a8439ecb"
FIXED = "87b7b583a697537aa68f47130b40884700b5f55f"
TARGET = "mlx_lm/sample_utils.py"
SOURCE_HASHES = {
    "baseline": "c93c1eef794725f9f7ce77b6212f61eb6d0fe17b9cd87c06cec6470ee12b07f2",
    "fixed": "3aac92d804b6b91c82a4e2844697a8723de26f6af57e10013a482b967e739cfa",
}
PUBLIC_UPSTREAM_TEST_HASH = "dbe871f37a2930eae6686ce67ffc0a641971ad11943a1e94833fa232c6938852"
LICENSE_HASH = "ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_evidence_manifest(out: Path, protocol_path: Path) -> None:
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "evidence-manifest.json":
            files[path.relative_to(out).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    write_json(out / "evidence-manifest.json", {
        "schema_version": 1,
        "protocol_sha256": sha256_file(protocol_path),
        "files": files,
    })


def resource_snapshot() -> dict[str, Any]:
    from scripts import smoke_local_mlx_model_load_v1 as monitor

    return {
        "swap_used_mib": monitor._swap_used_mib(),
        "memory_free_percent": monitor._memory_free_percent(),
    }


def process_rss_mib(pid: int) -> float:
    from scripts import smoke_local_mlx_model_load_v1 as monitor

    return monitor._rss_mib(pid)


def git_show(source: Path, revision: str, path: str) -> bytes:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", "-C", str(source), "--no-replace-objects", "-c",
         "core.hooksPath=/dev/null", "show", f"{revision}:{path}"],
        check=False, capture_output=True, env=env, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"cannot read pinned upstream object {revision[:12]}:{path}")
    return result.stdout


def verify_task_split() -> dict[str, Any]:
    path = ROOT / "results/mlsys-task-deep-qualification-v1/protocol.json"
    source = ROOT / "results/mlsys-task-deep-qualification-v1/qualification.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    qualification = json.loads(source.read_text(encoding="utf-8"))
    if protocol["source_manifest_sha256"] != sha256_file(source):
        raise RuntimeError("frozen deep-qualification source manifest hash mismatch")
    if TASK_ID not in protocol["split"]["development"]:
        raise RuntimeError("task is not assigned to the frozen Development split")
    if TASK_ID in protocol["split"]["heldout"]:
        raise RuntimeError("Development task overlaps the frozen Held-out split")
    entry = next((row for row in qualification["tasks"] if row["task_id"] == TASK_ID), None)
    if entry is None or entry.get("split") != "development":
        raise RuntimeError("frozen qualification inventory does not identify Development")
    if entry.get("exposure", {}).get("model_or_agent_seen_task") is not False:
        raise RuntimeError("task was previously exposed to a model or agent")
    return {
        "task_id": TASK_ID,
        "split": "development",
        "deep_qualification_protocol_sha256": sha256_file(path),
        "qualification_manifest_sha256": sha256_file(source),
        "previous_model_or_agent_exposure": False,
        "heldout_tasks_accessed": False,
    }


def verify_protocol(git_source: Path) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_NOT_RUN":
        raise RuntimeError("agent protocol is not in its frozen pre-run state")
    sidecar = PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    if sha256_file(PROTOCOL_PATH) != sidecar:
        raise RuntimeError("agent protocol SHA-256 sidecar mismatch")
    if protocol.get("runner_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("agent experiment runner hash differs from the frozen protocol")
    task_membership = verify_task_split()
    if any(protocol.get("task", {}).get(key) != value
           for key, value in task_membership.items()):
        raise RuntimeError("task split or exposure identity differs from the frozen protocol")

    expected_files = protocol["runtime"]["implementation_sha256"]
    if not isinstance(expected_files, dict):
        raise RuntimeError("source code hash inventory is malformed")
    for relative, expected in expected_files.items():
        file_path = ROOT / relative
        if not file_path.is_file() or sha256_file(file_path) != expected:
            raise RuntimeError(f"locked implementation changed: {relative}")
    for relative, expected in protocol["task_bundle_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"locked task bundle changed: {relative}")
    task_spec = json.loads((TASK_ROOT / "task.json").read_text(encoding="utf-8"))
    task_text = (TASK_ROOT / "TASK.md").read_text(encoding="utf-8")
    if (task_spec.get("task_id") != TASK_ID
            or task_spec.get("prompt") != task_text
            or task_spec.get("metadata", {}).get("exposure_classification") != "development_only"
            or task_spec.get("source", {}).get("upstream_base_commit") != BASELINE
            or task_spec.get("source", {}).get("upstream_fixed_commit") != FIXED):
        raise RuntimeError("task text, exposure, or upstream revisions differ from the frozen task")

    source = git_source.resolve(strict=True)
    baseline = git_show(source, BASELINE, TARGET)
    fixed = git_show(source, FIXED, TARGET)
    license_bytes = git_show(source, FIXED, "LICENSE")
    upstream_test = git_show(source, FIXED, "tests/test_sample_utils.py")
    observed_sources = {
        "baseline": sha256_bytes(baseline),
        "fixed": sha256_bytes(fixed),
    }
    if observed_sources != SOURCE_HASHES:
        raise RuntimeError("pinned upstream source bytes do not match the protocol")
    if sha256_bytes(license_bytes) != LICENSE_HASH:
        raise RuntimeError("pinned upstream license bytes do not match")
    if sha256_bytes(upstream_test) != PUBLIC_UPSTREAM_TEST_HASH:
        raise RuntimeError("pinned upstream public regression test differs")
    if sha256_file(TASK_ROOT / "test_public.py") != protocol["task"]["public_test_sha256"]:
        raise RuntimeError("candidate-visible public smoke test hash mismatch")

    loader_protocol_path = ROOT / protocol["model"]["load_protocol_path"]
    loader_protocol = json.loads(loader_protocol_path.read_text(encoding="utf-8"))
    if sha256_file(loader_protocol_path) != protocol["model"]["load_protocol_sha256"]:
        raise RuntimeError("successful model-load protocol hash mismatch")
    loader_result_path = ROOT / protocol["model"]["load_result_path"]
    if sha256_file(loader_result_path) != protocol["model"]["load_result_sha256"]:
        raise RuntimeError("successful model-load evidence hash mismatch")
    load_result = json.loads(loader_result_path.read_text(encoding="utf-8"))
    if (load_result.get("status") != "COMPLETED"
            or load_result.get("load_result", {}).get("loaded") is not True
            or load_result.get("generated_tokens") != 0):
        raise RuntimeError("model-load smoke does not establish a no-generation load pass")
    if (loader_protocol.get("model", {}).get("revision") != protocol["model"]["revision"]
            or loader_protocol.get("model", {}).get("files") != protocol["model"]["files"]):
        raise RuntimeError("agent checkpoint does not match the successful load-smoke snapshot")
    model_manifest_path = ROOT / protocol["model"]["manifest_path"]
    if sha256_file(model_manifest_path) != protocol["model"]["manifest_sha256"]:
        raise RuntimeError("model checkpoint file manifest hash mismatch")
    load_manifest_path = ROOT / protocol["model"]["load_evidence_manifest_path"]
    if sha256_file(load_manifest_path) != protocol["model"]["load_evidence_manifest_sha256"]:
        raise RuntimeError("successful model-load evidence manifest hash mismatch")
    load_manifest = json.loads(load_manifest_path.read_text(encoding="utf-8"))
    if load_manifest.get("files", {}).get("result.json", {}).get("sha256") != sha256_file(
        loader_result_path
    ):
        raise RuntimeError("model-load result is not covered by its evidence manifest")

    calibration_path = ROOT / protocol["grader"]["calibration_evidence_path"]
    if sha256_file(calibration_path) != protocol["grader"]["calibration_evidence_sha256"]:
        raise RuntimeError("independent grader calibration evidence hash mismatch")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    grader_protocol_path = ROOT / protocol["grader"]["protocol_path"]
    grader_lock_path = ROOT / protocol["grader"]["lock_path"]
    grader_protocol = json.loads(grader_protocol_path.read_text(encoding="utf-8"))
    grader_lock = json.loads(grader_lock_path.read_text(encoding="utf-8"))
    if (sha256_file(grader_protocol_path) != protocol["grader"]["protocol_sha256"]
            or sha256_file(grader_lock_path) != protocol["grader"]["lock_sha256"]
            or grader_protocol.get("protocol_id") != "mlxlm-1912-isolated-task-v4"
            or grader_lock.get("protocol_sha256") != sha256_file(grader_protocol_path)
            or protocol["grader"]["image"] != grader_protocol.get("runtime", {}).get("image")):
        raise RuntimeError("independent grader protocol or image differs from its frozen v4 lock")
    if (calibration.get("calibration_passed") is not True
            or calibration.get("baseline", {}).get("passed") is not False
            or calibration.get("known_fix", {}).get("passed") is not True
            or calibration.get("red_team_suite", {}).get("passed") is not True
            or calibration.get("source_swap_rejected") is not True):
        raise RuntimeError("v4 independent grader calibration did not pass")
    return protocol


def verify_runtime(protocol: dict[str, Any], model_path: Path) -> Path:
    runtime = protocol["runtime"]
    model = protocol["model"]
    if sys.platform != "darwin" or __import__("platform").machine() != "arm64":
        raise RuntimeError("protocol requires Apple Silicon macOS")
    if list(sys.version_info[:3]) != runtime["python_version"]:
        raise RuntimeError("Python version differs from the frozen protocol")
    for package, expected in runtime["packages"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"runtime package version differs: {package}")
    env_path = ROOT / runtime["environment_inventory_path"]
    if sha256_file(env_path) != runtime["environment_inventory_sha256"]:
        raise RuntimeError("runtime package inventory changed")

    snapshot = model_path.resolve(strict=True)
    if snapshot.name != model["revision"]:
        raise RuntimeError("local model snapshot revision differs from the frozen protocol")
    expected_files = model["files"]
    actual_names: set[str] = set()
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            actual = path.resolve(strict=True)
            if not actual.is_file():
                raise RuntimeError("model snapshot contains a non-file symlink")
        elif path.is_file():
            actual = path
        else:
            continue
        relative = path.relative_to(snapshot).as_posix()
        identity = expected_files.get(relative)
        if (identity is None or actual.stat().st_size != identity["bytes"]
                or sha256_file(actual) != identity["sha256"]):
            raise RuntimeError(f"model snapshot file differs from its lock: {relative}")
        actual_names.add(relative)
    if actual_names != set(expected_files):
        raise RuntimeError("model snapshot inventory differs from protocol")
    return snapshot


def _write_candidate_workspace(
    source: bytes, test_bytes: bytes, destination: Path, *, make_git: bool
) -> None:
    if destination.exists() or destination.is_symlink():
        raise RuntimeError("refusing to overwrite an existing candidate workspace")
    target = destination / TARGET
    target.parent.mkdir(parents=True)
    destination.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
    (destination / "test_public.py").write_bytes(test_bytes)
    if not make_git:
        return
    template = destination.parent / "empty-git-template"
    template.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TEMPLATE_DIR": str(template),
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    commands = [
        ["git", "-c", "core.hooksPath=/dev/null", "init", "-q", "--initial-branch=main"],
        ["git", "config", "user.name", "VARE task preparation"],
        ["git", "config", "user.email", "vare-task@invalid"],
        ["git", "add", "--", TARGET, "test_public.py"],
        ["git", "-c", "core.hooksPath=/dev/null", "commit", "-q", "-m", "Prepare pinned Development task"],
    ]
    for command in commands:
        subprocess.run(command, cwd=destination, check=True, env=env, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _resource_abort(
    sample: dict[str, Any], start_swap: float, *, limits: dict[str, Any], rss_mib: float,
    elapsed_seconds: float,
) -> str | None:
    if sample["swap_used_mib"] - start_swap > limits["max_swap_growth_mib"]:
        return "swap_growth_limit"
    if sample["swap_used_mib"] >= limits["max_absolute_swap_mib"]:
        return "absolute_swap_limit"
    if sample["memory_free_percent"] < limits["min_runtime_free_memory_percent"]:
        return "memory_free_limit"
    if rss_mib > limits["max_child_rss_mib"]:
        return "child_rss_limit"
    if elapsed_seconds >= limits["max_wall_seconds"]:
        return "wall_time_limit"
    return None


def run_monitored(
    command: list[str], *, cwd: Path, attempt_dir: Path, start_resources: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(Path.home()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": "0",
        "DOCKER_CONTEXT": "colima",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    abort_reason = None
    return_code: int | None = None
    with (attempt_dir / "launcher.stdout.log").open("xb") as stdout, (
        attempt_dir / "launcher.stderr.log"
    ).open("xb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        while process.poll() is None:
            elapsed = time.monotonic() - started
            try:
                current = resource_snapshot()
                rss = process_rss_mib(process.pid)
            except Exception as exc:
                abort_reason = f"resource_telemetry_error:{type(exc).__name__}"
                break
            sample = {
                "elapsed_seconds": round(elapsed, 3),
                **current,
                "child_rss_mib": round(rss, 2),
            }
            samples.append(sample)
            abort_reason = _resource_abort(
                current, start_resources["swap_used_mib"], limits=limits,
                rss_mib=rss, elapsed_seconds=elapsed,
            )
            if abort_reason:
                break
            time.sleep(limits["poll_interval_seconds"])
        if abort_reason:
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
        return_code = process.wait()

    final = resource_snapshot()
    peak_rss = max((sample["child_rss_mib"] for sample in samples), default=0.0)
    status = "ABORTED_RESOURCE" if abort_reason else "COMPLETED" if return_code == 0 else "FAILED"
    return {
        "status": status,
        "abort_reason": abort_reason,
        "return_code": return_code,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "start_resources": start_resources,
        "end_resources": final,
        "swap_growth_mib": round(final["swap_used_mib"] - start_resources["swap_used_mib"], 2),
        "minimum_memory_free_percent": min(
            [start_resources["memory_free_percent"],
             *(sample["memory_free_percent"] for sample in samples)]
        ),
        "peak_sampled_child_rss_mib": peak_rss,
        "samples": samples,
    }


def archive_workspace(workspace: Path, destination: Path) -> dict[str, Any]:
    files: list[Path] = []
    for path in sorted(workspace.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"candidate workspace contains a symlink: {path.relative_to(workspace)}")
        if path.is_file():
            files.append(path)
    with tarfile.open(destination, "w") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(workspace).as_posix(), recursive=False)
    return {"path": destination.name, "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination), "regular_file_count": len(files)}


def _agent_metrics(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {"record_present": False}
    calls = [call for entry in record.get("transcript", []) for call in entry.get("tool_calls", [])]
    names = [call.get("name") for call in calls if isinstance(call.get("name"), str)]
    public_runs = sum(1 for name in names if name == "run_public_tests")
    automatic_public_runs = sum(
        1 for entry in record.get("transcript", [])
        if isinstance(entry.get("automatic_public_test"), dict)
    )
    invalid_calls = sum(1 for call in calls if "error" in call)
    return {
        "record_present": True,
        "stop_reason": record.get("stop_reason"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "generated_tokens": record.get("new_tokens"),
        "prompt_tokens": sum(entry.get("prompt_tokens", 0) for entry in record.get("transcript", [])),
        "tool_turns": record.get("tool_turns"),
        "tool_call_count": len(names),
        "tool_call_names": names,
        "invalid_tool_call_count": invalid_calls,
        "format_parse_failure_count": record.get("format_parse_failure_count", 0),
        "edit_attempted": record.get("edit_attempted"),
        "valid_edit_applied": record.get("edit_applied"),
        "public_test_invocation_count": public_runs + automatic_public_runs,
        "public_test_invoked": bool(public_runs + automatic_public_runs),
        "public_test_passed": (record.get("public_test_result") or {}).get("passed"),
        "peak_rss_bytes": record.get("max_rss_bytes"),
        "candidate_diff_sha256": record.get("candidate_diff_sha256"),
        "source_hashes_before": record.get("source_hashes_before"),
        "source_hashes_after": record.get("source_hashes_after"),
    }


def read_raw_turn_log(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed durable model-turn record at line {line_number}") from exc
            if not isinstance(record, dict) or record.get("sequence") != line_number:
                raise RuntimeError(f"model-turn sequence mismatch at line {line_number}")
            records.append(record)
    return records


def _arm_command(
    *, python: Path, workspace: Path, task_root: Path, model_path: Path,
    attempt_dir: Path, protocol: dict[str, Any], arm: dict[str, Any],
) -> list[str]:
    agent = protocol["agent"]
    command = [
        str(python), str(ROOT / "scripts/run_local_agent_streaming_v1.py"),
        "--raw-turn-log", str(attempt_dir / "raw-turns.jsonl"),
        "--workspace", str(workspace), "--model-path", str(model_path),
        "--model-repository", protocol["model"]["repository"],
        "--backend", "mlx",
        "--public-test-image", protocol["grader"]["image"],
        "--task-root", str(task_root), "--output", str(attempt_dir / "agent.json"),
        "--instruction", agent["instruction"], "--seed", str(arm["seed"]),
        "--max-turns", str(agent["max_turns"]),
        "--max-no-tool-retries", str(agent["max_no_tool_retries"]),
        "--max-new-tokens-per-turn", str(agent["max_new_tokens_per_turn"]),
        "--max-total-new-tokens", str(agent["max_total_new_tokens"]),
        "--max-context-tokens", str(agent["max_context_tokens"]),
        "--wall-seconds", str(agent["wall_seconds"]),
        "--temperature", str(arm["temperature"]), "--top-p", str(arm["top_p"]),
        "--top-k", str(arm["top_k"]), "--edit-format", agent["edit_format"],
    ]
    command.append("--do-sample" if arm["do_sample"] else "--no-do-sample")
    return command


def run(args: argparse.Namespace) -> int:
    source = args.git_source.resolve(strict=True)
    model_path = verify_runtime(verify_protocol(source), args.model_path)
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite experiment output: {out}")
    out.mkdir(parents=True)
    (out / "protocol.json").write_bytes(PROTOCOL_PATH.read_bytes())

    limits = protocol["resources"]
    disk_free = shutil.disk_usage(out).free / (1024**3)
    resource_before = resource_snapshot()
    preflight_passed = (
        resource_before["swap_used_mib"] < limits["max_start_swap_mib"]
        and resource_before["memory_free_percent"] >= limits["min_start_free_memory_percent"]
        and disk_free >= limits["min_disk_free_gib"]
    )
    if not preflight_passed:
        write_json(out / "summary.json", {
            "schema_version": 1, "protocol_id": protocol["protocol_id"],
            "status": "BLOCKED", "reason": "preflight_resource_or_disk_limit",
            "resource_before_model_load": resource_before,
            "disk_free_gib": round(disk_free, 2), "attempts": [],
            "model_load_repeated": False, "generated_tokens": 0,
        })
        write_evidence_manifest(out, PROTOCOL_PATH)
        return 3

    base_source = git_show(source, BASELINE, TARGET)
    fixed_source = git_show(source, FIXED, TARGET)
    public_test = (TASK_ROOT / "test_public.py").read_bytes()
    test_hash = sha256_bytes(public_test)
    from scripts import run_local_agent as agent
    from vare.mlxlm_1912_repo_task import grade_workspace

    # Calibrate the candidate-visible smoke in the same isolated image before
    # any model prompt. Only the frozen Development baseline and known fix run.
    preflight_dir = out / "public-test-preflight"
    preflight_dir.mkdir()
    preflight_results = {}
    with tempfile.TemporaryDirectory(prefix="vare-1912-public-preflight-",
                                     dir=str(ROOT / "work")) as tmp:
        temp = Path(tmp)
        for label, source_bytes in (("baseline", base_source), ("known_fix", fixed_source)):
            workspace = temp / label / "workspace"
            _write_candidate_workspace(source_bytes, public_test, workspace, make_git=False)
            result = agent.run_public_tests(workspace, protocol["grader"]["image"], test_hash)
            preflight_results[label] = result
            write_json(preflight_dir / f"{label}.json", {
                "arm": label,
                "source_sha256": sha256_bytes(source_bytes),
                "public_test_sha256": test_hash,
                "passed": result.get("passed") is True,
                "returncode": result.get("returncode"),
                "error": result.get("error"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            })
    public_test_gate = (
        preflight_results["baseline"].get("passed") is False
        and preflight_results["known_fix"].get("passed") is True
    )
    write_json(preflight_dir / "decision.json", {
        "baseline_expected_fail": preflight_results["baseline"].get("passed") is False,
        "known_fix_expected_pass": preflight_results["known_fix"].get("passed") is True,
        "gate_passed": public_test_gate,
    })
    if not public_test_gate:
        write_json(out / "summary.json", {
            "schema_version": 1, "protocol_id": protocol["protocol_id"],
            "status": "NO_GO", "reason": "public_test_baseline_fix_gate_failed",
            "resource_before_model_load": resource_before,
            "preflight": preflight_results, "attempts": [],
            "model_load_repeated": False, "generated_tokens": 0,
        })
        write_evidence_manifest(out, PROTOCOL_PATH)
        return 2

    attempts: list[dict[str, Any]] = []
    should_stop = False
    for arm in protocol["arms"]:
        if should_stop:
            attempts.append({"arm": arm["arm"], "status": "NOT_RUN",
                             "reason": "earlier resource or infrastructure abort"})
            continue
        attempt_dir = out / arm["arm"]
        attempt_dir.mkdir()
        with tempfile.TemporaryDirectory(prefix=f"vare-1912-{arm['arm']}-",
                                         dir=str(ROOT / "work")) as tmp:
            work_root = Path(tmp)
            workspace = work_root / "workspace"
            _write_candidate_workspace(base_source, public_test, workspace, make_git=True)
            base_tree = subprocess.run(
                ["git", "-C", str(workspace), "rev-parse", "HEAD"], check=True,
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1",
                     "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
            ).stdout.strip()
            start_resources = resource_snapshot()
            if (start_resources["swap_used_mib"] >= limits["max_start_swap_mib"]
                    or start_resources["memory_free_percent"] < limits["min_start_free_memory_percent"]):
                attempt = {
                    "arm": arm["arm"], "seed": arm["seed"],
                    "status": "ABORTED_RESOURCE", "reason": "between-arm-preflight-limit",
                    "resource_before": start_resources, "workspace_base_commit": base_tree,
                    "generated_tokens": 0, "independent_grade_status": None,
                }
                write_json(attempt_dir / "attempt-summary.json", attempt)
                attempts.append(attempt)
                should_stop = True
                continue

            command = _arm_command(
                python=args.python.absolute(), workspace=workspace, task_root=TASK_ROOT,
                model_path=model_path, attempt_dir=attempt_dir, protocol=protocol, arm=arm,
            )
            safe_command = [
                "<pinned-python>", "scripts/run_local_agent_streaming_v1.py",
                "--raw-turn-log", "<attempt-output>/raw-turns.jsonl",
                "--workspace", "<candidate-workspace>", "--model-path", "<pinned-model-snapshot>",
                "--model-repository", protocol["model"]["repository"], "--backend", "mlx",
                "--public-test-image", protocol["grader"]["image"],
                "--task-root", "<frozen-task-root>", "--output", "<attempt-output>/agent.json",
                "--instruction", protocol["agent"]["instruction"], "--seed", str(arm["seed"]),
                "--max-turns", str(protocol["agent"]["max_turns"]),
                "--max-no-tool-retries", str(protocol["agent"]["max_no_tool_retries"]),
                "--max-new-tokens-per-turn", str(protocol["agent"]["max_new_tokens_per_turn"]),
                "--max-total-new-tokens", str(protocol["agent"]["max_total_new_tokens"]),
                "--max-context-tokens", str(protocol["agent"]["max_context_tokens"]),
                "--wall-seconds", str(protocol["agent"]["wall_seconds"]),
                "--temperature", str(arm["temperature"]), "--top-p", str(arm["top_p"]),
                "--top-k", str(arm["top_k"]), "--edit-format", protocol["agent"]["edit_format"],
                "--do-sample" if arm["do_sample"] else "--no-do-sample",
            ]
            write_json(attempt_dir / "launch.json", {
                "command_template": safe_command,
                "working_tree_base_commit": base_tree,
                "upstream_base_commit": BASELINE,
                "baseline_source_sha256": sha256_bytes(base_source),
                "public_test_sha256": test_hash,
                "resource_before": start_resources,
                "status": "FROZEN_ATTEMPT_PARAMETERS",
            })
            (attempt_dir / "raw-turns.jsonl").open("xb").close()
            monitor_result = run_monitored(
                command, cwd=ROOT, attempt_dir=attempt_dir,
                start_resources=start_resources, limits=limits,
            )
            write_json(attempt_dir / "resource-monitor.json", monitor_result)
            agent_path = attempt_dir / "agent.json"
            agent_record = json.loads(agent_path.read_text(encoding="utf-8")) if agent_path.is_file() else None
            raw_turn_path = attempt_dir / "raw-turns.jsonl"
            raw_turn_error = None
            try:
                raw_turns = read_raw_turn_log(raw_turn_path)
            except (OSError, RuntimeError) as exc:
                raw_turns = []
                raw_turn_error = f"{type(exc).__name__}: {exc}"
            grade_result = None
            grade_error = None
            workspace_symlinks = [
                path.relative_to(workspace).as_posix()
                for path in workspace.rglob("*") if path.is_symlink()
            ]
            public_file = workspace / "test_public.py"
            if workspace_symlinks:
                grade_error = f"candidate workspace contains symlinks: {workspace_symlinks}"
            elif not public_file.is_file() or public_file.is_symlink() or sha256_file(public_file) != test_hash:
                grade_error = "candidate public-test file is missing, linked, or changed"
            else:
                changed = subprocess.run(
                    ["git", "-C", str(workspace), "diff", "--name-only", "HEAD", "--"],
                    check=True, capture_output=True, text=True, timeout=10,
                    env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
                         "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1"},
                ).stdout.splitlines()
                untracked = subprocess.run(
                    ["git", "-C", str(workspace), "ls-files", "--others", "--exclude-standard"],
                    check=True, capture_output=True, text=True, timeout=10,
                    env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
                         "GIT_CONFIG_GLOBAL": os.devnull, "GIT_NO_REPLACE_OBJECTS": "1"},
                ).stdout.splitlines()
                if set(changed) - {TARGET} or untracked:
                    grade_error = "candidate modified protected files or added untracked files"
            if (monitor_result["status"] != "ABORTED_RESOURCE" and grade_error is None
                    and (workspace / TARGET).is_file() and not (workspace / TARGET).is_symlink()):
                try:
                    grade_result = grade_workspace(
                        workspace, image=protocol["grader"]["image"], timeout_s=30,
                        memory_mb=1536,
                    )
                    write_json(attempt_dir / "independent-grade.json", grade_result)
                except Exception as exc:
                    grade_error = f"{type(exc).__name__}: {exc}"
                    write_json(attempt_dir / "independent-grade-error.json", {
                        "classification": "grader_infrastructure_or_provenance_failure",
                        "error": grade_error,
                    })
            archive_error = None
            try:
                archive = archive_workspace(workspace, attempt_dir / "workspace.tar")
            except (OSError, RuntimeError) as exc:
                archive = None
                archive_error = f"{type(exc).__name__}: {exc}"
            metrics = _agent_metrics(agent_record)
            streamed_tokens = sum(
                row.get("generated_tokens", 0) for row in raw_turns
                if row.get("event") == "generation_completed"
            )
            metrics["durable_raw_turn_count"] = len(raw_turns)
            metrics["durable_raw_generated_tokens"] = streamed_tokens
            metrics["raw_turn_log_sha256"] = sha256_file(raw_turn_path)
            metrics["raw_turn_log_error"] = raw_turn_error
            metrics["agent_record_token_count_matches_raw_log"] = (
                None if agent_record is None else agent_record.get("new_tokens") == streamed_tokens
            )
            if agent_record is not None and agent_record.get("new_tokens") != streamed_tokens:
                grade_error = "agent token count differs from durable raw-turn log"
            if raw_turn_error and grade_error is None:
                grade_error = raw_turn_error
            if archive_error and grade_error is None:
                grade_error = f"workspace archive failed: {archive_error}"
            passed = (grade_result is not None and grade_error is None
                      and grade_result.get("passed") is True)
            useful_trajectory = (
                passed and metrics.get("stop_reason") == "agent_finished"
                and metrics.get("valid_edit_applied") is True
                and metrics.get("public_test_invoked") is True
                and metrics.get("public_test_passed") is True
            )
            if monitor_result["status"] == "ABORTED_RESOURCE":
                failure_classification = "resource_abort"
            elif grade_error is not None:
                failure_classification = "infrastructure_or_evidence_integrity"
            elif passed and useful_trajectory:
                failure_classification = "useful_trajectory_success"
            elif passed:
                failure_classification = "independent_patch_correct_but_trajectory_incomplete"
            elif not metrics.get("record_present"):
                failure_classification = "agent_runtime_failed_before_final_record"
            elif not metrics.get("valid_edit_applied"):
                failure_classification = "no_valid_edit"
            elif not metrics.get("public_test_invoked"):
                failure_classification = "public_test_not_invoked"
            elif metrics.get("public_test_passed") is not True:
                failure_classification = "public_test_failed"
            else:
                failure_classification = "independent_hidden_grade_failed"
            status = monitor_result["status"]
            if status not in {"ABORTED_RESOURCE"}:
                status = "COMPLETED" if grade_result is not None and grade_error is None else "FAILED"
            attempt = {
                "arm": arm["arm"], "seed": arm["seed"], "status": status,
                "monitor_abort_reason": monitor_result["abort_reason"],
                "monitor_wall_seconds": monitor_result["elapsed_seconds"],
                "monitor_swap_growth_mib": monitor_result["swap_growth_mib"],
                "monitor_minimum_free_memory_percent": monitor_result["minimum_memory_free_percent"],
                "monitor_peak_child_rss_mib": monitor_result["peak_sampled_child_rss_mib"],
                "agent": metrics,
                "model_runtime": None if agent_record is None else agent_record.get("runtime"),
                "independent_grade_status": None if grade_result is None else (
                    "PASS" if passed else "FAIL"
                ),
                "independent_success": passed,
                "useful_trajectory_success": useful_trajectory,
                "binary_reward": int(passed),
                "failure_classification": failure_classification,
                "grader_error": grade_error,
                "workspace_archive_error": archive_error,
                "workspace_base_commit": base_tree,
                "candidate_source_sha256": sha256_file(workspace / TARGET),
                "candidate_diff_sha256": None if agent_record is None else agent_record.get("candidate_diff_sha256"),
                "workspace_archive": archive,
            }
            write_json(attempt_dir / "attempt-summary.json", attempt)
            attempts.append(attempt)
            if monitor_result["status"] == "ABORTED_RESOURCE" or grade_error:
                should_stop = True

    completed = [row for row in attempts if row.get("status") == "COMPLETED"]
    rewards = [row["binary_reward"] for row in completed if isinstance(row.get("binary_reward"), int)]
    mean_reward = sum(rewards) / len(rewards) if rewards else None
    variance = (sum((value - mean_reward) ** 2 for value in rewards) / len(rewards)
                if rewards and mean_reward is not None else None)
    unique_success = int(any(row.get("independent_success") for row in attempts))
    useful_successes = sum(bool(row.get("useful_trajectory_success")) for row in attempts)
    measured_agent_wall = sum(
        row.get("monitor_wall_seconds", 0.0) for row in attempts
        if isinstance(row.get("monitor_wall_seconds"), (int, float))
    )
    final_resource = resource_snapshot()
    if any(row.get("status") == "ABORTED_RESOURCE" for row in attempts):
        status = "ABORTED_RESOURCE"
    elif len(completed) == len(protocol["arms"]):
        status = "COMPLETED"
    elif completed:
        status = "PARTIAL"
    else:
        status = "FAILED"
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "status": status,
        "task_id": TASK_ID,
        "split": "development",
        "independent_successes": sum(rewards),
        "graded_attempts": len(rewards),
        "unique_task_successes": unique_success if completed else None,
        "unique_task_denominator": 1 if completed else 0,
        "useful_trajectory_successes": useful_successes,
        "useful_trajectory_successes_per_agent_hour": (
            useful_successes * 3600 / measured_agent_wall if measured_agent_wall > 0 else None
        ),
        "reward_values": rewards,
        "reward_mean": mean_reward,
        "reward_population_variance_within_one_task": variance,
        "valid_edit_attempts": sum(bool(row.get("agent", {}).get("valid_edit_applied")) for row in attempts),
        "public_test_invoked_attempts": sum(bool(row.get("agent", {}).get("public_test_invoked")) for row in attempts),
        "public_test_passed_attempts": sum(bool(row.get("agent", {}).get("public_test_passed")) for row in attempts),
        "generated_tokens": sum(
            row.get("agent", {}).get("durable_raw_generated_tokens") or 0 for row in attempts
        ),
        "prompt_tokens": sum(
            row.get("agent", {}).get("prompt_tokens") or 0 for row in attempts
        ),
        "agent_wall_seconds": round(measured_agent_wall, 3),
        "model_load_confirmed_attempts": sum(
            bool(row.get("agent", {}).get("record_present")) for row in attempts
        ),
        "failure_classifications": {
            label: sum(row.get("failure_classification") == label for row in attempts)
            for label in sorted({
                row.get("failure_classification") for row in attempts
                if row.get("failure_classification") is not None
            })
        },
        "optimizer_steps": 0,
        "resource_before_any_agent_load": resource_before,
        "resource_after_all_attempts": final_resource,
        "disk_free_before_gib": round(disk_free, 2),
        "attempts": attempts,
        "heldout_tasks_accessed": False,
        "claim_limit": "Two attempts on one frozen Development task. This is feasibility evidence only, not a task-population success estimate, training signal qualification, or policy-improvement result.",
    }
    write_json(out / "summary.json", summary)
    write_evidence_manifest(out, PROTOCOL_PATH)
    print(json.dumps({"status": status, "independent_successes": summary["independent_successes"],
                      "graded_attempts": summary["graded_attempts"],
                      "reward_values": rewards, "generated_tokens": summary["generated_tokens"]}))
    return 0 if status == "COMPLETED" else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol = verify_protocol(args.git_source.resolve(strict=True))
    model_path = verify_runtime(protocol, args.model_path)
    if args.check_only:
        print(json.dumps({"protocol_id": protocol["protocol_id"], "status": "CHECKS_PASSED",
                          "model_snapshot": model_path.name, "model_load_authorized": True,
                          "task_id": TASK_ID, "split": "development", "inference_started": False}))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
