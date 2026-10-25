#!/usr/bin/env python3
"""One-shot no-network qualification runner for Transformers #47468 (Development only)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from typing import Any

import psutil

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results/transformers-47468-development-qualification-v1"
PROTOCOL = EXPERIMENT / "protocol/protocol-v1.json"
LOCK = EXPERIMENT / "protocol/protocol-v1.lock.json"
PROMPT = EXPERIMENT / "protocol/candidate-prompt.md"
SCRIPT = Path(__file__).resolve()
MANIFEST = EXPERIMENT / "records/manifest.jsonl"
PROFILE = "(version 1) (allow default) (deny network*)"
TARGET = "src/transformers/utils/generic.py"
OLD_MUTANT_EXPR = (
    b"return x.to(torch.float32) if torch.jit.is_tracing() "
    b"and isinstance(x, torch.Tensor) else float(x)"
)
NEW_MUTANT_EXPR = (
    b"return x.to(torch.float32) if torch.jit.is_tracing() "
    b"and isinstance(x, torch.Tensor) else int(x)"
)
SAMPLE_INTERVAL = 0.5
MAX_SECONDS = 180
MIN_AVAILABLE = 8 * 1024**3
MIN_FREE_SWAP = 256 * 1024**2
MAX_SWAP_GROWTH = 128 * 1024**2
MIN_DISK = 20 * 1024**3
MAX_TREE_RSS = 8 * 1024**3


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_new(path: Path, data: bytes) -> None:
    """Atomically create and fsync a file without replacing any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        os.unlink(temp)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def write_json_new(path: Path, value: Any) -> None:
    write_new(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def event_hash(event: dict[str, Any]) -> str:
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return sha_bytes(encoded)


def load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST.exists():
        return []
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    previous = "0" * 64
    for sequence, row in enumerate(rows, start=1):
        require(row.get("sequence") == sequence, "manifest sequence is not contiguous")
        require(row.get("previous_event_sha256") == previous, "manifest hash chain is broken")
        require(row.get("event_sha256") == event_hash(row), "manifest event hash mismatch")
        previous = row["event_sha256"]
    return rows


def append_event(event_type: str, fields: dict[str, Any]) -> dict[str, Any]:
    rows = load_manifest()
    previous = rows[-1]["event_sha256"] if rows else "0" * 64
    row = {
        "sequence": len(rows) + 1,
        "event_type": event_type,
        "timestamp_utc": utc_now(),
        "previous_event_sha256": previous,
        **fields,
    }
    row["event_sha256"] = event_hash(row)
    fd = os.open(MANIFEST, os.O_WRONLY | os.O_APPEND)
    try:
        payload = memoryview((json.dumps(row, sort_keys=True) + "\n").encode())
        while payload:
            payload = payload[os.write(fd, payload):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return row


def tree_hash(root: Path) -> tuple[int, int, str]:
    base = root.parent.parent
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        rel = path.relative_to(base).as_posix().encode()
        raw = path.read_bytes()
        total_bytes += len(raw)
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        digest.update(hashlib.sha256(raw).digest())
    return len(files), total_bytes, digest.hexdigest()


def make_mutant(fix_bytes: bytes) -> bytes:
    require(fix_bytes.count(OLD_MUTANT_EXPR) == 1, "fixed target does not match the frozen mutant transform")
    mutant = fix_bytes.replace(OLD_MUTANT_EXPR, NEW_MUTANT_EXPR, 1)
    require(sha_bytes(mutant) == PROTOCOL_DATA["task"]["mutant"]["sha256"], "mutant hash differs from protocol")
    return mutant


def resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def validate_network_deny(python: Path) -> dict[str, Any]:
    argv = [
        "/usr/bin/sandbox-exec", "-p", PROFILE, str(python), "-c",
        "import socket; s=socket.socket(); s.settimeout(1); s.connect(('1.1.1.1',443))",
    ]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=5)
    output = result.stdout + result.stderr
    denied = result.returncode != 0 and ("Operation not permitted" in output or "PermissionError" in output)
    require(denied, "sandbox network-deny preflight failed")
    return {"return_code": result.returncode, "denied": True, "output_sha256": sha_bytes(output.encode())}


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    require(PROTOCOL.is_file() and LOCK.is_file(), "protocol or lock is missing")
    protocol_raw = PROTOCOL.read_bytes()
    lock_data = json.loads(LOCK.read_text(encoding="utf-8"))
    require(sha_bytes(protocol_raw) == lock_data["protocol_sha256"], "protocol SHA mismatch")
    require(sha_file(SCRIPT) == lock_data["runner_sha256"], "runner SHA mismatch")
    protocol = json.loads(protocol_raw)
    require(sha_file(PROMPT) == protocol["task"]["candidate_prompt_sha256"], "candidate prompt SHA mismatch")

    archive = resolve(args.source_archive)
    grader_test = resolve(args.grader_test)
    fix_target = resolve(args.fix_target)
    python = resolve(args.python)
    overlay = resolve(args.overlay_site_packages)
    output_root = resolve(args.output)
    host_lock = EXPERIMENT / "locks/host-pip-freeze.txt"
    overlay_lock = EXPERIMENT / "locks/overlay-pip-freeze.txt"
    exposure = EXPERIMENT / "protocol/exposure-scan.json"
    preflight_record = EXPERIMENT / "records/preflight.json"
    require(archive.is_file() and sha_file(archive) == protocol["task"]["candidate_source_archive_sha256"], "baseline source archive missing or changed")
    require(grader_test.is_file() and sha_file(grader_test) == protocol["task"]["test_sha256"], "pinned grader test missing or changed")
    require(fix_target.is_file() and sha_file(fix_target) == protocol["task"]["fix_target_sha256"], "exact fix target missing or changed")
    require(sha_file(host_lock) == protocol["execution"]["dependency_freeze_sha256"]["locks/host-pip-freeze.txt"], "host Python freeze changed")
    require(sha_file(overlay_lock) == protocol["execution"]["dependency_freeze_sha256"]["locks/overlay-pip-freeze.txt"], "overlay Python freeze changed")
    require(sha_file(exposure) == protocol["exposure"]["local_vare_scan_sha256"], "exposure scan changed")
    require(sha_file(preflight_record) == lock_data["preflight_record_sha256"], "pre-outcome preflight record changed")
    require(python.is_file() and overlay.is_dir(), "pinned Python executable or package overlay missing")
    require(not grader_test.is_relative_to(output_root / "workspaces"),
            "grader test must remain outside candidate source workspaces")
    require(not grader_test.is_relative_to(archive.parent.resolve() / "candidate"),
            "grader test must remain outside the candidate input workspace")
    expected_python = Path(protocol["execution"]["python_executable"]).expanduser().resolve()
    require(python == expected_python, "Python executable differs from frozen protocol")
    with tarfile.open(archive, mode="r:") as tar:
        names = tar.getnames()
        require(names and all(name in {"src", "src/transformers"} or name.startswith("src/transformers/") for name in names), "source archive contains paths outside src/transformers")
        require(not any("/tests/" in f"/{name}/" or ".git" in name.split("/") for name in names), "source archive contains tests or Git history")
        require(not any(member.issym() or member.islnk() for member in tar.getmembers()), "source archive contains symlinks")
        scratch = Path(tempfile.mkdtemp(prefix="vare-transformers-47468-preflight-"))
        try:
            tar.extractall(scratch, filter="data")
            source_root = scratch / "src" / "transformers"
            count, size, digest = tree_hash(source_root)
            expected_task = protocol["task"]
            require((count, size, digest) == (
                expected_task["candidate_source_file_count"],
                expected_task["candidate_source_bytes"],
                expected_task["candidate_source_tree_sha256"],
            ), "source tree inventory differs from the frozen protocol")
            base_target = source_root.parent / "transformers/utils/generic.py"
            require(sha_file(base_target) == expected_task["base_target_sha256"], "baseline target differs from its upstream base hash")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    fix_bytes = fix_target.read_bytes()
    mutant_bytes = make_mutant(fix_bytes)
    network_check = validate_network_deny(python)
    return {
        "protocol_sha256": lock_data["protocol_sha256"],
        "runner_sha256": lock_data["runner_sha256"],
        "source_archive_sha256": sha_file(archive),
        "source_tree_sha256": protocol["task"]["candidate_source_tree_sha256"],
        "grader_test_sha256": sha_file(grader_test),
        "fix_target_sha256": sha_file(fix_target),
        "mutant_target_sha256": sha_bytes(mutant_bytes),
        "network_deny_probe": network_check,
        "python": str(python),
        "overlay_site_packages": str(overlay),
    }


def resource_sample(pid: int | None = None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(str(ROOT))
    tree_rss = 0
    if pid is not None:
        try:
            process = psutil.Process(pid)
            tree_rss = sum(item.memory_info().rss for item in [process, *process.children(recursive=True)] if item.is_running())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {
        "timestamp_utc": utc_now(),
        "available_memory_bytes": memory.available,
        "total_memory_bytes": memory.total,
        "free_swap_bytes": swap.free,
        "swap_used_bytes": swap.used,
        "disk_free_bytes": disk.free,
        "process_tree_rss_bytes": tree_rss,
    }


def test_matches(record: dict[str, Any], expected_pass: bool) -> bool:
    if record["status"] != "EXITED":
        return False
    output = Path(record["stdout_path"]).read_text(errors="replace") + Path(record["stderr_path"]).read_text(errors="replace")
    if expected_pass:
        return record["return_code"] == 0 and re.search(r"\b1 passed\b", output) is not None
    return (
        record["return_code"] == 1
        and "GenericTester::test_torch_float" in output
        and re.search(r"\b1 failed\b", output) is not None
        and "AssertionError" in output
        and "ERROR collecting" not in output
    )


def run_phase(
    phase: str, expected_pass: bool, target_bytes: bytes, args: argparse.Namespace,
    protocol: dict[str, Any], run_dir: Path, validation: dict[str, Any],
) -> dict[str, Any]:
    phase_id = str(uuid.uuid4())
    phase_dir = run_dir / "workspaces" / f"{phase}-{phase_id}"
    phase_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(resolve(args.source_archive), mode="r:") as tar:
        tar.extractall(phase_dir, filter="data")
    source_root = phase_dir / "src" / "transformers"
    target = phase_dir / TARGET
    source_tree_base = tree_hash(source_root)
    require(source_tree_base == (
        protocol["task"]["candidate_source_file_count"],
        protocol["task"]["candidate_source_bytes"],
        protocol["task"]["candidate_source_tree_sha256"],
    ), f"{phase}: reconstructed source inventory mismatch")
    target.write_bytes(target_bytes)
    target_hash = sha_file(target)
    source_tree_variant_before = tree_hash(source_root)
    expected_target = {
        "baseline": protocol["task"]["base_target_sha256"],
        "mutant": protocol["task"]["mutant"]["sha256"],
        "fix": protocol["task"]["fix_target_sha256"],
    }[phase]
    require(target_hash == expected_target, f"{phase}: target source hash mismatch")

    logs = run_dir / "logs"
    resources = run_dir / "resources"
    records = run_dir / "records"
    for folder in (logs, resources, records):
        folder.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{phase}-{phase_id}.stdout.log"
    stderr_path = logs / f"{phase}-{phase_id}.stderr.log"
    samples_path = resources / f"{phase}-{phase_id}.jsonl"
    record_path = records / f"{phase}-{phase_id}.json"
    grader_test = resolve(args.grader_test)
    overlay = resolve(args.overlay_site_packages)
    python = resolve(args.python)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str((run_dir / "tmp" / f"home-{phase_id}").resolve()),
        "TMPDIR": str((run_dir / "tmp" / f"tmp-{phase_id}").resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PIP_NO_INDEX": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(source_root.parent.resolve()), str(overlay))),
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
    }
    Path(env["HOME"]).mkdir(parents=True, exist_ok=False)
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=False)
    command = [
        "/usr/bin/sandbox-exec", "-p", PROFILE, str(python), "-m", "pytest", "-q",
        "-p", "no:cacheprovider",
        f"{grader_test}::GenericTester::test_torch_float",
    ]
    pre = resource_sample()
    baseline_swap = pre["swap_used_bytes"]
    min_available = pre["available_memory_bytes"]
    min_free_swap = pre["free_swap_bytes"]
    max_swap = baseline_swap
    min_disk_free = pre["disk_free_bytes"]
    max_rss = 0
    abort_reason = None
    started = time.monotonic()
    started_utc = utc_now()
    return_code = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr, samples_path.open("xb") as samples:
        def sample_and_write(pid: int | None) -> dict[str, Any]:
            nonlocal min_available, min_free_swap, max_swap, min_disk_free, max_rss
            row = resource_sample(pid)
            min_available = min(min_available, row["available_memory_bytes"])
            min_free_swap = min(min_free_swap, row["free_swap_bytes"])
            max_swap = max(max_swap, row["swap_used_bytes"])
            min_disk_free = min(min_disk_free, row["disk_free_bytes"])
            max_rss = max(max_rss, row["process_tree_rss_bytes"])
            samples.write((json.dumps(row, sort_keys=True) + "\n").encode())
            samples.flush()
            os.fsync(samples.fileno())
            return row

        if pre["available_memory_bytes"] < MIN_AVAILABLE:
            abort_reason = "preflight_available_memory_below_floor"
        elif pre["free_swap_bytes"] < MIN_FREE_SWAP:
            abort_reason = "preflight_free_swap_below_floor"
        elif pre["disk_free_bytes"] < MIN_DISK:
            abort_reason = "preflight_disk_below_floor"
        proc = None
        if abort_reason is None:
            proc = subprocess.Popen(command, cwd=grader_test.parent, env=env,
                                    stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
            while proc.poll() is None:
                row = sample_and_write(proc.pid)
                growth = row["swap_used_bytes"] - baseline_swap
                elapsed = time.monotonic() - started
                if row["available_memory_bytes"] < MIN_AVAILABLE:
                    abort_reason = "available_memory_below_floor"
                elif row["free_swap_bytes"] < MIN_FREE_SWAP:
                    abort_reason = "free_swap_below_floor"
                elif growth > MAX_SWAP_GROWTH:
                    abort_reason = "swap_growth_above_limit"
                elif row["disk_free_bytes"] < MIN_DISK:
                    abort_reason = "disk_below_floor"
                elif row["process_tree_rss_bytes"] > MAX_TREE_RSS:
                    abort_reason = "process_tree_rss_above_limit"
                elif elapsed > MAX_SECONDS:
                    abort_reason = "phase_timeout"
                if abort_reason:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    break
                time.sleep(SAMPLE_INTERVAL)
            return_code = proc.wait()
            final = sample_and_write(None)
            if abort_reason is None:
                if final["available_memory_bytes"] < MIN_AVAILABLE:
                    abort_reason = "completion_available_memory_below_floor"
                elif final["free_swap_bytes"] < MIN_FREE_SWAP:
                    abort_reason = "completion_free_swap_below_floor"
                elif final["swap_used_bytes"] - baseline_swap > MAX_SWAP_GROWTH:
                    abort_reason = "completion_swap_growth_above_limit"
                elif final["disk_free_bytes"] < MIN_DISK:
                    abort_reason = "completion_disk_below_floor"
        else:
            sample_and_write(None)
    elapsed = round(time.monotonic() - started, 3)
    status = "ABORTED" if abort_reason else "EXITED"
    source_tree_after = tree_hash(source_root)
    source_mutated_by_test = source_tree_variant_before != source_tree_after
    record = {
        "schema_version": 1,
        "invocation_id": phase_id,
        "phase": phase,
        "expected_pass": expected_pass,
        "started_at_utc": started_utc,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": elapsed,
        "command_argv": command,
        "cwd": str(grader_test.parent),
        "environment": env,
        "protocol_sha256": validation["protocol_sha256"],
        "runner_sha256": validation["runner_sha256"],
        "source_identity": {
            "base_commit": protocol["task"]["base_commit"],
            "fix_commit_for_control": protocol["task"]["fix_commit"] if phase == "fix" else None,
            "target_path": TARGET,
            "target_sha256": target_hash,
            "phase": phase,
        },
        "grader_identity": {
            "test_sha256": sha_file(grader_test),
            "selector": protocol["task"]["test_selector"],
            "test_outside_source_tree": True,
        },
        "return_code": return_code,
        "abort_reason": abort_reason,
        "status": status,
        "resources": {
            "preflight": pre,
            "minimum_available_memory_bytes": min_available,
            "minimum_free_swap_bytes": min_free_swap,
            "maximum_swap_growth_bytes": max(0, max_swap - baseline_swap),
            "minimum_disk_free_bytes": min_disk_free,
            "maximum_process_tree_rss_bytes": max_rss,
            "sample_count": sum(1 for _ in samples_path.open()),
            "sample_interval_seconds": SAMPLE_INTERVAL,
        },
        "stdout_path": str(stdout_path),
        "stdout_sha256": sha_file(stdout_path),
        "stderr_path": str(stderr_path),
        "stderr_sha256": sha_file(stderr_path),
        "resource_samples_path": str(samples_path),
        "resource_samples_sha256": sha_file(samples_path),
        "source_tree_base": source_tree_base,
        "source_tree_variant_before_test": source_tree_variant_before,
        "source_tree_after": source_tree_after,
        "source_mutated_by_test": source_mutated_by_test,
        "raw_record_path": str(record_path),
    }
    expected_outcome_matched = test_matches(record, expected_pass) and not source_mutated_by_test
    record["expected_outcome_matched"] = expected_outcome_matched
    write_json_new(record_path, record)
    record_sha = sha_file(record_path)
    append_event("phase_record", {
        "phase": phase,
        "invocation_id": phase_id,
        "record_path": str(record_path.relative_to(EXPERIMENT)),
        "record_sha256": record_sha,
        "stdout_sha256": record["stdout_sha256"],
        "stderr_sha256": record["stderr_sha256"],
        "resource_samples_sha256": record["resource_samples_sha256"],
        "status": status,
        "return_code": return_code,
        "abort_reason": abort_reason,
        "expected_outcome_matched": expected_outcome_matched,
        "source_mutated_by_test": source_mutated_by_test,
    })
    return record


def repo_identity() -> tuple[str, str]:
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True).strip()
    return commit, tree


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-record", action="store_true", help="append protocol identity only; runs no test")
    mode.add_argument("--execute", action="store_true", help="run the one frozen baseline/mutant/fix sequence")
    parser.add_argument("--source-archive", default="work/transformers-src-base.tar")
    parser.add_argument("--grader-test", default="work/transformers-grader-v1/tests/utils/test_generic.py")
    parser.add_argument("--fix-target", default="work/transformers-47468-controls-v1/fix-generic.py")
    parser.add_argument("--python", default="/opt/homebrew/Caskroom/miniforge/base/bin/python")
    parser.add_argument("--overlay-site-packages", default=(
        "/Users/user/Documents/Codex/2026-10-24/vare-flagship-research-verified-coding-agent/"
        "work/task-certification-20261025/work/mlx-core-development-agent-feasibility-v1/"
        "venv/lib/python3.12/site-packages"
    ))
    parser.add_argument("--output", default=str(EXPERIMENT / "run-v1"))
    args = parser.parse_args()

    global PROTOCOL_DATA
    PROTOCOL_DATA = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validation = validate_inputs(args)
    if not args.freeze_record and not args.execute:
        print(json.dumps({"status": "PREFLIGHT_VALID_NO_TEST_EXECUTED", **validation}, indent=2))
        return 0
    if args.freeze_record:
        require(not MANIFEST.exists(), "protocol manifest already exists; refusing overwrite")
        commit, tree = repo_identity()
        require(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip() == "",
                "worktree must be clean before freezing protocol identity")
        event = {
            "sequence": 1,
            "event_type": "protocol_frozen",
            "timestamp_utc": utc_now(),
            "previous_event_sha256": "0" * 64,
            "protocol_commit": commit,
            "protocol_tree": tree,
            "protocol_sha256": validation["protocol_sha256"],
            "lock_sha256": sha_file(LOCK),
            "runner_sha256": validation["runner_sha256"],
            "source_archive_sha256": validation["source_archive_sha256"],
            "grader_test_sha256": validation["grader_test_sha256"],
            "fix_target_sha256": validation["fix_target_sha256"],
            "mutant_target_sha256": validation["mutant_target_sha256"],
            "network_deny_probe": validation["network_deny_probe"],
            "claim_limit": "Development-only source/grader qualification; no model inference or held-out evidence.",
        }
        event["event_sha256"] = event_hash(event)
        write_new(MANIFEST, (json.dumps(event, sort_keys=True) + "\n").encode())
        print(json.dumps({"status": "PROTOCOL_IDENTITY_FROZEN_NO_TEST_EXECUTED", "manifest": str(MANIFEST), **validation}, indent=2))
        return 0

    frozen_events = load_manifest()
    require(frozen_events and frozen_events[0]["event_type"] == "protocol_frozen", "frozen protocol identity missing")
    commit, tree = repo_identity()
    frozen = frozen_events[0]
    require(commit == frozen["protocol_commit"] and tree == frozen["protocol_tree"], "code commit differs from frozen pre-outcome identity")
    require(not Path(args.output).exists(), "run output exists; one-shot runner refuses to repeat")
    run_dir = resolve(args.output)
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "records", "resources", "workspaces", "tmp"):
        (run_dir / name).mkdir()
    protocol = PROTOCOL_DATA
    fix_bytes = resolve(args.fix_target).read_bytes()
    mutant_bytes = make_mutant(fix_bytes)
    phases = [
        ("baseline", False, None),
        ("mutant", False, mutant_bytes),
        ("fix", True, fix_bytes),
    ]
    outcomes = []
    for phase, expected_pass, target_bytes in phases:
        if phase == "baseline":
            # Reconstruct the immutable base target directly from its source archive.
            with tarfile.open(resolve(args.source_archive), mode="r:") as tar:
                member = tar.extractfile(TARGET)
                require(member is not None, "baseline target is missing from source archive")
                target_bytes = member.read()
        result = run_phase(phase, expected_pass, target_bytes, args, protocol, run_dir, validation)
        outcomes.append(result)
        if not result["expected_outcome_matched"]:
            break
    expected_phases = ["baseline", "mutant", "fix"]
    qualified = (
        [row["phase"] for row in outcomes] == expected_phases
        and all(row["expected_outcome_matched"] for row in outcomes)
    )
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_commit": frozen["protocol_commit"],
        "protocol_sha256": validation["protocol_sha256"],
        "runner_sha256": validation["runner_sha256"],
        "status": "SOURCE_GRADER_QUALIFIED" if qualified else "UNQUALIFIED",
        "task_id": protocol["task"]["task_id"],
        "task_role": "DEVELOPMENT_ONLY",
        "phases": [{key: row[key] for key in ("phase", "expected_pass", "expected_outcome_matched", "return_code", "abort_reason", "raw_record_path")} for row in outcomes],
        "claim_limit": protocol["evidence_policy"]["claim_limit"],
    }
    summary_path = run_dir / "summary.json"
    write_json_new(summary_path, summary)
    append_event("run_summary", {"run_summary_path": str(summary_path.relative_to(EXPERIMENT)), "summary_sha256": sha_file(summary_path), "status": summary["status"]})
    print(json.dumps(summary, indent=2))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
