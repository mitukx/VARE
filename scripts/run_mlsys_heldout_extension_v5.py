#!/usr/bin/env python3
"""Run frozen deterministic baseline/fix/mutant controls for MLX-LM task extension v5."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "protocols/mlsys_heldout_extension_v5.json"
DEFAULT_LOCK = ROOT / "protocols/mlsys_heldout_extension_v5.lock.json"
GIT_DIR = ROOT / "source/mlx-lm.git/.git"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_archive(revision: str, paths: list[str], destination: Path) -> None:
    raw = subprocess.check_output(
        ["git", f"--git-dir={GIT_DIR}", "-c", "core.useReplaceRefs=false", "archive", "--format=tar", revision, *paths],
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    )
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError(f"unsafe Git archive member: {member.name}")
        archive.extractall(destination, filter="data")


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def collect_test_ids(value: str) -> list[str]:
    script = (
        "import unittest; "
        f"suite=unittest.defaultTestLoader.loadTestsFromName({value!r}); "
        "flatten=lambda s: [x for t in s for x in (flatten(t) if isinstance(t, unittest.TestSuite) else [t])]; "
        "ids=sorted(t.id() for t in flatten(suite)); "
        "print('VARE_COLLECTED_IDS='+__import__('json').dumps(ids)); "
        "assert ids, 'no test cases collected'"
    )
    return ["python", "-c", script]


def docker_run(protocol: dict, source: Path, grader: Path, command: list[str], label: str, timeout_s: int) -> tuple[int | None, bytes, float, str | None]:
    environment = protocol["environment"]
    args = [
        "docker", "--context", "colima", "run", "--rm", "--pull=never",
        "--name", f"vare-{protocol['protocol_id']}-{label}-{os.getpid()}",
        "--network=none", "--read-only", "--cpus=1", "--memory=1536m", "--pids-limit=128",
        "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m", "--user=65534:65534",
        "--mount", f"type=bind,src={source},dst=/subject,readonly",
        "--mount", f"type=bind,src={grader},dst=/grader,readonly",
        "--workdir=/grader", "--env", "PYTHONPATH=/subject:/grader",
        "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONHASHSEED=0",
        "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache",
        "--env", "HF_HUB_OFFLINE=1", "--env", "HF_DATASETS_OFFLINE=1",
        environment["image_id"], "python", "-c",
        "import pathlib,subprocess,sys; p=subprocess.run(" + repr(command) + "); "
        "print('VARE_CGROUP_MEMORY_PEAK='+pathlib.Path('/sys/fs/cgroup/memory.peak').read_text().strip() if pathlib.Path('/sys/fs/cgroup/memory.peak').exists() else 'VARE_CGROUP_MEMORY_PEAK=unavailable'); "
        "print('VARE_CGROUP_MEMORY_MAX='+pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip() if pathlib.Path('/sys/fs/cgroup/memory.max').exists() else 'VARE_CGROUP_MEMORY_MAX=unavailable'); "
        "sys.exit(p.returncode)",
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        return None, error.stdout or b"", time.monotonic() - started, "TIMEOUT"
    return result.returncode, result.stdout, time.monotonic() - started, None


def cgroup_value(text: str, label: str) -> int | None:
    prefix = f"VARE_CGROUP_{label}="
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):]
            return int(value) if value.isdigit() else None
    return None


def test_outcomes(text: str) -> dict[str, str]:
    outcomes = {}
    for line in text.splitlines():
        match = re.match(r"\s*(test_[A-Za-z0-9_]+)\s+\([^)]*\)\s+\.\.\.\s+(ok|FAIL|ERROR|skipped)", line)
        if match:
            outcomes[match.group(1)] = match.group(2)
    return outcomes


def write_exclusive(path: Path, raw: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"append-only artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return sha(raw)


def prepare_grader(task: dict, destination: Path) -> None:
    destination.mkdir(parents=True)
    if task.get("grader_delivery") == "copy_vare_file":
        source = ROOT / task["grader_path"]
        (destination / source.name).write_bytes(source.read_bytes())
    else:
        git_archive(task["grader_revision"], task["grader_archive_paths"], destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--staging", type=Path)
    args = parser.parse_args()
    protocol_raw = args.protocol.read_bytes()
    protocol = json.loads(protocol_raw)
    lock = json.loads(args.lock.read_text())
    protocol_sha = sha(protocol_raw)
    if lock.get("protocol_sha256") != protocol_sha or lock.get("protocol_id") != protocol.get("protocol_id"):
        raise SystemExit("frozen protocol lock mismatch")
    if lock.get("runner_sha256") != sha(Path(__file__).read_bytes()):
        raise SystemExit("runner source hash differs from frozen protocol")
    allocation_path = ROOT / protocol["allocation"]["path"]
    allocation_sha = sha(allocation_path.read_bytes())
    if allocation_sha != protocol["allocation"]["sha256"]:
        raise SystemExit("allocation hash differs from frozen protocol")
    image_id = protocol["environment"]["image_id"]
    image = subprocess.check_output(
        ["docker", "--context", "colima", "image", "inspect", image_id, "--format", "{{.Id}} {{.Os}}/{{.Architecture}} {{.Size}}"], text=True
    ).strip()
    if not image.startswith(image_id + " linux/arm64 "):
        raise SystemExit(f"unexpected container image identity: {image}")

    output = args.output or ROOT / protocol["output_directory"]
    staging = args.staging or ROOT / "work" / protocol["protocol_id"]
    if output.exists() or staging.exists():
        raise SystemExit("output or staging path exists; run is append-only")
    output.mkdir(parents=True)
    staging.mkdir(parents=True)
    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "allocation_sha256": allocation_sha,
        "runner_sha256": sha(Path(__file__).read_bytes()),
        "container_image_inspect": image,
        "candidate_model_or_agent_runs": 0,
        "training": False,
        "heldout_inference": False,
        "preflight_records": [],
        "task_decisions": {},
        "records": [],
    }

    prepared = {}
    # Import and collect the exact tests first. Test bodies are never invoked in this phase.
    for task in protocol["tasks"]:
        task_id = task["task_id"]
        stage = staging / task_id / "preflight"
        source = stage / "subject"
        grader = stage / "grader"
        source.mkdir(parents=True)
        git_archive(task["fixed_revision"], task["source_archive_paths"], source)
        prepare_grader(task, grader)
        if task_id.startswith("mlxlm-1877"):
            selector = Path(task["grader_path"]).stem
            expected_case_suffixes = task["selected_test_cases"]
        else:
            selector = task["test_command"][-1]
            expected_case_suffixes = task["selected_test_cases"]
        command = collect_test_ids(selector)
        code, raw, wall, timeout = docker_run(protocol, source, grader, command, "preflight-" + task_id, 150)
        text = raw.decode("utf-8", "replace")
        collected = []
        for line in text.splitlines():
            if line.startswith("VARE_COLLECTED_IDS="):
                try:
                    collected = json.loads(line.split("=", 1)[1])
                except json.JSONDecodeError:
                    pass
        actual_suffixes = sorted(item.rsplit(".", 1)[-1] for item in collected)
        valid = not timeout and code == 0 and actual_suffixes == sorted(expected_case_suffixes)
        log_name = f"{task_id}.preflight.log"
        write_exclusive(output / log_name, raw)
        record = {
            "task_id": task_id,
            "protocol_sha256": protocol_sha,
            "command": command,
            "exit_code": code,
            "timeout": timeout,
            "test_body_executed": False,
            "collected_test_ids": collected,
            "expected_test_case_suffixes": expected_case_suffixes,
            "actual_test_case_suffixes": actual_suffixes,
            "preflight_passed": valid,
            "wall_seconds": round(wall, 6),
            "cgroup_memory_peak_bytes": cgroup_value(text, "MEMORY_PEAK"),
            "cgroup_memory_max_bytes": cgroup_value(text, "MEMORY_MAX"),
            "raw_log": log_name,
            "raw_log_sha256": sha(raw),
        }
        record_raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
        write_exclusive(output / f"{task_id}.preflight.json", record_raw)
        summary["preflight_records"].append(record)
        prepared[task_id] = valid
        print("PREFLIGHT " + json.dumps(record, sort_keys=True), flush=True)

    # Each source/fix/mutant is extracted from its pinned immutable revision. The mutant is an explicit derived source tree.
    for task in protocol["tasks"]:
        task_id = task["task_id"]
        if not prepared[task_id]:
            summary["task_decisions"][task_id] = "BLOCKED_PREFLIGHT"
            continue
        controls = []
        for variant in protocol["controls"]["ordered_variants"]:
            revision = task["baseline_revision"] if variant == "baseline" else task["fixed_revision"]
            variant_root = staging / task_id / variant
            source = variant_root / "subject"
            grader = variant_root / "grader"
            source.mkdir(parents=True)
            git_archive(revision, task["source_archive_paths"], source)
            prepare_grader(task, grader)
            mutated = False
            if variant == "incorrect_mutant":
                target = source / task["target_path"]
                source_text = target.read_text()
                mutant = task["incorrect_mutant"]
                if task_id.startswith("mlxlm-1877"):
                    old = (
                        "    raise ValueError(\n"
                        "        f\"unsupported compressed-tensors format {fmt!r}; \"\n"
                        "        \"dequantize to bf16 before converting\"\n"
                        "    )"
                    )
                    new = '    return {"group_size": 32, "bits": 4, "mode": "affine"}'
                    if source_text.count(old) != 1:
                        raise RuntimeError("pinned #1877 mutant anchor is not unique")
                    target.write_text(source_text.replace(old, new))
                else:
                    old_a = "W[..., k:j] -= e @ Hinv[k : k + 1, k:j]"
                    new_a = "W[..., k : k + j] -= e @ Hinv[k : k + 1, k : k + j]"
                    old_b = "err[..., k - i : k - i + 1] = e"
                    new_b = "err[..., k : k + 1] = e"
                    if source_text.count(old_a) != 1 or source_text.count(old_b) != 1:
                        raise RuntimeError("pinned #1880 mutant anchors are not unique")
                    target.write_text(source_text.replace(old_a, new_a).replace(old_b, new_b))
                mutated = True
            command = task["test_command"]
            code, raw, wall, timeout = docker_run(protocol, source, grader, command, variant + "-" + task_id, protocol["controls"]["timeout_seconds_per_variant"])
            text = raw.decode("utf-8", "replace")
            outcomes = test_outcomes(text)
            expected_outcomes = task["expected_test_outcomes"][variant]
            outcome_match = outcomes == expected_outcomes
            memory_peak = cgroup_value(text, "MEMORY_PEAK")
            memory_max = cgroup_value(text, "MEMORY_MAX")
            resource_valid = memory_peak is not None and memory_max == protocol["controls"]["memory_limit_bytes"] and memory_peak <= memory_max
            expected_exit_code = 0 if variant == "known_fix" else 1
            valid = not timeout and code == expected_exit_code and outcome_match and resource_valid
            log_name = f"{task_id}.{variant}.log"
            write_exclusive(output / log_name, raw)
            record = {
                "task_id": task_id,
                "protocol_id": protocol["protocol_id"],
                "protocol_sha256": protocol_sha,
                "allocation_sha256": allocation_sha,
                "runner_sha256": summary["runner_sha256"],
                "variant": variant,
                "source_revision": revision,
                "source_tree_sha256": tree_sha256(source),
                "target_sha256": sha((source / task["target_path"]).read_bytes()),
                "grader_tree_sha256": tree_sha256(grader),
                "grader_sha256": sha((grader / Path(task["grader_path"]).name).read_bytes()) if task.get("grader_delivery") == "copy_vare_file" else sha((grader / task["grader_path"]).read_bytes()),
                "container_image_id": protocol["environment"]["image_id"],
                "command": command,
                "network": "none",
                "container_read_only": True,
                "source_and_grader_mounts_read_only": True,
                "container_uid": "65534:65534",
                "cpu_limit": protocol["controls"]["cpu_limit"],
                "memory_limit_bytes": protocol["controls"]["memory_limit_bytes"],
                "exit_code": code,
                "timeout": timeout,
                "test_outcomes": outcomes,
                "expected_test_outcomes": expected_outcomes,
                "test_body_executed": bool(outcomes) and not any(value == "skipped" for value in outcomes.values()),
                "control_matches_expectation": valid,
                "mutant_applied": mutated,
                "wall_seconds": round(wall, 6),
                "cgroup_memory_peak_bytes": memory_peak,
                "cgroup_memory_max_bytes": memory_max,
                "raw_log": log_name,
                "raw_log_sha256": sha(raw),
            }
            record_raw = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
            record_name = f"{task_id}.{variant}.json"
            record_sha = write_exclusive(output / record_name, record_raw)
            record["record_path"] = (output / record_name).relative_to(ROOT).as_posix()
            record["record_sha256"] = record_sha
            controls.append(record)
            summary["records"].append(record)
            print("VARIANT " + json.dumps(record, sort_keys=True), flush=True)
            if not valid:
                break
        summary["task_decisions"][task_id] = (
            "CERTIFIED_WITHIN_AUDITED_VARE_SCOPE"
            if len(controls) == len(protocol["controls"]["ordered_variants"]) and all(row["control_matches_expectation"] for row in controls)
            else "BLOCKED"
        )

    summary_raw = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()
    write_exclusive(output / "summary.json", summary_raw)
    write_exclusive(output / "summary.sha256", (sha(summary_raw) + "  summary.json\n").encode())
    print("SUMMARY_SHA256=" + sha(summary_raw), flush=True)
    return 0 if all(value == "CERTIFIED_WITHIN_AUDITED_VARE_SCOPE" for value in summary["task_decisions"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
