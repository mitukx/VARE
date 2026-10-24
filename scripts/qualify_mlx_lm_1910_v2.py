#!/usr/bin/env python3
"""Prospectively qualify MLX-LM #1910 with an isolated parser oracle.

The candidate checkout is mounted read-only in a digest-pinned, no-network
container. Hidden expected values remain in this host process. The container
receives only JSON inputs and returns parser outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

BASE = "bacd9e8e1f3a5be7dd64bae3a9ad8d74471ccc3a"
FIXED = "a537041adc86d73cfae4cc1375b2c8851c652c60"
BASE_FILE_SHA256 = "2d212e4e6f175178bd80b7db9963ccad613e29419f976e6d6b3272417f953722"
FIXED_FILE_SHA256 = "d1029aae22ed475e9d7c09d29c37e74dca8df52b0fb046f000f2e3f9b83a6583"
TEST_SHA256 = "09ba0fcd264ee83f5a3da1675345f05d839e5c1c09fb28f44cd8911d05045fcf"
IMAGE = "vare-mlx-cpu@sha256:a377e54dc56cbbf2b40627966ce8ff7c5152a0d580175921f45cbdbeff30bfa7"
TARGET = "mlx_lm/tool_parsers/qwen3_coder.py"
TEST = "tests/test_tool_parsing.py"

# Development-only cases. The oracle stays on the host; the candidate process
# sees input strings and emits values, but never receives these expected values.
CASES: list[tuple[str, str, Any]] = [
    ("object", '{"kind":"new","nested":{"n":2}}', {"kind": "new", "nested": {"n": 2}}),
    ("array", '["a",{"n":2}]', ["a", {"n": 2}]),
    ("plain_string", "plain text", "plain text"),
    ("malformed_json", '["unfinished"', '["unfinished"'),
    ("number_scalar", "123", "123"),
    ("quoted_scalar", '"quoted"', '"quoted"'),
    ("null_literal", "null", None),
    ("boolean_scalar", "true", "true"),
    ("empty_string", "", ""),
]

CHILD = r'''import json, sys
from mlx_lm.tool_parsers.qwen3_coder import parse_tool_call
payload = json.load(sys.stdin)
tools = [{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"value":{"anyOf":[{"type":"object"},{"type":"array"},{"type":"string"}]}}}}}]
outputs = []
for value in payload:
    call = "<function=f><parameter=value>" + value + "</parameter></function>"
    try:
        parsed = parse_tool_call(call, tools)
        outputs.append({"ok": True, "value": parsed["arguments"]["value"]})
    except BaseException as exc:
        outputs.append({"ok": False, "error": type(exc).__name__ + ": " + str(exc)})
print(json.dumps(outputs, sort_keys=True, separators=(",", ":")))
'''

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def run(args: list[str], *, timeout: int = 300, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp",
           "GIT_NO_REPLACE_OBJECTS": "1", "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": os.devnull, "DOCKER_HOST": os.environ.get("DOCKER_HOST", "")}
    return subprocess.run(args, input=input_bytes, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, timeout=timeout, env=env, check=False)

def archive_commit(repo: Path, commit: str, destination: Path) -> list[str]:
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    proc = run(["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null",
                "archive", "--format=tar", commit], timeout=120)
    if proc.returncode:
        raise RuntimeError(f"git archive {commit[:12]} failed: {proc.stdout.decode(errors='replace')[-2000:]}")
    import io
    skipped: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(proc.stdout), mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            p = Path(member.name)
            if p.is_absolute() or ".." in p.parts:
                raise RuntimeError(f"unsafe archive path: {member.name!r}")
            if member.issym() or member.islnk():
                skipped.append(member.name)
            elif not (member.isfile() or member.isdir()):
                skipped.append(member.name)
        for member in members:
            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                continue
            archive.extract(member, destination, filter="data")
    return skipped

def docker_run(source: Path, argv: list[str], *, stdin: bytes | None = None, timeout: int = 300) -> subprocess.CompletedProcess[bytes]:
    source = source.resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise RuntimeError("candidate source root must be a real directory")
    command = ["docker", "--context", "colima", "run", "--rm", "--init",
               "--network=none", "--read-only", "--cap-drop=ALL",
               "--security-opt=no-new-privileges", "--pids-limit=64",
               "--cpus=2", "--memory=1g", "--memory-swap=1g",
               "--user=65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m",
               "--mount", f"type=bind,src={source},dst=/candidate,readonly",
               "--workdir=/candidate", "--env=PYTHONPATH=/candidate",
               "--env=PYTHONDONTWRITEBYTECODE=1", "--env=HOME=/tmp", IMAGE, *argv]
    return run(command, timeout=timeout, input_bytes=stdin)

def edit_mutant(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"{label}: expected unique mutation anchor")
    path.write_text(source.replace(old, new), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--git-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.git_source.resolve(strict=True)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir()
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vare-mlx1910-", dir=str(out)) as temp:
        root = Path(temp)
        base = root / "base"
        fixed = root / "fixed"
        base_skipped = archive_commit(repo, BASE, base)
        fixed_skipped = archive_commit(repo, FIXED, fixed)
        (out / "source-export-manifest.json").write_text(
            json.dumps({"baseline_skipped_nonregular_paths": base_skipped,
                        "fixed_skipped_nonregular_paths": fixed_skipped,
                        "policy": "extract regular files and directories only; reject path traversal; no symlink is materialized"},
                       indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for label, tree in (("baseline", base), ("known_fix", fixed)):
            target = tree / TARGET
            test_path = fixed / TEST
            actual = sha256_bytes(target.read_bytes())
            expected_hash = BASE_FILE_SHA256 if label == "baseline" else FIXED_FILE_SHA256
            if actual != expected_hash:
                raise RuntimeError(f"{label} source hash mismatch: {actual}")
            if sha256_bytes(test_path.read_bytes()) != TEST_SHA256:
                raise RuntimeError("upstream test SHA-256 mismatch")
            if label == "baseline":
                shutil.copy2(test_path, tree / TEST)
            result = docker_run(tree, ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                       TEST, "-k", "test_qwen3_coder_untyped_param", "--disable-warnings",
                                       "--maxfail=1"], timeout=120)
            (out / "raw" / f"{label}-upstream-test.log").write_bytes(result.stdout)
            records.append({"arm": label, "stage": "upstream_test", "returncode": result.returncode,
                            "source_sha256": actual, "output_sha256": sha256_bytes(result.stdout)})
        mutant_specs = [
            ("mutant_restore_default_string",
             '        try:\n            value = json.loads(param_value)\n        except json.JSONDecodeError:\n            return param_value\n        return value if isinstance(value, (dict, list)) else param_value\n',
             '        param_type = "string"\n'),
            ("mutant_parse_any_json_scalar",
             '        try:\n            value = json.loads(param_value)\n        except json.JSONDecodeError:\n            return param_value\n        return value if isinstance(value, (dict, list)) else param_value\n',
             '        return json.loads(param_value)\n'),
            ("mutant_object_only",
             '        return value if isinstance(value, (dict, list)) else param_value\n',
             '        return value if isinstance(value, dict) else param_value\n'),
        ]
        for label, old, new in mutant_specs:
            tree = root / label
            shutil.copytree(fixed, tree, symlinks=False)
            edit_mutant(tree / TARGET, old, new, label)
            diff = subprocess.run(["diff", "-u", str(fixed / TARGET), str(tree / TARGET)],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            (out / "raw" / f"{label}.patch").write_bytes(diff.stdout)
            result = docker_run(tree, ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                       TEST, "-k", "test_qwen3_coder_untyped_param", "--disable-warnings",
                                       "--maxfail=1"], timeout=120)
            (out / "raw" / f"{label}-upstream-test.log").write_bytes(result.stdout)
            records.append({"arm": label, "stage": "upstream_test", "returncode": result.returncode,
                            "source_sha256": sha256_bytes((tree / TARGET).read_bytes()),
                            "output_sha256": sha256_bytes(result.stdout)})
        for label, tree in (("baseline", base), ("known_fix", fixed),
                            *((name, root / name) for name, _, _ in mutant_specs)):
            payload = json.dumps([case[1] for case in CASES], separators=(",", ":")).encode()
            result = docker_run(tree, ["python", "-c", CHILD], stdin=payload, timeout=60)
            log_path = out / "raw" / f"{label}-oracle.json"
            log_path.write_bytes(result.stdout)
            actual: list[Any] | None = None
            parse_error = None
            if result.returncode == 0:
                try:
                    actual = json.loads(result.stdout.decode().splitlines()[-1])
                except Exception as exc:
                    parse_error = type(exc).__name__ + ": " + str(exc)
            cases = []
            if actual is not None and len(actual) == len(CASES):
                for (case_id, _, expected), got in zip(CASES, actual):
                    passed = got == {"ok": True, "value": expected}
                    cases.append({"case_id": case_id, "passed": passed, "actual": got})
            passed = bool(cases) and len(cases) == len(CASES) and all(c["passed"] for c in cases)
            records.append({"arm": label, "stage": "independent_oracle", "returncode": result.returncode,
                            "source_sha256": sha256_bytes((tree / TARGET).read_bytes()),
                            "passed": passed, "cases": cases, "parse_error": parse_error,
                            "output_sha256": sha256_bytes(result.stdout)})
        expected_test = {"baseline": False, "known_fix": True,
                         "mutant_restore_default_string": False,
                         "mutant_parse_any_json_scalar": False, "mutant_object_only": False}
        by_name = {r["arm"]: r for r in records if r["stage"] == "upstream_test"}
        oracle = {r["arm"]: r for r in records if r["stage"] == "independent_oracle"}
        gate = all((by_name[k]["returncode"] == (0 if v else 1)) for k, v in expected_test.items())
        gate &= all(oracle[k]["passed"] == (k == "known_fix") for k in expected_test)
        if not gate:
            status = "NO_GO"
        else:
            status = "GRADER_CALIBRATION_PASS_DEVELOPMENT_ONLY"
    summary = {"schema_version": 1, "protocol_id": "mlsys-repo-task-mlx-lm-1910-v2",
               "status": status, "source_commits": {"baseline": BASE, "fixed": FIXED},
               "image": IMAGE, "records": records,
               "elapsed_seconds": round(time.monotonic() - started, 3)}
    (out / "qualification.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if status == "GRADER_CALIBRATION_PASS_DEVELOPMENT_ONLY" else 2

if __name__ == "__main__":
    raise SystemExit(main())
