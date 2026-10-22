"""Prospective v3 grader for MLX-LM #1971 with varied scales and KV-cache checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from .isolated_candidate import FunctionCall, grade_calls
from .mlxlm_1971_repo_task import (
    BASELINE_COMMIT,
    BASELINE_SOURCE_SHA256,
    CASE,
    FIXED_COMMIT,
    FIXED_SOURCE_SHA256,
    IMAGE,
    MEMORY_MB,
    TASK_DESCRIPTION,
    TASK_ID,
    TARGET_PATH,
    TIMEOUT_S,
    MlxLm1971TaskError,
    _copy_package,
    sha256_file,
)

TASK_ROOT = Path(__file__).resolve().parents[2] / (
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
)
PROTOCOL_PATH = TASK_ROOT / "protocol-v3.json"
LOCK_PATH = TASK_ROOT / "protocol-v3.lock.json"
EXPECTED_OUTPUT = TASK_ROOT / "expected-output-v2.json"
IMPLEMENTATION_FILES = {
    "src/vare/container_isolation.py",
    "src/vare/isolated_candidate.py",
    "src/vare/mlxlm_1971_repo_task.py",
    "src/vare/mlxlm_1971_repo_task_v3.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/Dockerfile",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/grader_adapter_v2.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/task.md",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/expected-output-v2.json",
    "scripts/run_mlxlm_1971_calibration_v3.py",
    "scripts/generate_mlxlm_1971_v2_goldens.py",
    "tests/test_mlxlm_1971_repo_task_v3.py",
}


def verify_protocol_v3() -> dict[str, Any]:
    try:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MlxLm1971TaskError("v3 protocol or lock unavailable") from exc
    protocol_sha256 = sha256_file(PROTOCOL_PATH)
    if (
        protocol.get("protocol_id") != "vare.mlsys-repository-task.mlxlm-1971.v3"
        or lock.get("protocol_id") != protocol.get("protocol_id")
        or lock.get("protocol_sha256") != protocol_sha256
    ):
        raise MlxLm1971TaskError("v3 protocol hash lock mismatch")
    hashes = protocol.get("implementation_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != IMPLEMENTATION_FILES
        or hashes != lock.get("implementation_sha256")
    ):
        raise MlxLm1971TaskError("v3 implementation inventory mismatch")
    try:
        repo_root = Path(__file__).resolve().parents[2]
        locked = set(hashes) | {
            str(PROTOCOL_PATH.relative_to(repo_root)),
            str(LOCK_PATH.relative_to(repo_root)),
        }
        flags = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-v", "--", *sorted(locked)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
        indexed: set[str] = set()
        for line in flags:
            if len(line) < 3 or line[1] != " ":
                raise MlxLm1971TaskError("cannot parse v3 Git index flags")
            tag, relative = line[0], line[2:]
            indexed.add(relative)
            if tag == "S" or tag.islower():
                raise MlxLm1971TaskError(f"v3 input has a hidden Git index flag: {relative}")
        if indexed != locked:
            raise MlxLm1971TaskError("v3 protocol input is missing or untracked")
        actual = {name: sha256_file(repo_root / name) for name in hashes}
    except (OSError, subprocess.SubprocessError) as exc:
        raise MlxLm1971TaskError("v3 implementation source missing") from exc
    if actual != hashes:
        raise MlxLm1971TaskError("v3 implementation hash mismatch")
    if (
        protocol.get("task_id") != TASK_ID
        or protocol.get("baseline_commit") != BASELINE_COMMIT
        or protocol.get("fixed_commit") != FIXED_COMMIT
        or protocol.get("baseline_source_sha256") != BASELINE_SOURCE_SHA256
        or protocol.get("fixed_source_sha256") != FIXED_SOURCE_SHA256
        or protocol.get("expected_output_sha256") != sha256_file(EXPECTED_OUTPUT)
        or protocol.get("task_prompt_sha256") != sha256_file(TASK_DESCRIPTION)
    ):
        raise MlxLm1971TaskError("v3 task, source, prompt, or golden identity mismatch")
    return protocol


def grade_workspace_v3(
    workspace: Path,
    *,
    image: str = IMAGE,
    timeout_s: float = TIMEOUT_S,
    memory_mb: int = MEMORY_MB,
) -> dict[str, Any]:
    protocol = verify_protocol_v3()
    runtime = protocol["runtime"]
    if image != runtime["image"]:
        raise MlxLm1971TaskError("requested image differs from v3 protocol")
    if timeout_s != runtime["timeout_seconds"] or memory_mb != runtime["memory_mb"]:
        raise MlxLm1971TaskError("resource limits differ from v3 protocol")
    target = workspace / TARGET_PATH
    if workspace.is_symlink() or not workspace.is_dir() or target.is_symlink() or not target.is_file():
        raise MlxLm1971TaskError("candidate workspace is incomplete or unsafe")
    expected = json.loads(EXPECTED_OUTPUT.read_text(encoding="utf-8"))
    calls = [
        FunctionCall(
            case_id=f"value-scale-{scale:g}",
            module="vare_task_adapter_v2",
            function="run",
            args=[{**CASE, "value_scale": scale}],
            expected=golden,
        )
        for scale, golden in zip(protocol["value_scales"], expected["cases"], strict=True)
    ]
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1971-v3-grade-") as folder:
        subject = Path(folder)
        _copy_package(workspace, subject)
        adapter = subject / "vare_task_adapter_v2.py"
        shutil.copyfile(TASK_ROOT / "grader_adapter_v2.py", adapter)
        for path in (subject, *(subject / "mlx_lm").rglob("*.py"), adapter):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        report = grade_calls(
            subject,
            calls,
            task_kind="correctness",
            image=image,
            timeout_s=timeout_s,
            rtol=1e-5,
            atol=1e-8,
            memory_mb=memory_mb,
            discard_candidate_stdout=True,
            preserve_session=True,
        )
    return {
        "task_id": TASK_ID,
        "protocol_id": protocol["protocol_id"],
        "passed": report.passed,
        "cases": [case.to_dict() for case in report.cases],
        "candidate_source_sha256": sha256_file(target),
        "grader_source_sha256": sha256_file(Path(__file__).resolve()),
        "grader_adapter_sha256": sha256_file(TASK_ROOT / "grader_adapter_v2.py"),
        "expected_output_sha256": sha256_file(EXPECTED_OUTPUT),
    }
