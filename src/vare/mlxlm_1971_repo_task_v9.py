"""MLX-LM #1971 v9 grader with a complete host-evaluator lock."""

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
    sha256_file,
    validate_patch,
)
from .mlxlm_1971_repo_task_v7 import (
    _candidate_inventory,
    _load_source_manifest,
    verify_pinned_source_tree,
)

TASK_ROOT = Path(__file__).resolve().parents[2] / (
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
)
PROTOCOL_PATH = TASK_ROOT / "protocol-v9.json"
LOCK_PATH = TASK_ROOT / "protocol-v9.lock.json"
SOURCE_MANIFEST_PATH = TASK_ROOT / "baseline-python-manifest-v6.json"
EXPECTED_OUTPUT = TASK_ROOT / "expected-output-v2.json"
ADAPTER = TASK_ROOT / "grader_adapter_v4.py"
IMPLEMENTATION_FILES = {
    "src/vare/__init__.py",
    "src/vare/_public.py",
    "src/vare/attestation.py",
    "src/vare/container_isolation.py",
    "src/vare/evidence.py",
    "src/vare/isolated_candidate.py",
    "src/vare/mlxlm_1971_repo_task.py",
    "src/vare/mlxlm_1971_repo_task_v4.py",
    "src/vare/mlxlm_1971_repo_task_v7.py",
    "src/vare/mlxlm_1971_repo_task_v9.py",
    "src/vare/types.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/Dockerfile",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/grader_adapter_v4.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/task.md",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/expected-output-v2.json",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/baseline-python-manifest-v6.json",
    "scripts/generate_mlxlm_1971_v2_goldens.py",
    "scripts/run_mlxlm_1971_calibration_v9.py",
    "tests/test_mlxlm_1971_repo_task_v9.py",
    "tests/test_mlxlm_1971_calibration_v9.py",
}


def verify_protocol_v9() -> dict[str, Any]:
    """Check v9 bytes and Git index metadata before using the grader."""
    try:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MlxLm1971TaskError("v9 protocol or lock unavailable") from exc
    if (
        protocol.get("protocol_id") != "vare.mlsys-repository-task.mlxlm-1971.v9"
        or lock.get("protocol_id") != protocol.get("protocol_id")
        or lock.get("protocol_sha256") != sha256_file(PROTOCOL_PATH)
    ):
        raise MlxLm1971TaskError("v9 protocol hash lock mismatch")
    hashes = protocol.get("implementation_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != IMPLEMENTATION_FILES
        or hashes != lock.get("implementation_sha256")
    ):
        raise MlxLm1971TaskError("v9 implementation inventory mismatch")

    root = Path(__file__).resolve().parents[2]
    task_inputs = {
        str(PROTOCOL_PATH.relative_to(root)),
        str(LOCK_PATH.relative_to(root)),
        str(SOURCE_MANIFEST_PATH.relative_to(root)),
        str(EXPECTED_OUTPUT.relative_to(root)),
        str(TASK_DESCRIPTION.relative_to(root)),
        str((TASK_ROOT / "protocol-v8.json").relative_to(root)),
    }
    locked_paths = set(hashes) | task_inputs
    try:
        rows = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-v", "--", *sorted(locked_paths)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise MlxLm1971TaskError("cannot verify v9 Git index flags") from exc
    indexed: set[str] = set()
    for row in rows:
        if len(row) < 3 or row[1] != " ":
            raise MlxLm1971TaskError("cannot parse v9 Git index flags")
        tag, relative = row[0], row[2:]
        indexed.add(relative)
        if tag == "S" or tag.islower():
            raise MlxLm1971TaskError(f"v9 input has a hidden Git index flag: {relative}")
    if indexed != locked_paths:
        raise MlxLm1971TaskError("v9 protocol input is missing or untracked")

    actual = {name: sha256_file(root / name) for name in hashes}
    if actual != hashes:
        raise MlxLm1971TaskError("v9 implementation hash mismatch")
    if protocol.get("protocol_parent_sha256") != sha256_file(TASK_ROOT / "protocol-v8.json"):
        raise MlxLm1971TaskError("v9 parent protocol hash mismatch")
    parent = json.loads((TASK_ROOT / "protocol-v8.json").read_text(encoding="utf-8"))
    for key in (
        "baseline_commit",
        "baseline_source_sha256",
        "fixed_commit",
        "fixed_source_sha256",
        "expected_output_sha256",
        "source_manifest_sha256",
        "task_prompt_sha256",
        "runtime",
        "value_scales",
    ):
        if protocol.get(key) != parent.get(key):
            raise MlxLm1971TaskError(f"v9 changed frozen v8 field: {key}")
    manifest = _load_source_manifest()
    if (
        protocol.get("baseline_commit") != BASELINE_COMMIT
        or protocol.get("baseline_source_sha256") != BASELINE_SOURCE_SHA256
        or protocol.get("fixed_commit") != FIXED_COMMIT
        or protocol.get("fixed_source_sha256") != FIXED_SOURCE_SHA256
        or protocol.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH)
        or protocol.get("expected_output_sha256") != sha256_file(EXPECTED_OUTPUT)
        or protocol.get("task_prompt_sha256") != sha256_file(TASK_DESCRIPTION)
        or protocol.get("runtime", {}).get("image") != IMAGE
        or protocol.get("source_file_count") != manifest["file_count"]
    ):
        raise MlxLm1971TaskError("v9 pinned task identity mismatch")
    return protocol


def export_candidate_v9(
    source_root: Path, workspace: Path, patch_path: Path | None = None
) -> dict[str, Any]:
    verify_protocol_v9()
    baseline = verify_pinned_source_tree(source_root)
    if workspace.exists() or workspace.is_symlink():
        raise MlxLm1971TaskError("candidate workspace already exists")
    if patch_path is not None:
        if patch_path.is_symlink() or not patch_path.is_file():
            raise MlxLm1971TaskError("patch must be a regular file")
        validate_patch(patch_path.read_text(encoding="utf-8"))
    source_root = source_root.resolve(strict=True)
    workspace.mkdir(parents=True)
    for relative in sorted(set(baseline) - {"mlx_lm/__init__.py"}):
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)
    if patch_path is not None:
        try:
            subprocess.run(
                ["git", "init", "-q"], cwd=workspace, check=True, timeout=10
            )
            subprocess.run(
                ["git", "apply", "--check", str(patch_path.resolve())],
                cwd=workspace,
                check=True,
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "apply", str(patch_path.resolve())],
                cwd=workspace,
                check=True,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            shutil.rmtree(workspace, ignore_errors=True)
            raise MlxLm1971TaskError("candidate patch did not apply cleanly") from exc
        shutil.rmtree(workspace / ".git", ignore_errors=True)
    _candidate_inventory(workspace, baseline)
    target = workspace / TARGET_PATH
    return {
        "task_id": TASK_ID,
        "baseline_commit": BASELINE_COMMIT,
        "candidate_patch_sha256": sha256_file(patch_path) if patch_path else None,
        "candidate_source_sha256": sha256_file(target),
        "fixed_source_match": sha256_file(target) == FIXED_SOURCE_SHA256,
        "changed_files": [TARGET_PATH] if patch_path else [],
        "baseline_python_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "git_history_exported": False,
        "tests_exported": False,
        "license_exported": False,
        "task_prompt_sha256": sha256_file(TASK_DESCRIPTION),
    }


def grade_workspace_v9(
    workspace: Path,
    *,
    image: str = IMAGE,
    timeout_s: float = TIMEOUT_S,
    memory_mb: int = MEMORY_MB,
) -> dict[str, Any]:
    protocol = verify_protocol_v9()
    runtime = protocol["runtime"]
    if image != runtime["image"]:
        raise MlxLm1971TaskError("requested image differs from v9 protocol")
    if timeout_s != runtime["timeout_seconds"] or memory_mb != runtime["memory_mb"]:
        raise MlxLm1971TaskError("resource limits differ from v9 protocol")
    baseline = _load_source_manifest()["files"]
    _candidate_inventory(workspace, baseline)
    expected = json.loads(EXPECTED_OUTPUT.read_text(encoding="utf-8"))
    calls: list[FunctionCall] = []
    for scale, golden in zip(protocol["value_scales"], expected["cases"], strict=True):
        case = {**CASE, "value_scale": scale}
        calls.append(
            FunctionCall(
                case_id=f"value-scale-{scale:g}-full",
                module="vare_task_adapter_v4",
                function="run",
                args=[{**case, "mode": "full"}],
                expected={
                    "full_reference": golden["full_reference"],
                    "full_configured": golden["full_configured"],
                },
            )
        )
        calls.append(
            FunctionCall(
                case_id=f"value-scale-{scale:g}-cached",
                module="vare_task_adapter_v4",
                function="run",
                args=[{**case, "mode": "cached"}],
                expected={
                    "incremental_reference": golden["incremental_reference"],
                    "incremental_configured": golden["incremental_configured"],
                },
            )
        )
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1971-v9-grade-") as folder:
        subject = Path(folder)
        shutil.copytree(workspace / "mlx_lm", subject / "mlx_lm")
        shutil.copyfile(ADAPTER, subject / "vare_task_adapter_v4.py")
        for path in (subject, *(subject / "mlx_lm").rglob("*.py"), subject / "vare_task_adapter_v4.py"):
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
        "candidate_source_sha256": sha256_file(workspace / TARGET_PATH),
        "candidate_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "grader_source_sha256": sha256_file(Path(__file__).resolve()),
        "grader_adapter_sha256": sha256_file(ADAPTER),
        "expected_output_sha256": sha256_file(EXPECTED_OUTPUT),
    }
