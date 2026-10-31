"""MLX-LM #1971 v7 grading with pinned source and candidate inventories."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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

TASK_ROOT = Path(__file__).resolve().parents[2] / (
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
)
PROTOCOL_PATH = TASK_ROOT / "protocol-v7.json"
LOCK_PATH = TASK_ROOT / "protocol-v7.lock.json"
SOURCE_MANIFEST_PATH = TASK_ROOT / "baseline-python-manifest-v6.json"
EXPECTED_OUTPUT = TASK_ROOT / "expected-output-v2.json"
ADAPTER = TASK_ROOT / "grader_adapter_v4.py"
IMPLEMENTATION_FILES = {
    "src/vare/container_isolation.py",
    "src/vare/isolated_candidate.py",
    "src/vare/mlxlm_1971_repo_task.py",
    "src/vare/mlxlm_1971_repo_task_v4.py",
    "src/vare/mlxlm_1971_repo_task_v7.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/Dockerfile",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/grader_adapter_v4.py",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/task.md",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/expected-output-v2.json",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/baseline-python-manifest-v6.json",
    "scripts/generate_mlxlm_1971_v2_goldens.py",
    "scripts/run_mlxlm_1971_calibration_v7.py",
    "tests/test_mlxlm_1971_repo_task_v7.py",
    "tests/test_mlxlm_1971_calibration_v7.py",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_source_manifest() -> dict[str, Any]:
    try:
        manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MlxLm1971TaskError("pinned source manifest unavailable") from exc
    if (
        manifest.get("baseline_commit") != BASELINE_COMMIT
        or manifest.get("repository") != "ml-explore/mlx-lm"
        or manifest.get("target_path") != TARGET_PATH
        or manifest.get("file_count") != len(manifest.get("files", {}))
        or SOURCE_MANIFEST_PATH.is_symlink()
    ):
        raise MlxLm1971TaskError("pinned source manifest identity mismatch")
    return manifest


def verify_protocol_v7() -> dict[str, Any]:
    try:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MlxLm1971TaskError("v7 protocol or lock unavailable") from exc
    if (
        protocol.get("protocol_id") != "vare.mlsys-repository-task.mlxlm-1971.v7"
        or lock.get("protocol_id") != protocol.get("protocol_id")
        or lock.get("protocol_sha256") != sha256_file(PROTOCOL_PATH)
    ):
        raise MlxLm1971TaskError("v7 protocol hash lock mismatch")
    hashes = protocol.get("implementation_sha256")
    if (
        not isinstance(hashes, dict)
        or set(hashes) != IMPLEMENTATION_FILES
        or hashes != lock.get("implementation_sha256")
    ):
        raise MlxLm1971TaskError("v7 implementation inventory mismatch")
    root = Path(__file__).resolve().parents[2]
    locked_paths = set(hashes) | {
        str(PROTOCOL_PATH.relative_to(root)),
        str(LOCK_PATH.relative_to(root)),
        str(SOURCE_MANIFEST_PATH.relative_to(root)),
    }
    try:
        index_rows = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-v", "--", *sorted(locked_paths)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise MlxLm1971TaskError("cannot verify v7 Git index flags") from exc
    indexed: set[str] = set()
    for row in index_rows:
        if len(row) < 3 or row[1] != " ":
            raise MlxLm1971TaskError("cannot parse v7 Git index flags")
        tag, relative = row[0], row[2:]
        indexed.add(relative)
        if tag == "S" or tag.islower():
            raise MlxLm1971TaskError(f"v7 input has a hidden Git index flag: {relative}")
    if indexed != locked_paths:
        raise MlxLm1971TaskError("v7 protocol input is missing or untracked")
    actual = {name: sha256_file(root / name) for name in hashes}
    if actual != hashes:
        raise MlxLm1971TaskError("v7 implementation hash mismatch")
    if protocol.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST_PATH):
        raise MlxLm1971TaskError("v7 pinned source manifest hash mismatch")
    if protocol.get("protocol_parent_sha256") != sha256_file(TASK_ROOT / "protocol-v6.json"):
        raise MlxLm1971TaskError("v7 parent protocol hash mismatch")
    manifest = _load_source_manifest()
    if (
        protocol.get("baseline_commit") != BASELINE_COMMIT
        or protocol.get("baseline_source_sha256") != BASELINE_SOURCE_SHA256
        or protocol.get("fixed_commit") != FIXED_COMMIT
        or protocol.get("fixed_source_sha256") != FIXED_SOURCE_SHA256
        or protocol.get("expected_output_sha256") != sha256_file(EXPECTED_OUTPUT)
        or protocol.get("task_prompt_sha256") != sha256_file(TASK_DESCRIPTION)
        or protocol.get("runtime", {}).get("image") != IMAGE
        or protocol.get("source_file_count") != manifest["file_count"]
    ):
        raise MlxLm1971TaskError("v7 task, source, prompt, golden, or image mismatch")
    return protocol


def _git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise MlxLm1971TaskError("cannot verify pinned source Git objects") from exc


def verify_pinned_source_tree(source_root: Path) -> dict[str, str]:
    """Verify all upstream Python bytes, inventory, and index flags before export."""
    if source_root.is_symlink():
        raise MlxLm1971TaskError("pinned source root cannot be a symlink")
    try:
        root = source_root.resolve(strict=True)
    except OSError as exc:
        raise MlxLm1971TaskError("pinned source root is unavailable") from exc
    if not root.is_dir():
        raise MlxLm1971TaskError("pinned source root must be a directory")
    manifest = _load_source_manifest()
    status = _git(root, "status", "--porcelain", "--untracked-files=all").decode()
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    if head != BASELINE_COMMIT or status:
        raise MlxLm1971TaskError("source checkout is not the clean pinned baseline")

    package = root / "mlx_lm"
    if package.is_symlink() or not package.is_dir():
        raise MlxLm1971TaskError("pinned mlx_lm package is missing or symlinked")
    actual_paths: set[str] = set()
    for current, dirs, files in os.walk(package, followlinks=False):
        here = Path(current)
        for name in dirs:
            if (here / name).is_symlink():
                raise MlxLm1971TaskError("symlink in pinned Python source tree")
        for name in files:
            path = here / name
            if path.is_symlink():
                raise MlxLm1971TaskError("symlink in pinned Python source tree")
            if path.suffix == ".py":
                actual_paths.add(path.relative_to(root).as_posix())

    expected: dict[str, str] = manifest["files"]
    if actual_paths != set(expected):
        raise MlxLm1971TaskError("pinned Python source inventory differs from manifest")

    tree_entries: dict[str, tuple[str, str]] = {}
    for row in _git(root, "ls-tree", "-r", "-z", "--full-tree", BASELINE_COMMIT, "--", "mlx_lm").split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split(" ")
        relative = raw_path.decode("utf-8")
        if relative.endswith(".py"):
            if mode not in {"100644", "100755"} or kind != "blob":
                raise MlxLm1971TaskError("pinned Python source is not a regular Git blob")
            tree_entries[relative] = (oid, mode)
    if set(tree_entries) != set(expected):
        raise MlxLm1971TaskError("pinned Git Python source inventory differs from manifest")

    flags = _git(root, "ls-files", "-v", "--", *sorted(expected)).decode().splitlines()
    indexed: set[str] = set()
    for line in flags:
        if len(line) < 3 or line[1] != " ":
            raise MlxLm1971TaskError("cannot parse pinned source index flags")
        tag, relative = line[0], line[2:]
        indexed.add(relative)
        if tag == "S" or tag.islower():
            raise MlxLm1971TaskError(f"pinned Python source has hidden index flag: {relative}")
    if indexed != set(expected):
        raise MlxLm1971TaskError("pinned Python source is missing from the Git index")

    for relative, expected_sha in expected.items():
        actual_sha = sha256_file(root / relative)
        oid, _ = tree_entries[relative]
        committed = _git(root, "cat-file", "blob", oid)
        if actual_sha != expected_sha or _sha256_bytes(committed) != expected_sha:
            raise MlxLm1971TaskError(f"pinned Python source bytes differ: {relative}")
    if expected.get(TARGET_PATH) != BASELINE_SOURCE_SHA256:
        raise MlxLm1971TaskError("target source hash differs from baseline identity")
    return expected


def _candidate_inventory(workspace: Path, baseline: dict[str, str]) -> None:
    if workspace.is_symlink() or not workspace.is_dir():
        raise MlxLm1971TaskError("candidate workspace must be a real directory")
    package = workspace / "mlx_lm"
    if package.is_symlink() or not package.is_dir():
        raise MlxLm1971TaskError("candidate package is missing or symlinked")
    expected_files = set(baseline) - {"mlx_lm/__init__.py"}
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for current, dirs, files in os.walk(workspace, followlinks=False):
        here = Path(current)
        for name in dirs:
            path = here / name
            if path.is_symlink():
                raise MlxLm1971TaskError("symlink in candidate workspace")
            actual_dirs.add(path.relative_to(workspace).as_posix())
        for name in files:
            path = here / name
            if path.is_symlink():
                raise MlxLm1971TaskError("symlink in candidate workspace")
            if not path.is_file():
                raise MlxLm1971TaskError("candidate workspace contains a non-regular file")
            actual_files.add(path.relative_to(workspace).as_posix())
    expected_dirs = {
        parent.as_posix()
        for relative in expected_files
        for parent in PurePosixPath(relative).parents
        if parent.as_posix() != "."
    }
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise MlxLm1971TaskError("candidate workspace file inventory differs from pinned baseline")
    for relative in expected_files - {TARGET_PATH}:
        if sha256_file(workspace / relative) != baseline[relative]:
            raise MlxLm1971TaskError(f"candidate changed a non-target dependency: {relative}")
    target = workspace / TARGET_PATH
    if target.is_symlink() or not target.is_file():
        raise MlxLm1971TaskError("candidate target is missing or symlinked")


def export_candidate_v7(
    source_root: Path, workspace: Path, patch_path: Path | None = None
) -> dict[str, Any]:
    verify_protocol_v7()
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
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, timeout=10)
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


def grade_workspace_v7(
    workspace: Path,
    *,
    image: str = IMAGE,
    timeout_s: float = TIMEOUT_S,
    memory_mb: int = MEMORY_MB,
) -> dict[str, Any]:
    protocol = verify_protocol_v7()
    runtime = protocol["runtime"]
    if image != runtime["image"]:
        raise MlxLm1971TaskError("requested image differs from v7 protocol")
    if timeout_s != runtime["timeout_seconds"] or memory_mb != runtime["memory_mb"]:
        raise MlxLm1971TaskError("resource limits differ from v7 protocol")
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
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1971-v7-grade-") as folder:
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
