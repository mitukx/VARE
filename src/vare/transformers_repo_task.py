"""Narrow isolated evaluator for the frozen Transformers #45852 task."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .isolated_candidate import FunctionCall, grade_calls

TASK_ID = "transformers-45852-4d-mask"
BASELINE_COMMIT = "381032b7e0a8896ae48f90dbe83eae3bf7630953"
BASELINE_TARGET_SHA256 = (
    "51a96b8d23e365014cfcd44049fda9a40a41e76ab5ba80a7b687138d3523f00a"
)
FIXED_COMMIT = "909ece888ed4b55402dc8acbb9c13ddf8ff977b5"
FIXED_TARGET_SHA256 = "5d7dfd5373e0a4b63ce0869f05f5559b01111e38bf68dd02b166ca551ec438b5"
TARGET_PATH = "src/transformers/tokenization_utils_base.py"
PACKAGE_PATH = "src/transformers"
ADAPTER_MODULE = "vare_transformers_task_adapter"
ENTRYPOINT = "run"
_DIFF_PATH = re.compile(r"(?:a|b)/([A-Za-z0-9_./-]+)\Z")

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "benchmarks/mlsys_repo_eval_v2/transformers_45852"
PROTOCOL_PATH = TASK_DIR / "protocol-v4.json"
PROTOCOL_LOCK_PATH = TASK_DIR / "protocol-v4.lock.json"
TASK_DESCRIPTION = TASK_DIR / "task.md"
GRADER_ADAPTER = TASK_DIR / "grader_adapter.py"
FROZEN_TIMEOUT_S = 30.0
FROZEN_MEMORY_MB = 1536

# This inventory is checked for every grade, not just calibration. Keep it in
# sync with the prospective protocol lock; v1 remains an immutable historical
# record of the original grader implementation.
IMPLEMENTATION_FILES = (
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/Dockerfile",
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/grader_adapter.py",
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/requirements.txt",
    "benchmarks/mlsys_repo_eval_v2/transformers_45852/task.md",
    "scripts/run_transformers_45852_calibration.py",
    "src/vare/attestation.py",
    "src/vare/container_isolation.py",
    "src/vare/isolated_candidate.py",
    "src/vare/transformers_repo_task.py",
    "src/vare/types.py",
    "tests/test_transformers_repo_task.py",
)


class TransformersTaskError(ValueError):
    """The pinned source, task workspace, or candidate patch is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_inventory(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise TransformersTaskError("candidate workspace may not contain symlinks")
        if path.is_file():
            entries.append(
                {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            )
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_protocol() -> dict[str, Any]:
    """Verify protocol bytes and every locked evaluator implementation file."""
    try:
        protocol_bytes = PROTOCOL_PATH.read_bytes()
        lock = json.loads(PROTOCOL_LOCK_PATH.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise TransformersTaskError("prospective protocol or lock is unavailable") from exc
    protocol_digest = hashlib.sha256(protocol_bytes).hexdigest()
    if (
        lock.get("protocol_id") != "transformers-45852-isolated-eval-v4"
        or lock.get("protocol_sha256") != protocol_digest
        or protocol.get("protocol_id") != lock.get("protocol_id")
    ):
        raise TransformersTaskError("protocol hash or identity does not match lock")
    locked = lock.get("implementation_sha256")
    declared = protocol.get("implementation_sha256")
    if not isinstance(locked, dict) or locked != declared:
        raise TransformersTaskError("protocol implementation manifest differs from lock")
    expected_names = set(IMPLEMENTATION_FILES)
    if set(locked) != expected_names:
        raise TransformersTaskError("protocol implementation manifest inventory differs")
    try:
        actual = {name: sha256_file(ROOT / name) for name in IMPLEMENTATION_FILES}
    except OSError as exc:
        raise TransformersTaskError("locked implementation file is unavailable") from exc
    if actual != locked:
        raise TransformersTaskError("protocol implementation hash mismatch")
    if (
        protocol.get("baseline_commit") != BASELINE_COMMIT
        or protocol.get("fixed_commit") != FIXED_COMMIT
        or protocol.get("task_id") != TASK_ID
    ):
        raise TransformersTaskError("protocol task/source identity differs from grader")
    return protocol


def verify_baseline(source_root: Path) -> None:
    """Verify the pinned checkout, target bytes, and package index flags."""
    if source_root.is_symlink():
        raise TransformersTaskError("source root may not be a symlink")
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise TransformersTaskError("source root must be a directory")
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        flags = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-v", "--", PACKAGE_PATH],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise TransformersTaskError(
            "source root is not a readable Git checkout"
        ) from exc
    if head != BASELINE_COMMIT or status:
        raise TransformersTaskError("source checkout is not the clean pinned baseline")
    if any(line[:1] in {"h", "S"} for line in flags):
        raise TransformersTaskError(
            "source package has assume-unchanged or skip-worktree flags"
        )
    target = root / TARGET_PATH
    package = root / PACKAGE_PATH
    if (
        target.is_symlink()
        or not target.is_file()
        or sha256_file(target) != BASELINE_TARGET_SHA256
    ):
        raise TransformersTaskError(
            "pinned target source SHA-256 does not match baseline"
        )
    if package.is_symlink() or not package.is_dir():
        raise TransformersTaskError("pinned source package is missing or a symlink")
    if any(path.is_symlink() for path in package.rglob("*")):
        raise TransformersTaskError("pinned source package contains a symlink")


def _validate_diff(patch_text: str) -> None:
    if not patch_text or len(patch_text.encode("utf-8")) > 256_000:
        raise TransformersTaskError("patch is empty or exceeds 256 KiB")
    if "GIT binary patch" in patch_text or "\x00" in patch_text:
        raise TransformersTaskError("binary patches are unsupported")
    paths: list[str] = []
    git_paths: list[tuple[str, str]] = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) != 4:
                raise TransformersTaskError("malformed git diff header")
            left = _DIFF_PATH.fullmatch(parts[2])
            right = _DIFF_PATH.fullmatch(parts[3])
            if left is None or right is None:
                raise TransformersTaskError(
                    "patch contains an absolute or unsupported path"
                )
            git_paths.append((left.group(1), right.group(1)))
        elif line.startswith(("--- ", "+++ ")):
            raw = line[4:].split("\t", 1)[0]
            match = _DIFF_PATH.fullmatch(raw)
            if match is None:
                raise TransformersTaskError(
                    "patch contains an absolute or unsupported path"
                )
            path = PurePosixPath(match.group(1))
            if path.is_absolute() or ".." in path.parts or "." in path.parts:
                raise TransformersTaskError("patch path traversal is forbidden")
            paths.append(path.as_posix())
        elif line.startswith(
            ("rename from ", "rename to ", "new file mode ", "deleted file mode ")
        ):
            raise TransformersTaskError(
                "renames and file creation/deletion are unsupported"
            )
    if paths != [TARGET_PATH, TARGET_PATH] or git_paths != [(TARGET_PATH, TARGET_PATH)]:
        raise TransformersTaskError(
            "patch must edit only the allowlisted target source"
        )


def _copy_candidate_package(source_root: Path, destination: Path) -> None:
    source = source_root / PACKAGE_PATH
    if source.is_symlink() or not source.is_dir():
        raise TransformersTaskError("pinned Transformers package is unavailable")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise TransformersTaskError("pinned source package contains a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def export_baseline(source_root: Path, workspace: Path) -> dict[str, Any]:
    verify_baseline(source_root)
    if workspace.exists() or workspace.is_symlink():
        raise TransformersTaskError("candidate workspace already exists")
    workspace.mkdir(parents=True)
    _copy_candidate_package(source_root.resolve(), workspace / PACKAGE_PATH)
    if TASK_DESCRIPTION.is_symlink() or not TASK_DESCRIPTION.is_file():
        raise TransformersTaskError("candidate task statement is unavailable")
    shutil.copyfile(TASK_DESCRIPTION, workspace / "TASK.md")
    baseline_files = {
        path.relative_to(workspace).as_posix(): sha256_file(path)
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    }
    return {
        "task_id": TASK_ID,
        "baseline_commit": BASELINE_COMMIT,
        "target_source_sha256": sha256_file(workspace / TARGET_PATH),
        "workspace_inventory_sha256": sha256_inventory(workspace),
        "visible_file_count": sum(1 for p in workspace.rglob("*") if p.is_file()),
        "git_history_exported": False,
        "tests_exported": False,
        "candidate_edit_allowlist": [TARGET_PATH],
        "baseline_files": baseline_files,
    }


def apply_patch(source_root: Path, patch_path: Path, workspace: Path) -> dict[str, Any]:
    verify_baseline(source_root)
    if patch_path.is_symlink() or not patch_path.is_file():
        raise TransformersTaskError("patch must be a regular file")
    _validate_diff(patch_path.read_text(encoding="utf-8"))
    exported = export_baseline(source_root, workspace)
    try:
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, timeout=10)
        subprocess.run(
            ["git", "apply", "--check", str(patch_path.resolve())],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        subprocess.run(
            ["git", "apply", str(patch_path.resolve())],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        raise TransformersTaskError("candidate patch did not apply cleanly") from exc
    shutil.rmtree(workspace / ".git", ignore_errors=True)
    return {
        **exported,
        "patch_sha256": sha256_file(patch_path),
        "baseline_files": exported["baseline_files"],
        "target_source_sha256": sha256_file(workspace / TARGET_PATH),
        "workspace_inventory_sha256": sha256_inventory(workspace),
        "changed_files": [TARGET_PATH],
    }


def _cases() -> list[FunctionCall]:
    return [
        FunctionCall(
            "ordinary_variable_length_padded_batch",
            ADAPTER_MODULE,
            ENTRYPOINT,
            [{"case_type": "ordinary_2d", "sequences": [[1, 2, 1], [4, 2]]}],
            {
                "input_ids": [[1, 2, 1], [4, 2, 3]],
                "attention_mask": [[1, 1, 1], [1, 1, 0]],
                "shape": [2, 3],
                "pad_token_id": 3,
            },
        ),
        FunctionCall(
            "torch_square_4d_mask",
            ADAPTER_MODULE,
            ENTRYPOINT,
            [{"input_kind": "torch", "q_len": 6, "kv_len": 6, "values": [0, 1, 2]}],
            {
                "shape": [3, 1, 6, 6],
                "dtype": "torch.float32",
                "values": [
                    [[[float(value) for _ in range(6)] for _ in range(6)]]
                    for value in (0, 1, 2)
                ],
            },
        ),
        FunctionCall(
            "numpy_square_4d_mask",
            ADAPTER_MODULE,
            ENTRYPOINT,
            [{"input_kind": "numpy", "q_len": 6, "kv_len": 6, "values": [0, 1, 2]}],
            {
                "shape": [3, 1, 6, 6],
                "dtype": "torch.float32",
                "values": [
                    [[[float(value) for _ in range(6)] for _ in range(6)]]
                    for value in (0, 1, 2)
                ],
            },
        ),
        FunctionCall(
            "torch_asymmetric_q_kv_mask",
            ADAPTER_MODULE,
            ENTRYPOINT,
            [{"input_kind": "torch", "q_len": 4, "kv_len": 9, "values": [0, 0]}],
            {
                "shape": [2, 1, 4, 9],
                "dtype": "torch.float32",
                "values": [[[[0.0] * 9 for _ in range(4)]] for _ in range(2)],
            },
        ),
        FunctionCall(
            "torch_nonuniform_4d_mask",
            ADAPTER_MODULE,
            ENTRYPOINT,
            [{"input_kind": "torch", "q_len": 3, "kv_len": 4,
              "values": [[[(i * 7 + j * 3) % 11 for j in range(4)] for i in range(3)],
                         [[(i * 5 + j * 2 + 1) % 13 for j in range(4)] for i in range(3)]]}],
            {
                "shape": [2, 1, 3, 4],
                "dtype": "torch.float32",
                "values": [[[
                    [float((i * 7 + j * 3) % 11) for j in range(4)]
                    for i in range(3)
                ]], [[
                    [float((i * 5 + j * 2 + 1) % 13) for j in range(4)]
                    for i in range(3)
                ]]],
            },
        ),
        FunctionCall(
            "numpy_nonuniform_4d_mask",
            ADAPTER_MODULE,
            ENTRYPOINT,
            [{"input_kind": "numpy", "q_len": 3, "kv_len": 4,
              "values": [[[(i * 3 + j * 5 + 2) % 17 for j in range(4)] for i in range(3)],
                         [[(i * 11 + j * 4 + 3) % 19 for j in range(4)] for i in range(3)]]}],
            {
                "shape": [2, 1, 3, 4],
                "dtype": "torch.float32",
                "values": [[[
                    [float((i * 3 + j * 5 + 2) % 17) for j in range(4)]
                    for i in range(3)
                ]], [[
                    [float((i * 11 + j * 4 + 3) % 19) for j in range(4)]
                    for i in range(3)
                ]]],
            },
        ),
    ]


def grade_workspace(
    workspace: Path,
    *,
    image: str,
    baseline_files: dict[str, str] | None = None,
    timeout_s: float = FROZEN_TIMEOUT_S,
    memory_mb: int = FROZEN_MEMORY_MB,
) -> dict[str, Any]:
    """Grade candidate behavior in the existing isolated-callee boundary."""
    protocol = verify_protocol()
    if timeout_s != FROZEN_TIMEOUT_S or memory_mb != FROZEN_MEMORY_MB:
        raise TransformersTaskError("grading limits differ from frozen protocol v2")
    if image != protocol["image"]:
        raise TransformersTaskError("candidate image differs from frozen task protocol")
    if workspace.is_symlink() or not workspace.is_dir():
        raise TransformersTaskError("candidate workspace is missing or unsafe")
    sha256_inventory(workspace)
    target = workspace / TARGET_PATH
    if target.is_symlink() or not target.is_file():
        raise TransformersTaskError("candidate target source is missing or unsafe")
    expected_files = {
        p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file()
    }
    if baseline_files is None:
        raise TransformersTaskError("trusted baseline file manifest is required")
    allowed = set(baseline_files) | {TARGET_PATH}
    if expected_files != allowed:
        raise TransformersTaskError(
            "candidate workspace file set differs from pinned baseline allowlist"
        )
    for name, expected_sha in baseline_files.items():
        if name == TARGET_PATH:
            continue
        path = workspace / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
            raise TransformersTaskError(
                f"candidate changed a non-allowlisted file: {name}"
            )
    if sha256_file(workspace / "TASK.md") != baseline_files.get("TASK.md"):
        raise TransformersTaskError("candidate changed the task statement")
    if "TASK.md" not in expected_files or not any(
        p.startswith(PACKAGE_PATH + "/") for p in expected_files
    ):
        raise TransformersTaskError("candidate workspace inventory is incomplete")
    if any(p.startswith(".git/") or p.startswith("tests/") for p in expected_files):
        raise TransformersTaskError(
            "Git history and tests are forbidden in candidate workspace"
        )
    with tempfile.TemporaryDirectory(prefix="vare-transformers-45852-grade-") as folder:
        isolated_subject = Path(folder)
        shutil.copytree(workspace / PACKAGE_PATH, isolated_subject / "transformers")
        adapter = isolated_subject / f"{ADAPTER_MODULE}.py"
        if adapter.exists() or adapter.is_symlink() or GRADER_ADAPTER.is_symlink():
            raise TransformersTaskError("trusted adapter path is occupied or unsafe")
        shutil.copyfile(GRADER_ADAPTER, adapter)
        for path in isolated_subject.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
            elif path.is_file():
                path.chmod(0o444)
        report = grade_calls(
            isolated_subject,
            _cases(),
            task_kind="correctness",
            image=image,
            timeout_s=timeout_s,
            memory_mb=memory_mb,
            discard_candidate_stdout=True,
        )
    return {
        "task_id": TASK_ID,
        "passed": report.passed,
        "cases": [case.to_dict() for case in report.cases],
        "candidate_target_sha256": sha256_file(target),
        "candidate_workspace_sha256": sha256_inventory(workspace),
        "grader_adapter_sha256": sha256_file(GRADER_ADAPTER),
        "grader_source_sha256": sha256_file(Path(__file__).resolve()),
        "task_statement_sha256": sha256_file(TASK_DESCRIPTION),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "candidate_image": image,
        "network": "none",
        "grader_files_mounted_to_candidate": False,
        "git_history_exported": False,
        "peak_memory_bytes": None,
        "configured_memory_limit_mb": memory_mb,
        "timeout_s": timeout_s,
        "candidate_wall_s": [case.wall_s for case in report.cases],
    }
