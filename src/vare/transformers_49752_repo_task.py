"""Isolated grader for the pinned Transformers #49752 video regression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any

from .isolated_candidate import FunctionCall, grade_calls

TASK_ID = "transformers-49752-stacked-frames"
BASELINE_COMMIT = "6f2e80bdad560947903bb4bde8043ae95387069a"
FIXED_COMMIT = "bc0e71ffef22474fc14fcaa4b399d824eebc0e50"
TARGET_PATH = "src/transformers/models/minicpmv4_6/video_processing_minicpmv4_6.py"
PACKAGE_PATH = "src/transformers"
BASELINE_TARGET_SHA256 = "c0479fc2f6a14e493f6be020f6685a7c3ebbc72891c8bdbfca66f3e24ed3d7ee"
FIXED_TARGET_SHA256 = "93297262c68d03a35d4b74cd6e30da71a83474c03fe07ab1cd3e3a0153e8277e"
ADAPTER_MODULE = "candidate_probe"
ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "benchmarks/mlsys_repo_eval_v3/transformers_49752"
TASK_DESCRIPTION = TASK_DIR / "task.md"
GRADER_ADAPTER = TASK_DIR / "candidate_probe.py"
PUBLIC_TEST = TASK_DIR / "test_public.py"
PROTOCOL = TASK_DIR / "protocol-v11.json"
_DIFF_HEADER = re.compile(r"(?:a|b)/([A-Za-z0-9_./-]+)\Z")


class Transformers49752Error(ValueError):
    """Pinned source or candidate workspace is invalid."""


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
            raise Transformers49752Error("workspace may not contain symlinks")
        if path.is_file():
            entries.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_baseline(source_root: Path) -> None:
    """Require the exact baseline commit, clean package, and pinned target bytes."""
    if source_root.is_symlink():
        raise Transformers49752Error("source root may not be a symlink")
    root = source_root.resolve(strict=True)
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        flags = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-v", "--", PACKAGE_PATH],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise Transformers49752Error("source root is not a readable Git checkout") from exc
    if head != BASELINE_COMMIT or status:
        raise Transformers49752Error("source checkout is not the clean pinned baseline")
    if any(line[:1] in {"h", "S"} for line in flags):
        raise Transformers49752Error("source package has hidden Git index flags")
    target = root / TARGET_PATH
    package = root / PACKAGE_PATH
    if target.is_symlink() or not target.is_file() or sha256_file(target) != BASELINE_TARGET_SHA256:
        raise Transformers49752Error("baseline target source SHA-256 mismatch")
    if package.is_symlink() or not package.is_dir() or any(p.is_symlink() for p in package.rglob("*")):
        raise Transformers49752Error("pinned source package is missing or unsafe")


def export_baseline(source_root: Path, workspace: Path) -> dict[str, Any]:
    verify_baseline(source_root)
    if workspace.exists() or workspace.is_symlink():
        raise Transformers49752Error("candidate workspace already exists")
    workspace.mkdir(parents=True)
    shutil.copytree(source_root.resolve() / PACKAGE_PATH, workspace / PACKAGE_PATH)
    for source, target in ((TASK_DESCRIPTION, "TASK.md"), (PUBLIC_TEST, "test_public.py"),
                           (GRADER_ADAPTER, "candidate_probe.py")):
        if source.is_symlink() or not source.is_file():
            raise Transformers49752Error(f"task asset is missing or unsafe: {source.name}")
        shutil.copyfile(source, workspace / target)
    files = {
        p.relative_to(workspace).as_posix(): sha256_file(p)
        for p in sorted(workspace.rglob("*")) if p.is_file()
    }
    return {
        "task_id": TASK_ID,
        "baseline_commit": BASELINE_COMMIT,
        "target_source_sha256": files[TARGET_PATH],
        "workspace_inventory_sha256": sha256_inventory(workspace),
        "candidate_edit_allowlist": [TARGET_PATH],
        "git_history_exported": False,
        "upstream_tests_exported": False,
        "baseline_files": files,
    }


def _validate_diff(text: str) -> None:
    if not text or len(text.encode("utf-8")) > 256_000 or "GIT binary patch" in text or "\x00" in text:
        raise Transformers49752Error("patch is empty, binary, or exceeds 256 KiB")
    paths: list[str] = []
    git_paths: list[tuple[str, str]] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) != 4:
                raise Transformers49752Error("malformed diff header")
            left, right = _DIFF_HEADER.fullmatch(parts[2]), _DIFF_HEADER.fullmatch(parts[3])
            if left is None or right is None:
                raise Transformers49752Error("unsupported diff path")
            git_paths.append((left.group(1), right.group(1)))
        elif line.startswith(("--- ", "+++ ")):
            match = _DIFF_HEADER.fullmatch(line[4:].split("\t", 1)[0])
            if match is None:
                raise Transformers49752Error("unsupported diff path")
            path = Path(match.group(1))
            if path.is_absolute() or ".." in path.parts or "." in path.parts:
                raise Transformers49752Error("path traversal is forbidden")
            paths.append(path.as_posix())
        elif line.startswith(("rename from ", "rename to ", "new file mode ", "deleted file mode ")):
            raise Transformers49752Error("renames and file creation/deletion are unsupported")
    if paths != [TARGET_PATH, TARGET_PATH] or git_paths != [(TARGET_PATH, TARGET_PATH)]:
        raise Transformers49752Error("patch must edit only the allowlisted source file")


def apply_patch(source_root: Path, patch_path: Path, workspace: Path) -> dict[str, Any]:
    verify_baseline(source_root)
    if patch_path.is_symlink() or not patch_path.is_file():
        raise Transformers49752Error("patch must be a regular file")
    _validate_diff(patch_path.read_text(encoding="utf-8"))
    exported = export_baseline(source_root, workspace)
    try:
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, timeout=10)
        subprocess.run(["git", "apply", "--check", str(patch_path.resolve())], cwd=workspace,
                       check=True, capture_output=True, text=True, timeout=10)
        subprocess.run(["git", "apply", str(patch_path.resolve())], cwd=workspace,
                       check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        raise Transformers49752Error("candidate patch did not apply cleanly") from exc
    shutil.rmtree(workspace / ".git", ignore_errors=True)
    return {
        **exported,
        "patch_sha256": sha256_file(patch_path),
        "target_source_sha256": sha256_file(workspace / TARGET_PATH),
        "workspace_inventory_sha256": sha256_inventory(workspace),
        "changed_files": [TARGET_PATH],
    }


def _cases() -> list[FunctionCall]:
    expected_sizes = [[32, 32], [48, 24], [32, 32], [48, 24],
                      [32, 32], [48, 24], [32, 32], [48, 24]]
    expected = {
        "num_frames_per_video": [8],
        "target_sizes_videos": expected_sizes,
        "num_patches_per_frame_count": 8,
        "main_grid": [32, 32],
    }
    return [
        FunctionCall(f"stack-frames-seed-{seed}", ADAPTER_MODULE, "run", [{"seed": seed}], expected)
        for seed in (20261021, 20261022)
    ]


def grade_workspace(workspace: Path, *, image: str, baseline_files: dict[str, str],
                    timeout_s: float = 30.0, memory_mb: int = 1536) -> dict[str, Any]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if image != protocol["runtime"]["candidate_container"]:
        raise Transformers49752Error("candidate image differs from frozen task protocol")
    if workspace.is_symlink() or not workspace.is_dir():
        raise Transformers49752Error("candidate workspace is missing or unsafe")
    sha256_inventory(workspace)
    target = workspace / TARGET_PATH
    if target.is_symlink() or not target.is_file():
        raise Transformers49752Error("candidate target is missing or unsafe")
    actual_files = {p.relative_to(workspace).as_posix() for p in workspace.rglob("*") if p.is_file()}
    if actual_files != set(baseline_files):
        raise Transformers49752Error("candidate workspace file manifest changed")
    for name, digest in baseline_files.items():
        if name != TARGET_PATH and (workspace / name).is_symlink() or name != TARGET_PATH and sha256_file(workspace / name) != digest:
            raise Transformers49752Error(f"candidate changed protected workspace file: {name}")
    with tempfile.TemporaryDirectory(prefix="vare-transformers-49752-grade-") as folder:
        subject = Path(folder)
        shutil.copytree(workspace / PACKAGE_PATH, subject / "transformers")
        shutil.copyfile(GRADER_ADAPTER, subject / f"{ADAPTER_MODULE}.py")
        for path in subject.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o444)
        report = grade_calls(
            subject, _cases(), task_kind="correctness", image=image,
            timeout_s=timeout_s, memory_mb=memory_mb,
            discard_candidate_stdout=True,
        )
    return {
        "task_id": TASK_ID,
        "passed": report.passed,
        "cases": [case.to_dict() for case in report.cases],
        "candidate_target_sha256": sha256_file(target),
        "candidate_workspace_sha256": sha256_inventory(workspace),
        "grader_adapter_sha256": sha256_file(GRADER_ADAPTER),
        "task_statement_sha256": sha256_file(TASK_DESCRIPTION),
        "protocol_sha256": sha256_file(PROTOCOL),
        "candidate_image": image,
        "candidate_workspace_contents_mounted_to_grader": False,
        "network": "none",
        "peak_memory_bytes": None,
        "configured_memory_limit_mb": memory_mb,
        "timeout_s": timeout_s,
    }
