"""Narrow isolated coding task for frozen MLX-LM #1912 regression behavior."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any

from .isolated_candidate import FunctionCall, grade_calls

TASK_ID = "mlxlm-1912-top-p-min-token"
BASELINE_COMMIT = "47e43a4526d42a6dcee758f07eb04b78a8439ecb"
BASELINE_SOURCE_SHA256 = (
    "c93c1eef794725f9f7ce77b6212f61eb6d0fe17b9cd87c06cec6470ee12b07f2"
)
FIXED_COMMIT = "87b7b583a697537aa68f47130b40884700b5f55f"
FIXED_SOURCE_SHA256 = "3aac92d804b6b91c82a4e2844697a8723de26f6af57e10013a482b967e739cfa"
TARGET_PATH = "mlx_lm/sample_utils.py"
_PATCH_PATH = re.compile(r"(?:a|b)/([A-Za-z0-9_./-]+)\Z")
TASK_DESCRIPTION = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/task.md"
)
GRADER_ADAPTER = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/mlsys_repo_eval_v2/mlxlm_1912_top_p/grader_adapter.py"
)


class MlxLm1912TaskError(ValueError):
    """Pinned source or candidate patch did not meet the task contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_baseline(source_root: Path) -> None:
    root = source_root.resolve(strict=True)
    if source_root.is_symlink() or not root.is_dir():
        raise MlxLm1912TaskError("pinned source root must be a real directory")
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
            ["git", "-C", str(root), "ls-files", "-v", "--", TARGET_PATH],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise MlxLm1912TaskError("source root is not a readable Git checkout") from exc
    target = root / TARGET_PATH
    if head != BASELINE_COMMIT or status:
        raise MlxLm1912TaskError("source checkout is not the clean pinned baseline")
    if (
        target.is_symlink()
        or not target.is_file()
        or sha256_file(target) != BASELINE_SOURCE_SHA256
    ):
        raise MlxLm1912TaskError("pinned baseline source SHA-256 mismatch")
    if any(line[:1] in {"h", "S"} for line in flags):
        raise MlxLm1912TaskError("target has assume-unchanged or skip-worktree flag")


def _validate_patch(text: str) -> None:
    if not text or len(text.encode("utf-8")) > 256_000 or "GIT binary patch" in text:
        raise MlxLm1912TaskError("patch is empty, binary, or too large")
    headers: list[tuple[str, str]] = []
    paths: list[str] = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            fields = line.split()
            if len(fields) != 4:
                raise MlxLm1912TaskError("malformed diff header")
            left, right = fields[2], fields[3]
            left_match, right_match = (
                _PATCH_PATH.fullmatch(left),
                _PATCH_PATH.fullmatch(right),
            )
            if left_match is None or right_match is None:
                raise MlxLm1912TaskError("unsupported patch path")
            headers.append((left_match.group(1), right_match.group(1)))
        elif line.startswith(("--- ", "+++ ")):
            match = _PATCH_PATH.fullmatch(line[4:].split("\t", 1)[0])
            if match is None:
                raise MlxLm1912TaskError("unsupported patch path")
            path = PurePosixPath(match.group(1))
            if path.is_absolute() or ".." in path.parts or "." in path.parts:
                raise MlxLm1912TaskError("patch traversal is forbidden")
            paths.append(path.as_posix())
        elif line.startswith(
            ("rename from ", "rename to ", "new file mode ", "deleted file mode ")
        ):
            raise MlxLm1912TaskError(
                "file creation, deletion, and rename are forbidden"
            )
    if headers != [(TARGET_PATH, TARGET_PATH)] or paths != [TARGET_PATH, TARGET_PATH]:
        raise MlxLm1912TaskError("patch must edit only mlx_lm/sample_utils.py")


def _copy_package(source_root: Path, destination: Path) -> None:
    package = source_root / "mlx_lm"
    if package.is_symlink() or not package.is_dir():
        raise MlxLm1912TaskError("pinned mlx_lm package is missing")
    for current, dirs, files in os.walk(package, followlinks=False):
        here = Path(current)
        dirs[:] = sorted(name for name in dirs if not (here / name).is_symlink())
        relative = here.relative_to(source_root)
        out = destination / relative
        out.mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            src = here / name
            if src.is_symlink() or src.suffix != ".py":
                continue
            shutil.copyfile(src, out / name)


def export_candidate(
    source_root: Path, workspace: Path, patch_path: Path | None = None
) -> dict[str, Any]:
    verify_baseline(source_root)
    if workspace.exists() or workspace.is_symlink():
        raise MlxLm1912TaskError("candidate workspace already exists")
    if patch_path is not None:
        if patch_path.is_symlink() or not patch_path.is_file():
            raise MlxLm1912TaskError("patch must be a regular file")
        _validate_patch(patch_path.read_text(encoding="utf-8"))
    workspace.mkdir(parents=True)
    _copy_package(source_root.resolve(), workspace)
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
            raise MlxLm1912TaskError("candidate patch did not apply cleanly") from exc
        shutil.rmtree(workspace / ".git", ignore_errors=True)
    target = workspace / TARGET_PATH
    expected = (
        FIXED_SOURCE_SHA256
        if patch_path is not None
        and target.exists()
        and sha256_file(target) == FIXED_SOURCE_SHA256
        else None
    )
    return {
        "task_id": TASK_ID,
        "baseline_commit": BASELINE_COMMIT,
        "candidate_patch_sha256": sha256_file(patch_path) if patch_path else None,
        "candidate_source_sha256": sha256_file(target),
        "fixed_source_match": expected == FIXED_SOURCE_SHA256,
        "changed_files": [TARGET_PATH] if patch_path else [],
        "git_history_exported": False,
        "tests_exported": False,
        "task_description_sha256": sha256_file(TASK_DESCRIPTION),
    }


def grade_workspace(
    workspace: Path, *, image: str, timeout_s: float = 30, memory_mb: int = 1536
) -> dict[str, Any]:
    target = workspace / TARGET_PATH
    if (
        workspace.is_symlink()
        or not workspace.is_dir()
        or target.is_symlink()
        or not target.is_file()
    ):
        raise MlxLm1912TaskError("candidate workspace is incomplete or unsafe")
    with __import__("tempfile").TemporaryDirectory(
        prefix="vare-mlxlm-1912-grade-"
    ) as folder:
        subject = Path(folder) / "mlx_lm"
        subject.mkdir()
        shutil.copyfile(target, subject / "sample_utils.py")
        adapter = subject.parent / "vare_task_adapter.py"
        shutil.copyfile(GRADER_ADAPTER, adapter)
        for path in (subject, subject / "sample_utils.py", adapter):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        calls = []
        for top_p in (1e-8, 1e-6, 1e-4):
            calls.append(
                FunctionCall(
                    f"tiny-threshold-{top_p:g}",
                    "vare_task_adapter",
                    "run",
                    [{"logits": [[0.0, None, None, -20.0]], "top_p": top_p}],
                    {"kept_indices": [[0]], "kept_values": [[0.0]]},
                )
            )
        large_logits = [[0.0, *([-20.0] * 4095)] for _ in range(2)]
        for dtype in ("float32", "float16", "bfloat16"):
            for top_p in (1e-8, 1e-4, 1e-3):
                calls.append(
                    FunctionCall(
                        f"{dtype}-threshold-{top_p:g}",
                        "vare_task_adapter",
                        "run",
                        [
                            {
                                "dtype": dtype,
                                "logits": large_logits,
                                "top_p": top_p,
                            }
                        ],
                        {
                            "kept_indices": [[0], [0]],
                            "kept_values": [[0.0], [0.0]],
                        },
                    )
                )
        report = grade_calls(
            subject.parent,
            calls,
            task_kind="correctness",
            image=image,
            timeout_s=timeout_s,
            memory_mb=memory_mb,
            discard_candidate_stdout=True,
            preserve_session=True,
        )
    return {
        "task_id": TASK_ID,
        "passed": report.passed,
        "cases": [case.to_dict() for case in report.cases],
        "grader_source_sha256": sha256_file(Path(__file__).resolve()),
        "grader_adapter_sha256": sha256_file(GRADER_ADAPTER),
        "candidate_source_sha256": sha256_file(target),
    }
