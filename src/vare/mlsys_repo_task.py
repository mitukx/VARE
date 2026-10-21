"""Narrow repository-patch calibration for one frozen MLX-LM train task.

The trusted grader evaluates only the pinned ``process_message_content``
function contract. Candidate code runs in the existing no-network Docker
callee; hidden expected values stay in this trusted parent module.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

from .container_isolation import DockerIsolationRunner
from .isolated_candidate import FunctionCall, grade_calls

TASK_ID = "mlxlm-1904-tool-arguments"
BASELINE_COMMIT = "2af5f7ce8620115bf37fbbddf9817f083118df59"
BASELINE_SERVER_SHA256 = (
    "57392576de114af80ede723976b9dd156dc234f856b9943b8b44b8b78bc27f79"
)
FIXED_COMMIT = "caed1943d38ecb58f6e466a72bed98310c9d45a5"
TARGET_PATH = "mlx_lm/server.py"
ENTRYPOINT_MODULE = "vare_task_adapter"
ENTRYPOINT_FUNCTION = "run"
TASK_DESCRIPTION = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/mlsys_repo_eval_v1/mlxlm_1904_tool_arguments/task.md"
)
GRADER_ADAPTER = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/mlsys_repo_eval_v1/mlxlm_1904_tool_arguments/grader_adapter.py"
)
_DIFF_PATH = re.compile(r"(?:a|b)/([A-Za-z0-9_./-]+)\Z")


class RepositoryTaskError(ValueError):
    """Task identity, checkout, or candidate patch failed closed."""


RED_TEAM_PROBES = frozenset(
    {
        "grader_path_probe",
        "git_history_probe",
        "environment_probe",
        "network_probe",
        "subprocess_write_probe",
        "grader_stdout_forgery",
        "timeout",
        "memory_exhaustion",
    }
)
LOCKED_IMPLEMENTATION_FILES = frozenset(
    {
        "src/vare/mlsys_repo_task.py",
        "benchmarks/mlsys_repo_eval_v1/mlxlm_1904_tool_arguments/grader_adapter.py",
        "benchmarks/mlsys_repo_eval_v1/mlxlm_1904_tool_arguments/known_fix.patch",
        "scripts/run_mlxlm_1904_calibration.py",
        "scripts/red_team_mlxlm_1904.py",
        "tests/test_mlsys_repo_task.py",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_red_team_evidence(
    path: Path,
    *,
    image: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a complete, fresh-format red-team record and bind its byte hash."""
    if path.is_symlink() or not path.is_file():
        raise RepositoryTaskError("red-team evidence is missing or not a regular file")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RepositoryTaskError(
            "red-team evidence SHA-256 does not match its recorded hash"
        )
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepositoryTaskError("red-team evidence is malformed") from exc
    if not isinstance(report, dict):
        raise RepositoryTaskError("red-team evidence root must be an object")
    attempts = report.get("attempts") if isinstance(report, dict) else None
    decision = report.get("decision") if isinstance(report, dict) else None
    if (
        report.get("experiment_id") != "mlsys-repository-task-mlxlm-1904-red-team-v1"
        or report.get("source_checkout") != BASELINE_COMMIT
        or report.get("image") != image
        or report.get("network") != "none"
        or not isinstance(attempts, dict)
        or set(attempts) != RED_TEAM_PROBES
        or not isinstance(decision, dict)
        or decision.get("all_attacks_checked") is not True
    ):
        raise RepositoryTaskError("red-team evidence does not match the required suite")
    for name, attempt in attempts.items():
        if not isinstance(attempt, dict):
            raise RepositoryTaskError(f"red-team result is malformed: {name}")
        expected_pass = name in {
            "grader_path_probe",
            "git_history_probe",
            "environment_probe",
            "network_probe",
            "subprocess_write_probe",
        }
        if (
            attempt.get("audit_check_passed") is not True
            or attempt.get("expected_grader_passed") is not expected_pass
            or attempt.get("grader_passed") is not expected_pass
        ):
            raise RepositoryTaskError(
                f"red-team probe failed or is inconsistent: {name}"
            )
    if attempts["subprocess_write_probe"].get("host_workspace_write") is not False:
        raise RepositoryTaskError("red-team subprocess escaped into the host workspace")
    return {
        "sha256": digest,
        "probe_count": len(attempts),
        "source_checkout": report["source_checkout"],
        "image": report["image"],
    }


def verify_protocol_implementation(
    protocol: dict[str, Any],
    repo_root: Path,
) -> dict[str, str]:
    """Require every protocol-locked evaluator artifact to match its exact bytes."""
    locked = protocol.get("implementation_sha256")
    if not isinstance(locked, dict) or set(locked) != LOCKED_IMPLEMENTATION_FILES:
        raise RepositoryTaskError(
            "protocol implementation lock is incomplete or unexpected"
        )
    root = repo_root.resolve(strict=True)
    verified: dict[str, str] = {}
    for relative, expected in locked.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not isinstance(expected, str):
            raise RepositoryTaskError(
                "protocol implementation lock contains an unsafe path"
            )
        source = root / path
        if source.is_symlink() or not source.is_file():
            raise RepositoryTaskError(
                f"protocol-locked implementation file is unavailable: {relative}"
            )
        actual = sha256_file(source)
        if actual != expected:
            raise RepositoryTaskError(
                f"protocol-locked implementation hash mismatch: {relative}"
            )
        verified[relative] = actual
    return verified


def workspace_manifest_sha256(root: Path) -> str:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_baseline(source_root: Path) -> None:
    """Require an unchanged checkout at the frozen baseline revision."""
    root = source_root.resolve(strict=True)
    if not root.is_dir() or source_root.is_symlink():
        raise RepositoryTaskError("pinned source root must be a real directory")
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
        index_flags = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-v", "--", TARGET_PATH],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RepositoryTaskError("source root is not a readable Git checkout") from exc
    if head != BASELINE_COMMIT or status:
        raise RepositoryTaskError("source checkout is not the clean pinned baseline")
    source = root / TARGET_PATH
    if source.is_symlink() or not source.is_file():
        raise RepositoryTaskError("pinned target source is unavailable or a symlink")
    if any(line[:1] in {"h", "S"} for line in index_flags):
        raise RepositoryTaskError(
            "target has assume-unchanged or skip-worktree index flags"
        )
    if sha256_file(source) != BASELINE_SERVER_SHA256:
        raise RepositoryTaskError(
            "pinned target source SHA-256 does not match the baseline"
        )


def _diff_paths(patch_text: str) -> list[str]:
    if not patch_text or len(patch_text.encode("utf-8")) > 256_000:
        raise RepositoryTaskError("patch is empty or exceeds 256 KiB")
    if "GIT binary patch" in patch_text or "\x00" in patch_text:
        raise RepositoryTaskError("binary patches are unsupported")
    paths: list[str] = []
    git_paths: list[tuple[str, str]] = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) != 4:
                raise RepositoryTaskError("malformed git diff header")
            left, right = parts[2], parts[3]
            left_match, right_match = (
                _DIFF_PATH.fullmatch(left),
                _DIFF_PATH.fullmatch(right),
            )
            if left_match is None or right_match is None:
                raise RepositoryTaskError(
                    "patch contains an absolute or unsupported path"
                )
            git_paths.append((left_match.group(1), right_match.group(1)))
        elif line.startswith(("--- ", "+++ ")):
            raw = line[4:].split("\t", 1)[0]
            match = _DIFF_PATH.fullmatch(raw)
            if match is None:
                raise RepositoryTaskError(
                    "patch contains an absolute or unsupported path"
                )
            path = PurePosixPath(match.group(1))
            if path.is_absolute() or ".." in path.parts or "." in path.parts:
                raise RepositoryTaskError("patch path traversal is forbidden")
            paths.append(path.as_posix())
        elif line.startswith(
            ("rename from ", "rename to ", "new file mode ", "deleted file mode ")
        ):
            raise RepositoryTaskError(
                "renames and file creation/deletion are unsupported"
            )
    if (
        len(paths) != 2
        or paths[0] != TARGET_PATH
        or paths[1] != TARGET_PATH
        or git_paths != [(TARGET_PATH, TARGET_PATH)]
    ):
        raise RepositoryTaskError("patch must edit only the allowlisted source file")
    return paths


def _copy_candidate_source(source_root: Path, destination: Path) -> None:
    """Export Python package context without Git history, tests, or symlinks."""
    package = source_root / "mlx_lm"
    if package.is_symlink() or not package.is_dir():
        raise RepositoryTaskError("pinned package directory is unavailable")
    destination.mkdir(parents=True)
    for current, dirs, files in os.walk(package, followlinks=False):
        current_path = Path(current)
        dirs[:] = sorted(
            name for name in dirs if not (current_path / name).is_symlink()
        )
        relative = current_path.relative_to(source_root)
        out_dir = destination / relative
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in sorted(files):
            source = current_path / name
            if source.is_symlink():
                continue
            if source.suffix != ".py":
                continue
            target = out_dir / name
            target.write_bytes(source.read_bytes())
    target_file = destination / TARGET_PATH
    if not target_file.is_file() or target_file.is_symlink():
        raise RepositoryTaskError("candidate export omitted the allowlisted target")
    if TASK_DESCRIPTION.is_symlink() or not TASK_DESCRIPTION.is_file():
        raise RepositoryTaskError("frozen task description is unavailable")
    shutil.copyfile(TASK_DESCRIPTION, destination / "TASK.md")


def export_baseline(source_root: Path, workspace: Path) -> dict[str, Any]:
    """Export the pinned task context without tests or Git objects."""
    verify_baseline(source_root)
    if workspace.exists() or workspace.is_symlink():
        raise RepositoryTaskError("candidate workspace already exists")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    _copy_candidate_source(source_root.resolve(), workspace)
    return {
        "task_id": TASK_ID,
        "baseline_commit": BASELINE_COMMIT,
        "candidate_patch_sha256": None,
        "candidate_source_sha256": sha256_file(workspace / TARGET_PATH),
        "candidate_inventory_sha256": workspace_manifest_sha256(workspace),
        "changed_files": [],
        "exported_python_file_count": sum(1 for _ in workspace.rglob("*.py")),
        "git_history_exported": False,
        "tests_exported": False,
        "task_description_sha256": sha256_file(workspace / "TASK.md"),
    }


def apply_candidate_patch(
    source_root: Path, patch_path: Path, workspace: Path
) -> dict[str, Any]:
    """Create a sanitized baseline export and apply one checked unified diff."""
    verify_baseline(source_root)
    if patch_path.is_symlink() or not patch_path.is_file():
        raise RepositoryTaskError("patch must be a regular file")
    patch_text = patch_path.read_text(encoding="utf-8")
    _diff_paths(patch_text)
    if workspace.exists() or workspace.is_symlink():
        raise RepositoryTaskError("candidate workspace already exists")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    _copy_candidate_source(source_root.resolve(), workspace)
    try:
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True, timeout=10)
        subprocess.run(
            ["git", "apply", "--unidiff-zero", "--check", str(patch_path.resolve())],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        raise RepositoryTaskError("candidate patch does not apply cleanly") from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        raise RepositoryTaskError("candidate patch validation failed") from exc
    target = workspace / TARGET_PATH
    if target.is_symlink() or not target.is_file():
        shutil.rmtree(workspace, ignore_errors=True)
        raise RepositoryTaskError("candidate target became a symlink or disappeared")
    try:
        subprocess.run(
            ["git", "apply", "--unidiff-zero", str(patch_path.resolve())],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(workspace, ignore_errors=True)
        raise RepositoryTaskError("candidate patch application failed") from exc
    shutil.rmtree(workspace / ".git", ignore_errors=True)
    actual = sorted(
        p.relative_to(workspace).as_posix()
        for p in workspace.rglob("*")
        if p.is_file() and not p.is_symlink()
    )
    allowed = sorted(
        p.relative_to(workspace).as_posix()
        for p in workspace.rglob("*.py")
        if p.is_file() and not p.is_symlink()
    )
    if TARGET_PATH not in allowed or any(p.startswith(".git/") for p in actual):
        shutil.rmtree(workspace, ignore_errors=True)
        raise RepositoryTaskError("candidate export inventory is invalid")
    return {
        "task_id": TASK_ID,
        "baseline_commit": BASELINE_COMMIT,
        "candidate_patch_sha256": sha256_file(patch_path),
        "candidate_source_sha256": sha256_file(target),
        "candidate_inventory_sha256": workspace_manifest_sha256(workspace),
        "changed_files": [TARGET_PATH],
        "exported_python_file_count": len(allowed),
        "git_history_exported": False,
        "tests_exported": False,
    }


def _calls() -> Sequence[FunctionCall]:
    """Parent-owned cases preserve the selected upstream regression contract."""
    return (
        FunctionCall(
            "dict_arguments",
            ENTRYPOINT_MODULE,
            ENTRYPOINT_FUNCTION,
            [
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"function": {"arguments": {"a": 2, "b": 3}}}],
                    }
                ]
            ],
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"arguments": {"a": 2, "b": 3}}}],
                }
            ],
        ),
        FunctionCall(
            "json_string_control",
            ENTRYPOINT_MODULE,
            ENTRYPOINT_FUNCTION,
            [
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"function": {"arguments": '{"a": 2, "b": 3}'}}],
                    }
                ]
            ],
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"arguments": {"a": 2, "b": 3}}}],
                }
            ],
        ),
    )


def grade_workspace(
    workspace: Path,
    *,
    image: str,
    runner: DockerIsolationRunner | None = None,
    timeout_s: float = 5.0,
    memory_mb: int = 512,
) -> dict[str, Any]:
    """Run the independent tests with only the candidate source mounted."""
    target = workspace / TARGET_PATH
    if (
        workspace.is_symlink()
        or not workspace.is_dir()
        or target.is_symlink()
        or not target.is_file()
    ):
        raise RepositoryTaskError("candidate workspace is incomplete or unsafe")
    with tempfile.TemporaryDirectory(prefix="vare-mlxlm-1904-grade-") as folder:
        isolated_subject = Path(folder)
        package = isolated_subject / "mlx_lm"
        package.mkdir()
        shutil.copyfile(target, package / "server.py")
        add_trusted_adapter(isolated_subject)
        os.chmod(isolated_subject, 0o755)
        os.chmod(package, 0o755)
        os.chmod(package / "server.py", 0o444)
        os.chmod(isolated_subject / "vare_task_adapter.py", 0o444)
        report = grade_calls(
            isolated_subject,
            _calls(),
            task_kind="correctness",
            image=image,
            timeout_s=timeout_s,
            memory_mb=memory_mb,
            runner=runner,
            discard_candidate_stdout=True,
        )
    return {
        "task_id": TASK_ID,
        "passed": report.passed,
        "cases": [case.to_dict() for case in report.cases],
        "grader_source_sha256": sha256_file(Path(__file__).resolve()),
        "grader_adapter_sha256": sha256_file(GRADER_ADAPTER),
        "candidate_source_sha256": sha256_file(target),
    }


def add_trusted_adapter(workspace: Path) -> None:
    """Install a fixed, non-editable bridge for the one pure-function task."""
    adapter = workspace / "vare_task_adapter.py"
    if adapter.exists() or adapter.is_symlink() or GRADER_ADAPTER.is_symlink():
        raise RepositoryTaskError(
            "candidate export already contains the grader adapter"
        )
    if not GRADER_ADAPTER.is_file():
        raise RepositoryTaskError("trusted adapter source is unavailable")
    shutil.copyfile(GRADER_ADAPTER, adapter)


def prepare_workspace(
    source_root: Path, patch_path: Path, workspace: Path
) -> dict[str, Any]:
    return apply_candidate_patch(source_root, patch_path, workspace)
