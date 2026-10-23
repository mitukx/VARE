#!/usr/bin/env python3
"""Prospective v11 operator: source-bound MLX-LM #1971 export/grading.

This is a new operator API, not a rewrite of frozen v9. Its trusted parent
process owns grader outputs; no model inference or qualification is implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale"
PROTOCOL = TASK / "protocol-v11.json"
LOCK = TASK / "protocol-v11.lock.json"
PROTOCOL_SHA256 = "b9643c163161ff7332446f3b3a729e3551f2ecc6a1a329038465ae60982181b6"
LOCK_SHA256 = "bdca66c4765d13e889c20cb4f22f479b027cb35fded2adb7c7daaf64d8b0129a"
OPERATOR_COMMIT = "8257a4dfe2106055a048ddd4b1520a08c7e4d658"
V9_COMMIT = "3f12eefd71d1d2213b37e3a174b446cb5155e979"
V9_PROTOCOL_SHA256 = "aa8fec4c3a3fbba119b83009c7f3c164b762758a5c1593faf3a78f3464678fb2"
SOURCE_PATH = "scripts/run_attested_mlxlm_1971_v9_calibration.py"
SPARSE_PATHS = (
    "scripts",
    "src/vare",
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale",
    "tests",
)
PINNED_OPERATOR_SOURCES = (
    SOURCE_PATH,
    "scripts/attest_mlxlm_1971_v9_current_git.py",
    "scripts/audit_mlxlm_1971_historical_implementation_hashes.py",
)
V9_PROTOCOL_PATH = (
    "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v9.json"
)
GIT_ENV = {
    **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_COUNT": "4",
    "GIT_CONFIG_KEY_0": "core.hooksPath",
    "GIT_CONFIG_VALUE_0": os.devnull,
    "GIT_CONFIG_KEY_1": "core.fsmonitor",
    "GIT_CONFIG_VALUE_1": "false",
    "GIT_CONFIG_KEY_2": "core.attributesFile",
    "GIT_CONFIG_VALUE_2": os.devnull,
    "GIT_CONFIG_KEY_3": "uploadpack.packObjectsHook",
    "GIT_CONFIG_VALUE_3": "",
}

_PINNED_CHILD = """
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd()))
from scripts.run_attested_mlxlm_1971_v9_calibration import run_pinned_v9_task

request = json.loads(sys.stdin.read())
if request["action"] == "grade":
    report = run_pinned_v9_task("grade", Path(request["workspace"]))
elif request["action"] == "export":
    patch = request.get("patch_path")
    report = run_pinned_v9_task(
        "export", Path(request["workspace"]),
        source_repo=Path(request["source_repo"]),
        patch_path=Path(patch) if patch is not None else None,
    )
else:
    raise ValueError("unsupported operator action")
print(json.dumps(report, sort_keys=True))
"""


def _git_bytes(revision: str, relative: str) -> bytes:
    result = subprocess.run(
        _git_command(ROOT, "show", f"{revision}:{relative}"),
        env=GIT_ENV, capture_output=True, check=True, timeout=30,
    )
    return result.stdout


def _git_command(repository: Path, *args: str) -> list[str]:
    """Run Git with hooks and ambient configuration disabled."""
    return [
        "git", "-c", f"core.hooksPath={os.devnull}",
        "-c", "core.fsmonitor=false", "-C", str(repository), *args,
    ]


def _materialize_sparse_operator_checkout(checkout: Path) -> None:
    """Materialize only trusted imports without invoking checkout hooks."""
    subprocess.run(
        _git_command(checkout, "sparse-checkout", "init", "--cone"),
        env=GIT_ENV, check=True, capture_output=True, timeout=120,
    )
    subprocess.run(
        _git_command(
            checkout, "sparse-checkout", "set", *SPARSE_PATHS,
        ),
        env=GIT_ENV, check=True, capture_output=True, timeout=120,
    )
    # read-tree materializes only the selected sparse paths and does not run
    # post-checkout hooks. The explicit no-checkout add above avoids running a
    # hook before the trusted files and import tree can be checked.
    subprocess.run(
        _git_command(checkout, "read-tree", "-mu", "HEAD"),
        env=GIT_ENV, check=True, capture_output=True, timeout=120,
    )


def _initialize_isolated_operator_repository(repository: Path) -> None:
    """Fetch the pinned commit into fresh Git metadata before materializing it.

    The caller's repository may contain executable checkout filters or local
    attributes. A fresh bare repository does not inherit either, and the
    pinned worktree is created there with hooks disabled and replacement refs
    unavailable.
    """
    repository.mkdir(mode=0o700)
    subprocess.run(
        ["git", "init", "--bare", "--quiet", str(repository)],
        env=GIT_ENV, check=True, capture_output=True, timeout=30,
    )
    subprocess.run(
        _git_command(
            repository, "fetch", "--no-tags", "--no-recurse-submodules",
            str(ROOT), OPERATOR_COMMIT,
        ),
        env=GIT_ENV, check=True, capture_output=True, timeout=120,
    )
    subprocess.run(
        _git_command(repository, "cat-file", "-e", f"{OPERATOR_COMMIT}^{{commit}}"),
        env=GIT_ENV, check=True, capture_output=True, timeout=30,
    )


def _trusted_v9_protocol() -> dict[str, Any]:
    raw = _git_bytes(V9_COMMIT, V9_PROTOCOL_PATH)
    if hashlib.sha256(raw).hexdigest() != V9_PROTOCOL_SHA256:
        raise ValueError("pinned v9 Git protocol digest mismatch")
    try:
        protocol = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("pinned v9 Git protocol is malformed") from exc
    if not isinstance(protocol, dict):
        raise ValueError("pinned v9 Git protocol is not an object")
    return protocol


def _reject_symlink_components(path: Path) -> Path:
    """Return an absolute path only after checking every existing component."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink path component is forbidden: {current}")
    return absolute


def _open_parent_without_symlinks(path: Path) -> tuple[int, str]:
    """Open a path's parent through directory descriptors with O_NOFOLLOW."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.name:
        raise ValueError("path must name a child of a directory")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:-1]:
            next_descriptor = os.open(
                component, directory_flags | nofollow, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, absolute.name
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file(source: Path, *, max_bytes: int, label: str) -> bytes:
    """Read a bounded, stable regular file without following a path link."""
    absolute = _reject_symlink_components(source)
    parent_fd, name = _open_parent_without_symlinks(absolute)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        os.close(parent_fd)
        raise ValueError(f"{label} must be a readable regular file") from exc
    os.close(parent_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError(f"{label} must be a regular file no larger than {max_bytes} bytes")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{label} must be a regular file no larger than {max_bytes} bytes")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        data = b"".join(chunks)
        if identity_before != identity_after or len(data) != before.st_size:
            raise ValueError(f"{label} changed while being read")
        return data
    finally:
        os.close(descriptor)


def _require_real_directory(path: Path) -> Path:
    absolute = _reject_symlink_components(path)
    parent_fd, name = _open_parent_without_symlinks(absolute)
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ValueError("source repository must be a real directory")
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ValueError("source repository must be a real directory without symlinks") from exc
    finally:
        os.close(parent_fd)
    return absolute


def _snapshot_patch(source: Path, destination: Path) -> Path:
    """Copy one stable, no-follow patch into the private operator directory."""
    data = _read_regular_file(source, max_bytes=256_000, label="patch")

    out_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    out_fd = os.open(destination, out_flags, 0o600)
    try:
        with os.fdopen(out_fd, "wb", closefd=True) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(destination, 0o400)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _require_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_child_report(
    action: str,
    report: dict[str, Any],
    protocol_v9: dict[str, Any],
    expected_patch_sha256: str | None = None,
) -> None:
    """Require the exact v9 evidence envelope before returning child output."""
    if action == "grade":
        fields = {
            "task_id", "protocol_id", "passed", "cases",
            "candidate_source_sha256", "candidate_manifest_sha256",
            "grader_source_sha256", "grader_adapter_sha256",
            "expected_output_sha256",
        }
        if set(report) != fields:
            raise RuntimeError("pinned child did not return complete grade evidence")
        if (
            report["task_id"] != protocol_v9.get("task_id")
            or report["protocol_id"] != protocol_v9.get("protocol_id")
            or type(report["passed"]) is not bool
            or not _require_sha256(report["candidate_source_sha256"])
            or report["candidate_manifest_sha256"]
            != protocol_v9.get("source_manifest_sha256")
            or report["grader_source_sha256"]
            != protocol_v9.get("implementation_sha256", {}).get(
                "src/vare/mlxlm_1971_repo_task_v9.py"
            )
            or report["grader_adapter_sha256"]
            != protocol_v9.get("implementation_sha256", {}).get(
                "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/grader_adapter_v4.py"
            )
            or report["expected_output_sha256"]
            != protocol_v9.get("expected_output_sha256")
        ):
            raise RuntimeError("pinned child grade identity differs from v9 evidence")
        expected_cases = [
            f"value-scale-{scale:g}-{phase}"
            for scale in protocol_v9.get("value_scales", [])
            for phase in ("full", "cached")
        ]
        cases = report["cases"]
        if not isinstance(cases, list) or [
            case.get("case_id") if isinstance(case, dict) else None
            for case in cases
        ] != expected_cases:
            raise RuntimeError("pinned child grade case inventory is incomplete")
        for case in cases:
            if (
                set(case) != {"case_id", "passed", "failure", "wall_s"}
                or type(case["passed"]) is not bool
                or (
                    case["failure"] is not None
                    and (not isinstance(case["failure"], str) or not case["failure"])
                )
                or (case["passed"] and case["failure"] is not None)
                or (not case["passed"] and case["failure"] is None)
            ):
                raise RuntimeError("pinned child grade case is malformed")
            wall_s = case["wall_s"]
            if wall_s is not None and (
                isinstance(wall_s, bool)
                or not isinstance(wall_s, (int, float))
                or not math.isfinite(wall_s)
                or wall_s < 0
            ):
                raise RuntimeError("pinned child grade timing is malformed")
        if report["passed"] is not all(case["passed"] for case in cases):
            raise RuntimeError("pinned child aggregate grade disagrees with cases")
        return

    if action == "export":
        fields = {
            "task_id", "baseline_commit", "candidate_patch_sha256",
            "candidate_source_sha256", "fixed_source_match", "changed_files",
            "baseline_python_manifest_sha256", "git_history_exported",
            "tests_exported", "license_exported", "task_prompt_sha256",
        }
        if set(report) != fields:
            raise RuntimeError("pinned child did not return complete export evidence")
        patch_digest = report["candidate_patch_sha256"]
        changed_files = report["changed_files"]
        if (
            report["task_id"] != protocol_v9.get("task_id")
            or report["baseline_commit"] != protocol_v9.get("baseline_commit")
            or patch_digest != expected_patch_sha256
            or (patch_digest is not None and not _require_sha256(patch_digest))
            or not _require_sha256(report["candidate_source_sha256"])
            or type(report["fixed_source_match"]) is not bool
            or changed_files != ([] if patch_digest is None else [protocol_v9.get("target_path")])
            or report["baseline_python_manifest_sha256"]
            != protocol_v9.get("source_manifest_sha256")
            or report["git_history_exported"] is not False
            or report["tests_exported"] is not False
            or report["license_exported"] is not False
            or report["task_prompt_sha256"] != protocol_v9.get("task_prompt_sha256")
        ):
            raise RuntimeError("pinned child export identity or provenance is invalid")
        return

    raise RuntimeError("cannot validate evidence for unsupported v11 action")


def _verify_clean_checkout(checkout: Path) -> None:
    result = subprocess.run(
        _git_command(
            checkout, "status", "--porcelain=v1", "--untracked-files=all",
            "--ignored=traditional",
        ),
        env=GIT_ENV, capture_output=True, check=True, text=True, timeout=30,
    )
    if result.stdout:
        raise ValueError("pinned operator worktree contains unexpected files or changes")


def verify_v11_protocol() -> dict[str, Any]:
    """Verify the v11 manifest and both historical trust roots."""
    source = _read_regular_file(PROTOCOL, max_bytes=256_000, label="v11 protocol")
    if hashlib.sha256(source).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("v11 protocol differs from independently pinned bytes")
    lock_bytes = _read_regular_file(LOCK, max_bytes=256_000, label="v11 lock")
    if hashlib.sha256(lock_bytes).hexdigest() != LOCK_SHA256:
        raise ValueError("v11 lock differs from independently pinned bytes")
    protocol = json.loads(source)
    lock = json.loads(lock_bytes)
    if (
        lock.get("protocol_id") != protocol.get("protocol_id")
        or lock.get("protocol_sha256") != PROTOCOL_SHA256
        or protocol.get("protocol_id")
        != "vare.mlsys-repository-task.mlxlm-1971.v11.operator"
        or protocol.get("supersedes_protocol_id")
        != "vare.mlsys-repository-task.mlxlm-1971.v10.operator"
        or protocol.get("status") != "PROSPECTIVE_OPERATOR_INTERFACE_NOT_EXECUTED"
        or protocol.get("operator_source_commit") != OPERATOR_COMMIT
        or protocol.get("operator_source_path") != SOURCE_PATH
        or protocol.get("sparse_checkout_paths") != list(SPARSE_PATHS)
        or protocol.get("required_nested_v9_test_paths") != [
            "tests/test_mlxlm_1971_calibration_v9.py",
            "tests/test_mlxlm_1971_repo_task_v9.py",
        ]
        or protocol.get("reference_v9_source_commit") != V9_COMMIT
        or protocol.get("reference_v9_protocol_sha256") != V9_PROTOCOL_SHA256
        or protocol.get("allowed_actions") != ["export", "grade"]
        or protocol.get("hidden_evaluation_output_to_candidate") is not False
        or protocol.get("requires_disable_git_replacement_refs") is not True
        or protocol.get("requires_immutable_pinned_operator_worktree") is not True
        or protocol.get("requires_nested_v9_source_attestation") is not True
    ):
        raise ValueError("v11 operator protocol identity mismatch")
    _trusted_v9_protocol()
    for relative in PINNED_OPERATOR_SOURCES:
        if not _git_bytes(OPERATOR_COMMIT, relative):
            raise ValueError(f"pinned v11 operator source missing from Git: {relative}")
    return protocol


def run_v11_operation(
    action: str,
    workspace: Path,
    *,
    source_repo: Path | None = None,
    patch_path: Path | None = None,
) -> dict[str, Any]:
    """Use the historical trusted grader only through pinned source commits."""
    if action not in {"grade", "export"}:
        raise ValueError("unsupported v11 operation")
    if action == "grade" and (source_repo is not None or patch_path is not None):
        raise ValueError("grade cannot receive export-only arguments")
    if action == "export" and source_repo is None:
        raise ValueError("export requires source repository")
    workspace = _reject_symlink_components(workspace)
    if workspace.exists() and workspace.is_symlink():
        raise ValueError("refusing symlinked candidate workspace")
    if action == "export":
        assert source_repo is not None
        source_repo = _require_real_directory(source_repo)
        if patch_path is not None:
            patch_path = _reject_symlink_components(patch_path)
            if not patch_path.is_file():
                raise ValueError("patch must be a regular file")
    verify_v11_protocol()
    protocol_v9 = _trusted_v9_protocol()
    request: dict[str, str] = {
        "action": action, "workspace": str(workspace),
    }
    if source_repo is not None:
        request["source_repo"] = str(source_repo)

    expected_patch_sha256: str | None = None
    with tempfile.TemporaryDirectory(prefix="vare-v11-operator-") as temporary:
        private = Path(temporary)
        if patch_path is not None:
            snapshot = _snapshot_patch(patch_path, private / "candidate.patch")
            request["patch_path"] = str(snapshot)
            expected_patch_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        checkout = private / "trusted-operator"
        operator_repository = private / "operator-objects.git"
        checkout_created = False
        try:
            _initialize_isolated_operator_repository(operator_repository)
            subprocess.run(
                _git_command(
                    operator_repository, "worktree", "add", "--detach", "--no-checkout",
                    str(checkout), OPERATOR_COMMIT,
                ),
                env=GIT_ENV, check=True, capture_output=True, timeout=120,
            )
            checkout_created = True
            _materialize_sparse_operator_checkout(checkout)
            _verify_clean_checkout(checkout)
            for relative in PINNED_OPERATOR_SOURCES:
                source_file = checkout / relative
                if source_file.is_symlink() or not source_file.is_file():
                    raise ValueError(f"untrusted operator source path: {relative}")
                if source_file.read_bytes() != _git_bytes(OPERATOR_COMMIT, relative):
                    raise ValueError(
                        f"operator worktree differs from pinned Git bytes: {relative}"
                    )
            result = subprocess.run(
                [sys.executable, "-I", "-c", _PINNED_CHILD],
                cwd=checkout, env=GIT_ENV,
                input=json.dumps(request), text=True, capture_output=True,
                check=False, timeout=1800,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"pinned v11 operator exited nonzero: {result.returncode}"
                )
            try:
                report = json.loads(result.stdout)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("v11 operator did not return one JSON object") from exc
            if not isinstance(report, dict):
                raise RuntimeError("v11 operator returned non-object evidence")
            _validate_child_report(
                action, report, protocol_v9, expected_patch_sha256
            )
            return report
        finally:
            if checkout_created:
                subprocess.run(
                    _git_command(
                        operator_repository, "worktree", "remove", "--force", str(checkout)
                    ),
                    env=GIT_ENV, check=True, capture_output=True, timeout=120,
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("grade", "export"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_v11_operation(
        args.action, args.workspace,
        source_repo=args.source_repo, patch_path=args.patch,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
