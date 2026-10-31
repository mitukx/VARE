"""Content-addressed snapshots for the code that executes an agent attempt."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


PROTOCOL_ID = "vare.agent-runtime-snapshot.v1"
MANIFEST_NAME = "manifest.json"
MANIFEST_DIGEST_NAME = "manifest.sha256"


class RuntimeSnapshotError(ValueError):
    """Raised when a runtime snapshot cannot be captured or verified safely."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or "\\" in value or "\x00" in value):
        raise RuntimeSnapshotError(f"unsafe runtime source path: {value!r}")
    return path


def _read_regular_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RuntimeSnapshotError(f"runtime source is missing or not a regular file: {path}")
    return path.read_bytes()


def _git_metadata(root: Path, source_paths: Sequence[str]) -> dict[str, Any]:
    env = os.environ.copy()
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0"})

    def run(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(root), *args], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env, check=False,
        )
        return result.stdout.decode("utf-8", "replace").strip() if result.returncode == 0 else None

    status = run("status", "--porcelain=v1", "--untracked-files=all") or ""
    runtime_status = run("status", "--porcelain=v1", "--untracked-files=no", "--", *source_paths) or ""
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_dirty": bool(status),
        "worktree_status_sha256": _sha256(status.encode("utf-8")),
        "runtime_sources_dirty": bool(runtime_status),
        "runtime_status_sha256": _sha256(runtime_status.encode("utf-8")),
    }


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write while publishing runtime snapshot")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def capture_runtime_snapshot(
    *,
    repository_root: Path,
    output_dir: Path,
    source_paths: Sequence[str],
    module_origins: Mapping[str, str],
    expected_hashes: Mapping[str, str] | None = None,
    grader_entrypoints: Sequence[Path] = (),
) -> dict[str, Any]:
    """Atomically retain and hash the exact repository source bytes for a run.

    Grader entrypoints are fingerprinted but not copied: they can embed hidden
    expected behavior. They remain outside candidate-visible workspaces.
    """
    root = repository_root.resolve(strict=True)
    destination = output_dir.absolute()
    if destination.exists() or destination.is_symlink():
        raise RuntimeSnapshotError(f"runtime snapshot destination already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise RuntimeSnapshotError("runtime snapshot parent must be an existing real directory")

    normalized_paths = sorted({_relative_path(item).as_posix() for item in source_paths})
    if not normalized_paths:
        raise RuntimeSnapshotError("runtime snapshot requires at least one source file")

    source_bytes: dict[str, bytes] = {}
    for relative in normalized_paths:
        source = root / relative
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise RuntimeSnapshotError(f"runtime source cannot be resolved: {relative}") from exc
        if resolved != source.absolute() or root not in resolved.parents:
            raise RuntimeSnapshotError(f"runtime source escapes repository or uses a symlink: {relative}")
        source_bytes[relative] = _read_regular_file(source)

    origin_records: dict[str, dict[str, str]] = {}
    for module_name, raw_origin in sorted(module_origins.items()):
        origin = Path(raw_origin)
        try:
            resolved_origin = origin.resolve(strict=True)
        except OSError as exc:
            raise RuntimeSnapshotError(f"loaded module origin is missing: {module_name}") from exc
        if origin.is_symlink() or resolved_origin != origin.absolute():
            raise RuntimeSnapshotError(f"loaded module origin uses a symlink: {module_name}")
        if root not in resolved_origin.parents:
            raise RuntimeSnapshotError(f"loaded module is shadowed outside repository: {module_name}")
        relative = resolved_origin.relative_to(root).as_posix()
        if relative not in source_bytes:
            raise RuntimeSnapshotError(f"loaded module was not included in snapshot: {module_name}")
        origin_bytes = _read_regular_file(origin)
        if origin_bytes != source_bytes[relative]:
            raise RuntimeSnapshotError(f"loaded module bytes differ from captured source: {module_name}")
        origin_records[module_name] = {"path": relative, "sha256": _sha256(origin_bytes)}

    hashes = {path: _sha256(data) for path, data in source_bytes.items()}
    for path, expected in (expected_hashes or {}).items():
        normalized = _relative_path(path).as_posix()
        if hashes.get(normalized) != expected:
            raise RuntimeSnapshotError(f"runtime source drift detected: {normalized}")

    grader_records: list[dict[str, str]] = []
    for grader in grader_entrypoints:
        if grader.is_symlink() or not grader.is_file():
            raise RuntimeSnapshotError(f"grader entrypoint is missing or not a regular file: {grader}")
        resolved = grader.resolve(strict=True)
        try:
            display_path = resolved.relative_to(root).as_posix()
        except ValueError:
            display_path = f"external/{resolved.name}"
        grader_records.append({"path": display_path, "sha256": _sha256(resolved.read_bytes())})

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": _git_metadata(root, normalized_paths),
        "runtime_files": [
            {"path": path, "sha256": hashes[path], "size_bytes": len(source_bytes[path])}
            for path in normalized_paths
        ],
        "module_origins": origin_records,
        "grader_entrypoints": sorted(grader_records, key=lambda item: item["path"]),
        "grader_sources_copied": False,
    }

    stage = Path(tempfile.mkdtemp(prefix=".runtime-snapshot-", dir=destination.parent))
    try:
        for relative, data in source_bytes.items():
            _write_exclusive(stage / "source" / relative, data)
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_exclusive(stage / MANIFEST_NAME, manifest_bytes)
        _write_exclusive(stage / MANIFEST_DIGEST_NAME, (_sha256(manifest_bytes) + "\n").encode())
        for directory, _subdirs, _files in os.walk(stage, topdown=False):
            fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        if destination.exists() or destination.is_symlink():
            raise RuntimeSnapshotError(f"runtime snapshot destination already exists: {destination}")
        os.rename(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    return {
        "path": str(destination),
        "manifest_sha256": _sha256(manifest_bytes),
        "source_hashes": hashes,
        "module_origins": origin_records,
        "repository": manifest["repository"],
        "grader_entrypoints": manifest["grader_entrypoints"],
    }


def audit_runtime_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """Independently verify that a published runtime snapshot is complete."""
    root = snapshot_dir.resolve(strict=True)
    if snapshot_dir.is_symlink() or not root.is_dir():
        raise RuntimeSnapshotError("runtime snapshot must be a real directory")
    manifest_path = root / MANIFEST_NAME
    digest_path = root / MANIFEST_DIGEST_NAME
    manifest_bytes = _read_regular_file(manifest_path)
    digest_bytes = _read_regular_file(digest_path)
    expected_manifest_digest = digest_bytes.decode("ascii").strip()
    if _sha256(manifest_bytes) != expected_manifest_digest:
        raise RuntimeSnapshotError("runtime snapshot manifest digest mismatch")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeSnapshotError("runtime snapshot manifest is malformed") from exc
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("protocol_id") != PROTOCOL_ID):
        raise RuntimeSnapshotError("unsupported runtime snapshot manifest")

    files = manifest.get("runtime_files")
    if not isinstance(files, list) or not files:
        raise RuntimeSnapshotError("runtime snapshot file inventory is empty")
    expected_paths: set[str] = set()
    verified: dict[str, str] = {}
    for record in files:
        if not isinstance(record, dict):
            raise RuntimeSnapshotError("runtime snapshot file record is malformed")
        relative = _relative_path(record.get("path", "")).as_posix()
        if relative in expected_paths:
            raise RuntimeSnapshotError(f"duplicate runtime source path: {relative}")
        expected_paths.add(relative)
        source = root / "source" / relative
        data = _read_regular_file(source)
        actual = _sha256(data)
        if actual != record.get("sha256") or len(data) != record.get("size_bytes"):
            raise RuntimeSnapshotError(f"runtime source snapshot mismatch: {relative}")
        verified[relative] = actual

    actual_paths: set[str] = set()
    source_root = root / "source"
    for current, directories, filenames in os.walk(source_root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise RuntimeSnapshotError("symlink directory in runtime snapshot")
        for name in filenames:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise RuntimeSnapshotError("non-regular file in runtime snapshot")
            actual_paths.add(path.relative_to(source_root).as_posix())
    if actual_paths != expected_paths:
        raise RuntimeSnapshotError("runtime snapshot has missing or unexpected source files")

    return {
        "status": "PASS",
        "manifest_sha256": expected_manifest_digest,
        "verified_source_hashes": verified,
        "module_origins": manifest.get("module_origins", {}),
        "grader_entrypoints": manifest.get("grader_entrypoints", []),
        "repository": manifest.get("repository", {}),
    }
