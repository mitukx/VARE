#!/usr/bin/env python3
"""Prepare a pinned, remote-free urllib3 task workspace for calibration or screening."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from typing import Any

from vare.environments.runner import Workspace
from vare.environments.spec import EngineeringTaskSpec


TASK_ID = "urllib3-preserve-ambiguous-request-target-v1"
URL = "https://github.com/urllib3/urllib3.git"
ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/historical/urllib3_3352"
BASE = "166f66faa5ef23e255f654e8c6b66ad8590fe630"
BASE_TREE = "0929cc7c1f30a972cd280fcd41faef3c80aa84bd"
FIX = "d150cea52b37f4633a0dff4e26339c51d23d987b"
FIX_TREE = "de9be60beb39803cb681fae6f6236a885f999948"
VERSION_SHIM = b'__version__ = "2.2.1+vare"\n'


def run(argv: list[str], *, cwd: Path | None = None, stdout=None) -> str:
    if stdout is None:
        result = subprocess.run(argv, cwd=cwd, check=True, text=True, capture_output=True)
    else:
        result = subprocess.run(argv, cwd=cwd, check=True, stdout=stdout, stderr=subprocess.PIPE)
    return result.stdout.strip() if stdout is None else ""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_digest(root: Path) -> str:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def extract_archive(bare: Path, revision: str, target: Path) -> None:
    target.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "-C", str(bare), "archive", "--format=tar", revision],
        check=True,
        capture_output=True,
    )
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as bundle:
        members = bundle.getmembers()
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts or member.issym() or member.islnk():
                raise RuntimeError(f"unsafe path in pinned source archive: {member.name!r}")
        bundle.extractall(target, members=members)


def git(cwd: Path, *args: str) -> str:
    return run(["git", "-C", str(cwd), *args])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new, empty run directory outside the VARE checkout")
    parser.add_argument("--calibration-fix", action="store_true",
                        help="apply the pinned upstream fix after creating the base workspace")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing run directory: {output}")
    output.mkdir(parents=True)
    source_dir = output / "source"
    workspace = output / "workspace"

    with tempfile.TemporaryDirectory(prefix="vare-urllib3-fetch-") as temporary:
        bare = Path(temporary) / "upstream.git"
        run(["git", "init", "--bare", "-q", str(bare)])
        run(["git", "-C", str(bare), "fetch", "--quiet", "--no-tags", "--depth=1", URL, BASE])
        base_commit = git(bare, "rev-parse", "FETCH_HEAD")
        base_tree = git(bare, "rev-parse", "FETCH_HEAD^{tree}")
        if base_commit != BASE or base_tree != BASE_TREE:
            raise RuntimeError(f"base revision/tree mismatch: {base_commit} {base_tree}")
        run(["git", "-C", str(bare), "update-ref", "refs/vare/base", base_commit])
        if args.calibration_fix:
            run(["git", "-C", str(bare), "fetch", "--quiet", "--no-tags", "--depth=1", URL, FIX])
            fix_commit = git(bare, "rev-parse", "FETCH_HEAD")
            fix_tree = git(bare, "rev-parse", "FETCH_HEAD^{tree}")
            if fix_commit != FIX or fix_tree != FIX_TREE:
                raise RuntimeError(f"calibration revision/tree mismatch: {fix_commit} {fix_tree}")
            run(["git", "-C", str(bare), "update-ref", "refs/vare/fix", fix_commit])
        else:
            fix_commit, fix_tree = None, None

        extract_archive(bare, "refs/vare/base", source_dir)
        upstream_shim = source_dir / "src/urllib3/_version.py"
        if upstream_shim.exists():
            raise RuntimeError("expected generated _version.py to be absent at the pinned source tree")
        upstream_shim.write_bytes(VERSION_SHIM)

        spec_raw = json.loads((TASK_ROOT / "vare_spec.json").read_text(encoding="utf-8"))
        spec_raw["source"]["local_path"] = str(source_dir)
        spec = EngineeringTaskSpec.from_dict(spec_raw)
        Workspace(spec, root=workspace)
        if git(workspace, "status", "--porcelain"):
            raise RuntimeError("new VARE workspace is not clean after baseline commit")
        baseline_tree = git(workspace, "rev-parse", "HEAD^{tree}")

        reference_file_sha256 = None
        if args.calibration_fix:
            relative = "src/urllib3/connectionpool.py"
            fixed_bytes = subprocess.run(
                ["git", "-C", str(bare), "show", f"refs/vare/fix:{relative}"],
                check=True, capture_output=True,
            ).stdout
            (workspace / relative).write_bytes(fixed_bytes)
            reference_file_sha256 = sha256_bytes(fixed_bytes)

        record: dict[str, Any] = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "source_repository": URL,
            "source_revision": BASE,
            "source_tree": BASE_TREE,
            "workspace_path": str(workspace),
            "workspace_baseline_commit": git(workspace, "rev-parse", "HEAD"),
            "workspace_baseline_tree": baseline_tree,
            "source_snapshot_sha256": snapshot_digest(source_dir),
            "version_shim_sha256": sha256_bytes(VERSION_SHIM),
            "calibration_reference_revision": fix_commit,
            "calibration_reference_tree": fix_tree,
            "calibration_reference_file_sha256": reference_file_sha256,
            "remote_count": int(bool(git(workspace, "remote"))),
        }
        (output / "source-record.json").write_text(
            json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"output": str(output), "source_record": record}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
