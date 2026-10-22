"""Materialize pinned base or known-fix workspaces for the HF task screen."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks/hf_coding_dev_v1"
TASK_IDS = (
    "accelerate-4296-skip-first-batches-v1",
    "accelerate-4307-tuple-padding-v1",
    "accelerate-4354-multi-cpu-barrier-v1",
    "trl-6625-entropy-gradient-v1",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _archive(repository: Path, revision: str, upstream_prefix: str, destination: Path) -> None:
    raw = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", revision, upstream_prefix],
        check=True, capture_output=True, timeout=120,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if (name.is_absolute() or ".." in name.parts
                    or not (member.isdir() or member.isfile())):
                raise ValueError(f"unsafe or non-regular archive member: {member.name}")
        archive.extractall(destination, filter="data")


def _source_snapshot(root: Path) -> str:
    entries: list[bytes] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"workspace contains symlink: {path}")
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            entries.extend((rel.encode("utf-8"), b"\0", bytes.fromhex(sha256_bytes(path.read_bytes())), b"\n"))
    return sha256_bytes(b"".join(entries))


def prepare(task_id: str, arm: str, repository: Path, output: Path) -> dict[str, Any]:
    if task_id not in TASK_IDS or arm not in {"baseline", "known-fix"}:
        raise ValueError("unknown task or workspace arm")
    task_root = PACK / task_id
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    manifest = json.loads((task_root / "baseline-source-manifest-v1.json").read_text(encoding="utf-8"))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite workspace: {output}")
    repository = repository.resolve(strict=True)
    base_revision = task["source"]["base_revision"]
    source = task["source"]
    target_path = source["target_path"]
    baseline_target_sha = manifest["files"][target_path]["sha256"]
    if baseline_target_sha != source["baseline_target_sha256"]:
        raise ValueError("task baseline target hash differs from the pinned source manifest")
    license_bytes = subprocess.run(
        ["git", "-C", str(repository), "show", f"{base_revision}:LICENSE"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    if sha256_bytes(license_bytes) != source["license_file_sha256"]:
        raise ValueError("source license bytes differ from the pinned task record")
    upstream_prefix = "src/accelerate" if task["source"]["repository"].endswith("accelerate") else "trl"
    output.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory(prefix="vare-hf-source-archive-") as temporary:
            extracted = Path(temporary)
            _archive(repository, base_revision, upstream_prefix, extracted)
            if upstream_prefix == "trl":
                (output / "src").mkdir()
                shutil.move(str(extracted / "trl"), str(output / "src/trl"))
            else:
                shutil.copytree(extracted / "src", output / "src")
        if arm == "known-fix":
            fixed_revision = task["source"]["known_fix_revision"]
            upstream_target = task["source"]["upstream_target_path"]
            content = subprocess.run(
                ["git", "-C", str(repository), "show", f"{fixed_revision}:{upstream_target}"],
                check=True, capture_output=True, timeout=30,
            ).stdout
            if sha256_bytes(content) != source["known_fix_target_sha256"]:
                raise ValueError("known-fix target bytes differ from the pinned task record")
            target = output / task["source"]["target_path"]
            target.write_bytes(content)
            target.chmod(manifest["files"][task["source"]["target_path"]]["mode"])
        shutil.copyfile(task_root / "test_public.py", output / "test_public.py")
        for name, mode in manifest["directories"].items():
            (output / name).chmod(mode)
        for name, record in manifest["files"].items():
            (output / name).chmod(record["mode"])
        actual = _source_snapshot(output)
        if arm == "baseline" and actual != manifest["workspace_snapshot_sha256"]:
            raise ValueError("materialized baseline does not match the pinned workspace snapshot")
        return {
            "task_id": task_id,
            "arm": arm,
            "source_revision": base_revision,
            "known_fix_revision": task["source"]["known_fix_revision"] if arm == "known-fix" else None,
            "workspace": str(output.resolve()),
            "workspace_snapshot_sha256": actual,
            "baseline_snapshot_sha256": manifest["workspace_snapshot_sha256"],
        }
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--arm", choices=("baseline", "known-fix"), required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.task_id, args.arm, args.repository, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
