"""Prepare an exact source-only workspace for the frozen Transformers cohort."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks/mlsys_repo_eval_v5"
TASKS: dict[str, dict[str, str]] = {
    "transformers-49481-bloom-static-cache-v1": {
        "directory": "transformers-49481-bloom-static-cache-v1",
        "base": "536ecc007387a50e77603bb5d92100e9b07514cc",
        "fixed": "bc4d8df2c4604727ea63414c7af48de7e6eb9c0b",
        "target": "src/transformers/models/bloom/modeling_bloom.py",
        "family": "generation-static-cache",
    },
    "transformers-49361-phi3-longrope-v1": {
        "directory": "transformers-49361-phi3-longrope-v1",
        "base": "6d3802a45f50ce6f2262029926265629010b2217",
        "fixed": "4a2f93bbb5594bcccab9612b095de6f900822601",
        "target": "src/transformers/models/phi3/modeling_phi3.py",
        "family": "longrope-cache-boundary",
    },
    "transformers-49515-rope-setter-v1": {
        "directory": "transformers-49515-rope-setter-v1",
        "base": "1573af48b1c5cd8bfb0cfba4dc1ecf6e882d0348",
        "fixed": "93d5be6b2f7b479ea5efa5a35989ef805365d6d9",
        "target": "src/transformers/configuration_utils.py",
        "family": "legacy-rope-configuration",
    },
}
PROMPTS = {
    "transformers-49481-bloom-static-cache-v1": (
        "Generation with a static cache should work when callers omit an attention mask "
        "for an unpadded input. A small local BLOOM configuration currently fails on this "
        "path. Make the smallest compatible change so generation completes and preserves "
        "the input prefix and requested output length. Keep unrelated generation behavior unchanged."
    ),
    "transformers-49361-phi3-longrope-v1": (
        "For a small Phi3 model using LongRoPE, cached generation should produce the same "
        "next-token logits as a full-prefix forward pass at every generated step, including "
        "when the configured sliding window is enabled. The current behavior diverges after "
        "crossing the original context boundary. Correct the cache behavior while preserving "
        "ordinary generation semantics."
    ),
    "transformers-49515-rope-setter-v1": (
        "Legacy RoPE configuration dictionaries remain accepted through the compatibility "
        "setter and should be normalized before rotary embeddings are initialized. A small "
        "Llama configuration assigned a legacy scaling dictionary currently fails during "
        "initialization. Preserve the legacy input form while making initialization succeed "
        "with the intended scaling type."
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def workspace_snapshot(root: Path, files: dict[str, dict[str, Any]]) -> str:
    entries: list[bytes] = []
    for name in sorted(files):
        entries.extend((name.encode("utf-8"), b"\0", bytes.fromhex(files[name]["sha256"]), b"\n"))
    return sha256(b"".join(entries))


def manifest_for(root: Path, task_id: str, revision: str) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    directories: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        parsed = PurePosixPath(rel)
        if path.is_symlink() or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"unsafe source path in checkout: {rel}")
        if path.is_dir():
            directories[rel] = stat.S_IMODE(path.stat().st_mode)
        elif path.is_file():
            data = path.read_bytes()
            files[rel] = {"sha256": sha256(data), "mode": stat.S_IMODE(path.stat().st_mode), "size": len(data)}
        else:
            raise ValueError(f"non-regular source path in checkout: {rel}")
    return {
        "schema_version": 1,
        "task_id": task_id,
        "source_revision": revision,
        "files": files,
        "directories": directories,
        "workspace_snapshot_sha256": workspace_snapshot(root, files),
    }


def extract_archive(repository: Path, revision: str, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite workspace: {output}")
    output.mkdir(parents=True)
    raw = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", revision, "src/transformers"],
        check=True, capture_output=True, timeout=120,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isdir() or member.isfile()):
                raise ValueError(f"unsafe or non-regular archive member: {member.name}")
        archive.extractall(output, filter="data")


def prepare(task_id: str, kind: str, repository: Path, output: Path) -> dict[str, Any]:
    spec = TASKS[task_id]
    task_root = PACK / spec["directory"]
    if kind == "baseline":
        frozen = (
            task_root / "task.json",
            task_root / "protocol.lock.json",
            task_root / "baseline-source-manifest-v1.json",
        )
        present = [path.name for path in frozen if path.exists() or path.is_symlink()]
        if present:
            raise FileExistsError(
                f"refusing to rewrite frozen task artifacts for {task_id}: {present}"
            )
    revision = spec["base"] if kind == "baseline" else spec["fixed"]
    extract_archive(repository.resolve(), spec["base"], output.resolve())
    if kind == "known-fix":
        with tempfile.TemporaryDirectory(prefix="vare-fixed-target-") as temporary:
            fixed_root = Path(temporary) / "fixed"
            extract_archive(repository.resolve(), spec["fixed"], fixed_root)
            source_target = fixed_root / spec["target"]
            destination = output / spec["target"]
            destination.write_bytes(source_target.read_bytes())
            destination.chmod(stat.S_IMODE(source_target.stat().st_mode))
    shutil.copyfile(task_root / "test_public.py", output / "test_public.py")
    if kind == "baseline":
        manifest = manifest_for(output, task_id, revision)
        target = spec["target"]
        task = {
            "schema": "historical",
            "task_id": task_id,
            "metadata": {
                "pack": "mlsys_repo_eval_v5",
                "split": "development",
                "task_type": "correctness",
                "family": spec["family"],
                "license": "Apache-2.0",
            },
            "prompt": PROMPTS[task_id],
            "source": {
                "repository": "huggingface/transformers",
                "base_revision": spec["base"],
                "calibration_revision": spec["fixed"],
                "fixed_revision": spec["fixed"],
                "git_history_exported": False,
                "files": [target, "test_public.py"],
                "file_sha256": {
                    target: manifest["files"][target]["sha256"],
                    "test_public.py": manifest["files"]["test_public.py"]["sha256"],
                },
                "snapshot_sha256": manifest["workspace_snapshot_sha256"],
            },
            "grader": {"visibility": "hidden", "adapter": "evaluator/adapter.py"},
            "limits": {"cpus": 1, "memory_mb": 1900, "pids": 32, "wall_time_s": 120},
        }
        (task_root / "baseline-source-manifest-v1.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        (task_root / "task.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lock = {
            "protocol_id": "vare-transformers-fresh-development-task-lock-v6",
            "task_id": task_id,
            "task_json_sha256": sha256((task_root / "task.json").read_bytes()),
            "baseline_manifest_sha256": sha256(
                (task_root / "baseline-source-manifest-v1.json").read_bytes(),
            ),
            "workspace_snapshot_sha256": manifest["workspace_snapshot_sha256"],
        }
        (task_root / "protocol.lock.json").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        wrapper = (
            "from __future__ import annotations\n"
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[4]))\n"
            "from scripts.grade_transformers_fresh_dev_v1 import main\n"
            f"TASK_ID = {task_id!r}\n"
            "if __name__ == '__main__':\n"
            "    sys.argv[1:1] = ['--task-id', TASK_ID]\n"
            "    raise SystemExit(main())\n"
        )
        (task_root / "evaluator/grade.py").write_text(wrapper, encoding="utf-8")
        return {"task_id": task_id, "revision": revision, "manifest": manifest, "task": task}
    return {"task_id": task_id, "revision": revision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASKS), required=True)
    parser.add_argument("--kind", choices=["baseline", "known-fix"], required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.task_id, args.kind, args.repository, args.output)
    print(json.dumps({"task_id": result["task_id"], "revision": result["revision"], "kind": args.kind}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
