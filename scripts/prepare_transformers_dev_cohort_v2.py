"""Prepare baseline workspaces for the fresh v6 Development cohort."""
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
PACK = ROOT / "benchmarks/mlsys_repo_eval_v6"
TASKS: dict[str, dict[str, str]] = {
    "transformers-49315-watermark-repeat-v1": {
        "base": "26c0a7fd940b7d14381a39de73f6fbaaa570cf4f",
        "fixed": "ec56e0e1383198065dd3a838629323c9fe228cf4",
        "target": "src/transformers/generation/watermarking.py",
        "family": "watermark-scoring",
        "title": "Repeated n-gram scoring in the watermark detector",
        "prompt": "The watermark detector's option to ignore repeated n-grams should count each unique n-gram once. On repeated-token passages, detection currently overcounts the scored windows. Correct the scoring behavior while preserving the existing seeding and green-token semantics.",
        "worker": "watermark_repeated_ngrams",
        "test": "assert detector_reports_unique_repeated_ngram_count()",
    },
    "transformers-49515-rope-setter-v1": {
        "base": "1573af48b1c5cd8bfb0cfba4dc1ecf6e882d0348",
        "fixed": "93d5be6b2f7b479ea5efa5a35989ef805365d6d9",
        "target": "src/transformers/configuration_utils.py",
        "family": "rotary-configuration-compatibility",
        "title": "Legacy RoPE setter normalization",
        "prompt": "A legacy RoPE scaling dictionary assigned through the compatibility setter must be normalized before rotary embeddings are constructed. Keep the legacy input form usable, provide the deprecation notice, and preserve assignment of None.",
        "worker": "rope_legacy_setter",
        "test": "assert legacy_rope_scaling_initializes_rotary_embedding()",
    },
    "transformers-49731-generation-device-type-v1": {
        "base": "01269bc99768ad45f8aeaeb701e1315371e25239",
        "fixed": "d806b8c84fea7247bc2a066254e72d4a72583430",
        "target": "src/transformers/pytorch_utils.py",
        "family": "nested-output-device-transfer",
        "title": "Preserve typed model outputs during recursive device transfer",
        "prompt": "Recursive device transfer of a model output must preserve its structured output type and nested tuple fields. A generation preparation path currently turns a typed model output into a plain mapping. Correct the recursive conversion without changing tensor values or sequence containers.",
        "worker": "move_to_device_preserves_output_type",
        "test": "assert move_to_device_preserves_model_output_structure()",
    },
    "transformers-49767-janus-background-v1": {
        "base": "849fa754e2dd24f5dd49d05f61ff40b1abc2fd36",
        "fixed": "a3c7fcf3608e2a045d8ae1df6947de8c4ae60b3f",
        "target": "src/transformers/models/janus/image_processing_janus.py",
        "family": "image-preprocessing-configuration",
        "title": "Derive Janus background color from resolved image mean",
        "prompt": "Janus image preprocessing should derive its default padding background from the resolved image mean, including class defaults, while preserving an explicitly supplied color. The PIL and torchvision paths should use consistent configuration semantics.",
        "worker": "janus_background_color",
        "test": "assert janus_padding_color_matches_resolved_mean()",
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_for(root: Path, task_id: str, revision: str) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    directories: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parsed = PurePosixPath(relative)
        if path.is_symlink() or parsed.is_absolute() or ".." in parsed.parts:
            raise ValueError(f"unsafe source path in checkout: {relative}")
        if path.is_dir():
            directories[relative] = stat.S_IMODE(path.stat().st_mode)
        elif path.is_file():
            content = path.read_bytes()
            files[relative] = {
                "sha256": sha256(content),
                "mode": stat.S_IMODE(path.stat().st_mode),
                "size": len(content),
            }
        else:
            raise ValueError(f"non-regular source path in checkout: {relative}")
    snapshot_input = b"".join(
        name.encode() + b"\0" + bytes.fromhex(row["sha256"]) + b"\n"
        for name, row in sorted(files.items())
    )
    return {
        "schema_version": 1,
        "task_id": task_id,
        "source_revision": revision,
        "files": files,
        "directories": directories,
        "workspace_snapshot_sha256": sha256(snapshot_input),
    }


def extract_source(repository: Path, revision: str, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite workspace: {destination}")
    destination.mkdir(parents=True)
    archive_bytes = subprocess.run(
        ["git", "-C", str(repository.resolve()), "archive", "--format=tar", revision, "src/transformers"],
        check=True, capture_output=True, timeout=120,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
        for member in archive.getmembers():
            parsed = PurePosixPath(member.name)
            if parsed.is_absolute() or ".." in parsed.parts or not (member.isdir() or member.isfile()):
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def prepare(task_id: str, kind: str, repository: Path, output: Path) -> dict[str, Any]:
    spec = TASKS[task_id]
    task_root = PACK / task_id
    revision = spec["base"] if kind == "baseline" else spec["fixed"]
    extract_source(repository, spec["base"], output.resolve())
    if kind == "known-fix":
        with tempfile.TemporaryDirectory(prefix="vare-v6-fixed-") as temporary:
            fixed_root = Path(temporary) / "fixed"
            extract_source(repository, spec["fixed"], fixed_root)
            fixed_target = fixed_root / spec["target"]
            output_target = output / spec["target"]
            output_target.write_bytes(fixed_target.read_bytes())
            output_target.chmod(stat.S_IMODE(fixed_target.stat().st_mode))
    test_bytes = (task_root / "test_public.py").read_bytes()
    (output / "test_public.py").write_bytes(test_bytes)
    if kind != "baseline":
        return {"task_id": task_id, "revision": revision}
    manifest = manifest_for(output, task_id, revision)
    target = spec["target"]
    task = {
        "schema": "vare-repository-task-v1",
        "task_id": task_id,
        "metadata": {
            "pack": "mlsys_repo_eval_v6",
            "split": "development",
            "task_type": "correctness",
            "family": spec["family"],
            "license": "Apache-2.0",
        },
        "title": spec["title"],
        "prompt": spec["prompt"],
        "source": {
            "repository": "huggingface/transformers",
            "base_revision": spec["base"],
            "calibration_revision": spec["fixed"],
            "fixed_revision": spec["fixed"],
            "git_history_exported": False,
            "editable_files": [target],
            "baseline_target_sha256": manifest["files"][target]["sha256"],
            "baseline_workspace_sha256": manifest["workspace_snapshot_sha256"],
        },
        "grader": {"visibility": "hidden", "adapter": "scripts/grade_transformers_fresh_dev_v6.py"},
        "limits": {"cpus": 2, "memory_mb": 1900, "pids": 32, "timeout_s": 120},
    }
    task_root.mkdir(parents=True, exist_ok=True)
    (task_root / "task.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n")
    (task_root / "baseline-source-manifest-v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return {"task_id": task_id, "revision": revision, "manifest": manifest, "task": task}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASKS), required=True)
    parser.add_argument("--kind", choices=["baseline", "known-fix"], required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.task_id, args.kind, args.repository, args.output)
    print(json.dumps({"task_id": result["task_id"], "revision": result["revision"], "kind": args.kind}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
