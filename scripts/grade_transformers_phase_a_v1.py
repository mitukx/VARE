"""Grade a Phase A candidate against a frozen baseline and parent-owned oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Any

from vare.isolated_candidate import FunctionCall, grade_calls

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/coding_agent_phase_a_structured_v1/protocol-v6.json"
PROTOCOL_LOCK = ROOT / "experiments/coding_agent_phase_a_structured_v1/protocol-v6.lock.json"
TASKS: dict[str, tuple[str, str, str, dict[str, Any], dict[str, Any]]] = {
    "transformers-49116-tokenizer-routing-v2": (
        "transformers_49116_tokenizer_routing_v2",
        "src/transformers/models/auto/tokenization_auto.py",
        "routes",
        {"model_types": ["qwen3_5_moe_text", "qwen4_exp", "llama", "qwen3"]},
        {
            "routes": {
                "qwen3_5_moe_text": "Qwen3_5Tokenizer",
                "qwen4_exp": "Qwen3_5Tokenizer",
                "llama": "<unregistered>",
                "qwen3": "Qwen2Tokenizer",
            },
            "override_types": ["qwen3_5_moe_text", "qwen4_exp"],
        },
    ),
    "transformers-49412-fusion-registration-idempotence-v2": (
        "transformers_49412_fusion_idempotence_v2",
        "src/transformers/fusion_mapping.py",
        "repeated_registration",
        {"model_type": "vare_hidden_fusion_probe"},
        {
            "repeat_accepted": True,
            "mapping_unchanged": True,
            "conflict_rejected": True,
            "mapping_unchanged_after_conflict": True,
        },
    ),
}


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_protocol() -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_sha = sha256_bytes(protocol_bytes)
    if protocol_sha != lock.get("protocol_sha256"):
        raise ValueError("frozen Phase A protocol hash differs from its lock")
    protocol = json.loads(protocol_bytes)
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("frozen Phase A protocol identity differs from its lock")
    if protocol.get("container_image") != lock.get("container_image"):
        raise ValueError("frozen Phase A grading image differs from its lock")
    implementation_hash = lock.get("implementation_sha256", {}).get(
        "scripts/grade_transformers_phase_a_v1.py"
    )
    if not isinstance(implementation_hash, str):
        raise ValueError("frozen grader implementation hash is missing")
    if sha256_bytes(Path(__file__).resolve().read_bytes()) != implementation_hash:
        raise ValueError("running grader implementation differs from its frozen lock")
    return protocol, lock, protocol_sha


def _load_manifest(task_id: str, lock: dict[str, Any]) -> tuple[dict[str, Any], str]:
    entry = lock.get("baseline_manifests", {}).get(task_id)
    if not isinstance(entry, dict):
        raise ValueError("task has no independently pinned baseline source manifest")
    path = ROOT / entry["path"]
    content = path.read_bytes()
    digest = sha256_bytes(content)
    if digest != entry.get("sha256"):
        raise ValueError("pinned baseline source manifest hash differs")
    manifest = json.loads(content)
    if manifest.get("task_id") != task_id:
        raise ValueError("pinned baseline source manifest task id differs")
    if manifest.get("workspace_snapshot_sha256") != entry.get("workspace_snapshot_sha256"):
        raise ValueError("pinned baseline workspace snapshot differs from protocol lock")
    return manifest, digest


def _relative_name(value: str) -> str:
    parsed = PurePosixPath(value)
    if (not value or parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts
            or "\\" in value or value != parsed.as_posix()):
        raise ValueError("candidate workspace contains an unsafe path")
    return value


def _scan_candidate(workspace: Path, manifest: dict[str, Any], target: str) -> tuple[dict[str, bytes], int, int]:
    """Verify exact tree inventory and return the bytes that will be staged.

    Reads the candidate bytes once. The graded subject is rebuilt from this
    captured byte map, so later workspace mutations cannot alter the code that
    is executed. Only the target file may differ from its pinned baseline.
    """
    root = Path(os.path.abspath(os.fspath(workspace)))
    try:
        root_info = root.lstat()
    except OSError as exc:
        raise ValueError("candidate workspace is missing") from exc
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("candidate workspace is missing or unsafe")

    expected_files = manifest.get("files")
    expected_dirs = manifest.get("directories")
    if not isinstance(expected_files, dict) or not isinstance(expected_dirs, dict):
        raise ValueError("pinned baseline manifest is malformed")
    if target not in expected_files or target == "test_public.py":
        raise ValueError("candidate target is not the pinned editable source")

    actual_files: dict[str, bytes] = {}
    actual_dirs: dict[str, int] = {}
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in list(dirnames):
            path = base / name
            rel = _relative_name(path.relative_to(root).as_posix())
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"candidate workspace contains a symlink: {rel}")
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"candidate workspace contains a non-directory path: {rel}")
            if info.st_dev != root_info.st_dev:
                raise ValueError(f"candidate workspace contains a mounted directory: {rel}")
            actual_dirs[rel] = stat.S_IMODE(info.st_mode)
        for name in filenames:
            path = base / name
            rel = _relative_name(path.relative_to(root).as_posix())
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"candidate workspace contains a symlink: {rel}")
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"candidate workspace contains a non-regular file: {rel}")
            if info.st_dev != root_info.st_dev or info.st_nlink != 1:
                raise ValueError(f"candidate workspace contains a mounted or hard-linked file: {rel}")
            record = expected_files.get(rel)
            if not isinstance(record, dict):
                raise ValueError(f"candidate workspace contains an unpinned file: {rel}")
            mode = stat.S_IMODE(info.st_mode)
            if mode != record.get("mode"):
                raise ValueError(f"candidate file mode differs from baseline: {rel}")
            content = path.read_bytes()
            digest = sha256_bytes(content)
            if rel != target and digest != record.get("sha256"):
                raise ValueError(f"candidate changed a protected baseline file: {rel}")
            actual_files[rel] = content

    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        raise ValueError(f"candidate file inventory differs: missing={missing[:3]} extra={extra[:3]}")
    if actual_dirs != expected_dirs:
        missing = sorted(set(expected_dirs) - set(actual_dirs))
        extra = sorted(set(actual_dirs) - set(expected_dirs))
        changed = sorted(name for name in set(expected_dirs) & set(actual_dirs)
                         if expected_dirs[name] != actual_dirs[name])
        raise ValueError(
            f"candidate directory inventory differs: missing={missing[:3]} "
            f"extra={extra[:3]} mode_changed={changed[:3]}"
        )
    return actual_files, len(actual_files), len(actual_dirs)


def _stage_verified_snapshot(subject: Path, manifest: dict[str, Any], files: dict[str, bytes]) -> None:
    """Rebuild the grader subject only from the already verified file bytes."""
    directories = manifest["directories"]
    for name in sorted(directories, key=lambda item: (item.count("/"), item)):
        _relative_name(name)
        (subject / name).mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        _relative_name(name)
        path = subject / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(manifest["files"][name]["mode"])
    for name, mode in sorted(directories.items(), key=lambda item: (-item[0].count("/"), item[0])):
        (subject / name).chmod(mode)


def _check_image(image: str, lock: dict[str, Any]) -> str:
    expected = lock.get("container_image")
    if not isinstance(expected, str) or image != expected or "@sha256:" not in expected:
        raise ValueError("grading image must exactly match the frozen protocol digest")
    return expected


def _inspect_frozen_image(reference: str) -> str:
    """Require the daemon to have the pinned manifest digest locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", reference, "--format", "{{json .RepoDigests}} {{.Id}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("frozen grading image is unavailable or cannot be inspected") from exc
    try:
        raw_digests, image_id = result.stdout.strip().rsplit(" ", 1)
        repo_digests = json.loads(raw_digests)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Docker image inspection returned malformed digest evidence") from exc
    digest = reference.split("@", 1)[1]
    if not isinstance(repo_digests, list) or not any(item.endswith("@" + digest) for item in repo_digests):
        raise ValueError("local grading image manifest digest differs from the frozen protocol")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ValueError("Docker image inspection omitted the immutable image id")
    return image_id


def grade(task_id: str, workspace: Path, image: str) -> dict[str, Any]:
    if task_id not in TASKS:
        raise ValueError("unknown Development task")
    protocol, lock, protocol_sha = _load_protocol()
    frozen_image = _check_image(image, lock)
    task_dir_name, target, function, args, expected = TASKS[task_id]
    manifest, manifest_sha = _load_manifest(task_id, lock)
    protocol_task = protocol.get("tasks", {}).get(task_id)
    if (not isinstance(protocol_task, dict) or protocol_task.get("target_path") != target
            or protocol_task.get("source_revision") != manifest.get("source_revision")
            or protocol_task.get("baseline_manifest_sha256") != manifest_sha):
        raise ValueError("frozen protocol task definition differs from grader task")

    if Path(workspace).is_symlink():
        raise ValueError("candidate workspace root must not be a symlink")
    files, file_count, directory_count = _scan_candidate(workspace, manifest, target)
    if task_id == "transformers-49116-tokenizer-routing-v2":
        expected_registry = lock.get("tokenizer_registry_expected")
        if not isinstance(expected_registry, dict):
            raise ValueError("frozen tokenizer registry expectation is missing")
        registry_path = ROOT / expected_registry["path"]
        registry_bytes = registry_path.read_bytes()
        registry_sha = sha256_bytes(registry_bytes)
        if registry_sha != expected_registry.get("sha256"):
            raise ValueError("pinned tokenizer registry expectation hash differs")
        registry_expectation = json.loads(registry_bytes)
        expected["registry_ast_sha256"] = sha256_bytes(
            json.dumps(
                registry_expectation["mapping"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        expected["all_override_types_sha256"] = sha256_bytes(
            json.dumps(
                registry_expectation["all_override_types"], separators=(",", ":")
            ).encode("utf-8")
        )
    task_root = ROOT / "benchmarks/mlsys_repo_eval_v4" / task_dir_name
    adapter = task_root / "grader_adapter.py"
    adapter_hash = sha256_bytes(adapter.read_bytes())
    if adapter_hash != protocol_task.get("grader_adapter_sha256"):
        raise ValueError("trusted grader adapter hash differs from frozen protocol")
    container_image_id = _inspect_frozen_image(frozen_image)

    # Stage only verified bytes. Candidate-added hooks/modules never enter the
    # subject, even if a workspace changes after validation.
    staging_root = Path.home() / ".cache"
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vare-phase-a-subject-", dir=staging_root) as directory:
        subject = Path(directory) / "subject"
        subject.mkdir()
        _stage_verified_snapshot(subject, manifest, files)
        shutil.copy2(adapter, subject / "vare_phase_a_adapter.py")
        call = FunctionCall(
            case_id=task_id,
            module="vare_phase_a_adapter",
            function=function,
            args=[args],
            expected=expected,
        )
        report = grade_calls(
            subject,
            [call],
            task_kind="correctness",
            image=frozen_image,
            timeout_s=60,
            memory_mb=1900,
            discard_candidate_stdout=True,
        )
    payload = report.to_dict()
    payload.update(
        {
            "task_id": task_id,
            "candidate_target": target,
            "expected_held_by_parent": True,
            "grader_mounted": False,
            "candidate_git_history_present": False,
            "candidate_workspace_inventory_verified": True,
            "candidate_file_count": file_count,
            "candidate_directory_count": directory_count,
            "candidate_target_sha256": sha256_bytes(files[target]),
            "baseline_manifest_sha256": manifest_sha,
            "trusted_adapter_sha256": adapter_hash,
            "protocol_id": protocol.get("protocol_id"),
            "protocol_sha256": protocol_sha,
            "grader_implementation_sha256": lock["implementation_sha256"][
                "scripts/grade_transformers_phase_a_v1.py"
            ],
            "grading_image": frozen_image,
            "container_image_id": container_image_id,
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASKS), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to overwrite grade evidence")
    workspace_abs = Path(os.path.abspath(os.fspath(args.workspace)))
    output_abs = Path(os.path.abspath(os.fspath(args.output)))
    try:
        output_abs.relative_to(workspace_abs)
    except ValueError:
        pass
    else:
        parser.error("grade evidence output must be outside the candidate workspace")
    result = grade(args.task_id, args.workspace, args.image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
