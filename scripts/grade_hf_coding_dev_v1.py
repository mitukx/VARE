"""Parent-owned isolated verifier for the HF coding-development candidate screen.

Candidate source runs in a separate, network-disabled Docker container through
the existing JSON function-call boundary. This process owns all cases and
expected values; the callee API copied into the subject contains no assertions,
test vectors, or expected results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
PACK = ROOT / "benchmarks/hf_coding_dev_v1"
PROTOCOL = PACK / "protocol-v1.json"
LOCK = PACK / "protocol-v1.lock.json"
IMAGE = "vare/hf-coding-dev-arm64@sha256:0363cbea4db3e3739f09eaa6d402ed2328d1bf768bfead190af9ee706e663d97"
TASK_IDS = (
    "accelerate-4296-skip-first-batches-v1",
    "accelerate-4307-tuple-padding-v1",
    "accelerate-4354-multi-cpu-barrier-v1",
    "trl-6625-entropy-gradient-v1",
)

from scripts.grade_transformers_phase_a_v1 import (  # noqa: E402
    _inspect_frozen_image,
    _scan_candidate,
    _stage_verified_snapshot,
)
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _protocol() -> tuple[dict[str, Any], dict[str, Any], str]:
    lock_bytes = LOCK.read_bytes()
    lock = json.loads(lock_bytes)
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_sha = sha256_bytes(protocol_bytes)
    if protocol_sha != lock.get("protocol_sha256"):
        raise ValueError("candidate-screen protocol differs from its lock")
    protocol = json.loads(protocol_bytes)
    if protocol.get("protocol_id") != lock.get("protocol_id"):
        raise ValueError("candidate-screen protocol identity differs from its lock")
    if protocol.get("candidate_image") != IMAGE:
        raise ValueError("candidate-screen image differs from the frozen image")
    if protocol.get("task_ids") != list(TASK_IDS):
        raise ValueError("candidate-screen task membership differs from its lock")
    _verify_runtime_lock(protocol.get("runtime"))
    implementations = lock.get("implementation_sha256")
    if not isinstance(implementations, dict):
        raise ValueError("candidate-screen implementation lock is missing")
    for relative, expected in implementations.items():
        path = ROOT / relative
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"candidate-screen implementation differs from its lock: {relative}")
    for task_id in TASK_IDS:
        spec = protocol.get("tasks", {}).get(task_id)
        if not isinstance(spec, dict):
            raise ValueError(f"candidate-screen task is missing: {task_id}")
        task_root = PACK / spec["directory"]
        for name, expected in (
            ("task.json", spec["task_json_sha256"]),
            ("baseline-source-manifest-v1.json", spec["baseline_manifest_sha256"]),
            ("evaluator/task_api.py", spec["task_api_sha256"]),
        ):
            path = task_root / name
            if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
                raise ValueError(f"candidate-screen artifact differs from lock: {task_id}/{name}")
        task = json.loads((task_root / "task.json").read_bytes())
        source = task.get("source")
        if not isinstance(source, dict) or (
            source.get("known_fix_revision") != spec.get("fixed_revision")
            or source.get("baseline_target_sha256") is None
            or source.get("known_fix_target_sha256") is None
        ):
            raise ValueError(f"candidate-screen source provenance is incomplete: {task_id}")
    return protocol, lock, protocol_sha


def _verify_runtime_lock(runtime: Any) -> None:
    """Verify the recorded build inputs as well as the immutable runtime image."""
    if not isinstance(runtime, dict):
        raise ValueError("candidate-screen runtime provenance is missing")
    for file_key, digest_key in (
        ("dockerfile", "dockerfile_sha256"),
        ("requirements_direct", "requirements_direct_sha256"),
        ("requirements_lock", "requirements_lock_sha256"),
    ):
        relative = runtime.get(file_key)
        expected = runtime.get(digest_key)
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"candidate-screen runtime lock is incomplete: {file_key}")
        path = PACK / relative
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"candidate-screen runtime input differs from lock: {file_key}")


def _load_task(task_id: str, protocol: dict[str, Any], lock: dict[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, bytes], str,
]:
    if task_id not in TASK_IDS:
        raise ValueError("unknown candidate-screen task")
    spec = protocol["tasks"][task_id]
    root = PACK / spec["directory"]
    task_bytes = (root / "task.json").read_bytes()
    manifest_bytes = (root / "baseline-source-manifest-v1.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    task = json.loads(task_bytes)
    if task.get("task_id") != task_id or manifest.get("task_id") != task_id:
        raise ValueError("candidate-screen task identity mismatch")
    if task.get("source", {}).get("base_revision") != spec.get("baseline_revision"):
        raise ValueError("candidate-screen baseline revision mismatch")
    if task.get("source", {}).get("target_path") != spec.get("target_path"):
        raise ValueError("candidate-screen target path mismatch")
    if sha256_bytes((root / "test_public.py").read_bytes()) != spec["public_test_sha256"]:
        raise ValueError("candidate-screen public test differs from its lock")
    return task, manifest, manifest["files"], sha256_bytes(manifest_bytes)


def _expected_entropy(args: list[Any]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    seed, rows, hidden_size, vocab_size, temperature, _chunk_size = args
    torch.manual_seed(int(seed))
    hidden = torch.randn(int(rows), int(hidden_size), requires_grad=True)
    weight = torch.randn(int(vocab_size), int(hidden_size), requires_grad=True)
    logits = (hidden @ weight.T) / float(temperature)
    log_probabilities = functional.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    entropy.sum().backward()
    assert hidden.grad is not None and weight.grad is not None
    return {"hidden": hidden.grad.tolist(), "weight": weight.grad.tolist()}


def _cases(task_id: str) -> list[FunctionCall]:
    if task_id == "accelerate-4296-skip-first-batches-v1":
        loader_cases = ((2, True, True, 1), (3, False, True, 2), (4, True, False, 1))
        return [
            FunctionCall(
                case_id=f"loader-config-{index}", module="vare_task_api",
                function="skip_first_batches_config", args=list(case),
                expected=[
                    {"loader_type": name, "drop_last": case[1], "non_blocking": case[2]}
                    for name in ("DataLoaderShard", "DataLoaderDispatcher")
                ],
            )
            for index, case in enumerate(loader_cases)
        ]
    if task_id == "accelerate-4307-tuple-padding-v1":
        tuple_cases = (
            ([0, 1, 2], 2, 1, [2, 2]),
            ([0, 1, 2, 3, 4], 3, 2, [4, 4]),
            ([5, 7, 9, 11, 13, 15, 17], 4, 3, [17, 17]),
        )
        return [
            FunctionCall(
                case_id=f"tuple-padding-{index}", module="vare_task_api",
                function="split_tuple", args=[values, count, rank],
                expected={"container": "tuple", "values": expected},
            )
            for index, (values, count, rank, expected) in enumerate(tuple_cases)
        ]
    if task_id == "accelerate-4354-multi-cpu-barrier-v1":
        return [
            FunctionCall(
                case_id=f"multi-cpu-barrier-{index}", module="vare_task_api",
                function="multi_cpu_barrier", args=[rank],
                expected=[{"args": [], "kwargs": {}}],
            )
            for index, rank in enumerate((0, 2))
        ]
    entropy_cases = (
        (17, 4, 3, 7, 1.0, 3),
        (23, 5, 4, 11, 0.7, 4),
        (41, 3, 5, 13, 1.3, 5),
    )
    return [
        FunctionCall(
            case_id=f"entropy-gradient-{index}", module="vare_task_api",
            function="entropy_gradients", args=list(case),
            expected=_expected_entropy(list(case)),
        )
        for index, case in enumerate(entropy_cases)
    ]


def grade(task_id: str, workspace: Path) -> dict[str, Any]:
    protocol, lock, protocol_sha = _protocol()
    task, manifest, _manifest_files, manifest_sha = _load_task(task_id, protocol, lock)
    spec = protocol["tasks"][task_id]
    files, file_count, directory_count = _scan_candidate(
        workspace, manifest, spec["target_path"],
    )
    # The candidate container receives the pinned package and a request adapter.
    # Public tests and every parent-owned expected value remain outside the subject.
    source_files = {name: value for name, value in files.items() if name.startswith("src/")}
    source_manifest = {
        "files": {name: manifest["files"][name] for name in source_files},
        "directories": {
            name: mode for name, mode in manifest["directories"].items()
            if name == "src" or name.startswith("src/")
        },
    }
    image_id = _inspect_frozen_image(IMAGE)
    cache = Path.home() / ".cache" / "vare" / "candidate-workspaces"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="vare-hf-task-subject-", dir=cache) as directory:
        subject = Path(directory) / "subject"
        subject.mkdir()
        _stage_verified_snapshot(subject, source_manifest, source_files)
        api = PACK / spec["directory"] / "evaluator/task_api.py"
        shutil.copyfile(api, subject / "vare_task_api.py")
        report = grade_calls(
            subject,
            _cases(task_id),
            task_kind="correctness",
            image=IMAGE,
            timeout_s=float(protocol["limits"]["case_timeout_s"]),
            memory_mb=int(protocol["limits"]["memory_mb"]),
            rtol=float(spec.get("rtol", 0.0)),
            atol=float(spec.get("atol", 0.0)),
            discard_candidate_stdout=True,
        )
    result = report.to_dict()
    result.update({
        "task_id": task_id,
        "candidate_target": spec["target_path"],
        "candidate_source_sha256": sha256_bytes(files[spec["target_path"]]),
        "candidate_file_count": file_count,
        "candidate_directory_count": directory_count,
        "candidate_inventory_verified": True,
        "grader_code_mounted": False,
        "public_tests_mounted": False,
        "expected_values_held_by_parent": True,
        "task_json_sha256": sha256_bytes((PACK / spec["directory"] / "task.json").read_bytes()),
        "baseline_manifest_sha256": manifest_sha,
        "task_api_sha256": sha256_bytes((PACK / spec["directory"] / "evaluator/task_api.py").read_bytes()),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "grader_sha256": lock["implementation_sha256"]["scripts/grade_hf_coding_dev_v1.py"],
        "grading_image": IMAGE,
        "container_image_id": image_id,
        "source_revision": task["source"]["base_revision"],
        "source_repository": task["source"]["repository"],
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    workspace = args.workspace.absolute()
    if output.exists() or output.is_symlink():
        parser.error("refusing to overwrite evidence")
    if output == workspace or workspace in output.parents:
        parser.error("grader result must be outside the candidate workspace")
    result = grade(args.task_id, workspace)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
