"""Adapter for frozen MLSys graders to VARE's isolated subject grader.

The benchmark pack's adapter discovers this symbol at runtime. This module
uses only task IDs with a trusted operator-side call materializer; unknown
tasks fail closed. Candidate code runs through ``vare.isolated_candidate``
and never in the grader's Python process.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

from .container_isolation import IsolationLimits
from .isolated_candidate import _MAX_LINE
from .separated_grading import grade_isolated_subject


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "protocols/mlsys_separated_candidate_harness_v1.json"
PACK_ROOTS = {
    "mlsys-v1-": ROOT / "benchmarks/mlsys_coding_v1",
    "mlsys-v3-": ROOT / "benchmarks/mlsys_coding_v3",
}
_IMAGE_BY_HOST = {
    ("Linux", "x86_64"): "linux_x86_64",
    ("Darwin", "arm64"): "darwin_arm64",
    ("Darwin", "aarch64"): "darwin_arm64",
}


class SeparatedHarnessError(ValueError):
    """Trusted task identity or pack integrity could not be established."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_protocol() -> dict[str, Any]:
    try:
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeparatedHarnessError("isolated harness protocol is unavailable") from exc
    if protocol.get("protocol_id") != "mlsys-separated-candidate-harness-v1":
        raise SeparatedHarnessError("isolated harness protocol identity mismatch")
    return protocol


def _verify_pack(pack_root: Path, protocol: dict[str, Any]) -> None:
    relative = pack_root.relative_to(ROOT).as_posix()
    locked = protocol["supported_packs"].get(relative)
    if type(locked) is not dict:
        raise SeparatedHarnessError("task pack is not in the frozen allowlist")
    manifest_path = pack_root / "manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SeparatedHarnessError("task pack manifest is unreadable") from exc
    if _sha256(raw) != locked.get("manifest_sha256"):
        raise SeparatedHarnessError("task pack manifest hash differs from the protocol")
    entries = manifest.get("files")
    if type(entries) is not list:
        raise SeparatedHarnessError("task pack manifest file list is missing")
    expected: dict[str, tuple[int, str]] = {}
    for item in entries:
        if type(item) is not dict:
            raise SeparatedHarnessError("task pack manifest entry is malformed")
        name, size, digest = item.get("path"), item.get("bytes"), item.get("sha256")
        if (
            type(name) is not str or not name or Path(name).is_absolute()
            or ".." in Path(name).parts or type(size) is not int
            or type(digest) is not str or len(digest) != 64
            or name in expected
        ):
            raise SeparatedHarnessError("task pack manifest entry is invalid")
        expected[name] = (size, digest)
    actual_files: dict[str, Path] = {}
    for path in pack_root.rglob("*"):
        if path.is_symlink():
            raise SeparatedHarnessError("task pack contains a symlink")
        if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts:
            actual_files[path.relative_to(pack_root).as_posix()] = path
    if set(actual_files) != set(expected):
        raise SeparatedHarnessError("task pack file inventory differs from the manifest")
    for name, path in actual_files.items():
        data = path.read_bytes()
        size, digest = expected[name]
        if len(data) != size or _sha256(data) != digest:
            raise SeparatedHarnessError(f"task pack file hash mismatch: {name}")


def _task_adapter(task_id: str) -> tuple[Path, str, str, str]:
    pack_root = next(
        (root for prefix, root in PACK_ROOTS.items() if task_id.startswith(prefix)), None,
    )
    if pack_root is None:
        raise SeparatedHarnessError("task ID is not in a supported MLSys pack")
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    try:
        from mlsys_task_isolated_calls import TASKS
    except ImportError as exc:
        raise SeparatedHarnessError("trusted task call adapter is unavailable") from exc
    spec = TASKS.get(task_id)
    if spec is None:
        raise SeparatedHarnessError("task has no isolated call materializer")
    _grader_file, module, function, _count, task_kind = spec
    if task_id in {"mlsys-v3-ema-bias", "mlsys-v3-label-smooth"}:
        task_name = task_id
    else:
        task_name = task_id.removeprefix("mlsys-v1-").removeprefix("mlsys-v3-").replace("-", "_")
    task_dir = pack_root / "tasks" / task_name
    if not (task_dir / "task.json").is_file():
        raise SeparatedHarnessError("trusted task descriptor is missing")
    return task_dir, module, function, task_kind


def _resolve_image(protocol: dict[str, Any]) -> str:
    key = _IMAGE_BY_HOST.get((platform.system(), platform.machine()))
    if key is None:
        raise SeparatedHarnessError("host architecture has no locked grader image")
    image = protocol["container_images"].get(key)
    if type(image) is not str or "@sha256:" not in image:
        raise SeparatedHarnessError("locked grader image is missing")
    return image


def _verify_runtime_limits(protocol: dict[str, Any], image: str) -> None:
    """Keep this adapter's locked limits equal to the existing runner defaults."""
    expected = protocol["resource_limits"]
    actual = IsolationLimits(image=image, max_output_bytes=_MAX_LINE)
    if (
        actual.memory_mb != expected.get("memory_mb")
        or actual.cpus != expected.get("cpus")
        or actual.pids_limit != expected.get("pids_limit")
        or actual.max_output_bytes != expected.get("max_output_bytes")
        or actual.uid != expected.get("unprivileged_uid")
    ):
        raise SeparatedHarnessError("isolated grader defaults differ from the frozen limits")


def grade_function(request: dict[str, Any]) -> dict[str, Any]:
    """Grade one frozen task through the existing Docker-isolated grader."""
    if type(request) is not dict:
        raise SeparatedHarnessError("harness request must be an object")
    task_path_input = Path(str(request.get("trusted_task_path", "")))
    workspace_input = Path(str(request.get("workspace", "")))
    if task_path_input.is_symlink() or workspace_input.is_symlink():
        raise SeparatedHarnessError("task and candidate roots cannot be symlinks")
    task_path = task_path_input.resolve()
    workspace = workspace_input.resolve()
    if not task_path.is_file() or task_path.name != "task.json":
        raise SeparatedHarnessError("trusted task descriptor path is invalid")
    if not workspace.is_dir():
        raise SeparatedHarnessError("candidate workspace is invalid")
    nonce = request.get("nonce")
    if type(nonce) is not str or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
        raise SeparatedHarnessError("harness nonce is invalid")
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeparatedHarnessError("trusted task descriptor is unreadable") from exc
    task_id = task.get("id") if type(task) is dict else None
    if type(task_id) is not str:
        raise SeparatedHarnessError("trusted task ID is missing")
    task_dir, module, function, task_kind = _task_adapter(task_id)
    if task_path != (task_dir / "task.json").resolve():
        raise SeparatedHarnessError("task descriptor is outside its canonical pack path")
    if request.get("task_type") != task_kind or task.get("metadata", {}).get("task_type") != task_kind:
        raise SeparatedHarnessError("task type differs from trusted metadata")
    if request.get("entrypoint") != f"{module}:{function}":
        raise SeparatedHarnessError("entrypoint differs from the trusted task adapter")

    protocol = _load_protocol()
    pack_root = task_dir.parents[1]
    _verify_pack(pack_root, protocol)
    image = _resolve_image(protocol)
    _verify_runtime_limits(protocol, image)
    try:
        from mlsys_task_isolated_calls import materialize_isolated_calls
        calls, _provenance = materialize_isolated_calls(task_id, task_dir)
    except Exception as exc:  # trusted adapter failures never become a pass
        raise SeparatedHarnessError("trusted call materialization failed") from exc
    _verify_pack(pack_root, protocol)
    with tempfile.TemporaryDirectory(prefix="vare-isolated-calls-") as folder:
        calls_path = Path(folder) / "calls.json"
        calls_path.write_text(
            json.dumps(calls, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        )
        try:
            result = grade_isolated_subject(workspace, calls_path, image=image)
        except Exception as exc:  # noqa: BLE001 - isolation failures are fail-closed
            return {
                "passed": False,
                "error": f"isolated grader failed: {type(exc).__name__}",
                "baseline_samples_s": [],
                "candidate_samples_s": [],
                "nonce_enforced": True,
            }
    decision = result.get("decision")
    if type(decision) is not dict or type(decision.get("success")) is not bool:
        raise SeparatedHarnessError("isolated grader returned a malformed decision")
    grade = result.get("grade", {})
    performance = grade.get("performance", {}) if type(grade) is dict else {}
    pairs = performance.get("pairs", []) if type(performance) is dict else []
    if pairs:
        baseline = [item.get("baseline_s") for item in pairs]
        candidate = [item.get("candidate_s") for item in pairs]
    else:
        correctness = grade.get("cases", []) if type(grade) is dict else []
        baseline = []
        candidate = [item.get("wall_s") for item in correctness if type(item) is dict]
    baseline = [sample for sample in baseline if type(sample) in (int, float)]
    candidate = [sample for sample in candidate if type(sample) in (int, float)]
    return {
        "passed": decision["success"],
        "error": None if decision["success"] else str(decision.get("reason", "grade_failed")),
        "baseline_samples_s": baseline,
        "candidate_samples_s": candidate,
        "nonce": nonce,
        "nonce_enforced": True,
        "task_id": task_id,
        "grade_evidence": grade,
    }
