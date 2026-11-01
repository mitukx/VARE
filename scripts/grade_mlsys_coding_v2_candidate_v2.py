#!/usr/bin/env python3
"""Grade authored MLSys v2 candidates through the locked isolated runner.

This adapter is deliberately separate from the frozen generic MLSys v2
adapter: the authored task pack is not allowlisted by that protocol. Trusted
oracles materialize JSON calls in the operator process; candidate modules run
only through ``grade_isolated_subject`` in the pinned, network-disabled
container.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _repository_path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_repository_path) not in sys.path:
        sys.path.insert(0, str(_repository_path))
PROTOCOL_PATH = ROOT / "benchmarks/corrected_template_agent_v3/isolation-protocol.json"
PROTOCOL_SHA256 = "d87ad20c1a477b313e3dd8ec4cd35e30efc82273155d4c4bbc306e5c19fa2b88"
PACK_ROOT = ROOT / "benchmarks/mlsys_coding_v2"
TASK_SPECS = {
    "label-smooth": {
        "task_id": "mlsys-v2-label-smooth",
        "source": "label-smooth.py.txt",
        "target": "smooth_targets",
        "module": "label_smooth",
        "control_names": ("label-smooth-baseline.py.txt", "label-smooth-known-fix.py.txt",
                          "label-smooth-incorrect-mutant.py.txt"),
    },
    "ema-bias": {
        "task_id": "mlsys-v2-ema-bias",
        "source": "ema-bias.py.txt",
        "target": "bias_corrected_ema",
        "module": "ema_bias",
        "control_names": ("ema-bias-baseline.py.txt", "ema-bias-known-fix.py.txt",
                          "ema-bias-incorrect-mutant.py.txt"),
    },
}
IMAGE_BY_HOST = {
    ("Linux", "x86_64"): "linux_x86_64",
    ("Darwin", "arm64"): "darwin_arm64",
    ("Darwin", "aarch64"): "darwin_arm64",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_protocol() -> dict[str, Any]:
    raw = PROTOCOL_PATH.read_bytes()
    if _sha(raw) != PROTOCOL_SHA256:
        raise ValueError("authored-task isolation protocol hash mismatch")
    protocol = json.loads(raw)
    if protocol.get("protocol_id") != "vare-authored-mlsys-v2-isolated-grade-v1":
        raise ValueError("authored-task isolation protocol identity mismatch")
    return protocol


def _verify_task_pack(protocol: dict[str, Any]) -> None:
    pack = protocol["task_pack"]
    if pack.get("path") != PACK_ROOT.relative_to(ROOT).as_posix():
        raise ValueError("task pack path differs from the isolation protocol")
    manifest_path = PACK_ROOT / "manifest.json"
    raw = manifest_path.read_bytes()
    if _sha(raw) != pack.get("manifest_sha256"):
        raise ValueError("task pack manifest hash differs from the isolation protocol")
    manifest = json.loads(raw)
    entries = manifest.get("files")
    if type(entries) is not list:
        raise ValueError("task pack manifest file inventory is missing")
    expected: dict[str, tuple[int, str]] = {}
    for item in entries:
        if type(item) is not dict:
            raise ValueError("task pack manifest entry is malformed")
        name, size, digest = item.get("path"), item.get("bytes"), item.get("sha256")
        if (
            type(name) is not str or not name or Path(name).is_absolute()
            or ".." in Path(name).parts or type(size) is not int
            or type(digest) is not str or len(digest) != 64 or name in expected
        ):
            raise ValueError("task pack manifest entry is invalid")
        expected[name] = (size, digest)
    actual: dict[str, Path] = {}
    for path in PACK_ROOT.rglob("*"):
        if path.is_symlink():
            raise ValueError("task pack contains a symlink")
        if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts:
            actual[path.relative_to(PACK_ROOT).as_posix()] = path
    if set(actual) != set(expected):
        raise ValueError("task pack file inventory differs from its manifest")
    for name, path in actual.items():
        data = path.read_bytes()
        size, digest = expected[name]
        if len(data) != size or _sha(data) != digest:
            raise ValueError(f"task pack file hash mismatch: {name}")


def _load_hidden_evaluator(path: Path):
    v1_pack = ROOT / "benchmarks/mlsys_coding_v1"
    for candidate in (ROOT / "src", ROOT / "scripts", v1_pack):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location("vare_authored_v2_hidden_evaluator", path)
    if spec is None or spec.loader is None:
        raise ValueError("trusted hidden evaluator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _isolated_calls(task: str, protocol: dict[str, Any]) -> dict[str, Any]:
    task_spec = TASK_SPECS[task]
    trusted = protocol["supported_tasks"][task_spec["task_id"]]
    evaluator_path = ROOT / trusted["evaluator"]
    if _sha(evaluator_path.read_bytes()) != trusted.get("evaluator_sha256"):
        raise ValueError("trusted evaluator hash differs from the isolation protocol")
    evaluator = _load_hidden_evaluator(evaluator_path)
    descriptor_path = PACK_ROOT / "tasks" / task_spec["task_id"].removeprefix("mlsys-v2-").replace("-", "_") / "task.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    if descriptor.get("id") != task_spec["task_id"]:
        raise ValueError("trusted task descriptor identity mismatch")
    repeat_count = trusted.get("case_count")
    if type(repeat_count) is not int or repeat_count < 1:
        raise ValueError("isolation protocol case count is invalid")
    cases = []
    for index in range(repeat_count):
        args, kwargs = evaluator._generator(index)
        try:
            expected = evaluator._reference(*args, **kwargs)
        except ValueError:
            row = {
                "case_id": f"{task_spec['task_id']}-hidden-{index}",
                "args": json.loads(json.dumps(list(args), allow_nan=False)),
                "kwargs": json.loads(json.dumps(kwargs, allow_nan=False)),
                "expected": None,
                "expected_exception": "ValueError",
                "expected_exception_prefix": "ValueError",
            }
        else:
            row = {
                "case_id": f"{task_spec['task_id']}-hidden-{index}",
                "args": json.loads(json.dumps(list(args), allow_nan=False)),
                "kwargs": json.loads(json.dumps(kwargs, allow_nan=False)),
                "expected": json.loads(json.dumps(expected, allow_nan=False)),
            }
        cases.append(row)
    limits = protocol["resource_limits"]
    image_key = IMAGE_BY_HOST.get((platform.system(), platform.machine()))
    if image_key is None:
        raise ValueError("host platform has no pinned authored-task grader image")
    image = protocol["container_images"].get(image_key)
    if type(image) is not str or "@sha256:" not in image:
        raise ValueError("pinned authored-task grader image is missing")
    from vare.container_isolation import IsolationLimits
    from vare.isolated_candidate import _MAX_LINE

    actual_limits = IsolationLimits(
        image=image,
        memory_mb=limits["memory_mb"],
        cpus=limits["cpus"],
        pids_limit=limits["pids_limit"],
        uid=limits["unprivileged_uid"],
        gid=limits["unprivileged_uid"],
        max_output_bytes=_MAX_LINE,
    )
    if _MAX_LINE != limits["max_output_bytes"] or actual_limits.max_output_bytes != limits["max_output_bytes"]:
        raise ValueError("candidate output limit differs from the isolation protocol")
    return {
        "image": image,
        "task_timeout_s": limits["task_wall_seconds"],
        "calls": {
            "schema_version": 1,
            "task_id": task_spec["task_id"],
            "task_kind": "correctness",
            "module": task_spec["module"],
            "function": task_spec["target"],
            "rtol": 0.0,
            "atol": 1e-12,
            "timeout_s": limits["task_wall_seconds"],
            "memory_mb": limits["memory_mb"],
            "preserve_session": True,
            "discard_candidate_stdout": True,
            "calls": cases,
        },
    }


def grade(task: str, source: Path, *, public: bool = False) -> dict[str, Any]:
    if task not in TASK_SPECS:
        raise ValueError("unsupported authored MLSys task")
    data = source.read_bytes()
    result: dict[str, Any] = {
        "task": task,
        "public": public,
        "source_sha256": _sha(data),
    }
    try:
        protocol = _load_protocol()
        _verify_task_pack(protocol)
        # The repaired function language is statically admitted before any
        # candidate code is compiled or executed. Parent-side observations
        # use only that closed function body; the independent full cases below
        # execute in the pinned isolated container.
        import scripts.grade_mlsys_coding_v2_candidate_v1 as prior_contract

        function, _trusted_module = prior_contract.boundary.load_candidate(task, data.decode("utf-8"))
        checks = (
            prior_contract.evaluate_label_smooth(function, public=public)
            if task == "label-smooth"
            else prior_contract.evaluate_ema(function, public=public)
        )
        if public:
            result.update(
                admitted=True,
                checks=checks,
                isolated_candidate_execution=False,
                success=bool(checks) and all(value is True for value in checks.values()),
            )
            return result
        isolated = _isolated_calls(task, protocol)
        task_spec = TASK_SPECS[task]
        descriptor_path = PACK_ROOT / "tasks" / task_spec["task_id"].removeprefix("mlsys-v2-").replace("-", "_") / "task.json"
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        memory_mb = descriptor["limits"]["memory_mb"]
        if memory_mb != protocol["resource_limits"]["memory_mb"]:
            raise ValueError("task and isolation protocol memory limits differ")
        from vare.separated_grading import grade_isolated_subject

        with tempfile.TemporaryDirectory(prefix="vare-authored-mlsys-v2-") as tmp:
            root = Path(tmp)
            workspace = root / "candidate"
            workspace.mkdir()
            (workspace / f"{task_spec['module']}.py").write_bytes(data)
            calls_path = root / "calls.json"
            calls_path.write_text(json.dumps(isolated["calls"], sort_keys=True) + "\n", encoding="utf-8")
            isolated_grade = grade_isolated_subject(
                workspace,
                calls_path,
                image=isolated["image"],
                task_timeout_s=isolated["task_timeout_s"],
            )
        isolated_success = isolated_grade.get("decision", {}).get("success") is True
        result.update(
            admitted=True,
            checks=checks,
            isolated_grade=isolated_grade,
            isolated_candidate_execution=True,
            success=bool(checks) and all(value is True for value in checks.values()) and isolated_success,
        )
    except Exception as error:  # failure to isolate or grade is always a failure
        result.update(
            admitted=False,
            success=False,
            error=f"{type(error).__name__}: {error}"[:512],
        )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_SPECS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public", action="store_true")
    args = parser.parse_args()
    print(json.dumps(grade(args.task, args.source, public=args.public), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
