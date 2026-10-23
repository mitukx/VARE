"""Prospective v9 calibration entry point for the consumed tokenizer task."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import FunctionType
from typing import Any

from vare.isolated_candidate import values_close as _values_close

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "experiments/coding_agent_phase_a_structured_v1/protocol-v9-integrity.json"
PROTOCOL_LOCK = ROOT / "experiments/coding_agent_phase_a_structured_v1/protocol-v9-integrity.lock.json"
TASK_ID = "transformers-49116-tokenizer-routing-v2"
TASK_DIRECTORY = "transformers_49116_tokenizer_routing_v2_grader_v9"
TASK_ADAPTER = (
    "benchmarks/mlsys_repo_eval_v4/"
    "transformers_49116_tokenizer_routing_v2_grader_v9/grader_adapter.py"
)
PHASE_A_GRADER = "scripts/grade_transformers_phase_a_v1.py"
RUNNER = "scripts/grade_transformers_phase_a_integrity_v9.py"
PARENT_COMPARATOR = "src/vare/isolated_candidate.py"
PARENT_ISOLATION = "src/vare/container_isolation.py"
TIMEOUT_S = 45
MEMORY_MB = 1536


def _values_close_with_null(actual: Any, expected: Any, *, rtol: float, atol: float) -> bool:
    """Compare JSON values while treating matching nulls as equal.

    The shared phase-A comparator intentionally rejects unsupported values,
    including None. This integrity task compares the complete registry, whose
    legitimate runtime mapping contains optional routes represented as null.
    """
    if actual is None or expected is None:
        return actual is None and expected is None
    if type(actual) is list and type(expected) is list:
        return len(actual) == len(expected) and all(
            _values_close_with_null(left, right, rtol=rtol, atol=atol)
            for left, right in zip(actual, expected, strict=True)
        )
    if type(actual) is dict and type(expected) is dict:
        return set(actual) == set(expected) and all(
            _values_close_with_null(actual[key], expected[key], rtol=rtol, atol=atol)
            for key in actual
        )
    return _values_close(actual, expected, rtol=rtol, atol=atol)


def _with_null_safe_comparator(original: Callable[..., Any]) -> Callable[..., Any]:
    """Clone the trusted host comparator call with a private JSON comparator."""
    if not isinstance(original, FunctionType) or "values_close" not in original.__globals__:
        raise ValueError("v9 parent grader does not expose its comparison primitive")
    function_globals = original.__globals__.copy()
    function_globals["values_close"] = _values_close_with_null
    wrapped = FunctionType(
        original.__code__, function_globals, original.__name__,
        original.__defaults__, original.__closure__,
    )
    wrapped.__kwdefaults__ = original.__kwdefaults__
    wrapped.__annotations__ = original.__annotations__
    wrapped.__doc__ = original.__doc__
    return wrapped


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_protocol() -> tuple[dict[str, Any], dict[str, Any], str]:
    lock = json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_sha = _sha256(protocol_bytes)
    if protocol_sha != lock.get("protocol_sha256"):
        raise ValueError("v9 tokenizer integrity protocol hash differs from its lock")
    protocol = json.loads(protocol_bytes)
    protocol_id = "vare-coding-agent-phase-a-integrity-v9"
    if protocol.get("protocol_id") != protocol_id or lock.get("protocol_id") != protocol_id:
        raise ValueError("v9 tokenizer integrity protocol identity differs")
    if protocol.get("container_image") != lock.get("container_image"):
        raise ValueError("v9 tokenizer integrity grading image differs from its lock")

    implementation_hashes = lock.get("implementation_sha256")
    if protocol.get("parent_comparison_dependencies") != [
        PARENT_COMPARATOR, PARENT_ISOLATION,
    ]:
        raise ValueError("v9 parent comparison dependency inventory is malformed")
    expected_paths = {
        PHASE_A_GRADER, RUNNER, TASK_ADAPTER, PARENT_COMPARATOR, PARENT_ISOLATION,
    }
    if not isinstance(implementation_hashes, dict) or set(implementation_hashes) != expected_paths:
        raise ValueError("v9 tokenizer integrity implementation inventory is malformed")
    for relative_path, expected_sha in implementation_hashes.items():
        content = (ROOT / relative_path).read_bytes()
        if _sha256(content) != expected_sha:
            raise ValueError(f"locked v9 implementation differs: {relative_path}")

    task = protocol.get("tasks", {}).get(TASK_ID)
    if not isinstance(task, dict) or task.get("grader_adapter_path") != TASK_ADAPTER:
        raise ValueError("v9 tokenizer task definition is missing or malformed")
    if task.get("exposure_classification") != "retrospective-development-consumed":
        raise ValueError("v9 tokenizer task exposure classification differs")

    manifest_entry = lock.get("baseline_manifests", {}).get(TASK_ID)
    if not isinstance(manifest_entry, dict):
        raise ValueError("v9 tokenizer baseline manifest pin is missing")
    manifest_bytes = (ROOT / manifest_entry["path"]).read_bytes()
    if _sha256(manifest_bytes) != manifest_entry.get("sha256"):
        raise ValueError("v9 tokenizer baseline manifest hash differs")
    registry_entry = lock.get("tokenizer_registry_expected")
    if not isinstance(registry_entry, dict):
        raise ValueError("v9 tokenizer registry expectation pin is missing")
    registry_bytes = (ROOT / registry_entry["path"]).read_bytes()
    if _sha256(registry_bytes) != registry_entry.get("sha256"):
        raise ValueError("v9 tokenizer registry expectation hash differs")
    return protocol, lock, protocol_sha


def _bounded_grade_calls(original: Callable[..., Any]) -> Callable[..., Any]:
    """Apply the task's locked resource ceilings to the reused grader."""
    def invoke(subject: Path, calls: list[Any], **kwargs: Any) -> Any:
        if len(calls) != 1 or calls[0].case_id != TASK_ID:
            raise ValueError("v9 parent oracle received an unexpected grader call set")
        expected = kwargs.pop("v9_expected_registry_output")
        rewritten_calls = [replace(calls[0], expected=expected)]
        kwargs["timeout_s"] = TIMEOUT_S
        kwargs["memory_mb"] = MEMORY_MB
        return original(subject, rewritten_calls, **kwargs)

    return invoke


def _complete_registry_expected_output(
    protocol: dict[str, Any],
    lock: dict[str, Any],
    phase_a: Any,
) -> dict[str, Any]:
    """Build the full expected registry in the trusted host process."""
    entry = lock["tokenizer_registry_expected"]
    registry_bytes = (ROOT / entry["path"]).read_bytes()
    if _sha256(registry_bytes) != entry["sha256"]:
        raise ValueError("v9 tokenizer registry expectation changed before comparison")
    registry = json.loads(registry_bytes)
    mapping = registry.get("mapping")
    override_types = registry.get("all_override_types")
    features = protocol.get("tokenizer_registry_features")
    if not isinstance(mapping, dict) or not isinstance(override_types, list):
        raise ValueError("v9 tokenizer registry expectation has an invalid schema")
    required_features = {"tokenizers", "sentencepiece", "g2p_en", "mistral_common"}
    if not isinstance(features, dict) or set(features) != required_features:
        raise ValueError("v9 tokenizer registry feature state is malformed")
    if any(type(value) is not bool for value in features.values()):
        raise ValueError("v9 tokenizer registry feature state is malformed")

    runtime_mapping = {
        name: phase_a._resolve_registry_expression(
            phase_a._decode_ast_dump(expression), features
        )
        for name, expression in mapping.items()
    }
    for name in override_types:
        runtime_mapping.setdefault(
            name, "TokenizersBackend" if features["tokenizers"] else None
        )
    task = protocol["tasks"][TASK_ID]
    expected_routes = task.get("expected_routes")
    if not isinstance(expected_routes, dict):
        raise ValueError("v9 tokenizer route expectation is malformed")
    return {
        **expected_routes,
        "registry_ast": mapping,
        "all_override_types": override_types,
        "runtime_registry": runtime_mapping,
        "runtime_override_types": sorted(override_types),
    }


def grade(task_id: str, workspace: Path, image: str) -> dict[str, Any]:
    """Grade only the retained Development task using the prospective adapter."""
    if task_id != TASK_ID:
        raise ValueError("v9 integrity calibration accepts only its pinned Development task")
    protocol, lock, protocol_sha = _load_protocol()
    task = protocol["tasks"][TASK_ID]
    from scripts import grade_transformers_phase_a_v1 as phase_a

    expected_registry_output = _complete_registry_expected_output(
        protocol, lock, phase_a
    )
    prior_loader = phase_a._load_protocol
    prior_tasks = phase_a.TASKS
    prior_grade_calls = phase_a.grade_calls

    bounded = _bounded_grade_calls(_with_null_safe_comparator(prior_grade_calls))

    def compare_full_registry(subject: Path, calls: list[Any], **kwargs: Any) -> Any:
        kwargs["v9_expected_registry_output"] = expected_registry_output
        return bounded(subject, calls, **kwargs)

    setattr(phase_a, "_load_protocol", lambda: (protocol, lock, protocol_sha))
    setattr(phase_a, "grade_calls", compare_full_registry)
    setattr(phase_a, "TASKS", {
        TASK_ID: (
            TASK_DIRECTORY,
            task["target_path"],
            "routes",
            {"model_types": task["model_types"]},
            task["expected_routes"],
        )
    })
    try:
        return phase_a.grade(task_id, workspace, image)
    finally:
        setattr(phase_a, "_load_protocol", prior_loader)
        setattr(phase_a, "grade_calls", prior_grade_calls)
        setattr(phase_a, "TASKS", prior_tasks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = grade(args.task_id, args.workspace, args.image)
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    sys.stdout.write(encoded)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
