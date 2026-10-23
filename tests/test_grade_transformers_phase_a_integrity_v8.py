from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

from scripts import grade_transformers_phase_a_v1 as phase_a
from scripts import grade_transformers_phase_a_integrity_v8 as grader
from vare.isolated_candidate import FunctionCall, values_close


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = (
    ROOT
    / "benchmarks/mlsys_repo_eval_v4/"
    / "transformers_49116_tokenizer_routing_v2_grader_v8/grader_adapter.py"
)
IMAGE = (
    "vare/transformers-phase-a-arm64:1@sha256:"
    "7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"
)


def _load_adapter():
    spec = importlib.util.spec_from_file_location("v8_fingerprint_probe", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hashlib_and_json_monkeypatch_cannot_forge_registry_comparison(monkeypatch):
    adapter = _load_adapter()
    protocol, lock, _ = grader._load_protocol()
    expected = grader._complete_registry_expected_output(protocol, lock, phase_a)
    mutated = deepcopy(expected)
    registry_key = next(iter(mutated["runtime_registry"]))
    mutated["runtime_registry"][registry_key] = "ForgedTokenizer"

    class ForgedDigest:
        def hexdigest(self) -> str:
            return "0" * 64

    assert not hasattr(adapter, "hashlib")
    assert not hasattr(adapter, "_fingerprint")
    monkeypatch.setattr(hashlib, "sha256", lambda _content: ForgedDigest())
    monkeypatch.setattr(json, "dumps", lambda *_args, **_kwargs: "forged-json")

    # The adapter returns complete structures and the host equality oracle
    # compares them directly; no candidate-side digest participates.
    assert not values_close(mutated, expected, rtol=0, atol=0)


def test_v8_protocol_and_implementation_identities_are_locked():
    protocol, lock, protocol_sha = grader._load_protocol()

    assert protocol["protocol_id"] == "vare-coding-agent-phase-a-integrity-v8"
    assert protocol_sha == lock["protocol_sha256"]
    assert protocol["exposure_classification"] == "retrospective-development-consumed"
    assert set(lock["implementation_sha256"]) == {
        "scripts/grade_transformers_phase_a_v1.py",
        "scripts/grade_transformers_phase_a_integrity_v8.py",
        "benchmarks/mlsys_repo_eval_v4/"
        "transformers_49116_tokenizer_routing_v2_grader_v8/grader_adapter.py",
    }
    assert grader._sha256(ADAPTER_PATH.read_bytes()) == lock["implementation_sha256"][
        protocol["tasks"][grader.TASK_ID]["grader_adapter_path"]
    ]


def test_v8_refuses_other_or_fresh_task_ids(tmp_path: Path):
    try:
        grader.grade("transformers-49412-fusion-registration-idempotence-v2", tmp_path, IMAGE)
    except ValueError as exc:
        assert "only its pinned Development task" in str(exc)
    else:
        raise AssertionError("v8 accepted a task outside its retrospective scope")


def test_v8_dispatches_to_the_pinned_phase_a_grader_before_container_startup(tmp_path: Path):
    try:
        grader.grade(grader.TASK_ID, tmp_path, IMAGE)
    except ValueError as exc:
        assert "candidate file inventory differs" in str(exc)
    else:
        raise AssertionError("v8 unexpectedly graded a nonexistent candidate workspace")


def test_reused_grader_cannot_exceed_task_resource_limits():
    seen: dict[str, object] = {}

    def original(_subject, _calls, **kwargs):
        seen.update(kwargs)
        return "ok"

    bounded = grader._bounded_grade_calls(original)
    assert bounded(
        Path("subject"),
        [FunctionCall(grader.TASK_ID, "adapter", "routes", [], {})],
        timeout_s=600,
        memory_mb=16_000,
        v8_expected_registry_output={},
    ) == "ok"
    assert seen["timeout_s"] == 45
    assert seen["memory_mb"] == 1536
