"""Protocol-only regressions for the T4-compatible v5 inference gate."""

import hashlib
import json
from pathlib import Path

from scripts.gpu_gate_vram_v5 import maximum_reserved_vram_bytes


V4_PATH = Path("protocols/qwen25_deepmath_grpo_math500_v4.lock.json")
V5_PATH = Path("protocols/qwen25_deepmath_grpo_math500_v5.lock.json")
V5 = json.loads(V5_PATH.read_text(encoding="utf-8"))


def _get_path(value, path):
    for key in path:
        value = value[key]
    return value


def _replace_path(value, path, replacement):
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def _delete_path(value, path):
    current = value
    for key in path[:-1]:
        current = current[key]
    current.pop(path[-1])


def test_v5_changes_only_runtime_and_hardware_eligibility_from_v4():
    v4 = json.loads(V4_PATH.read_text(encoding="utf-8"))
    normalized_v5 = json.loads(json.dumps(V5))
    for path in (
        ("protocol_id",), ("phase",), ("claim_boundary",),
        ("model", "device_for_first_gpu_gate"),
        ("first_gpu_gate", "resource_budget", "minimum_total_vram_bytes"),
        ("first_gpu_gate", "resource_budget", "maximum_peak_reserved_vram_bytes"),
        ("first_gpu_gate", "budget", "minimum_gpu_vram_bytes"),
        ("first_gpu_gate", "budget", "max_peak_reserved_vram_bytes"),
        ("software_for_gate", "note"),
        ("software_for_gate", "colab_reference_runtime"),
        ("freeze", "gpu_gate_runner"),
        ("protocol_revision",), ("execution_code", "files_sha256"),
    ):
        _replace_path(normalized_v5, path, _get_path(v4, path))
    for path in (
        ("first_gpu_gate", "resource_budget", "peak_reserved_vram_limit_rule"),
        ("first_gpu_gate", "resource_budget", "minimum_vram_rationale"),
        ("first_gpu_gate", "budget", "max_peak_reserved_vram_fraction"),
        ("first_gpu_gate", "budget", "peak_reserved_vram_limit_rule"),
    ):
        _delete_path(normalized_v5, path)
    assert normalized_v5 == v4
    assert V5["protocol_revision"]["v1_immutable"] is True
    assert V5["protocol_revision"]["v2_immutable"] is True
    assert V5["protocol_revision"]["v3_immutable"] is True
    assert V5["protocol_revision"]["v4_immutable"] is True


def test_t4_minimum_and_reserved_vram_cap_are_preregistered():
    budget = V5["first_gpu_gate"]["budget"]
    observed_t4_bytes = 15_637_086_208
    assert budget["minimum_gpu_vram_bytes"] == 15_000_000_000
    assert budget["minimum_gpu_vram_bytes"] <= observed_t4_bytes < 16 * 1024**3
    assert budget["max_peak_reserved_vram_bytes"] == 14_073_377_587
    assert budget["max_peak_reserved_vram_fraction"] == 0.90
    assert maximum_reserved_vram_bytes(observed_t4_bytes, 0.90) == 14_073_377_587
    assert maximum_reserved_vram_bytes(16 * 1024**3, 0.90) == 14 * 1024**3


def test_v5_runner_preserves_study_conditions_and_enforces_t4_memory_rules():
    runner = Path("scripts/run_qwen25_deepmath_base_gate_v5.py").read_text(encoding="utf-8")
    assert V5["model"]["model_files_sha256"] == json.loads(
        V4_PATH.read_text(encoding="utf-8")
    )["model"]["model_files_sha256"]
    assert V5["training_data"]["sha256"] == "e0c5b2fc11978d735a7710273920676977b533e185284044c3eafa63a24479d7"
    assert V5["training_data"]["sample"]["base_gate_id_list_sha256"] == (
        "0ab6b16559db37f6ff84bb312e18c29046063fac32fae026d5b1265037d03c9a"
    )
    assert V5["first_gpu_gate"]["generation"] == json.loads(
        V4_PATH.read_text(encoding="utf-8")
    )["first_gpu_gate"]["generation"]
    assert '"T4" not in device.name' in runner
    assert "journal.peak_reserved_vram_bytes > maximum_reserved_vram_bytes" in runner


def test_v5_execution_manifest_matches_all_pinned_source_files():
    assert V5["freeze"]["gpu_gate_runner"] == "scripts/run_qwen25_deepmath_base_gate_v5.py"
    for name, expected in V5["execution_code"]["files_sha256"].items():
        observed = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        assert observed == expected, name
