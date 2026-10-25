from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.audit_output_safety import preflight_new_output, write_new_output


ROOT = Path(__file__).resolve().parents[1]


def test_preflight_rejects_new_destination_inside_retained_input(tmp_path):
    retained = tmp_path / "retained-run"
    retained.mkdir()
    destination = retained / "reports" / "audit.json"

    with pytest.raises(ValueError, match="inside retained input"):
        preflight_new_output(destination, [retained])

    assert not destination.exists()
    assert not destination.parent.exists()


def test_preflight_resolves_symlinked_parent_before_creating_output(tmp_path):
    retained = tmp_path / "retained-run"
    retained.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(retained, target_is_directory=True)
    destination = alias / "new-report.json"

    with pytest.raises(ValueError, match="inside retained input"):
        preflight_new_output(destination, [retained])

    assert not (retained / "new-report.json").exists()


def test_preflight_preserves_existing_manifest_inside_retained_run(tmp_path):
    retained = tmp_path / "retained-run"
    retained.mkdir()
    manifest = retained / "SHA256.json"
    original = b'{"frozen":"inventory"}\n'
    manifest.write_bytes(original)

    with pytest.raises(ValueError, match="inside retained input"):
        preflight_new_output(manifest, [retained])

    assert manifest.read_bytes() == original


def test_exclusive_writer_preserves_destination_created_after_preflight(tmp_path):
    destination = preflight_new_output(tmp_path / "report.json", [])
    sentinel = b"another writer won the race\n"
    destination.write_bytes(sentinel)

    with pytest.raises(ValueError, match="appeared after preflight"):
        write_new_output(destination, "replacement\n")

    assert destination.read_bytes() == sentinel


def test_hash_pinned_v1_crash_auditor_remains_unchanged():
    protocol = json.loads(
        (ROOT / "benchmarks/verifier_consistent_async_grpo_v7/protocol.json")
        .read_text(encoding="utf-8")
    )
    auditor = ROOT / "scripts/audit_sustained_crash_recovery.py"
    digest = hashlib.sha256(auditor.read_bytes()).hexdigest()

    assert digest == protocol["v7_support_sha256"][
        "scripts/audit_sustained_crash_recovery.py"
    ]


def test_scenario_auditor_rejects_nested_report_path_before_reading_run(tmp_path):
    retained = tmp_path / "scenario-run"
    retained.mkdir()
    destination = retained / "new-report.json"
    result = subprocess.run(
        [sys.executable, "-S", "scripts/audit_verifier_drift_scenarios.py",
         str(retained), "--output", str(destination)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )

    assert result.returncode != 0
    assert "inside retained input" in result.stderr
    assert not destination.exists()


def test_sustained_recovery_v2_rejects_nested_report_path_before_audit(tmp_path):
    retained = tmp_path / "crash-run"
    retained.mkdir()
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    source_audit = tmp_path / "source-audit.json"
    source_audit.write_text("{}\n", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    destination = retained / "nested" / "report.json"
    result = subprocess.run(
        [sys.executable, "-S", "scripts/audit_sustained_crash_recovery_v2.py",
         "--crash-output", str(retained), "--fixture", str(fixture),
         "--source-run", str(source_run), "--source-audit", str(source_audit),
         "--protocol", str(protocol), "--output", str(destination)],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )

    assert result.returncode != 0
    assert "inside retained input" in result.stderr
    assert not destination.exists()


def test_final_arm_wall_time_is_captured_after_final_evaluation():
    runner = (ROOT / "scripts/benchmark_verifier_drift_scenarios.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(runner)
    run_arm = next(
        node for node in module.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_arm"
    )
    final_eval_line = next(
        node.lineno for node in ast.walk(run_arm)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "capture_evaluation"
        and len(node.value.args) >= 2
        and isinstance(node.value.args[1], ast.Constant)
        and node.value.args[1].value == "final"
    )
    total_wall_line = next(
        node.lineno for node in ast.walk(run_arm)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "total_wall"
                for target in node.targets)
    )

    assert total_wall_line > final_eval_line
