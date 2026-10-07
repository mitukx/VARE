from __future__ import annotations

from pathlib import Path
import subprocess

from vare.environments import EngineeringTaskSpec, ExecutableEvaluator, Workspace


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "benchmarks/smoke/stable_logsumexp/task.json"
ORACLE = ROOT / "tests/fixtures/oracle/fix_stable_logsumexp.py"


def test_task_spec_roundtrip_and_baseline_fails():
    spec = EngineeringTaskSpec.load(SPEC)
    assert spec.id == "smoke-stable-logsumexp"
    with Workspace(spec) as ws:
        result = ExecutableEvaluator(spec).evaluate(ws.path)
    assert not result.passed
    assert result.integrity_ok
    assert result.correctness < 1.0


def test_oracle_patch_passes_and_has_diff():
    spec = EngineeringTaskSpec.load(SPEC)
    with Workspace(spec) as ws:
        subprocess.run(["python", str(ORACLE), str(ws.path)], check=True)
        result = ExecutableEvaluator(spec).evaluate(ws.path)
    assert result.passed
    assert result.correctness == 1.0
    assert "subject.py" in result.workspace_diff


def test_protected_evaluator_tampering_fails_closed():
    spec = EngineeringTaskSpec.load(SPEC)
    with Workspace(spec) as ws:
        (ws.path / "test_subject.py").write_text("def test_fake(): assert True\n", encoding="utf-8")
        result = ExecutableEvaluator(spec).evaluate(ws.path)
    assert not result.passed
    assert not result.integrity_ok
    assert any("protected file modified" in x for x in result.integrity_failures)


def test_relative_local_source_is_portable_and_not_serialized(monkeypatch, tmp_path):
    spec = EngineeringTaskSpec.load(SPEC)
    assert spec.to_dict()["source"]["local_path"] == "."
    assert "_source_base" not in spec.to_dict()
    monkeypatch.chdir(tmp_path)
    with Workspace(spec) as ws:
        assert (ws.path / "subject.py").is_file()
