"""Calibration for the authored MLSys coding pack.

The checks execute this repository's baseline and oracle fixtures through
the in-process evaluator. They do not load a model or claim agent success.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vare.environments.catalog import TaskCatalog
from vare.environments.runner import ExecutableEvaluator, Workspace
from vare.environments.spec import EngineeringTaskSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks" / "mlsys_coding_v1"
TASKS = PACK / "tasks"
EVAL_PROTOCOL = ROOT / "protocols" / "mlsys_coding_agent_eval_v1.json"
COMPARISON_PROTOCOL = ROOT / "protocols" / "mlsys_coding_policy_comparison_v1.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward = _load(PACK / "reward.py", "mlsys_coding_v1_reward")
hasher = _load(ROOT / "scripts" / "hash_mlsys_coding_pack_v1.py", "hash_mlsys_coding_pack_v1")


def _task_dirs() -> list[Path]:
    return sorted(path.parent for path in TASKS.rglob("task.json"))


def test_reward_recomputes_speedup_and_fails_closed():
    identical = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50]
    assert reward.conservative_speedup(identical, identical) < reward.SUCCESS_THRESHOLD
    separated = reward.conservative_speedup([0.40] * 7, [0.05] * 7)
    assert separated == pytest.approx(8.0)
    assert reward.shaped_reward(separated) == 1.0
    assert reward.shaped_reward(1.0) == pytest.approx(0.25)

    passed = {
        "integrity_ok": True,
        "passed": True,
        "correctness": 1.0,
        "tests": [{"returncode": 0, "timed_out": False, "output_limited": False, "stdout": ""}],
    }
    assert reward.decide(passed)["success"] is True
    assert reward.decide(passed)["reward"] == 1.0

    slow = dict(passed)
    slow["passed"] = False
    slow["correctness"] = 0.0
    slow["tests"] = [{
        "returncode": 1,
        "timed_out": False,
        "output_limited": False,
        "stdout": json.dumps({
            "measurement": reward.MEASUREMENT,
            "correctness_pass": True,
            "baseline_samples_s": identical,
            "candidate_samples_s": identical,
        }),
    }]
    slow_decision = reward.decide(slow)
    assert slow_decision["success"] is False
    assert slow_decision["reason"] == "performance_below_threshold"
    assert 0.0 < slow_decision["reward"] < 0.5
    assert slow_decision["baseline_stdev_s"] is not None

    inconsistent = dict(passed)
    inconsistent["tests"] = [{
        "returncode": 1,
        "timed_out": False,
        "output_limited": False,
        "stdout": json.dumps({
            "measurement": reward.MEASUREMENT,
            "correctness_pass": True,
            "baseline_samples_s": [0.40] * 7,
            "candidate_samples_s": [0.05] * 7,
        }),
    }]
    inconsistent["passed"] = False
    denied = reward.decide(inconsistent)
    assert denied["success"] is False
    assert denied["reward"] == 0.0
    assert denied["reason"] == "threshold_met_but_grade_failed"

    broken = dict(passed)
    broken["integrity_ok"] = False
    assert reward.decide(broken)["reason"] == "integrity"


def test_pack_hash_split_and_protocol_agree():
    manifest_text = hasher.dump_manifest(hasher.build_manifest(PACK))
    assert (PACK / "manifest.json").read_text(encoding="utf-8") == manifest_text
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
    protocol = json.loads(EVAL_PROTOCOL.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON_PROTOCOL.read_text(encoding="utf-8"))
    split = json.loads((PACK / "split.json").read_text(encoding="utf-8"))
    assert protocol["pack"]["manifest_sha256"] == manifest_sha
    assert comparison["task_pack_manifest_sha256"] == manifest_sha
    assert protocol["split"] == {"heldout": split["heldout"], "train": split["train"]}
    assert set(split["train"]).isdisjoint(split["heldout"])
    assert protocol["success_threshold"] == reward.SUCCESS_THRESHOLD
    assert protocol["repeats"] == reward.REPEATS
    assert protocol["t_crit_95_df6"] == reward.T_CRIT_95_DF6
    assert protocol["reward_saturation"] == reward.REWARD_SATURATION
    assert protocol["no_update_baseline"]["status"] == "not_run"
    assert protocol["no_update_baseline"]["measured"] is False
    assert comparison["authorization"]["status"] == "not_authorized"
    assert comparison["runner"]["implemented"] is False

    catalog = TaskCatalog.discover(TASKS)
    ids = {spec.id for spec in catalog.specs()}
    assert ids == set(split["train"]) | set(split["heldout"])
    timing_bytes = (PACK / "evaluator_timing.py").read_bytes()
    copies = list(PACK.rglob("evaluator/timing.py"))
    assert len(copies) == 2
    assert all(path.read_bytes() == timing_bytes for path in copies)
    timing = _load(PACK / "evaluator_timing.py", "mlsys_evaluator_timing")
    assert timing.REPEATS == reward.REPEATS
    assert timing.T_CRIT_95_DF6 == reward.T_CRIT_95_DF6
    assert timing.THRESHOLD == reward.SUCCESS_THRESHOLD


def test_policy_comparison_refuses_without_running():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_mlsys_coding_policy_comparison_v1.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 3
    payload = json.loads(completed.stdout)
    assert payload["status"] == "not_run"
    assert any("not implemented" in reason for reason in payload["reasons"])
    assert "torch" not in completed.stdout


@pytest.mark.parametrize("task_dir", _task_dirs(), ids=lambda path: path.name)
def test_hidden_grade_rejects_baseline_and_accepts_oracle(task_dir: Path):
    spec = EngineeringTaskSpec.load(task_dir / "task.json")
    assert spec.source.local_path == "subject"
    assert spec.metadata["grader_visibility"] == "hidden"
    assert spec.metadata["pack"] == "mlsys-coding-v1"
    assert spec.tests[0].cwd_base == "evaluator"
    assert "evaluator/" not in spec.prompt
    performance = spec.metadata["reward_kind"] == "hidden-correctness-and-repeated-timing"
    if performance:
        assert spec.tests[0].argv[1] == "grade.py"
    else:
        assert spec.metrics == ()

    outcomes = {}
    for label, use_oracle in (("baseline", False), ("oracle", True)):
        with Workspace(spec) as workspace:
            names = {path.name for path in workspace.path.rglob("*") if path.is_file()}
            assert "test_hidden.py" not in names
            assert "grade.py" not in names
            assert "timing.py" not in names
            public = subprocess.run(
                [sys.executable, "test_public.py"],
                cwd=workspace.path,
                check=False,
                capture_output=True,
                text=True,
            )
            assert public.returncode == 0, public.stderr
            if use_oracle:
                oracle_files = list((task_dir / "evaluator" / "oracle").glob("*.py"))
                assert len(oracle_files) == 1
                assert oracle_files[0].read_bytes() != (workspace.path / oracle_files[0].name).read_bytes()
                shutil.copy(oracle_files[0], workspace.path / oracle_files[0].name)
            result = ExecutableEvaluator(spec, evaluator_root=task_dir).evaluate(workspace.path)
        decision = reward.decide(result.to_dict())
        assert decision["success"] is result.passed
        outcomes[label] = (result, decision)

    baseline_result, baseline_decision = outcomes["baseline"]
    oracle_result, oracle_decision = outcomes["oracle"]
    assert baseline_result.passed is False
    assert baseline_decision["success"] is False
    assert oracle_result.passed is True
    assert oracle_decision["success"] is True
    assert oracle_decision["reward"] == 1.0
    if performance:
        assert baseline_decision["conservative_speedup"] < reward.SUCCESS_THRESHOLD
        assert baseline_decision["baseline_stdev_s"] is not None
        assert baseline_decision["candidate_stdev_s"] is not None
        assert 0.0 < baseline_decision["reward"] < 0.5
        assert oracle_decision["conservative_speedup"] >= reward.SUCCESS_THRESHOLD
        payload = json.loads(oracle_result.tests[0].stdout)
        assert len(payload["baseline_samples_s"]) == reward.REPEATS
        assert len(payload["candidate_samples_s"]) == reward.REPEATS
    else:
        assert baseline_decision["reward"] == 0.0
        assert oracle_decision["conservative_speedup"] is None
