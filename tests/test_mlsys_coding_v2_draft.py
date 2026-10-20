"""Draft v2 pack checks. Fixture grades are not a held-out policy result."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vare.environments.runner import ExecutableEvaluator, Workspace
from vare.environments.spec import EngineeringTaskSpec

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "benchmarks" / "mlsys_coding_v2"
V1 = ROOT / "benchmarks" / "mlsys_coding_v1"
PROTOCOL = ROOT / "protocols" / "mlsys_coding_policy_comparison_v2.json"


def _manifest_text() -> str:
    files = []
    for path in sorted(item for item in PACK.rglob("*") if item.is_file()):
        if path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "bytes": path.stat().st_size,
                "path": path.relative_to(PACK).as_posix(),
                "sha256": digest,
            }
        )
    payload = {"files": files, "pack_id": "mlsys-coding-v2", "status": "draft"}
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def test_draft_split_replaces_causal_attention_and_matches_the_protocol():
    split = json.loads((PACK / "split.json").read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    v1_split = json.loads((V1 / "split.json").read_text(encoding="utf-8"))
    assert split["status"] == "draft"
    assert protocol["status"] == "draft"
    assert protocol["heldout_evaluation_run"] is False
    assert split["train"] == v1_split["train"]
    assert "mlsys-v1-causal-attention" in v1_split["heldout"]
    assert split["heldout"] is None
    assert split["heldout_membership_frozen"] is False
    assert protocol["split"]["heldout"] is None
    assert split["excluded_from_future_heldout"] == ["mlsys-v1-causal-attention"]
    assert "mlsys-v1-causal-attention" not in split["pool_candidates_unassigned"]
    assert set(split["train"]).isdisjoint(split["pool_candidates_unassigned"])
    text = _manifest_text()
    assert (PACK / "manifest.json").read_text(encoding="utf-8") == text
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert protocol["draft_pack"]["manifest_sha256"] == digest
    hasher_path = ROOT / "scripts" / "hash_mlsys_coding_pack_v1.py"
    spec = importlib.util.spec_from_file_location("hash_v1", hasher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    v1_text = module.dump_manifest(module.build_manifest(V1))
    v1_digest = hashlib.sha256(v1_text.encode("utf-8")).hexdigest()
    assert protocol["base_pack"]["manifest_sha256"] == v1_digest
    assert protocol["base_pack"]["previous_manifest_sha256"] == (
        "18c5999f823f9a550a561a913023b87f1fc7bf5a86f481d319e4778212d97499"
    )


@pytest.mark.parametrize("task_name", ["gqa_repeat", "label_smooth", "ema_bias"])
def test_draft_fixture_calibration_is_not_a_heldout_policy_result(task_name: str):
    task_dir = PACK / "tasks" / task_name
    spec = EngineeringTaskSpec.load(task_dir / "task.json")
    assert spec.metadata["task_type"] == "correctness"
    assert spec.metadata["split"] == "unassigned"
    assert spec.metadata["grader_visibility"] == "hidden"
    outcomes = {}
    for label, use_oracle in (("baseline", False), ("oracle", True)):
        with Workspace(spec) as workspace:
            names = {path.name for path in workspace.path.rglob("*") if path.is_file()}
            assert "test_hidden.py" not in names
            public = subprocess.run(
                [sys.executable, "test_public.py"],
                cwd=workspace.path,
                check=False,
                capture_output=True,
                text=True,
            )
            assert public.returncode == 0, public.stderr
            if use_oracle:
                oracle = next((task_dir / "evaluator" / "oracle").glob("*.py"))
                shutil.copy(oracle, workspace.path / oracle.name)
            result = ExecutableEvaluator(spec, evaluator_root=task_dir).evaluate(
                workspace.path
            )
        outcomes[label] = result.passed
    assert outcomes == {"baseline": False, "oracle": True}
