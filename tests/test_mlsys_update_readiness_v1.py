from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.audit_mlsys_update_readiness_v1 import audit_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/mlsys_actual_model_update_no_go_v1.lock.json"


def _copy_audit_inputs(destination: Path) -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    for pack in protocol["observed_inventory"]["packs"]:
        source = ROOT / pack["pack_root"]
        target = destination / pack["pack_root"]
        shutil.copytree(source, target)
    return protocol


def test_frozen_inventory_reproduces_readiness_no_go():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    report = audit_protocol(protocol, ROOT)

    assert report["audit_passed"] is True
    assert report["status"] == "NO_GO"
    assert report["eligible_unique_heldout_tasks"] == 0
    assert report["maximum_possible_unique_heldout_tasks"] == 14
    assert report["inventory_alone_fails_minimum"] is True
    assert report["required_unique_heldout_tasks"] == 15
    assert report["unique_task_definitions_by_sha256"] == 14


def test_task_mutation_fails_readiness_audit(tmp_path: Path):
    protocol = _copy_audit_inputs(tmp_path)
    task_path = tmp_path / protocol["observed_inventory"]["packs"][0]["tasks"][0][
        "task_json_path"
    ]
    task_path.write_text(task_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = audit_protocol(protocol, tmp_path)

    assert report["audit_passed"] is False
    assert any("task hash mismatch" in error for error in report["errors"])


def test_grader_file_mutation_fails_manifest_audit(tmp_path: Path):
    protocol = _copy_audit_inputs(tmp_path)
    grader_path = tmp_path / "benchmarks/mlsys_coding_v3/tasks/mlsys-v3-ema-bias/evaluator/test_hidden.py"
    grader_path.write_text(grader_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = audit_protocol(protocol, tmp_path)

    assert report["audit_passed"] is False
    assert any("manifest file hash mismatch" in error for error in report["errors"])

