from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/audit_mlsys_update_readiness_v2.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location("readiness_v2_auditor", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_readiness_audit_binds_exact_main_commit_and_tree() -> None:
    module = _load_auditor()
    lock = json.loads(module.LOCK.read_text(encoding="utf-8"))
    report = module.run_audit(lock)

    assert report["audit_passed"] is True
    assert report["audited_vare_commit"] == lock["source"]["commit"]
    assert report["audited_vare_tree"] == lock["source"]["tree"]
    assert report["source_inputs_match_pinned_commit"] is True
    assert report["status"] == "NO_GO"
    assert report["eligible_unique_heldout_tasks"] == 0
    assert report["required_unique_heldout_tasks"] == 15


def test_v2_readiness_audit_rejects_wrong_pinned_tree() -> None:
    module = _load_auditor()
    lock = json.loads(module.LOCK.read_text(encoding="utf-8"))
    lock["source"]["tree"] = "0" * 40

    with pytest.raises(ValueError, match="source tree does not match"):
        module.verify_source_binding(lock)
