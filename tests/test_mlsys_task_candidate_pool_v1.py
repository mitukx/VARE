from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/mlsys-task-candidate-survey-v1"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_pool_audit_rejects_nested_audit_named_rerun_file(tmp_path: Path) -> None:
    module = _load_script(
        "audit_mlsys_task_candidate_pool_v1_test",
        ROOT / "scripts/audit_mlsys_task_candidate_pool_v1.py",
    )
    bundle = tmp_path / "frozen-evidence"
    shutil.copytree(EVIDENCE, bundle)
    extra = bundle / "run-2/audit.json"
    extra.parent.mkdir()
    extra.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="frozen evidence inventory differs"):
        module.audit(bundle)


@pytest.mark.parametrize("entry_kind", ["dangling_file", "directory"])
def test_candidate_pool_audit_rejects_unmanifested_symlinks(
    tmp_path: Path, entry_kind: str,
) -> None:
    module = _load_script(
        f"audit_mlsys_task_candidate_pool_v1_symlink_{entry_kind}",
        ROOT / "scripts/audit_mlsys_task_candidate_pool_v1.py",
    )
    bundle = tmp_path / "frozen-evidence"
    shutil.copytree(EVIDENCE, bundle)
    if entry_kind == "dangling_file":
        (bundle / "run-1/unmanifested.json").symlink_to(tmp_path / "missing.json")
    else:
        target = tmp_path / "rerun"
        target.mkdir()
        (target / "payload.json").write_text("{}\n", encoding="utf-8")
        (bundle / "run-2").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink in frozen evidence inventory"):
        module.audit(bundle)


def test_candidate_pool_audit_does_not_rewrite_protocol_commitment(tmp_path: Path) -> None:
    module = _load_script(
        "audit_mlsys_task_candidate_pool_v1_manifest_test",
        ROOT / "scripts/audit_mlsys_task_candidate_pool_v1.py",
    )
    bundle = tmp_path / "frozen-evidence"
    shutil.copytree(EVIDENCE, bundle)
    manifest = bundle / "evidence-manifest.json"
    original = manifest.read_bytes()

    assert module.audit(bundle)["audit_passed"] is True
    assert manifest.read_bytes() == original


def test_candidate_pool_audit_rejects_protocol_edit(tmp_path: Path, monkeypatch) -> None:
    module = _load_script(
        "audit_mlsys_task_candidate_pool_v1_protocol_test",
        ROOT / "scripts/audit_mlsys_task_candidate_pool_v1.py",
    )
    bundle = tmp_path / "frozen-evidence"
    shutil.copytree(EVIDENCE, bundle)
    changed_protocol = tmp_path / "changed-protocol.json"
    changed_protocol.write_bytes(module.PROTOCOL.read_bytes() + b"\n")
    monkeypatch.setattr(module, "PROTOCOL", changed_protocol)

    with pytest.raises(ValueError, match="frozen protocol hash differs"):
        module.audit(bundle)


def test_incomplete_candidate_collection_exits_nonzero_and_retains_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(
        "collect_mlsys_task_candidate_pool_v1_test",
        ROOT / "scripts/collect_mlsys_task_candidate_pool_v1.py",
    )
    output = tmp_path / "partial-run"
    monkeypatch.setattr(module, "CANDIDATES", {"example/project": [7]})

    def fail_collection(_repo: str, _number: int) -> dict:
        raise TimeoutError("fixture timeout")

    monkeypatch.setattr(module, "collect_one", fail_collection)
    monkeypatch.setattr(sys, "argv", ["collect_mlsys_task_candidate_pool_v1.py", "--output", str(output)])

    assert module.main() == 1
    record = json.loads((output / "candidate-pool.json").read_text(encoding="utf-8"))
    assert record["candidate_count_requested"] == 1
    assert record["candidate_count_captured"] == 0
    assert len(record["failures"]) == 1
