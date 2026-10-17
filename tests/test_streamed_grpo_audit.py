import importlib.util
import hashlib
from pathlib import Path
import subprocess
import pytest


def test_numerical_artifact_mismatch_and_missing_tensor_fail_closed(monkeypatch):
    torch=pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec=importlib.util.spec_from_file_location("stream_audit",Path("scripts/audit_streamed_grpo_cpu.py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference={"weight":torch.tensor([.2,.3])}
    assert module.tensor_error(reference,reference)==0
    with pytest.raises(ValueError,match="coverage"):
        module.tensor_error({},reference)
    with pytest.raises(ValueError,match="numerical tensor mismatch"):
        module.tensor_error({"weight":torch.tensor([.2,.4])},reference)


def test_snapshot_runtime_digest_must_match_archived_source(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec=importlib.util.spec_from_file_location("stream_audit_source",Path("scripts/audit_streamed_grpo_cpu.py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source="a"*64
    module.verify_snapshot_source({"source_sha256":source},{"src/vare/trajectory_store.py":source})
    with pytest.raises(ValueError,match="snapshot runtime source"):
        module.verify_snapshot_source({"source_sha256":"b"*64},{"src/vare/trajectory_store.py":source})


def test_snapshot_store_loader_executes_the_pinned_git_tree(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec=importlib.util.spec_from_file_location("stream_audit_pinned",Path("scripts/audit_streamed_grpo_cpu.py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root=Path(__file__).resolve().parents[1]
    revision=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    source=subprocess.check_output(
        ["git","-C",str(root),"show",f"{revision}:src/vare/trajectory_store.py"]
    )
    expected=hashlib.sha256(source).hexdigest()

    pinned,cleanup=module.load_snapshot_store(revision,expected)
    try:
        assert pinned.__package__.startswith("_vare_snapshot_")
        assert callable(pinned.audit_trajectory_store)
    finally:
        cleanup()

    with pytest.raises(ValueError,match="recorded revision and archived trajectory store differ"):
        module.load_snapshot_store(revision,"f"*64)
