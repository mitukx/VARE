import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

EXPECTED_V31_PROTOCOL_HASH = "344a28a6e0b66eec362512fc43b9c42cf65bd9bf45bbb7578746326c96ecbfed"


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


def test_snapshot_store_loader_executes_the_pinned_archive_without_git_history(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec=importlib.util.spec_from_file_location("stream_audit_pinned",Path("scripts/audit_streamed_grpo_cpu.py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected="e7ee76826b3aed19f1c381b95dbc9c5f17c0d78cecf58f1dcb5d60514fb08179"
    monkeypatch.setattr(module.subprocess,"check_output",
                        lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError("Git history must not be consulted")))

    for revision,(archive,archive_hash) in module.PINNED_VARE_SOURCES.items():
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_hash
        pinned,cleanup=module.load_snapshot_store(revision,expected)
        try:
            assert pinned.__package__.startswith("_vare_snapshot_")
            assert callable(pinned.audit_trajectory_store)
        finally:
            cleanup()

    with pytest.raises(ValueError,match="recorded revision and archived trajectory store differ"):
        module.load_snapshot_store("e4155f0d6d7b6781a6d0c3fdcf13a9e3ef1342b7","f"*64)

    damaged=tmp_path/"source.tar.xz"
    source_archive=module.PINNED_VARE_SOURCES["e4155f0d6d7b6781a6d0c3fdcf13a9e3ef1342b7"][0]
    damaged.write_bytes(source_archive.read_bytes()+b"tamper")
    with pytest.raises(ValueError,match="source archive hash mismatch"):
        module.load_snapshot_store("e4155f0d6d7b6781a6d0c3fdcf13a9e3ef1342b7",expected,damaged)


def test_run_protocol_is_preserved_and_later_protocol_is_versioned():
    root=Path(__file__).resolve().parents[1]
    run_path=root/"benchmarks/streamed_grpo_cpu_v3/protocol.json"
    next_path=root/"benchmarks/streamed_grpo_cpu_v3/protocol-v3.1.json"
    run_protocol=json.loads(run_path.read_text())
    next_protocol=json.loads(next_path.read_text())

    assert hashlib.sha256(run_path.read_bytes()).hexdigest() == (
        "7954ac550a302335bf6c22e6ec009f24eb03d55063ce8441d137c8591d77a9d4"
    )
    assert run_protocol["schema"] == "vare.streamed-grpo-systems.v3"
    assert next_protocol["schema"] == "vare.streamed-grpo-systems.v3.1"
    assert hashlib.sha256(next_path.read_bytes()).hexdigest() == EXPECTED_V31_PROTOCOL_HASH
    assert next_protocol["shift_reference_families"] == ["train"]
    runner_hash=hashlib.sha256((root/"scripts/validate_streamed_grpo_cpu.py").read_bytes()).hexdigest()
    assert next_protocol["runner_sha256"] == runner_hash
