import importlib.util
from pathlib import Path
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
