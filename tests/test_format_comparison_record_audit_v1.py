import json
import io
from pathlib import Path
import struct
import tarfile

import pytest

from scripts import reproduce_format_comparison_record_v1 as audit


def test_independent_expression_oracle():
    for expression, expected in (("2 + 3 * 4", 14), ("(2 + 3) * 4", 20),
                                 ("12 / 3 - 6", -2), ("-(-2) + 1", 3)):
        assert audit.truth(expression) == expected


@pytest.mark.parametrize("expression", ["3 / 2", "f()", "True"])
def test_expression_oracle_rejects_unsupported_values(expression):
    with pytest.raises(ValueError):
        audit.truth(expression)


def test_strict_output_contract():
    assert audit.parsed("work\n#### -2\n\n ") == -2
    assert audit.parsed("work\u2028#### 2") == 2
    for text in ("", "#### +2", "####  2", "#### 2\nother", "answer 2"):
        assert audit.parsed(text) is None


def test_zero_tensor_witness_rejects_nonzero_b():
    path = audit.RECORD / "format-zero-adapters/20261016.safetensors"
    blob = bytearray(path.read_bytes())
    assert len(audit.zero_b_tensors(blob)) == 2
    size = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + size])
    spec = next(value for key, value in header.items() if key.endswith(".lora_b"))
    offset = 8 + size + spec["data_offsets"][0]
    blob[offset:offset + 4] = struct.pack("<f", 1.0)
    with pytest.raises(ValueError, match="nonzero"):
        audit.zero_b_tensors(blob)


def test_audit_rejects_corrupted_archive_before_reporting(tmp_path, monkeypatch):
    source = audit.RECORD
    record = tmp_path / "record"
    record.mkdir()
    manifest = json.loads((source / "record-manifest.json").read_text())
    manifest["files_sha256"] = {"progress.jsonl.gz": manifest["files_sha256"]["progress.jsonl.gz"]}
    (record / "record-manifest.json").write_text(json.dumps(manifest))
    (record / "progress.jsonl.gz").write_bytes(b"corrupt")
    monkeypatch.setattr(audit, "RECORD", record)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="artifact changed"):
        audit.audit(output)
    assert list(Path(output).iterdir()) == []


def test_optimizer_witness_rejects_wrong_step_and_nonzero_momentum():
    with tarfile.open(fileobj=io.BytesIO((audit.RECORD / "format-zero-optimizers.tar.gz").read_bytes()), mode="r:gz") as archive:
        blob = bytearray(archive.extractfile("format_reward_rl/20261016/step-0016/optimizer.safetensors").read())
    audit.zero_adam_moments(blob, 16)
    with pytest.raises(ValueError, match="step/moments"):
        audit.zero_adam_moments(blob, 15)
    size = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + size])
    spec = next(value for key, value in header.items() if key.endswith(".m"))
    offset = 8 + size + spec["data_offsets"][0]
    blob[offset:offset + 4] = struct.pack("<f", 1.0)
    with pytest.raises(ValueError, match="nonzero"):
        audit.zero_adam_moments(blob, 16)
