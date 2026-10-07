import json
from dataclasses import asdict

import pytest

from vare.evidence import CapabilityScore, HashChainLedger, artifact_manifest
from vare.protocol import ProtocolLock


def test_hash_chain_ledger_detects_tampering(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = HashChainLedger(path)
    ledger.append("one", {"x": 1})
    ledger.append("two", {"x": 2})
    recovered = HashChainLedger(path)
    assert recovered.seq == 2
    assert recovered.head == ledger.head
    rows = path.read_text().splitlines()
    row = json.loads(rows[0])
    row["payload"]["x"] = 999
    rows[0] = json.dumps(row)
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        HashChainLedger(path)


def test_protocol_lock_roundtrip_and_score(tmp_path):
    lock = ProtocolLock.create(
        name="l2",
        config={"seed": 17},
        hypotheses=["VARE > fixed curriculum"],
        primary_metrics=["held_out_accuracy_delta"],
        acceptance={"min_delta": 0.01},
    )
    path = tmp_path / "lock.json"
    lock.write(path)
    assert ProtocolLock.read(path).sha256 == lock.sha256
    score = CapabilityScore(before=0.5, after=0.6, gpu_hours=2.0)
    assert score.gain == pytest.approx(0.1)
    assert score.gain_per_gpu_hour == pytest.approx(0.05)
    manifest = artifact_manifest([path])
    assert len(manifest[0].sha256) == 64
    assert asdict(manifest[0])["bytes"] > 0
