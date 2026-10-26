import hashlib
import json

import pytest

from scripts.verify_coding_agent_evidence_manifest_v1 import manifest_entries, verify


@pytest.mark.parametrize("manifest", [{}, {"files": {}}, {"files": []}, {"files": None}])
def test_empty_inventory_is_rejected(manifest, tmp_path):
    (tmp_path / "evidence-manifest.json").write_text(json.dumps(manifest))
    verified, errors = verify(tmp_path)
    assert verified == 0
    assert errors


@pytest.mark.parametrize("array", [False, True])
def test_nonempty_inventory_verifies_bytes_and_rejects_corruption(tmp_path, array):
    payload = b"actual evidence\n"
    (tmp_path / "record.json").write_bytes(payload)
    entry = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    files = [{"path": "record.json", **entry}] if array else {"record.json": entry}
    (tmp_path / "evidence-manifest.json").write_text(json.dumps({"files": files}))
    assert verify(tmp_path) == (1, [])
    (tmp_path / "record.json").write_bytes(b"corrupt")
    verified, errors = verify(tmp_path)
    assert verified == 0 and errors


def test_duplicate_paths_are_rejected():
    entries, errors = manifest_entries({"files": [{"path": "x"}, {"path": "x"}]})
    assert len(entries) == 1 and errors
