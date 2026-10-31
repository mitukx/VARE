from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "sprint_xiii" / "audit_online_actor_v2.py"
spec = importlib.util.spec_from_file_location("sprint_xiii_audit_online_actor_v2", SCRIPT)
auditor = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(auditor)


def test_group_enumerator_excludes_acknowledgement_receipts(tmp_path: Path) -> None:
    groups = tmp_path / "groups"
    groups.mkdir()
    (groups / "group-0.json").write_text("{}")
    (groups / "group-0.ack.json").write_text("{}")
    (groups / "fresh-policy-probe.json").write_text("{}")
    (groups / "fresh-policy-probe.ack.json").write_text("{}")
    assert [p.name for p in auditor.data_group_paths(tmp_path)] == [
        "fresh-policy-probe.json",
        "group-0.json",
    ]


def test_score_enumerator_excludes_acknowledgement_receipts(tmp_path: Path) -> None:
    scores = tmp_path / "scores"
    scores.mkdir()
    (scores / "group-0-v0.json").write_text('{"verifier_version":0}')
    (scores / "group-0-v0.ack.json").write_text('{"score_sha256":"abc"}')
    assert [p.name for p in auditor.data_score_paths(tmp_path, "group-0")] == [
        "group-0-v0.json"
    ]
