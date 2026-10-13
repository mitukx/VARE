import json
import sqlite3

import pytest

from vare.integrations.rvl import RVLTokenReplayReader
from vare.rvl_failure_triage import adjudicate_replay, replay_manifest, load_strict_json
from test_rvl_integration import _make_replay


def _setup(tmp_path):
    path = tmp_path / "replay.sqlite"
    _make_replay(path)
    manifest = replay_manifest(RVLTokenReplayReader(path), current_policy_version=6,
                               current_verifier_version=5)
    audit = {
        "selection_sha256": manifest.selection_sha256,
        "rows": [{"sample_id": x["sample_id"], "response_sha256": x["response_sha256"],
                  "format_valid": True, "trusted_pass": x["proxy_pass"]}
                 for x in manifest.rows],
    }
    bindings = {
        "run_id": "test", "training_cohort_sha256": "a" * 64,
        "heldout_cohort_sha256": "b" * 64, "proxy_grader_sha256": "c" * 64,
        "trusted_grader_sha256": "d" * 64, "active_tokens": 10, "clipped_tokens": 1,
    }
    return path, manifest, audit, bindings


def test_manifest_is_read_only_and_does_not_conflate_prompts(tmp_path):
    path, manifest, _, _ = _setup(tmp_path)
    assert len(manifest.rows) == 2
    assert len({r["sample_id"] for r in manifest.rows}) == 2
    assert manifest.reward_groups == ()
    assert manifest.skipped_noncomparison_groups == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM groups").fetchone()[0] == 2


def test_complete_external_audit_reports_insufficient_evidence(tmp_path):
    _, manifest, audit, bindings = _setup(tmp_path)
    report = adjudicate_replay(manifest, audit, bindings)
    assert report["triage"]["status"] == "insufficient_evidence"
    assert report["triage"]["counts"]["heldout_tasks"] == 2
    assert "unverified" in report["limitations"][1]


@pytest.mark.parametrize("tamper, error", [
    ("selection", "does not match"),
    ("response", "response hash mismatch"),
    ("missing", "incomplete independent"),
    ("duplicate", "duplicate audit"),
    ("boolean", "verdicts must be booleans"),
    ("cohort", "cohort digests must differ"),
])
def test_adversarial_audit_fails_closed(tmp_path, tamper, error):
    _, manifest, audit, bindings = _setup(tmp_path)
    if tamper == "selection":
        audit["selection_sha256"] = "f" * 64
    elif tamper == "response":
        audit["rows"][0]["response_sha256"] = "e" * 64
    elif tamper == "missing":
        audit["rows"].pop()
    elif tamper == "duplicate":
        audit["rows"].append(dict(audit["rows"][0]))
    elif tamper == "boolean":
        audit["rows"][0]["trusted_pass"] = 1
    else:
        bindings["heldout_cohort_sha256"] = "a" * 64
    with pytest.raises(ValueError, match=error):
        adjudicate_replay(manifest, audit, bindings)


def test_replay_change_invalidates_external_selection(tmp_path):
    path, manifest, audit, bindings = _setup(tmp_path)
    with sqlite3.connect(path) as db:
        raw = db.execute("SELECT payload FROM groups WHERE id='g1'").fetchone()[0]
        payload = json.loads(raw)
        payload[0]["generation"]["response"] = "altered"
        db.execute("UPDATE groups SET payload=? WHERE id='g1'", (json.dumps(payload),))
    new = replay_manifest(RVLTokenReplayReader(path), current_policy_version=6,
                          current_verifier_version=5)
    assert new.selection_sha256 != manifest.selection_sha256
    with pytest.raises(ValueError, match="does not match"):
        adjudicate_replay(new, audit, bindings)


def test_external_labels_are_not_inferred_from_rvl_proxy(tmp_path):
    _, manifest, audit, bindings = _setup(tmp_path)
    audit["rows"][0]["trusted_pass"] = not audit["rows"][0]["trusted_pass"]
    report = adjudicate_replay(manifest, audit, bindings)
    assert report["triage"]["counts"]["false_accepts"] == 1


def test_strict_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_strict_json(path)
