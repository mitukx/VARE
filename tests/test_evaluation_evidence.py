import asyncio
import json
import re

import pytest

from scripts.verifier_drift_evaluation_evidence import (
    EvaluationEvidenceJournal,
    audit_evaluation_evidence,
    utc_now,
)
from scripts import benchmark_verifier_drift_scenarios as runner


def _record(journal: EvaluationEvidenceJournal, invocation: str = "call-001") -> dict:
    started = utc_now()
    common = {
        "invocation_id": invocation,
        "role": "candidate",
        "run_id": "fixture-run",
        "scenario_id": "semantic-drift",
        "arm": "vare_async",
        "round": 0,
        "started_at_utc": started,
        "model": {"name": "fixture-model", "revision": "revision-1"},
        "checkpoint": {"policy_id": "policy-1", "file_sha256": "a" * 64},
        "verifier": {
            "name": "landing", "version": 1, "identity_sha256": "b" * 64,
            "definition": {"rule": "whole_word_ground"}, "pass_threshold": 0.5,
        },
        "evaluator": {"name": "independent-fixture", "identity_sha256": "c" * 64},
        "source_hashes": {"runner": "d" * 64, "evaluator": "e" * 64},
    }
    journal.record_evaluation_started({
        "schema": "vare.evaluation-start.v1", **common,
        "tasks": [{"task_id": "task-1", "prompt": "The bird landed on the"}],
    })
    completed = utc_now()
    return {
        "schema": "vare.evaluation-evidence.v1", **common,
        "completed_at_utc": completed,
        "aggregate": {"primary": 1.0, "n": 1},
        "tasks": [{"task_id": "task-1", "prompt": "The bird landed on the",
                   "output_text": "ground", "output_token_ids": [17],
                   "score": 1.0, "passed": True}],
        "evaluation_invocation": {"id": invocation, "started_at_utc": started,
                                  "completed_at_utc": completed},
    }


def _independent_score(definition, _prompt, output):
    if definition["rule"] != "whole_word_ground":
        raise ValueError("unknown fixture verifier")
    return float(re.search(r"(?<![A-Za-z0-9_])ground(?![A-Za-z0-9_])", output,
                           flags=re.IGNORECASE) is not None)


@pytest.mark.parametrize(
    ("point", "stage"),
    [("after_evaluation", "evaluation_durable"),
     ("during_promotion", "promotion_applied_before_commit"),
     ("before_arm_completion", "before_arm_completion")],
)
def test_completed_evaluation_survives_injected_failure(tmp_path, monkeypatch, point, stage):
    journal = EvaluationEvidenceJournal(tmp_path / point)
    evaluation = _record(journal)
    journal.record_evaluation(evaluation)
    context = {"journal": journal, "run_id": point, "scenario_id": "semantic-drift",
               "arm": "vare_async", "stage": stage}
    root = str(tmp_path / point)
    runner.ACTIVE_EVALUATION_JOURNALS[root] = context
    runner.INJECTED_FAILURE_POINTS.clear()
    monkeypatch.setenv("VARE_EXPERIMENT_FAILPOINT", point)
    if point == "during_promotion":
        state = {"policy": "policy-0"}

        async def promote(candidate_id):
            state["policy"] = candidate_id

        def mark_applied():
            context["stage"] = "promotion_applied_before_commit"

        with pytest.raises(RuntimeError, match=f"injected failure at {point}"):
            asyncio.run(runner.promote_then_inject(promote, "policy-1", mark_applied))
        assert state["policy"] == "policy-1"
    else:
        with pytest.raises(RuntimeError, match=f"injected failure at {point}"):
            runner.inject_failure(point)
    runner.persist_active_evidence_failures(RuntimeError(f"injected failure at {point}"))
    runner.ACTIVE_EVALUATION_JOURNALS.pop(root)

    result = audit_evaluation_evidence(tmp_path / point, scorer=_independent_score)
    assert result["pass"] is True
    assert result["evaluation_records"] == 1
    assert result["failure_records"] == 1
    assert result["incomplete_evaluations"] == []
    retained = json.loads((tmp_path / point / "evaluation-call-001.json").read_text())
    assert retained["tasks"][0]["prompt"] == "The bird landed on the"
    assert retained["tasks"][0]["output_text"] == "ground"


def test_evaluation_evidence_is_non_overwriting_and_tamper_evident(tmp_path):
    journal = EvaluationEvidenceJournal(tmp_path / "journal")
    journal.record_evaluation(_record(journal))
    original = (tmp_path / "journal" / "evaluation-call-001.json").read_bytes()
    with pytest.raises(ValueError, match="duplicate started evaluation id"):
        journal.record_evaluation(_record(journal))
    assert (tmp_path / "journal" / "evaluation-call-001.json").read_bytes() == original

    payload = json.loads(original)
    payload["tasks"][0]["output_text"] = "sky"
    (tmp_path / "journal" / "evaluation-call-001.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="self-hash mismatch"):
        audit_evaluation_evidence(tmp_path / "journal")


def test_started_but_incomplete_evaluation_is_retained_as_partial(tmp_path):
    journal = EvaluationEvidenceJournal(tmp_path / "partial")
    _record(journal, invocation="partial-001")
    started_path = tmp_path / "partial" / "started-partial-001.json"
    # _record prepares the durable start; model evaluation is interrupted before
    # a completion record can be written.
    started = json.loads(started_path.read_text())
    journal.record_failure(failure_type="TimeoutError", message="interrupted decode",
                           stage="evaluation:baseline:partial-001", run_id="fixture-run",
                           scenario_id="semantic-drift", arm="vare_async")
    result = audit_evaluation_evidence(tmp_path / "partial")
    assert result["evaluation_records"] == 0
    assert result["incomplete_evaluations"] == ["partial-001"]
    failure = next((tmp_path / "partial").glob("failure-*.json"))
    failure_record = json.loads(failure.read_text())
    assert failure_record["started_invocation_ids"] == ["partial-001"]
    assert started["tasks"][0]["prompt"] == "The bird landed on the"


def test_prospective_manifest_binds_runner_and_preserves_v7_3_identity():
    runner.validate_retention_manifest()
