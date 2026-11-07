from pathlib import Path

from vare.research_lab.controller import create_run
from vare.research_lab.isolation import ISOLATION_UNVERIFIED
from vare.research_lab.spec import load_catalog, suite_manifest


def _run(tmp_path: Path, task_id: str, proposer: str, kind: str):
    task = load_catalog()[task_id]
    controller = create_run(
        tmp_path / f"{task_id}-{proposer}",
        task,
        run_id=f"{task_id}-{proposer}",
        proposer_name=proposer,
        proposer_kind=kind,
        suite_sha256=suite_manifest()["manifest_sha256"],
    )
    summary = controller.run()
    return controller, summary


def _assert_grading_result_or_unavailable(controller, summary) -> None:
    records = controller.store.read_journal()
    first_public = records[0].get("public") or {}
    if first_public.get("classification") == ISOLATION_UNVERIFIED:
        assert summary["verified"] is False
        assert summary["counts_as_autonomous_model_success"] is False
        assert not any(row["status"] == "accepted" for row in records)
        assert all(
            row.get("verification") is None
            or row["verification"].get("classification") == ISOLATION_UNVERIFIED
            for row in records
        )
        return
    assert summary["verified"] is True


def test_diagnostic_policy_recovers_after_a_rejected_hypothesis(tmp_path: Path):
    for task_id in (
        "grad_accum_scale_v1",
        "loop_invariant_score_v1",
        "descent_sign_debug_v1",
    ):
        controller, summary = _run(tmp_path, task_id, "diagnostic", "diagnostic_policy")
        _assert_grading_result_or_unavailable(controller, summary)
        if not summary["verified"]:
            assert summary["counts_as_autonomous_model_success"] is False
            assert summary["success_class"] == "not_verified"
            continue
        assert summary["success_class"] == "diagnostic_policy"
        assert summary["verified"] is True
        assert summary["counts_as_autonomous_model_success"] is False
        assert summary["model_backed"] is False
        assert summary["rejected_hypotheses"]
        statuses = [row["status"] for row in controller.store.read_journal()]
        assert "rejected" in statuses
        assert statuses[-1] == "accepted"


def test_first_hypothesis_control_fails_closed(tmp_path: Path):
    controller, summary = _run(
        tmp_path,
        "descent_sign_debug_v1",
        "first-hypothesis",
        "single_attempt_control",
    )
    _assert_grading_result_or_unavailable(controller, summary)
    if not summary["verified"]:
        assert summary["success_class"] == "single_attempt_control"
        return
    assert summary["verified"] is False
    assert summary["success_class"] == "single_attempt_control"
    assert controller.store.read_journal()[-1]["status"] == "rejected"


def test_oracle_control_is_not_labeled_as_agent_success(tmp_path: Path):
    controller, summary = _run(tmp_path, "grad_accum_scale_v1", "oracle", "oracle_control")
    _assert_grading_result_or_unavailable(controller, summary)
    if not summary["verified"]:
        assert summary["oracle_calibration_passed"] is False
        assert summary["success_class"] == "oracle_control"
        assert summary["counts_as_autonomous_model_success"] is False
        return
    assert summary["oracle_calibration_passed"] is True
    assert summary["verified"] is False
    assert summary["success_class"] == "oracle_control"
    assert summary["counts_as_autonomous_model_success"] is False
