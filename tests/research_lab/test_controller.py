from pathlib import Path

from vare.research_lab.controller import create_run
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


def test_diagnostic_policy_recovers_after_a_rejected_hypothesis(tmp_path: Path):
    for task_id in (
        "grad_accum_scale_v1",
        "loop_invariant_score_v1",
        "descent_sign_debug_v1",
    ):
        controller, summary = _run(tmp_path, task_id, "diagnostic", "diagnostic_policy")
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
    assert summary["verified"] is False
    assert summary["success_class"] == "single_attempt_control"
    assert controller.store.read_journal()[-1]["status"] == "rejected"


def test_oracle_control_is_not_labeled_as_agent_success(tmp_path: Path):
    _controller, summary = _run(tmp_path, "grad_accum_scale_v1", "oracle", "oracle_control")
    assert summary["oracle_calibration_passed"] is True
    assert summary["verified"] is False
    assert summary["success_class"] == "oracle_control"
    assert summary["counts_as_autonomous_model_success"] is False
