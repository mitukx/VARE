"""Regression tests for the reproducible CPU end-to-end benchmark harness."""

import asyncio
import pytest

from benchmarks.e2e_rl_system.runner import _child_trial, _run_fault_case, main, run_experiment


def test_paired_perf_arms_use_same_seeded_workload_and_run_complete_pipeline():
    baseline = _child_trial("baseline", 20261001, 2, 64, 128)
    vare = _child_trial("vare", 20261001, 2, 64, 128)

    assert baseline["rollout_task_ids_sha256"] == vare["rollout_task_ids_sha256"]
    assert [row["eval_task_set_sha256"] for row in baseline["round_results"]] == [
        row["eval_task_set_sha256"] for row in vare["round_results"]
    ]
    assert len({row["eval_task_set_sha256"] for row in vare["round_results"]}) == 2
    assert baseline["rollout_count_per_round"] == vare["rollout_count_per_round"] == 64
    assert all(row["admitted"] == 64 for row in baseline["round_results"])
    assert all(row["admitted"] == 64 for row in vare["round_results"])
    assert baseline["active_policy"][1] == vare["active_policy"][1] == 2
    assert baseline["peak_rss_bytes"] > 0
    assert vare["peak_python_bytes"] > 0
    assert vare["evaluation_seconds"] > 0
    assert vare["orchestration_replay_residual_seconds"] > 0


def test_stale_rollouts_are_rejected_before_training_only_in_vare_arm():
    baseline = asyncio.run(_run_fault_case("baseline", "stale_rollouts", 20261991))
    vare = asyncio.run(_run_fault_case("vare", "stale_rollouts", 20261991))

    assert baseline["invalid_update_accepted"] is True
    assert baseline["outcome"]["dropped_stale_rollouts"] == 0
    assert baseline["outcome"]["trained_on_stale_rows"] > 0
    assert vare["invalid_update_accepted"] is False
    assert vare["outcome"]["dropped_stale_rollouts"] == 64
    assert vare["outcome"]["trained_on_stale_rows"] == 0
    assert vare["outcome"]["max_policy_lag"] == 3
    assert vare["outcome"]["max_verifier_lag"] == 4


def test_evaluator_failure_restores_and_discards_candidate_in_vare():
    baseline = asyncio.run(_run_fault_case("baseline", "evaluator_failure", 20261992))
    vare = asyncio.run(_run_fault_case("vare", "evaluator_failure", 20261992))

    assert baseline["recovered_exactly_after_fault"] is False
    assert baseline["outcome"]["candidate_discarded"] is False
    assert vare["recovered_exactly_after_fault"] is True
    assert vare["outcome"]["candidate_discarded"] is True


def test_interrupted_promotion_is_fenced_without_false_success():
    baseline = asyncio.run(_run_fault_case("baseline", "interrupted_promotion", 20261993))
    vare = asyncio.run(_run_fault_case("vare", "interrupted_promotion", 20261993))

    assert baseline["outcome"]["recovery_fenced"] is False
    assert vare["outcome"]["recovery_fenced"] is True
    assert vare["outcome"]["reentry_error"] == "PromotionRecoveryRequired"
    # Publication can have mutated the live policy before the interrupted
    # acknowledgment. VARE correctly quarantines this unresolved state; it does
    # not claim exact rollback without an independently reconciled journal.
    assert vare["outcome"]["exact_rollback"] is False


def test_real_worker_process_exit_is_observed_and_round_can_resume():
    baseline = asyncio.run(_run_fault_case("baseline", "worker_crash", 20261994))
    vare = asyncio.run(_run_fault_case("vare", "worker_crash", 20261994))

    assert "status 73" in baseline["recovery_error"]
    assert "status 73" in vare["recovery_error"]
    assert baseline["recovered_exactly_after_fault"] is True
    assert vare["recovered_exactly_after_fault"] is True
    assert baseline["outcome"]["retry_completed"] is True
    assert vare["outcome"]["retry_completed"] is True


def test_concurrent_rounds_are_serialized_by_vare_control_plane():
    baseline = asyncio.run(_run_fault_case("baseline", "concurrent_execution", 20261995))
    vare = asyncio.run(_run_fault_case("vare", "concurrent_execution", 20261995))

    assert baseline["outcome"]["duplicate_candidate_ids"] is True
    assert vare["outcome"]["duplicate_candidate_ids"] is False
    assert baseline["outcome"]["concurrent_rounds_completed"] == 2
    assert vare["outcome"]["concurrent_rounds_completed"] == 2


@pytest.mark.parametrize("args", [
    ["--seeds", "2"], ["--rounds", "2"], ["--fault-seeds", "1"],
    ["--skip-faults"],
    ["--child-trial", "vare", "20261001", "2", "64", "256"],
])
def test_locked_protocol_rejects_reduced_cli_runs_before_writing(tmp_path, args):
    output = tmp_path / "must-not-exist"
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output), *args])
    assert error.value.code == 2
    assert not output.exists()


def test_direct_campaign_cannot_publish_reduced_protocol_evidence(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="locked v1"):
        run_experiment(output, (20261001, 20261002), 2)
    assert not output.exists()
