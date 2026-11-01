"""Contract and protocol checks for the prospective long-horizon cohort."""

from __future__ import annotations

from scripts import vare_ig_long_horizon_cohort as cohort


def test_cohort_is_development_only_and_requires_multifile_navigation():
    assert len(cohort.TASKS) == 4
    assert all(task.split == "development" for task in cohort.TASKS)
    assert all(len(task.readable_files) >= 5 for task in cohort.TASKS)
    assert all(task.editable_files == (cohort.TARGET,) for task in cohort.TASKS)
    assert all(not any("test" in path.lower() for path in task.readable_files) for task in cohort.TASKS)


def test_final_parent_grade_cases_are_disjoint_from_visible_public_examples():
    for task in cohort.TASKS:
        public_ids = {call.case_id for call in cohort.CALLS_BY_ID[task.task_id][:2]}
        hidden_ids = {call.case_id for call in cohort.CALLS_BY_ID[task.task_id][2:]}
        grader_ids = {call.case_id for call in cohort.grader_for(task.task_id).calls}
        assert public_ids.isdisjoint(hidden_ids)
        assert grader_ids == hidden_ids


def test_protocol_payload_binds_tasks_controller_and_source_hashes():
    protocol = cohort.protocol_payload()
    assert protocol["preregistered_before_model_inference"] is True
    assert protocol["split"] == "development"
    assert protocol["execution"]["task_ids"] == [task.task_id for task in cohort.TASKS]
    assert protocol["execution"]["task_sha256"] == {task.task_id: task.sha256() for task in cohort.TASKS}
    assert protocol["controller"]["config"]["reveal_target"] is False
