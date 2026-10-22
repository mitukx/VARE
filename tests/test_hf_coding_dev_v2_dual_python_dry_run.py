from __future__ import annotations

import json

import pytest

from scripts.dry_run_hf_coding_dev_v2_dual_python import (
    GRADER_PYTHON,
    MLX_PYTHON,
    ROOT,
    build_dual_python_plan,
    plan_if_ready,
    validate,
    verify_locked_inputs,
)


def test_frozen_rollout_plan_binds_distinct_interpreters() -> None:
    protocol = json.loads(
        (ROOT / 'experiments/hf_coding_dev_inference_v1/protocol-v2.json').read_text()
    )

    plan = build_dual_python_plan(protocol)

    assert len(plan) == 12
    assert all(item['agent']['argv'][0] == str(MLX_PYTHON) for item in plan)
    assert all(item['grader']['argv'][0] == str(GRADER_PYTHON) for item in plan)
    assert all(item['agent']['env']['DOCKER_CONTEXT'] == 'colima' for item in plan)
    assert all(item['grader']['env']['DOCKER_CONTEXT'] == 'colima' for item in plan)
    assert all('{SHARED_DEADLINE_EPOCH}' in item['agent']['argv'] for item in plan)
    assert {item['seed'] for item in plan} == {303, 404, 505}
    assert all('--do-sample' in item['agent']['argv'] for item in plan)
    assert protocol['task_cohort']['prior_model_exposure']['status'] == 'previously_model_exposed'
    assert all(status == 'previously_model_exposed_not_fresh'
               for status in protocol['task_cohort']['task_exposure_status'].values())
    assert protocol['launch']['grader_argv_template'][0] == str(GRADER_PYTHON)


def test_dry_run_planner_refuses_execution_mode() -> None:
    with pytest.raises(RuntimeError, match='only supports --dry-run'):
        validate(dry_run=False)


def test_failed_prerequisite_returns_no_launcher_commands() -> None:
    protocol = json.loads(
        (ROOT / 'experiments/hf_coding_dev_inference_v1/protocol-v2.json').read_text()
    )

    assert plan_if_ready(protocol, [{'name': 'torch', 'passed': False}]) == []


def test_v2_protocol_and_source_hash_locks_match() -> None:
    protocol_path = ROOT / 'experiments/hf_coding_dev_inference_v1/protocol-v2.json'
    lock_path = ROOT / 'experiments/hf_coding_dev_inference_v1/protocol-v2.lock.json'
    protocol = json.loads(protocol_path.read_text())
    lock = json.loads(lock_path.read_text())

    assert all(check['passed'] for check in verify_locked_inputs(protocol, lock))
