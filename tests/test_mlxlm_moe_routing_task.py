"""Contract tests for the pinned MLX-LM MoE routing task and grader lock."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from vare.agent_task_contract import assert_train_only, load_agent_task


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "benchmarks/mlsys_repo_eval_v1/mlxlm_moe_routing_gradient"
GRADER = TASK_ROOT / "evaluator/grade.py"
CALIBRATION_RUNNER = ROOT / "scripts/calibrate_mlxlm_moe_routing_grader.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grader_module():
    spec = importlib.util.spec_from_file_location("mlxlm_moe_task_grader", GRADER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_contract_is_train_only_and_scoped_to_three_model_files():
    contract = load_agent_task(TASK_ROOT)
    assert contract.task_id == "mlxlm-moe-routing-backward-train-01"
    assert contract.split == "train"
    assert contract.allowed_files == (
        "mlx_lm/models/granitemoe.py",
        "mlx_lm/models/granitemoehybrid.py",
        "mlx_lm/models/lfm2_moe.py",
    )
    assert_train_only(contract, {"train": [contract.task_id], "heldout": []})


def test_candidate_prompt_does_not_disclose_grader_or_patch_identifier():
    prompt = (TASK_ROOT / "TASK.md").read_text(encoding="utf-8").lower()
    task = json.loads((TASK_ROOT / "task.json").read_text(encoding="utf-8"))
    prompt += task["prompt"].lower()
    assert "test_granitemoe_backward" not in prompt
    assert "test_granitemoehybrid_backward" not in prompt
    assert "test_lfm2_moe_backward" not in prompt
    assert "stop_gradient" not in prompt
    assert "1795" not in prompt


def test_protocol_lock_binds_task_definition_and_independent_grader():
    lock = json.loads((TASK_ROOT / "protocol.lock.json").read_text(encoding="utf-8"))
    assert lock["status"] == "FROZEN_BEFORE_AGENT_ATTEMPTS"
    assert lock["protocol_id"] == "mlxlm-moe-routing-gradient-independent-grader-v6"
    assert lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v5.json"
    )
    prior_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v3.json").read_text(encoding="utf-8")
    )
    v2_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v2.json").read_text(encoding="utf-8")
    )
    assert prior_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v2.json"
    )
    prior_v4_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v4.json").read_text(encoding="utf-8")
    )
    prior_v5_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v5.json").read_text(encoding="utf-8")
    )
    assert prior_v4_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v3.json"
    )
    assert prior_v5_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v4.json"
    )
    assert v2_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v1.json"
    )
    assert (
        "Not independently verified"
        in lock["exposure"]["researcher_source_qualification"]
    )
    assert lock["task_definition"]["task_json_sha256"] == _sha256(
        TASK_ROOT / "task.json"
    )
    assert lock["task_definition"]["task_md_sha256"] == _sha256(TASK_ROOT / "TASK.md")
    assert lock["grader"]["sha256"] == _sha256(GRADER)
    assert lock["calibration"]["runner_sha256"] == _sha256(CALIBRATION_RUNNER)
    assert lock["calibration"]["runner_path"] == (
        "scripts/calibrate_mlxlm_moe_routing_grader.py"
    )
    assert (
        lock["grader"]["upstream_test_sha256"]
        == "92c8954f802681b8ddecc4874f3cc60f2379e568580b6b4e959846c99e8c6f17"
    )
    assert lock["grader"]["selectors"] == [
        "TestTunerTrainer::test_granitemoe_backward",
        "TestTunerTrainer::test_granitemoehybrid_backward",
        "TestTunerTrainer::test_lfm2_moe_backward",
    ]
    assert lock["source"]["known_fix_file_sha256"] == {
        "mlx_lm/models/granitemoe.py": "ed6bc92f0a190ecdc78e67dfd4de9e9e2132c61ea92538977eb480c959ebd61f",
        "mlx_lm/models/granitemoehybrid.py": "9f9ba35f3806bae835d88e390421345d88002ee70fab6852144c588195194b54",
        "mlx_lm/models/lfm2_moe.py": "2268bcf9617d0dcdebd3c57b5dc02d2813b1abb88899ae49e998250f63b7902a",
    }
    assert lock["calibration"]["required_arms_in_order"] == [
        "baseline",
        "known-fix",
        "wrong-mutant",
    ]
    loaded_lock, lock_hash = _grader_module()._load_lock(TASK_ROOT)
    assert loaded_lock == lock
    assert lock_hash == _sha256(TASK_ROOT / "protocol.lock.json")


def test_signal_pilot_v7_supersedes_unexecuted_selector_protocol():
    v1_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v1.json"
    v2_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v2.json"
    v3_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v3.json"
    v4_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v4.json"
    v5_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v5.json"
    v6_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v6.json"
    v7_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v7.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    v5 = json.loads(v5_path.read_text(encoding="utf-8"))
    v6 = json.loads(v6_path.read_text(encoding="utf-8"))
    v7 = json.loads(v7_path.read_text(encoding="utf-8"))
    lock = json.loads((TASK_ROOT / "protocol.lock.json").read_text(encoding="utf-8"))
    assert v2["supersedes"]["protocol_sha256"] == _sha256(v1_path)
    assert v2["supersedes"]["experiment_id"] == v1["experiment_id"]
    assert v3["supersedes"]["protocol_sha256"] == _sha256(v2_path)
    assert v4["supersedes"]["protocol_sha256"] == _sha256(v3_path)
    assert v5["supersedes"]["protocol_sha256"] == _sha256(v4_path)
    assert v6["supersedes"]["protocol_sha256"] == _sha256(v5_path)
    assert v7["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v7"
    assert v7["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert v7["task"]["grader_protocol_id"] == lock["protocol_id"]
    assert v7["supersedes"]["protocol_sha256"] == _sha256(v6_path)
    assert v7["supersedes"]["prior_outcomes_or_model_exposure"] == "none"
    assert v6["experiment_id"] == v7["supersedes"]["experiment_id"]
    assert v7["policy"]["controller_source_sha256"] == _sha256(
        ROOT / "scripts/run_local_agent.py"
    )
    assert v7["update"]["trainer_source_sha256"] == _sha256(
        ROOT / "src/vare/grpo_mlx.py"
    )
    assert v1["experiment_id"] != v7["experiment_id"]


def test_grader_sandbox_defaults_to_network_denied_and_caps_are_frozen():
    grader = _grader_module()
    candidate = Path("/tmp/candidate-task")
    source = Path("/tmp/trusted-source")
    test_file = Path("/tmp/trusted-evaluator/test_tuner_trainer.py")
    runtime = Path("/opt/mlx-runtime/site-packages")
    scratch = Path("/tmp/grader-scratch")
    profile = grader.sandbox_profile(
        candidate=candidate,
        trusted_source=source,
        grader_test=test_file,
        runtime_root=runtime,
        scratch=scratch,
        python_prefix=Path("/opt/python"),
        metal_cache_paths=(),
    )
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert f'(allow file-write* (subpath "{scratch.resolve()}"))' in profile
    assert "(allow network-outbound)" not in profile
    assert grader.MAX_WALL_SECONDS == 240
    assert grader.MAX_RSS_BYTES == 8 * 1024**3
    assert grader.MAX_OUTPUT_BYTES == 1_048_576


def test_grader_parses_pytest_counts_as_string_keys():
    grader = _grader_module()
    assert grader._parse_pytest_counts(b"3 passed in 0.12s") == {"passed": 3}
    assert grader._parse_pytest_counts(b"2 passed, 1 failed, 0 errors in 0.12s") == {
        "passed": 2,
        "failed": 1,
        "errors": 0,
    }
    assert grader._parse_pytest_counts(b"1 error, 2 skipped") == {
        "errors": 1,
        "skipped": 2,
    }
