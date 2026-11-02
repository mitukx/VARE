"""Contract tests for the pinned MLX-LM MoE routing task and grader lock."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

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


def _calibration_module():
    spec = importlib.util.spec_from_file_location(
        "mlxlm_moe_calibration", CALIBRATION_RUNNER
    )
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
    assert lock["status"] == "POST_EXPOSURE_SECURITY_CORRECTION"
    assert lock["protocol_id"] == "mlxlm-moe-routing-gradient-independent-grader-v13"
    assert lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v12.json"
    )
    assert _sha256(TASK_ROOT / "protocol.lock.v12.json") == (
        "0f5687a75638825f7660b2e31fd20cbdecfbab600e23d867365eebb58a7f3b49"
    )
    v12_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v12.json").read_text(encoding="utf-8")
    )
    assert (
        v12_lock["protocol_id"] == "mlxlm-moe-routing-gradient-independent-grader-v12"
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
    prior_v6_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v6.json").read_text(encoding="utf-8")
    )
    prior_v7_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v7.json").read_text(encoding="utf-8")
    )
    prior_v8_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v8.json").read_text(encoding="utf-8")
    )
    prior_v9_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v9.json").read_text(encoding="utf-8")
    )
    prior_v10_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v10.json").read_text(encoding="utf-8")
    )
    prior_v11_lock = json.loads(
        (TASK_ROOT / "protocol.lock.v11.json").read_text(encoding="utf-8")
    )
    assert prior_v4_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v3.json"
    )
    assert prior_v5_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v4.json"
    )
    assert prior_v6_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v5.json"
    )
    assert prior_v7_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v6.json"
    )
    assert prior_v8_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v7.json"
    )
    assert prior_v9_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v8.json"
    )
    assert prior_v10_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v9.json"
    )
    assert prior_v11_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v10.json"
    )
    assert lock["isolation"]["metal_gpu"].startswith(
        "Only AGXDeviceUserClient and IOSurfaceRootUserClient"
    )
    assert lock["isolation"]["entropy_device"].startswith(
        "/dev/urandom is readable by exact literal path"
    )
    assert lock["isolation"]["trusted_tests_imported_by_candidate"] is False
    assert lock["isolation"]["candidate_workspace_git_visible"] is False
    assert lock["isolation"]["pytest_verdict_parser_used"] is False
    assert v2_lock["supersedes"]["protocol_lock_sha256"] == _sha256(
        TASK_ROOT / "protocol.lock.v1.json"
    )
    assert (
        "Four model-generated v14 attempts occurred before this correction"
        in lock["exposure"]["researcher_source_qualification"]
    )
    assert lock["task_definition"]["task_json_sha256"] == _sha256(
        TASK_ROOT / "task.json"
    )
    assert lock["task_definition"]["task_md_sha256"] == _sha256(TASK_ROOT / "TASK.md")
    assert lock["grader"]["sha256"] == _sha256(GRADER)
    assert lock["grader"]["worker_sha256"] == _sha256(
        TASK_ROOT / "evaluator/candidate_worker.py"
    )
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
    assert lock["calibration"]["status"] == "NOT_RUN_TASK_CONSUMED"
    assert lock["calibration"]["upstream_test_executed"] is False
    assert lock["calibration"]["per_worker_wall_seconds"] == 240
    assert lock["calibration"]["worker_processes_per_arm"] == 6
    assert lock["calibration"]["outer_process_timeout_seconds"] == 1560
    assert lock["exposure"]["independently_valid_grades"] == 0
    assert lock["exposure"]["optimizer_updates"] == 0
    assert lock["exposure"]["historical_v14_grades_valid"] is False
    assert lock["grader"]["success"].find("not imported or executed") >= 0
    loaded_lock, lock_hash = _grader_module()._load_lock(TASK_ROOT)
    assert loaded_lock == lock
    assert lock_hash == _sha256(TASK_ROOT / "protocol.lock.json")


def test_signal_pilot_v13_supersedes_prior_selector_protocol():
    v1_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v1.json"
    v2_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v2.json"
    v3_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v3.json"
    v4_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v4.json"
    v5_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v5.json"
    v6_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v6.json"
    v7_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v7.json"
    v8_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v8.json"
    v9_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v9.json"
    v10_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v10.json"
    v11_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v11.json"
    v12_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v12.json"
    v13_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v13.json"
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    v5 = json.loads(v5_path.read_text(encoding="utf-8"))
    v6 = json.loads(v6_path.read_text(encoding="utf-8"))
    v7 = json.loads(v7_path.read_text(encoding="utf-8"))
    v8 = json.loads(v8_path.read_text(encoding="utf-8"))
    v9 = json.loads(v9_path.read_text(encoding="utf-8"))
    v10 = json.loads(v10_path.read_text(encoding="utf-8"))
    v11 = json.loads(v11_path.read_text(encoding="utf-8"))
    v12 = json.loads(v12_path.read_text(encoding="utf-8"))
    v13 = json.loads(v13_path.read_text(encoding="utf-8"))
    assert v2["supersedes"]["protocol_sha256"] == _sha256(v1_path)
    assert v2["supersedes"]["experiment_id"] == v1["experiment_id"]
    assert v3["supersedes"]["protocol_sha256"] == _sha256(v2_path)
    assert v4["supersedes"]["protocol_sha256"] == _sha256(v3_path)
    assert v5["supersedes"]["protocol_sha256"] == _sha256(v4_path)
    assert v6["supersedes"]["protocol_sha256"] == _sha256(v5_path)
    assert v7["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v7"
    assert v7["supersedes"]["protocol_sha256"] == _sha256(v6_path)
    assert v8["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v8"
    assert v8["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert (
        v8["task"]["grader_protocol_id"]
        == "mlxlm-moe-routing-gradient-independent-grader-v7"
    )
    assert v8["supersedes"]["protocol_sha256"] == _sha256(v7_path)
    assert v8["supersedes"]["prior_outcomes_or_model_exposure"] != "none"
    assert v7["experiment_id"] == v8["supersedes"]["experiment_id"]
    assert v9["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v9"
    assert v9["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert (
        v9["task"]["grader_protocol_id"]
        == "mlxlm-moe-routing-gradient-independent-grader-v8"
    )
    assert v9["supersedes"]["protocol_sha256"] == _sha256(v8_path)
    assert v9["supersedes"]["prior_outcomes_or_model_exposure"].startswith(
        "V6 failed in dyld startup and V7 failed in pytest collection"
    )
    assert v8["experiment_id"] == v9["supersedes"]["experiment_id"]
    assert v10["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v10"
    assert v10["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert (
        v10["task"]["grader_protocol_id"]
        == "mlxlm-moe-routing-gradient-independent-grader-v9"
    )
    assert v10["supersedes"]["protocol_sha256"] == _sha256(v9_path)
    assert v10["supersedes"]["prior_outcomes_or_model_exposure"].startswith(
        "V6 failed during dyld startup, V7 failed at denied /dev metadata"
    )
    assert v9["experiment_id"] == v10["supersedes"]["experiment_id"]
    assert v11["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v11"
    assert v11["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert (
        v11["task"]["grader_protocol_id"]
        == "mlxlm-moe-routing-gradient-independent-grader-v10"
    )
    assert v11["supersedes"]["protocol_sha256"] == _sha256(v10_path)
    assert v11["supersedes"]["prior_outcomes_or_model_exposure"].startswith(
        "V6 through V9 calibration attempts stopped before any test ran"
    )
    assert v10["experiment_id"] == v11["supersedes"]["experiment_id"]
    assert v12["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v12"
    assert v12["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert (
        v12["task"]["grader_protocol_id"]
        == "mlxlm-moe-routing-gradient-independent-grader-v11"
    )
    assert v12["supersedes"]["protocol_sha256"] == _sha256(v11_path)
    assert v12["supersedes"]["prior_outcomes_or_model_exposure"].startswith(
        "V6 through V10 calibration stopped before any test body ran"
    )
    assert v11["experiment_id"] == v12["supersedes"]["experiment_id"]
    assert v13["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v13"
    assert v13["status"] == "PREREGISTERED_NOT_EXECUTED"
    assert (
        v13["task"]["grader_protocol_id"]
        == "mlxlm-moe-routing-gradient-independent-grader-v12"
    )
    assert v13["supersedes"]["protocol_sha256"] == _sha256(v12_path)
    assert v13["supersedes"]["prior_outcomes_or_model_exposure"].startswith(
        "V6 through V11 calibration stopped before any test body ran"
    )
    assert v12["experiment_id"] == v13["supersedes"]["experiment_id"]
    assert v8["policy"]["controller_source_sha256"] == _sha256(
        ROOT / "scripts/run_local_agent.py"
    )
    assert v13["update"]["trainer_source_sha256"] == _sha256(
        ROOT / "src/vare/grpo_mlx.py"
    )
    assert v1["experiment_id"] != v13["experiment_id"]


def test_pilot_v15_preserves_consumed_task_nogo():
    path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v15.json"
    prior_path = ROOT / "protocols/mlxlm_moe_routing_signal_pilot_v14.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    lock = json.loads((TASK_ROOT / "protocol.lock.json").read_text(encoding="utf-8"))
    assert protocol["status"] == "NO_GO_TASK_CONSUMED_GRADER_INVALID"
    assert protocol["experiment_id"] == "mlxlm-moe-routing-train-signal-pilot-v15"
    assert protocol["supersedes"]["protocol_sha256"] == _sha256(prior_path)
    assert protocol["task"]["grader_protocol_id"] == lock["protocol_id"]
    assert protocol["outcome"]["model_generated_attempts"] == 4
    assert protocol["outcome"]["independently_valid_grades"] == 0
    assert protocol["outcome"]["optimizer_updates"] == 0
    assert protocol["outcome"]["new_model_execution"] is False
    assert protocol["outcome"]["new_inference_or_grading_started"] is False
    assert protocol["outcome"]["attempts_without_usable_source_diff"] == 2
    assert protocol["outcome"]["attempts_with_source_diff"] == 2
    assert protocol["execution_gate"]["task_eligible"] is False
    assert protocol["execution_gate"]["shared_metal_lease_present_at_audit"] is False
    assert prior["experiment_id"] == protocol["supersedes"]["experiment_id"]


def test_grader_sandbox_blocks_trusted_files_imports_and_writes(tmp_path):
    grader = _grader_module()
    if sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        pytest.skip("macOS Seatbelt is required for this adversarial boundary test")

    task_root = tmp_path / "task"
    evaluator_root = task_root / "evaluator"
    evaluator_root.mkdir(parents=True)
    hidden_test = evaluator_root / "test_tuner_trainer.py"
    hidden_test.write_text("HIDDEN_SENTINEL = 'must-not-leak'\n", encoding="utf-8")
    grader_file = evaluator_root / "grade.py"
    grader_file.write_text("TRUSTED_GRADER = True\n", encoding="utf-8")
    grader_output = tmp_path / "grade-output.json"
    grader_output.write_text("trusted output\n", encoding="utf-8")
    known_fix_root = tmp_path / "trusted-known-fix"
    known_fix_models = known_fix_root / "mlx_lm/models"
    known_fix_models.mkdir(parents=True)
    known_fix = known_fix_models / "granitemoe.py"
    known_fix.write_text("KNOWN_REPAIR = 'must-not-load'\n", encoding="utf-8")
    known_fix_pyc = known_fix_models / "__pycache__/granitemoe.cpython-312.pyc"
    known_fix_pyc.parent.mkdir()
    known_fix_pyc.write_bytes(b"compiled known repair")
    workspace = tmp_path / "candidate-workspace"
    (workspace / ".git").mkdir(parents=True)
    git_config = workspace / ".git/config"
    git_config.write_text("private candidate history\n", encoding="utf-8")
    candidate_source = tmp_path / "candidate-source"
    candidate_source.mkdir()
    allowed_file = candidate_source / "probe.py"
    allowed_file.write_text("# candidate-controlled file\n", encoding="utf-8")
    escape_link = candidate_source / "escape.py"
    escape_link.symlink_to(hidden_test)
    reference_source = tmp_path / "reference-source"
    reference_source.mkdir()
    runtime = tmp_path / "runtime/site-packages"
    installed_models = runtime / "mlx_lm/models"
    installed_models.mkdir(parents=True)
    installed_source = installed_models / "granitemoe.py"
    installed_source.write_text(
        "INSTALLED_REPAIR = 'must-not-load'\n", encoding="utf-8"
    )
    installed_pyc = installed_models / "__pycache__/granitemoe.cpython-312.pyc"
    installed_pyc.parent.mkdir()
    installed_pyc.write_bytes(b"compiled installed repair")
    metadata = runtime / "mlx_lm-0.32.2.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: mlx-lm\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    installed_paths = grader._installed_mlx_lm_paths(
        runtime,
        tmp_path / "venv",
        tmp_path / "python-base",
    )
    assert runtime.resolve() / "mlx_lm" in installed_paths
    assert metadata.resolve() in installed_paths
    profile = grader.sandbox_profile(
        trusted_source=candidate_source,
        runtime_root=runtime,
        scratch=scratch,
        python=Path(sys.executable),
        python_prefix=Path(sys.prefix),
        python_base_prefix=Path(sys.base_prefix),
        forbidden_paths=(
            task_root,
            hidden_test,
            hidden_test.parent,
            grader_file,
            grader_output,
            known_fix_root,
            *installed_paths,
            workspace,
            reference_source,
        ),
        metal_cache_paths=(),
    )
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert "(allow process*)" not in profile
    assert '(iokit-user-client-class "AGXDeviceUserClient")' in profile
    assert '(iokit-user-client-class "IOSurfaceRootUserClient")' in profile
    assert '(global-name "com.apple.MTLCompilerService")' in profile
    assert '(xpc-service-name "com.apple.MTLCompilerService")' in profile
    assert f'(allow file-write* (subpath "{scratch.resolve()}"))' in profile
    assert '(allow file-read-data (literal "/"))' in profile
    assert '(allow file-read* (subpath "/dev"))' not in profile
    assert '(allow file-read-metadata (subpath "/dev"))' not in profile
    assert '(allow file-read* (literal "/dev/urandom"))' in profile
    assert f'(allow file-read* (subpath "{candidate_source.resolve()}"))' in profile
    assert f'(allow file-read* (subpath "{workspace.resolve()}"))' not in profile
    assert f'(allow file-read* (subpath "{task_root.resolve()}"))' not in profile
    assert f'(allow file-read* (subpath "{known_fix_root.resolve()}"))' not in profile
    assert f'(allow file-read* (subpath "{runtime.resolve()}/mlx_lm"))' not in profile
    assert '(allow file-read* (subpath "/usr"))' not in profile
    assert '(allow file-read* (subpath "/"))' not in profile
    assert "(allow network-outbound)" not in profile

    targets = {
        "hidden_test": str(hidden_test),
        "known_fix": str(known_fix),
        "known_fix_pyc": str(known_fix_pyc),
        "installed_source": str(installed_source),
        "installed_pyc": str(installed_pyc),
        "metadata": str(metadata / "METADATA"),
        "escape_link": str(escape_link),
        "grader": str(grader_file),
        "output": str(grader_output),
        "git_config": str(git_config),
        "candidate_file": str(allowed_file),
    }
    probe = candidate_source / "probe.py"
    probe.write_text(
        "import importlib, importlib.machinery, json, pathlib, subprocess, sys\n"
        f"targets = {targets!r}\n"
        "results = {}\n"
        "def attempt(name, fn):\n"
        "    try:\n"
        "        value = fn()\n"
        "        if isinstance(value, bytes): value = value.decode('utf-8', 'replace')\n"
        "        results[name] = 'OPEN:' + str(value)\n"
        "    except BaseException as exc:\n"
        "        results[name] = 'DENIED:' + type(exc).__name__\n"
        "attempt('hidden_open', lambda: pathlib.Path(targets['hidden_test']).read_text())\n"
        "attempt('hidden_import', lambda: importlib.machinery.SourceFileLoader('hidden', targets['hidden_test']).load_module())\n"
        "attempt('hidden_module_import', lambda: importlib.import_module('test_tuner_trainer'))\n"
        "attempt('symlink_escape', lambda: pathlib.Path(targets['escape_link']).read_text())\n"
        "attempt('known_fix_source', lambda: pathlib.Path(targets['known_fix']).read_text())\n"
        "attempt('known_fix_pyc', lambda: importlib.machinery.SourcelessFileLoader('fixed', targets['known_fix_pyc']).load_module())\n"
        "attempt('installed_source', lambda: pathlib.Path(targets['installed_source']).read_text())\n"
        "attempt('installed_module_import', lambda: importlib.import_module('mlx_lm.models.granitemoe'))\n"
        "attempt('installed_pyc', lambda: importlib.machinery.SourcelessFileLoader('installed', targets['installed_pyc']).load_module())\n"
        "attempt('installed_metadata', lambda: pathlib.Path(targets['metadata']).read_text())\n"
        "attempt('grader_write', lambda: pathlib.Path(targets['grader']).write_text('tampered'))\n"
        "attempt('output_write', lambda: pathlib.Path(targets['output']).write_text('forged'))\n"
        "attempt('workspace_git', lambda: pathlib.Path(targets['git_config']).read_text())\n"
        "attempt('candidate_self_read', lambda: pathlib.Path(targets['candidate_file']).read_text())\n"
        "attempt('subprocess', lambda: subprocess.run(['/bin/cat', targets['hidden_test']], check=True))\n"
        "results['grader_test_loaded'] = any('test_tuner_trainer' in name for name in sys.modules)\n"
        "results['grader_parent_on_sys_path'] = str(pathlib.Path(targets['hidden_test']).parent) in sys.path\n"
        "results['pytest_loaded'] = 'pytest' in sys.modules\n"
        "print(json.dumps(results, sort_keys=True))\n",
        encoding="utf-8",
    )
    profile_path = tmp_path / "candidate.sb"
    profile_path.write_text(profile, encoding="utf-8")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONPATH": os.pathsep.join((str(candidate_source), str(runtime))),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile_path),
            sys.executable,
            "-S",
            str(probe),
        ],
        cwd=candidate_source,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    results = json.loads(completed.stdout)
    for name in (
        "hidden_open",
        "hidden_import",
        "hidden_module_import",
        "symlink_escape",
        "known_fix_source",
        "known_fix_pyc",
        "installed_source",
        "installed_module_import",
        "installed_pyc",
        "installed_metadata",
        "grader_write",
        "output_write",
        "workspace_git",
        "subprocess",
    ):
        assert results[name].startswith("DENIED:"), (name, results[name])
    assert results["candidate_self_read"].startswith("OPEN:")
    assert results["grader_test_loaded"] is False
    assert results["grader_parent_on_sys_path"] is False
    assert results["pytest_loaded"] is False
    assert "must-not-leak" not in completed.stdout
    assert "must-not-load" not in completed.stdout
    assert "tampered" not in grader_file.read_text(encoding="utf-8")
    assert grader_output.read_text(encoding="utf-8") == "trusted output\n"
    assert grader.MAX_WALL_SECONDS == 240
    assert grader.MAX_RSS_BYTES == 8 * 1024**3
    assert grader.MAX_OUTPUT_BYTES == 1_048_576


def test_worker_protocol_rejects_pytest_verdicts_and_forged_fields():
    grader = _grader_module()
    with pytest.raises(ValueError, match="one JSON report"):
        grader._parse_worker_report(
            b"3 passed in 0.12s",
            expected_model="granitemoe",
            expected_nonce="a" * 64,
            expects_gradients=True,
        )
    forged = {
        "schema_version": 1,
        "model": "granitemoe",
        "nonce": "a" * 64,
        "module_origin": "/tmp/candidate/granitemoe.py",
        "logits": [[1.0]],
        "logits_finite": True,
        "loss": 0.0,
        "loss_finite": True,
        "gradients": [],
        "passed": True,
    }
    with pytest.raises(ValueError, match="fields mismatch"):
        grader._parse_worker_report(
            json.dumps(forged).encode(),
            expected_model="granitemoe",
            expected_nonce="a" * 64,
            expects_gradients=True,
        )


def test_routing_ast_contract_rejects_decoy_stop_gradient(tmp_path):
    grader = _grader_module()
    baseline_root = tmp_path / "baseline"
    source_root = tmp_path / "candidate"
    baseline_dir = baseline_root / "mlx_lm/models"
    model_dir = source_root / "mlx_lm/models"
    baseline_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    baseline = {
        "granitemoe": (
            "class GraniteMoeTopKGating:\n"
            " def __call__(self, x):\n"
            "    top_k_idx = mx.argpartition(x, kth=-2)[..., -2:]\n"
            "    return mx.take_along_axis(x, top_k_idx, axis=-1)\n"
        ),
        "granitemoehybrid": (
            "class GraniteMoeHybridTopKGating:\n"
            " def __call__(self, x):\n"
            "    top_k_idx = mx.argpartition(x, kth=-2)[..., -2:]\n"
            "    return mx.take_along_axis(x, top_k_idx, axis=-1)\n"
        ),
        "lfm2_moe": (
            "class Lfm2MoeSparseMoeBlock:\n"
            " def __call__(self, x):\n"
            "    if self.flag:\n"
            "        inds = mx.argpartition(x, kth=-2)[..., -2:]\n"
            "    else:\n"
            "        inds = mx.argpartition(x, kth=-2)[..., -2:]\n"
            "    return mx.take_along_axis(x, inds, axis=-1)\n"
        ),
    }
    valid = {
        "granitemoe": (
            "class GraniteMoeTopKGating:\n"
            " def __call__(self, x):\n"
            "    top_k_idx = mx.stop_gradient(mx.argpartition(x, kth=-2)[..., -2:])\n"
            "    return mx.take_along_axis(x, top_k_idx, axis=-1)\n"
        ),
        "granitemoehybrid": (
            "class GraniteMoeHybridTopKGating:\n"
            " def __call__(self, x):\n"
            "    top_k_idx = mx.stop_gradient(mx.argpartition(x, kth=-2)[..., -2:])\n"
            "    return mx.take_along_axis(x, top_k_idx, axis=-1)\n"
        ),
        "lfm2_moe": (
            "class Lfm2MoeSparseMoeBlock:\n"
            " def __call__(self, x):\n"
            "    if self.flag:\n"
            "        inds = mx.argpartition(x, kth=-2)[..., -2:]\n"
            "    else:\n"
            "        inds = mx.argpartition(x, kth=-2)[..., -2:]\n"
            "    inds = mx.stop_gradient(inds)\n"
            "    return mx.take_along_axis(x, inds, axis=-1)\n"
        ),
    }
    for model, text in baseline.items():
        (baseline_dir / f"{model}.py").write_text(text, encoding="utf-8")
    for model, text in valid.items():
        (model_dir / f"{model}.py").write_text(text, encoding="utf-8")
    assert grader._routing_fix_failures(source_root, baseline_root) == {
        "granitemoe": [],
        "granitemoehybrid": [],
        "lfm2_moe": [],
    }

    (model_dir / "granitemoe.py").write_text(
        "class GraniteMoeTopKGating:\n"
        " def __call__(self, x):\n"
        "    top_k_idx = mx.argpartition(x, kth=-2)[..., -2:]\n"
        "    unused = mx.stop_gradient(mx.argpartition(x, kth=-2)[..., -2:])\n"
        "    return mx.take_along_axis(x, top_k_idx, axis=-1)\n",
        encoding="utf-8",
    )
    (model_dir / "lfm2_moe.py").write_text(
        "class Lfm2MoeSparseMoeBlock:\n"
        " def __call__(self, x):\n"
        "    inds = mx.argpartition(x, kth=-2)[..., -2:]\n"
        "    scores = mx.take_along_axis(x, inds, axis=-1)\n"
        "    unused = mx.stop_gradient(mx.argpartition(x, kth=-2)[..., -2:])\n"
        "    return scores\n",
        encoding="utf-8",
    )
    failures = grader._routing_fix_failures(source_root, baseline_root)
    assert failures["granitemoe"] == ["routing_indices_not_detached_at_selection"]
    assert failures["lfm2_moe"] == ["routing_indices_not_detached_at_selection"]

    (model_dir / "granitemoe.py").write_text(
        "import os\n" + valid["granitemoe"], encoding="utf-8"
    )
    assert grader._routing_fix_failures(source_root, baseline_root)["granitemoe"] == [
        "source_modified_outside_routing_fix"
    ]


def test_candidate_report_alone_cannot_override_source_contract(tmp_path):
    grader = _grader_module()
    reference = {"logits": [[1.0, 2.0]]}
    fake_candidate_report = {
        "module_origin": str(tmp_path / "mlx_lm/models/granitemoe.py"),
        "logits_finite": True,
        "logits": [[1.0, 2.0]],
        "loss_finite": True,
        "loss": 0.0,
        "gradients": [
            {
                "name": "switch_mlp.weight",
                "nonzero_count": 1,
                "max_abs": 1.0,
                "finite": True,
            },
            {
                "name": "router.layer.weight",
                "nonzero_count": 1,
                "max_abs": 1.0,
                "finite": True,
            },
        ],
    }
    assert (
        grader._candidate_report_failures(
            fake_candidate_report,
            reference,
            model="granitemoe",
            expected_origin=tmp_path / "mlx_lm/models/granitemoe.py",
        )
        == []
    )
    source_root = tmp_path / "source"
    model_dir = source_root / "mlx_lm/models"
    model_dir.mkdir(parents=True)
    (model_dir / "granitemoe.py").write_text(
        "class GraniteMoeTopKGating:\n"
        " def __call__(self, x):\n"
        "    top_k_idx = mx.argpartition(x, kth=-2)[..., -2:]\n"
        "    unused = mx.stop_gradient(mx.argpartition(x, kth=-2)[..., -2:])\n"
        "    return mx.take_along_axis(x, top_k_idx, axis=-1)\n",
        encoding="utf-8",
    )
    for model in ("granitemoehybrid", "lfm2_moe"):
        (model_dir / f"{model}.py").write_text(
            "class Placeholder: pass\n", encoding="utf-8"
        )
    assert grader._routing_fix_failures(source_root, source_root)["granitemoe"]


def test_candidate_path_checks_reject_symlink_hardlink_and_escape(tmp_path):
    grader = _grader_module()
    external = tmp_path / "outside.py"
    external.write_text("trusted\n", encoding="utf-8")
    symlinked = tmp_path / "symlink.py"
    symlinked.symlink_to(external)
    with pytest.raises(ValueError, match="regular file"):
        grader._validate_regular_source(symlinked)

    source = tmp_path / "source.py"
    source.write_text("candidate\n", encoding="utf-8")
    hard_link = tmp_path / "hardlink.py"
    os.link(source, hard_link)
    with pytest.raises(ValueError, match="hard-linked"):
        grader._validate_regular_source(source)

    with pytest.raises(ValueError, match="outside editable allowlist"):
        grader._validate_changed_paths(("../evaluator/grade.py",))


def test_calibration_oracle_and_consumed_task_guard():
    grader = _grader_module()
    calibration = _calibration_module()
    assert calibration.expected_model_passes("baseline") == {
        "granitemoe": False,
        "granitemoehybrid": False,
        "lfm2_moe": False,
    }
    assert calibration.expected_model_passes("known-fix") == {
        "granitemoe": True,
        "granitemoehybrid": True,
        "lfm2_moe": True,
    }
    assert calibration.expected_model_passes("wrong-mutant") == {
        "granitemoe": True,
        "granitemoehybrid": True,
        "lfm2_moe": False,
    }
    assert calibration.expected_structure_rejections("baseline") == {
        "mlx_lm/models/granitemoe.py",
        "mlx_lm/models/granitemoehybrid.py",
        "mlx_lm/models/lfm2_moe.py",
    }
    assert calibration.expected_structure_rejections("known-fix") == set()
    assert calibration.expected_structure_rejections("wrong-mutant") == {
        "mlx_lm/models/lfm2_moe.py"
    }
    assert calibration.ARM_PROCESS_TIMEOUT_SECONDS == 1560
    with pytest.raises(SystemExit, match="consumed task"):
        calibration.refuse_post_exposure_calibration(
            {"status": "POST_EXPOSURE_SECURITY_CORRECTION"}
        )
    with pytest.raises(RuntimeError, match="fresh task identity"):
        grader._assert_task_eligible({"status": "POST_EXPOSURE_SECURITY_CORRECTION"})
