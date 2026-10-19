from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import reproduce_trl_async_stale_resume_review_v2 as replay_suffix_v2
from scripts import reproduce_trl_async_stale_resume_v1 as stale_resume
from scripts.reproduce_trl_async_stale_resume_v1 import (
    audit_bundle,
    execution_platform,
    expected_runtime_environment,
    parse_pytest_outcome,
    parse_run_log,
    protocol_timeout_seconds,
    pytest_environment,
    resolved_requirements,
    run_pytest_arm,
    runner_requires_source_revision_provenance,
    runtime_environment,
    sha256_file,
    source_revision,
    validate_new_run_protocol,
    write_json,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
LOCKED_RUN = ROOT / "results/trl-async-stale-resume-v1/run-2-locked-repetitions"
EPOCH_AWARE_RUN = ROOT / "results/trl-async-stale-resume-v2/run-1-epoch-aware-replay"
V2_REVIEW_FIX_RUN = ROOT / "results/trl-async-stale-resume-v2/run-2-review-fixes"
V1_FALSE_NEGATIVE = ROOT / "results/trl-async-stale-resume-v1/run-3-final-runner-replay"
V3_PROTOCOL = ROOT / "protocols/trl_async_stale_resume_v3.lock.json"
V4_PROTOCOL = ROOT / "protocols/trl_async_stale_resume_v4.lock.json"
V4_PLATFORM_RUN = ROOT / "results/trl-async-stale-resume-v4/run-1-platform-locked"
V4_SANITIZED_RUN = ROOT / "results/trl-async-stale-resume-v4/run-2-python-env-sanitized"
REPLAY_SUFFIX_PROTOCOL = ROOT / "protocols/trl_async_stale_resume_replay_suffix_v1.lock.json"
REPLAY_SUFFIX_RUN = ROOT / "results/trl-async-stale-resume-v5/review-p1-suffix-epoch-count"
REPLAY_SUFFIX_V2_RUN = ROOT / "results/trl-async-stale-resume-v6/review-p1-suffix-epoch-count-provenance-v2-final"


def test_reproduction_parser_extracts_stale_cursor_and_replay_count():
    text = (ROOT / "results/trl-async-stale-resume-v1/run-1-original-threshold/baseline-2.log").read_text(
        encoding="utf-8"
    )
    observed = parse_run_log(text)
    assert observed["saved_prompt_index"] == 2
    assert observed["resumed_row"] == 2
    assert observed["duplicated_rows"] == 144


@pytest.mark.parametrize(
    "contradictory_measurement, field",
    [("saved prompt_index: 2", "saved_prompt_index"), ("rows trained TWICE: 999", "duplicated_rows")],
)
def test_audit_rejects_resealed_bundle_with_duplicate_measurement(
    tmp_path: Path, contradictory_measurement: str, field: str
):
    copied = tmp_path / "run"
    shutil.copytree(V4_PLATFORM_RUN, copied)
    summary = json.loads((copied / "summary.json").read_text(encoding="utf-8"))
    candidate_run = next(run for run in summary["runs"] if run["arm"] == "candidate")
    log = copied / candidate_run["log"]
    log.write_text(
        log.read_text(encoding="utf-8") + f"\n{contradictory_measurement}\n",
        encoding="utf-8",
    )
    write_manifest(copied)

    with pytest.raises(ValueError, match=f"repeats {field}"):
        audit_bundle(copied)


def test_pytest_outcome_parser_preserves_failed_and_passed_counts():
    assert parse_pytest_outcome("1 failed, 1 passed, 2 warnings in 8.32s") == {
        "failed": 1,
        "passed": 1,
        "warnings": 2,
    }


def test_pytest_outcome_parser_rejects_multiple_summaries():
    with pytest.raises(ValueError, match="multiple pytest result summaries"):
        parse_pytest_outcome("1 failed in 0.01s\n1 passed in 0.01s\n")


def test_pytest_outcome_parser_rejects_duplicate_categories_in_one_summary():
    with pytest.raises(ValueError, match="repeats an outcome category"):
        parse_pytest_outcome("1 failed, 2 passed, 0 failed in 0.01s\n")


def test_pytest_outcome_parser_allows_missing_summary_only_for_failed_arm():
    assert parse_pytest_outcome("ERROR collecting test_example.py\n", allow_missing=True) == {}
    with pytest.raises(ValueError, match="lacks a pytest result summary"):
        parse_pytest_outcome("ERROR collecting test_example.py\n")


def test_execution_platform_uses_the_selected_python(monkeypatch: pytest.MonkeyPatch):
    expected = {
        "sys_platform": "darwin",
        "system": "Darwin",
        "release": "27.0.0",
        "machine": "x86_64",
        "mac_ver": "27.0.1",
        "hardware_model": "MacBookPro18,3",
        "processor": "Apple M1 Pro",
        "memory_bytes": 34359738368,
    }
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(expected))

    monkeypatch.setattr("scripts.reproduce_trl_async_stale_resume_v1.subprocess.run", fake_run)

    assert execution_platform("/selected/python") == expected
    assert calls[0][:3] == ["/selected/python", "-I", "-c"]


def test_pytest_environment_removes_inherited_python_path_overrides(monkeypatch: pytest.MonkeyPatch):
    names = ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "PYTHONEXECUTABLE")
    for name in names:
        monkeypatch.setenv(name, "/untrusted/python-path")

    env = pytest_environment()

    assert all(name not in env for name in names)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["TRL_EXPERIMENTAL_SILENCE"] == "1"


def test_pytest_arm_does_not_import_from_inherited_pythonpath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    override = tmp_path / "override"
    override.mkdir()
    (override / "json.py").write_text("raise RuntimeError('untrusted PYTHONPATH module imported')\n")
    monkeypatch.setenv("PYTHONPATH", str(override))
    env = pytest_environment()

    output, exit_code, timed_out = run_pytest_arm(
        [sys.executable, "-c", "import json; print(json.__file__)"],
        tmp_path,
        env,
        timeout_seconds=5,
    )

    assert exit_code == 0
    assert timed_out is False
    assert "untrusted PYTHONPATH module imported" not in output
    assert str(override) not in output


def test_runtime_probes_ignore_pythonpath_and_user_site_hooks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    marker = tmp_path / "sitecustomize-loaded"
    hook = (
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['VARE_SITE_CUSTOMIZE_MARKER']).write_text('loaded')\n"
    )
    pythonpath = tmp_path / "pythonpath"
    user_site = tmp_path / "userbase" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    pythonpath.mkdir()
    user_site.mkdir(parents=True)
    (pythonpath / "sitecustomize.py").write_text(hook, encoding="utf-8")
    (user_site / "sitecustomize.py").write_text(hook, encoding="utf-8")

    package_names = {"pytest": "pytest"}
    monkeypatch.setattr(stale_resume, "PACKAGE_NAMES", package_names)
    versions = {name: importlib.metadata.version(dist) for name, dist in package_names.items()}
    expected = {
        "python": ".".join(map(str, sys.version_info[:3])),
        **versions,
        "execution_platform": execution_platform(sys.executable),
        "device_used": "CPU",
    }
    monkeypatch.setenv("PYTHONPATH", str(pythonpath))
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "userbase"))
    monkeypatch.setenv("VARE_SITE_CUSTOMIZE_MARKER", str(marker))

    observed = runtime_environment(sys.executable, tmp_path, expected)

    assert observed == expected_runtime_environment(expected)
    assert not marker.exists()


def test_replay_suffix_review_bundle_matches_its_protocol_and_raw_logs():
    protocol = json.loads(REPLAY_SUFFIX_PROTOCOL.read_text(encoding="utf-8"))
    summary = json.loads((REPLAY_SUFFIX_RUN / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((REPLAY_SUFFIX_RUN / "manifest.json").read_text(encoding="utf-8"))
    patches = protocol["source"]

    assert protocol["protocol_id"] == "trl-async-stale-resume-replay-suffix-v1"
    assert summary["protocol_id"] == protocol["protocol_id"]
    assert summary["protocol_sha256"] == sha256_file(REPLAY_SUFFIX_PROTOCOL)
    assert sha256_file(ROOT / patches["original_candidate_patch"]) == patches["original_candidate_patch_sha256"]
    assert sha256_file(ROOT / patches["regression_test_patch"]) == patches["regression_test_patch_sha256"]
    assert sha256_file(ROOT / patches["correction_patch"]) == patches["correction_patch_sha256"]
    assert patches["candidate_base_revision"] == "a85900829d86bcd45b7db0310725f28d3152289b"

    for run in ("baseline-failing-test.log", "corrected-targeted-test.log", "corrected-full-test-file.log", "summary.json"):
        path = REPLAY_SUFFIX_RUN / run
        assert manifest["files"][run] == sha256_file(path)

    baseline = parse_pytest_outcome((REPLAY_SUFFIX_RUN / "baseline-failing-test.log").read_text())
    targeted = parse_pytest_outcome((REPLAY_SUFFIX_RUN / "corrected-targeted-test.log").read_text())
    full = parse_pytest_outcome((REPLAY_SUFFIX_RUN / "corrected-full-test-file.log").read_text())
    assert summary["baseline"]["raw_log_sha256"] == sha256_file(REPLAY_SUFFIX_RUN / "baseline-failing-test.log")
    assert summary["corrected_targeted"]["raw_log_sha256"] == sha256_file(REPLAY_SUFFIX_RUN / "corrected-targeted-test.log")
    assert summary["corrected_full_file"]["raw_log_sha256"] == sha256_file(REPLAY_SUFFIX_RUN / "corrected-full-test-file.log")
    assert summary["baseline"]["exit_code"] == 1
    assert baseline["failed"] == 2
    assert summary["corrected_targeted"]["exit_code"] == 0
    assert targeted["passed"] == 2
    assert summary["corrected_full_file"]["exit_code"] == 0
    assert full["passed"] == 11


def test_sanitized_v4_rerun_audits_with_retained_runner():
    summary = audit_bundle(V4_SANITIZED_RUN)
    runner = (V4_SANITIZED_RUN / "runner.py").read_text(encoding="utf-8")

    assert summary["all_checks_passed"] is True
    assert summary["run_count"] == 6
    assert '"PYTHONPATH"' in runner
    assert "PYTHONNOUSERSITE" in runner


def test_replay_suffix_v2_bundle_binds_logs_to_source_and_protocol():
    summary = replay_suffix_v2.audit_bundle(REPLAY_SUFFIX_V2_RUN)

    assert summary["all_checks_passed"] is True
    assert summary["source_revision"] == "a85900829d86bcd45b7db0310725f28d3152289b"
    assert [run["pytest_outcome"] for run in summary["runs"]] == [
        {"failed": 2},
        {"passed": 2},
        {"passed": 11},
    ]


def test_replay_suffix_v2_stage_metadata_binds_each_raw_log_digest():
    summary = json.loads((REPLAY_SUFFIX_V2_RUN / "summary.json").read_text(encoding="utf-8"))

    for run in summary["runs"]:
        stage_record = json.loads((REPLAY_SUFFIX_V2_RUN / f"{run['name']}.json").read_text(encoding="utf-8"))
        assert stage_record == run
        assert stage_record["raw_log_sha256"] == sha256_file(REPLAY_SUFFIX_V2_RUN / run["log"])


def test_frozen_v5_digest_mismatch_is_superseded_by_audited_v6_record():
    old_record = json.loads((REPLAY_SUFFIX_RUN / "baseline-failing-test.json").read_text())
    old_log = REPLAY_SUFFIX_RUN / "baseline-failing-test.log"
    current_summary = replay_suffix_v2.audit_bundle(REPLAY_SUFFIX_V2_RUN)
    current_baseline = next(
        run for run in current_summary["runs"] if run["name"] == "baseline-failing-test"
    )

    assert old_record["raw_log_sha256"] != sha256_file(old_log)
    assert current_summary["all_checks_passed"] is True
    assert current_baseline["raw_log_sha256"] == sha256_file(
        REPLAY_SUFFIX_V2_RUN / current_baseline["log"]
    )


def test_replay_suffix_v2_audit_rejects_logs_without_source_provenance(tmp_path: Path):
    copied = tmp_path / "v2-run"
    shutil.copytree(REPLAY_SUFFIX_V2_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = summary["runs"][0]
    log_path = copied / record["log"]
    log = log_path.read_text(encoding="utf-8").replace(
        f"VARE_REVIEW_SOURCE_REVISION={record['revision']}",
        "VARE_REVIEW_SOURCE_REVISION=unverified",
    )
    log_path.write_text(log, encoding="utf-8")
    record["raw_log_sha256"] = sha256_file(log_path)
    write_json(copied / "baseline-failing-test.json", record)
    write_json(summary_path, summary)
    stale_resume.write_manifest(copied)

    with pytest.raises(ValueError, match="raw log provenance marker mismatch"):
        replay_suffix_v2.audit_bundle(copied)


def _reseal_replay_suffix_v2_stage(copied: Path, stage_name: str, mutate) -> dict:
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = next(run for run in summary["runs"] if run["name"] == stage_name)
    log_path = copied / record["log"]
    log = log_path.read_text(encoding="utf-8")
    mutate(record, log_path, log)
    record["raw_log_sha256"] = sha256_file(log_path)
    write_json(copied / f"{stage_name}.json", record)
    write_json(summary_path, summary)
    stale_resume.write_manifest(copied)
    return summary


def test_replay_suffix_v2_audit_rejects_resealed_unrelated_source_diff(tmp_path: Path):
    copied = tmp_path / "v2-run"
    shutil.copytree(REPLAY_SUFFIX_V2_RUN, copied)

    def replace_diff(record: dict, log_path: Path, log: str) -> None:
        diff_path = copied / record["diff_artifact"]
        diff_path.write_text("unrelated bytes presented as the correction\n", encoding="utf-8")
        digest = sha256_file(diff_path)
        log_path.write_text(
            log.replace(record["diff_artifact_sha256"], digest), encoding="utf-8"
        )
        record["diff_sha256"] = digest
        record["diff_artifact_sha256"] = digest

    _reseal_replay_suffix_v2_stage(copied, "corrected-targeted-test", replace_diff)

    with pytest.raises(ValueError, match="does not match the frozen patches"):
        replay_suffix_v2.audit_bundle(copied)


def test_replay_suffix_v2_audit_rejects_resealed_unexpected_candidate_tree(tmp_path: Path):
    copied = tmp_path / "v2-run"
    shutil.copytree(REPLAY_SUFFIX_V2_RUN, copied)

    def replace_tree(record: dict, log_path: Path, log: str) -> None:
        expected = record["candidate_tree"]
        record["candidate_tree"] = "0" * 40
        log_path.write_text(log.replace(expected, record["candidate_tree"]), encoding="utf-8")

    _reseal_replay_suffix_v2_stage(copied, "corrected-targeted-test", replace_tree)

    with pytest.raises(ValueError, match="candidate tree differs from the frozen source state"):
        replay_suffix_v2.audit_bundle(copied)


def test_replay_suffix_v2_audit_rejects_passing_summary_with_failures(tmp_path: Path):
    copied = tmp_path / "v2-run"
    shutil.copytree(REPLAY_SUFFIX_V2_RUN, copied)

    def add_failure(record: dict, log_path: Path, log: str) -> None:
        changed = False
        lines = []
        for line in log.splitlines():
            if not changed and "2 passed" in line and " in " in line:
                line = line.replace("2 passed", "1 failed, 2 passed", 1)
                changed = True
            lines.append(line)
        assert changed
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record["pytest_outcome"] = {"failed": 1, "passed": 2}

    _reseal_replay_suffix_v2_stage(copied, "corrected-targeted-test", add_failure)

    with pytest.raises(ValueError, match="does not satisfy the locked stage"):
        replay_suffix_v2.audit_bundle(copied)


def test_replay_suffix_v2_audit_cli_returns_nonzero_for_negative_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(replay_suffix_v2, "audit_bundle", lambda _: {"all_checks_passed": False})
    monkeypatch.setattr(sys, "argv", ["reproduce_trl_async_stale_resume_review_v2.py", "--audit", str(tmp_path)])

    assert replay_suffix_v2.main() == 1


def test_v3_lock_pins_transitive_dependencies_and_timeout():
    lock = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
    requirements = resolved_requirements(lock["environment"])
    assert requirements is not None
    assert len(requirements) > 40
    assert requirements["packaging"]
    assert requirements["pluggy"]
    assert requirements["numpy"]
    assert lock["reproducer"]["per_run_timeout_seconds"] == 600
    assert protocol_timeout_seconds(lock) == 600
    v2_lock = json.loads((ROOT / "protocols/trl_async_stale_resume_v2.lock.json").read_text())
    with pytest.raises(ValueError, match="no valid locked per-run timeout"):
        protocol_timeout_seconds(v2_lock)


def test_new_reproductions_require_a_platform_bound_protocol():
    v3_lock = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="execution-platform fields"):
        validate_new_run_protocol(v3_lock)
    validate_new_run_protocol(json.loads(V4_PROTOCOL.read_text(encoding="utf-8")))


def test_pytest_arm_timeout_preserves_partial_output_as_failed_result(tmp_path: Path):
    output, exit_code, timed_out = run_pytest_arm(
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
        tmp_path,
        dict(os.environ),
        timeout_seconds=2,
    )
    assert timed_out is True
    assert exit_code == 124
    assert "started" in output
    assert "VARE_RUNNER_TIMEOUT_SECONDS=2" in output


def test_pytest_arm_timeout_kills_descendant_processes(tmp_path: Path):
    marker = tmp_path / "child-survived"
    child_code = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"time.sleep(0.3); Path({str(marker)!r}).touch()"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        "print('spawned', flush=True); time.sleep(5)"
    )

    output, exit_code, timed_out = run_pytest_arm(
        [sys.executable, "-c", parent_code],
        tmp_path,
        dict(os.environ),
        timeout_seconds=0.1,
    )
    time.sleep(0.4)

    assert timed_out is True
    assert exit_code == 124
    assert "spawned" in output
    assert not marker.exists()


def _convert_bundle_to_v3(copied: Path) -> dict:
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock = json.loads(V3_PROTOCOL.read_text(encoding="utf-8"))
    summary["protocol_id"] = lock["protocol_id"]
    summary["protocol_sha256"] = sha256_file(V3_PROTOCOL)
    summary["environment"] = expected_runtime_environment(lock["environment"])
    expected_revisions = {
        "baseline": lock["source"]["baseline_revision"],
        "candidate": lock["source"]["candidate_revision"],
    }
    runner_path = ROOT / "results/trl-async-stale-resume-v3/run-1-full-lock-timeout/runner.py"
    runner_bytes = runner_path.read_bytes()
    (copied / "runner.py").write_bytes(runner_bytes)
    summary["runner_sha256"] = hashlib.sha256(runner_bytes).hexdigest()
    summary["runner_source_valid"] = True
    for run in summary["runs"]:
        run["timed_out"] = False
        run["timeout_seconds"] = lock["reproducer"]["per_run_timeout_seconds"]
        run["source_revision_valid"] = True
        log_path = copied / run["log"]
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"VARE_RUNNER_SOURCE_REVISION={expected_revisions[run['arm']]}\n")
            stream.write("VARE_RUNNER_SOURCE_REVISION_VALID=true\n")
    write_json(summary_path, summary)
    write_manifest(copied)
    return summary


def _convert_v4_bundle_to_current_runner(copied: Path) -> dict:
    shutil.copytree(V4_PLATFORM_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runner_path = V4_SANITIZED_RUN / "runner.py"
    runner_bytes = runner_path.read_bytes()
    (copied / "runner.py").write_bytes(runner_bytes)
    summary["runner_sha256"] = hashlib.sha256(runner_bytes).hexdigest()
    summary["runner_source_valid"] = True
    for run in summary["runs"]:
        run["source_revision_valid"] = True
        log_path = copied / run["log"]
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"VARE_RUNNER_SOURCE_REVISION={summary['source_revisions'][run['arm']]}\n")
            stream.write("VARE_RUNNER_SOURCE_REVISION_VALID=true\n")
    write_json(summary_path, summary)
    write_manifest(copied)
    return summary


def test_audit_rejects_resealed_v3_bundle_with_changed_transitive_dependency(tmp_path: Path):
    copied = tmp_path / "run"
    _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["environment"]["versions"]["packaging"] = "0.0.0-tampered"
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="recorded runtime environment"):
        audit_bundle(copied)


def test_current_retained_runner_requires_source_provenance_after_manifest_reseal(tmp_path: Path):
    copied = tmp_path / "run"
    summary = _convert_v4_bundle_to_current_runner(copied)
    assert runner_requires_source_revision_provenance(copied / "runner.py") is True

    for run in summary["runs"]:
        run.pop("source_revision_valid")
        log_path = copied / run["log"]
        log_path.write_text(
            "\n".join(
                line
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if not line.startswith("VARE_RUNNER_SOURCE_REVISION=")
                and not line.startswith("VARE_RUNNER_SOURCE_REVISION_VALID=")
            )
            + "\n",
            encoding="utf-8",
        )
    write_json(copied / "summary.json", summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="current runner source revision provenance is incomplete"):
        audit_bundle(copied)


def test_audit_rejects_resealed_v4_bundle_with_changed_execution_platform(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lock = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    summary["protocol_id"] = lock["protocol_id"]
    summary["protocol_sha256"] = sha256_file(V4_PROTOCOL)
    summary["environment"] = expected_runtime_environment(lock["environment"])
    runner_bytes = (V4_PLATFORM_RUN / "runner.py").read_bytes()
    (copied / "runner.py").write_bytes(runner_bytes)
    summary["runner_sha256"] = hashlib.sha256(runner_bytes).hexdigest()
    summary["runner_source_valid"] = True
    for run in summary["runs"]:
        run["timed_out"] = False
        run["timeout_seconds"] = lock["reproducer"]["per_run_timeout_seconds"]
    summary["environment"]["execution_platform"]["machine"] = "x86_64"
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="recorded runtime environment"):
        audit_bundle(copied)


def test_audit_retains_v3_timeout_as_nonpassing_arm(tmp_path: Path):
    copied = tmp_path / "run"
    summary = _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    failed_run = next(run for run in summary["runs"] if run["arm"] == "candidate")
    failed_run["exit_code"] = 124
    failed_run["timed_out"] = True
    failed_run["accepted"] = False
    log_path = copied / failed_run["log"]
    log_path.write_text(
        log_path.read_text(encoding="utf-8") + "\nVARE_RUNNER_TIMEOUT_SECONDS=600\n",
        encoding="utf-8",
    )
    summary["all_checks_passed"] = False
    write_json(summary_path, summary)
    write_manifest(copied)

    assert audit_bundle(copied)["all_checks_passed"] is False


def test_audit_seals_early_failed_arm_with_partial_measurements(tmp_path: Path):
    copied = tmp_path / "run"
    summary = _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    failed_run = next(run for run in summary["runs"] if run["arm"] == "candidate")
    failed_run["exit_code"] = 2
    failed_run["timed_out"] = False
    failed_run["accepted"] = False
    failed_run["observed"] = {"first_run_trained_rows": 144}
    failed_run["source_revision_valid"] = True
    (copied / failed_run["log"]).write_text(
        "ERROR collecting test_repro.py\n"
        "run 1 trained rows: 144\n"
        "1 error in 0.01s\n"
        "VARE_RUNNER_EXIT_CODE=2\n"
        f"VARE_RUNNER_SOURCE_REVISION={summary['source_revisions']['candidate']}\n"
        "VARE_RUNNER_SOURCE_REVISION_VALID=true\n",
        encoding="utf-8",
    )
    summary["all_checks_passed"] = False
    write_json(summary_path, summary)
    write_manifest(copied)

    audited = audit_bundle(copied)

    assert audited["all_checks_passed"] is False
    assert failed_run["observed"] == {"first_run_trained_rows": 144}


def test_audit_retains_runner_source_change_as_a_nonpassing_bundle(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(V4_SANITIZED_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runner_source_valid"] = False
    summary["all_checks_passed"] = False
    write_json(summary_path, summary)
    write_manifest(copied)

    audited = audit_bundle(copied)

    assert audited["runner_source_valid"] is False
    assert audited["all_checks_passed"] is False



def test_reproducer_seals_all_arms_after_early_collection_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    lock = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    revisions = {
        "baseline": lock["source"]["baseline_revision"],
        "candidate": lock["source"]["candidate_revision"],
    }
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    output = tmp_path / "bundle"
    python_link = tmp_path / "python"
    python_link.symlink_to(sys.executable)
    calls: dict[str, int] = {"baseline": 0, "candidate": 0}

    monkeypatch.setattr(stale_resume, "validate_new_run_protocol", lambda _: None)
    monkeypatch.setattr(stale_resume, "validate_local_inputs", lambda _: None)
    monkeypatch.setattr(stale_resume, "protocol_timeout_seconds", lambda _: 600)
    monkeypatch.setattr(stale_resume, "source_revision", lambda source: revisions[Path(source).name])
    monkeypatch.setattr(
        stale_resume, "observed_source_revision", lambda source: revisions[Path(source).name]
    )
    monkeypatch.setattr(
        stale_resume,
        "runtime_environment",
        lambda _python, _base, expected: expected_runtime_environment(expected),
    )
    monkeypatch.setattr(stale_resume, "download_gist", lambda _lock, _path: b"locked source")

    def prepare_runner(_gist: bytes, temp_root: Path, arm: str, _lock: dict[str, object]) -> Path:
        path = temp_root / f"{arm}.py"
        path.write_text("# synthetic test arm\n", encoding="utf-8")
        return path

    def run_arm(
        _command: list[str], source: Path, _env: dict[str, str], _timeout: int
    ) -> tuple[str, int, bool]:
        arm = source.name
        calls[arm] += 1
        if sum(calls.values()) == 1:
            return (
                (
                    "ERROR collecting test_repro.py\n"
                    "run 1 trained rows: 144\n"
                    "1 error in 0.01s"
                ),
                2,
                False,
            )
        log = V4_SANITIZED_RUN / f"{arm}-{calls[arm]}.log"
        output_text = "\n".join(
            line
            for line in log.read_text(encoding="utf-8").splitlines()
            if not line.startswith(
                (
                    "VARE_RUNNER_EXIT_CODE=",
                    "VARE_RUNNER_SOURCE_REVISION=",
                    "VARE_RUNNER_SOURCE_REVISION_VALID=",
                )
            )
        )
        return output_text + "\n", 0, False

    def audit_written_bundle(bundle: Path) -> dict[str, object]:
        summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        failed = summary["runs"][0]

        assert summary["python_executable"] == str(python_link.absolute())
        assert calls == {"baseline": 3, "candidate": 3}
        assert summary["run_count"] == 6
        assert failed["exit_code"] == 2
        assert failed["timed_out"] is False
        assert failed["accepted"] is False
        assert failed["observed"] == {"first_run_trained_rows": 144}
        assert summary["runner_source_valid"] is False
        assert len(list(bundle.glob("*.log"))) == 6
        assert all(f"{arm}-{repetition}.log" in manifest["files"]
                   for arm in ("baseline", "candidate")
                   for repetition in range(1, 4))
        assert manifest["files"]["baseline-1.log"] == sha256_file(bundle / "baseline-1.log")
        assert summary["all_checks_passed"] is False
        return summary

    monkeypatch.setattr(stale_resume, "prepare_runner", prepare_runner)
    monkeypatch.setattr(stale_resume, "run_pytest_arm", run_arm)
    monkeypatch.setattr(stale_resume, "audit_bundle", audit_written_bundle)
    original_sha256_file = stale_resume.sha256_file
    runner_path = Path(stale_resume.__file__).resolve()
    monkeypatch.setattr(
        stale_resume,
        "sha256_file",
        lambda path: "0" * 64 if Path(path).resolve() == runner_path else original_sha256_file(path),
    )

    result = stale_resume.reproduce(
        SimpleNamespace(
            protocol="trl-async-stale-resume-v4",
            python=str(python_link),
            baseline_source=baseline,
            candidate_source=candidate,
            output=output,
            gist_file=None,
        )
    )

    assert result == 1


def test_audit_rejects_promotion_after_source_checkout_changes_during_arm(tmp_path: Path):
    copied = tmp_path / "run"
    summary = _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    changed_run = summary["runs"][0]
    changed_run["source_revision_valid"] = False
    changed_run["accepted"] = False
    summary["all_checks_passed"] = False
    log_path = copied / changed_run["log"]
    raw_log = [
        line
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("VARE_RUNNER_SOURCE_REVISION=")
        and not line.startswith("VARE_RUNNER_SOURCE_REVISION_VALID=")
    ]
    log_path.write_text("\n".join(raw_log) + "\n", encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("VARE_RUNNER_SOURCE_REVISION=unavailable\n")
        stream.write("VARE_RUNNER_SOURCE_REVISION_VALID=false\n")
    write_json(summary_path, summary)
    write_manifest(copied)

    audited = audit_bundle(copied)

    assert audited["all_checks_passed"] is False


def test_audit_cli_returns_nonzero_for_valid_nonpassing_bundle(tmp_path: Path):
    copied = tmp_path / "run"
    summary = _convert_bundle_to_v3(copied)
    summary_path = copied / "summary.json"
    failed_run = next(run for run in summary["runs"] if run["arm"] == "candidate")
    failed_run["exit_code"] = 124
    failed_run["timed_out"] = True
    failed_run["accepted"] = False
    log_path = copied / failed_run["log"]
    log_path.write_text(
        log_path.read_text(encoding="utf-8") + "\nVARE_RUNNER_TIMEOUT_SECONDS=600\n",
        encoding="utf-8",
    )
    summary["all_checks_passed"] = False
    write_json(summary_path, summary)
    write_manifest(copied)

    result = subprocess.run(
        [sys.executable, "scripts/audit_trl_async_stale_resume_v1.py", str(copied)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert json.loads(result.stdout)["all_checks_passed"] is False


def test_v2_review_fix_bundle_recomputes_from_hashed_logs_and_retained_runner():
    summary = audit_bundle(V2_REVIEW_FIX_RUN)
    assert summary["run_count"] == 6
    assert summary["all_checks_passed"] is True


def test_v4_bundle_binds_and_audits_the_execution_platform():
    summary = audit_bundle(V4_PLATFORM_RUN)
    lock = json.loads(V4_PROTOCOL.read_text(encoding="utf-8"))
    assert summary["all_checks_passed"] is True
    assert summary["environment"]["execution_platform"] == lock["environment"]["execution_platform"]
    assert summary["environment"]["device_used"] == "CPU"


def test_epoch_aware_bundle_distinguishes_next_epoch_from_replayed_position():
    summary = audit_bundle(EPOCH_AWARE_RUN)
    assert summary["protocol_id"] == "trl-async-stale-resume-v2"
    baseline = [run["observed"]["duplicated_rows"] for run in summary["runs"] if run["arm"] == "baseline"]
    candidate = [run["observed"]["duplicated_rows"] for run in summary["runs"] if run["arm"] == "candidate"]
    assert baseline == [144, 145, 148]
    assert candidate == [1, 1, 1]
    assert summary["all_checks_passed"] is True


def test_superseded_v1_nonpass_is_retained_but_rejected_without_runner_source():
    summary = json.loads((V1_FALSE_NEGATIVE / "summary.json").read_text(encoding="utf-8"))
    assert summary["protocol_id"] == "trl-async-stale-resume-v1"
    assert summary["all_checks_passed"] is False
    with pytest.raises(ValueError, match="runner source is not retained"):
        audit_bundle(V1_FALSE_NEGATIVE)
    candidate_failures = [
        run for run in summary["runs"] if run["arm"] == "candidate" and not run["accepted"]
    ]
    assert [run["repetition"] for run in candidate_failures] == [1]
    failed_log = (V1_FALSE_NEGATIVE / candidate_failures[0]["log"]).read_text(encoding="utf-8")
    assert parse_pytest_outcome(failed_log)["failed"] == 1


def test_legacy_v1_bundle_is_not_accepted_without_its_runner_source():
    with pytest.raises(ValueError, match="runner source is not retained"):
        audit_bundle(LOCKED_RUN)


def test_audit_rejects_resealed_runner_hash_without_retained_source(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runner_sha256"] = "0" * 64
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="runner source hash is not approved for protocol"):
        audit_bundle(copied)


def test_audit_rejects_unapproved_embedded_runner_before_inspecting_capabilities(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    runner = copied / "runner.py"
    runner.write_text(
        'SOURCE_REVISION_PROVENANCE_CAPABILITY = "source-revision-provenance-v1"\n',
        encoding="utf-8",
    )
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runner_sha256"] = sha256_file(runner)
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="runner source hash is not approved for protocol"):
        audit_bundle(copied)


def test_audit_accepts_exact_runner_source_embedded_in_bundle(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    runner_sha = json.loads((copied / "summary.json").read_text(encoding="utf-8"))["runner_sha256"]
    snapshot = ROOT / "results/trl-async-stale-resume-runner-snapshots" / f"runner-{runner_sha}.py"
    shutil.copyfile(snapshot, copied / "runner.py")
    write_manifest(copied)

    assert audit_bundle(copied)["all_checks_passed"] is True


def test_audit_rejects_resealed_success_bundle_with_failure_in_raw_pytest_log(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    log = copied / "candidate-1.log"
    log.write_text(
        log.read_text(encoding="utf-8")
        + "\n1 failed in 0.01s\n1 passed in 0.01s\n",
        encoding="utf-8",
    )
    write_manifest(copied)

    with pytest.raises(ValueError, match="multiple pytest result summaries"):
        audit_bundle(copied)


@pytest.mark.parametrize("decision_field", ["accepted", "all_checks_passed"])
def test_audit_rejects_integer_values_for_boolean_decisions(tmp_path: Path, decision_field: str):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if decision_field == "accepted":
        summary["runs"][0][decision_field] = 1
    else:
        summary[decision_field] = 1
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="must be a boolean"):
        audit_bundle(copied)


def test_locked_run_bundle_rejects_changed_raw_log(tmp_path: Path):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    log = copied / "baseline-1.log"
    log.write_text(log.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_bundle(copied)


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked", "ignored"])
def test_reproduction_rejects_dirty_source_checkout(tmp_path: Path, dirty_kind: str):
    checkout = tmp_path / "source"
    checkout.mkdir()
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Evidence Test"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "evidence@example.invalid"], check=True
    )
    tracked = checkout / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "tracked.py"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "--quiet", "-m", "base"], check=True)

    if dirty_kind == "tracked":
        tracked.write_text("value = 2\n", encoding="utf-8")
    elif dirty_kind == "untracked":
        (checkout / "rogue.py").write_text("value = 2\n", encoding="utf-8")
    else:
        exclude = checkout / ".git/info/exclude"
        exclude.write_text("/rogue.py\n", encoding="utf-8")
        (checkout / "rogue.py").write_text("value = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source checkout must be clean"):
        source_revision(checkout)


@pytest.mark.parametrize("inventory_error", ["missing_arm", "duplicate"])
def test_audit_rejects_resealed_bundle_with_invalid_run_inventory(
    tmp_path: Path, inventory_error: str
):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = [run for run in summary["runs"] if run["arm"] == "candidate"]
    if inventory_error == "missing_arm":
        summary["runs"] = candidates + [
            {**candidates[0], "repetition": repetition, "log": f"candidate-{repetition}.log"}
            for repetition in range(4, 7)
        ]
    else:
        summary["runs"] = [candidates[0].copy() for _ in range(6)]
    # Keep the self-reported six-run count and reseal every file to exercise
    # arm/repetition validation rather than manifest integrity.
    summary["run_count"] = 6
    summary["all_checks_passed"] = True
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="duplicate run identity|run inventory mismatch"):
        audit_bundle(copied)


@pytest.mark.parametrize("provenance_field", ["environment", "gist_source_sha256"])
def test_audit_rejects_resealed_bundle_with_unlocked_execution_provenance(
    tmp_path: Path, provenance_field: str
):
    copied = tmp_path / "run"
    shutil.copytree(EPOCH_AWARE_RUN, copied)
    summary_path = copied / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if provenance_field == "environment":
        summary["environment"]["versions"]["torch"] = "0.0.0-tampered"
    else:
        summary["gist_source_sha256"] = "0" * 64
    write_json(summary_path, summary)
    write_manifest(copied)

    with pytest.raises(ValueError, match="recorded runtime environment|recorded Gist source hash"):
        audit_bundle(copied)
