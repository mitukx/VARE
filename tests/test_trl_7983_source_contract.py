from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

from benchmarks.mlsys_repo_eval_v1.trl_7983_support_logprob.evaluator.source_contract import (
    ast_sha256,
    canonical_ast_sha256,
    check_candidate_source,
)
from benchmarks.mlsys_repo_eval_v1.trl_7983_support_logprob.evaluator.grade import (
    EXPECTED_KNOWN_FIX_AST_SHA256,
    grade_candidate,
    generate_cases,
    reference,
)
from benchmarks.mlsys_repo_eval_v1.trl_7983_support_logprob.evaluator.runner import (
    WorkerConfig,
    _profile_text,
    _write_exact,
    grade_workspace,
)


TASK_ROOT = Path("benchmarks/mlsys_repo_eval_v1/trl_7983_support_logprob")
PROTOCOL = TASK_ROOT / "protocol.lock.v1.json"
V9_RECORD = TASK_ROOT / "calibration/v9/record.json"


def _candidate(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "trl" / "kernels" / "chunked_logprob.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_source_contract_accepts_only_pinned_ast_without_importing_candidate(
    tmp_path: Path,
):
    reference = "def normalize(x):\n    return x / 2\n"
    candidate = (
        "# comment changes are not executable\ndef normalize(x):\n    return x / 2\n"
    )
    _candidate(tmp_path, candidate)

    result, checked_bytes = check_candidate_source(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        ast_sha256(reference.encode("utf-8")),
    )

    assert result.accepted is True
    assert result.status == "SOURCE_CONTRACT_ACCEPT"
    assert result.candidate_code_imported is False
    assert result.worker_started is False
    assert checked_bytes == candidate.encode("utf-8")


def test_empty_type_params_are_normalized_but_nonempty_params_are_preserved(
    monkeypatch,
):
    tree = ast.parse("def normalize(x):\n    return x / 2\n")
    function = tree.body[0]
    baseline = canonical_ast_sha256(tree)
    fields = type(function)._fields
    monkeypatch.setattr(type(function), "_fields", fields + ("type_params",))
    function.type_params = []

    assert canonical_ast_sha256(tree) == baseline

    function.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert canonical_ast_sha256(tree) != baseline


def test_worker_report_forgery_source_is_rejected_before_execution(tmp_path: Path):
    expected = "def evaluate(x):\n    return x / 2\n"
    forged = (
        "import os\n"
        "def evaluate(x):\n"
        '    os.write(1, b\'{\\"status\\":\\"PASS\\"}\')\n'
        "    return 123.0\n"
    )
    _candidate(tmp_path, forged)

    result, checked_bytes = check_candidate_source(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        ast_sha256(expected.encode("utf-8")),
    )

    assert result.accepted is False
    assert result.status == "SOURCE_CONTRACT_REJECT"
    assert result.candidate_code_imported is False
    assert result.worker_started is False
    assert checked_bytes is None

    worker_calls = []

    def worker_runner(_source: bytes, _cases: list[dict[str, object]]) -> bytes:
        worker_calls.append(True)
        return b'{"python_version":"3.9.6","torch_version":"2.8.0","rows":[]}\n'

    grade = grade_candidate(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        ast_sha256(expected.encode("utf-8")),
        seed=101,
        worker_runner=worker_runner,
    )
    assert grade["grader_status"] == "TEST_FAIL"
    assert grade["worker_started"] is False
    assert worker_calls == []


def test_syntactically_different_equivalent_form_is_rejected(tmp_path: Path):
    expected = "def support_exp(mask, z):\n    return where(mask, exp(z), 0.0)\n"
    equivalent = (
        "def support_exp(mask, z):\n    return exp(z).masked_fill(~mask, 0.0)\n"
    )
    _candidate(tmp_path, equivalent)

    result, checked_bytes = check_candidate_source(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        ast_sha256(expected.encode("utf-8")),
    )

    assert result.accepted is False
    assert result.ast_sha256 is not None
    assert checked_bytes is None


def test_symlink_hardlink_and_path_escape_are_rejected(tmp_path: Path):
    source = "value = 1\n"
    expected = ast_sha256(source.encode("utf-8"))
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.py"
    workspace.mkdir()
    outside.write_text(source, encoding="utf-8")

    symlink = workspace / "linked.py"
    symlink.symlink_to(outside)
    symlink_result, _ = check_candidate_source(workspace, "linked.py", expected)
    assert symlink_result.accepted is False
    assert "symlink" in symlink_result.reason

    hardlink = workspace / "hardlinked.py"
    os.link(outside, hardlink)
    hardlink_result, _ = check_candidate_source(workspace, "hardlinked.py", expected)
    assert hardlink_result.accepted is False
    assert "single-link" in hardlink_result.reason

    escape_result, _ = check_candidate_source(workspace, "../outside.py", expected)
    assert escape_result.accepted is False
    assert "relative path" in escape_result.reason


def test_parent_symlink_components_are_rejected(tmp_path: Path):
    source = "value = 1\n"
    expected = ast_sha256(source.encode("utf-8"))
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "candidate.py").write_text(source, encoding="utf-8")
    (workspace / "linked-dir").symlink_to(outside, target_is_directory=True)

    result, _ = check_candidate_source(workspace, "linked-dir/candidate.py", expected)

    assert result.accepted is False
    assert "symlink" in result.reason


def test_staging_writer_rejects_zero_length_write(monkeypatch):
    monkeypatch.setattr(os, "write", lambda _descriptor, _data: 0)
    try:
        _write_exact(5, b"candidate")
    except OSError as exc:
        assert "short write" in str(exc)
    else:
        raise AssertionError("zero-length write was accepted")


def test_sandbox_profile_denies_protected_paths_after_allow_rules(tmp_path: Path):
    python_prefix = tmp_path / "python"
    runtime_root = tmp_path / "runtime"
    run_dir = tmp_path / "run"
    temp_dir = run_dir / "tmp"
    scratch_root = tmp_path / "scratch"
    worker = tmp_path / "worker.py"
    protected = tmp_path / "protected"
    for directory in (
        python_prefix,
        runtime_root,
        run_dir,
        temp_dir,
        scratch_root,
        protected,
    ):
        directory.mkdir(parents=True)
    python = python_prefix / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.write_text("pass\n", encoding="utf-8")
    config = WorkerConfig(
        python=python,
        python_sha256="0" * 64,
        python_prefix=python_prefix,
        runtime_root=runtime_root,
        runtime_read_paths=(),
        worker=worker,
        worker_sha256="0" * 64,
        protected_paths=(protected,),
        scratch_root=scratch_root,
    )

    profile = _profile_text(config, run_dir, temp_dir)
    deny_start = profile.rfind("(deny file-read*")
    allow_runtime = profile.find(f'(allow file-read* (subpath "{runtime_root}"))')

    assert allow_runtime >= 0
    assert deny_start > allow_runtime
    assert f'(deny file-read* (subpath "{protected}"))' in profile


def test_protocol_keeps_task_unallocated_and_report_untrusted():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    task = json.loads((TASK_ROOT / "task.json").read_text(encoding="utf-8"))

    assert protocol["status"] == "SOURCE_QUALIFICATION_ONLY_NO_TRAIN_ALLOCATION"
    assert protocol["grader"]["contract_class"] == "restricted_patch_contract"
    assert protocol["grader"]["worker_report_authenticated"] is False
    assert protocol["grader"]["ast_digest_algorithm"].startswith("canonical_ast_v1:")
    assert (
        protocol["source"]["known_fix_ast_sha256_canonical_v1"]
        == EXPECTED_KNOWN_FIX_AST_SHA256
    )
    assert protocol["exposure"]["pretraining_exposure"] == "UNKNOWN"
    assert (
        protocol["exposure"]["candidate_level_train_eligibility"] == "NOT_ESTABLISHED"
    )
    assert task["allocation"]["train_eligible"] is False
    assert (
        task["evaluation"]["alternate_equivalent_source_forms_may_be_rejected"] is True
    )
    task_definition_sha = hashlib.sha256(
        (TASK_ROOT / "TASK.md").read_bytes()
        + b"\0"
        + (TASK_ROOT / "task.json").read_bytes()
    ).hexdigest()
    assert protocol["task_definition_sha256"] == task_definition_sha
    for name in ("source_contract.py", "grade.py", "runner.py", "candidate_worker.py"):
        path = TASK_ROOT / "evaluator" / name
        assert (
            protocol["grader"]["source_sha256"][name]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_v9_public_record_is_bound_to_local_frozen_calibration_hashes():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    record = json.loads(V9_RECORD.read_text(encoding="utf-8"))
    calibration = protocol["calibration_history"][
        "trl-7983-source-contract-three-arm-v9"
    ]

    assert calibration["public_record_path"] == "calibration/v9/record.json"
    assert (
        calibration["public_record_sha256"]
        == hashlib.sha256(V9_RECORD.read_bytes()).hexdigest()
    )
    assert (
        record["raw_evidence_hashes"]["frozen_protocol_sha256"]
        == calibration["protocol_sha256"]
    )
    assert (
        record["raw_evidence_hashes"]["raw_summary_sha256"]
        == calibration["summary_sha256"]
    )
    assert record["status"] == "PASS_SOURCE_QUALIFICATION_ONLY"
    assert record["model_or_agent_run"] is False
    assert record["optimizer_update"] is False
    assert record["trust_limits"]["worker_report_authenticated"] is False
    assert record["trust_limits"]["candidate_level_train_eligibility"] == (
        "NOT_ESTABLISHED"
    )
    assert {row["observed_status"] for row in record["source_arms"].values()} == {
        "PASS",
        "TEST_FAIL",
    }


def test_parent_oracle_rejects_wrong_worker_values_for_pinned_source(tmp_path: Path):
    source = "def evaluate(x):\n    return x / 2\n"
    _candidate(tmp_path, source)
    expected_ast = ast_sha256(source.encode("utf-8"))
    cases = generate_cases(7)
    rows = []
    for case in cases:
        value, gradient = reference(case)
        rows.append(
            {
                "case": case["name"],
                "status": "EXECUTED",
                "logprobs": [value + 1.0],
                "weight_grad": gradient,
                "finite": True,
            }
        )
    report = {
        "python_version": "3.9.6",
        "torch_version": "2.8.0",
        "rows": rows,
    }

    result = grade_candidate(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        expected_ast,
        seed=7,
        worker_runner=lambda _source, _cases: (
            json.dumps(report, separators=(",", ":")).encode("utf-8") + b"\n"
        ),
    )

    assert result["grader_status"] == "TEST_FAIL"
    assert result["worker_started"] is True
    assert result["worker_report_authenticated"] is False
    assert not all(case["pass"] for case in result["cases"])


def test_nonfinite_candidate_values_are_semantic_failures_not_infrastructure(
    tmp_path: Path,
):
    source = "def evaluate(x):\n    return x / 2\n"
    _candidate(tmp_path, source)
    cases = generate_cases(23)
    rows = []
    for index, case in enumerate(cases):
        value, gradient = reference(case)
        if index == 0:
            rows.append(
                {
                    "case": case["name"],
                    "status": "EXECUTED",
                    "logprobs": [-0.0],
                    "weight_grad": [0.0, float("inf"), float("nan"), float("nan")],
                    "finite": False,
                }
            )
        else:
            rows.append(
                {
                    "case": case["name"],
                    "status": "EXECUTED",
                    "logprobs": [value],
                    "weight_grad": gradient,
                    "finite": True,
                }
            )
    report = {
        "python_version": "3.9.6",
        "torch_version": "2.8.0",
        "rows": rows,
    }

    result = grade_candidate(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        ast_sha256(source.encode("utf-8")),
        seed=23,
        worker_runner=lambda _source, _cases: (
            json.dumps(report, separators=(",", ":")).encode("utf-8") + b"\n"
        ),
    )

    assert result["grader_status"] == "TEST_FAIL"
    assert result["worker_started"] is True
    assert result["cases"][0]["value_ok"] is False
    assert result["cases"][0]["gradient_ok"] is False


def test_candidate_exception_row_is_a_test_failure(tmp_path: Path):
    source = "def evaluate(x):\n    return x / 2\n"
    _candidate(tmp_path, source)
    cases = generate_cases(29)
    report = {
        "python_version": "3.9.6",
        "torch_version": "2.8.0",
        "rows": [
            {
                "case": case["name"],
                "status": "TEST_ERROR",
                "error_type": "IndexError",
                "error": "candidate execution failed",
            }
            for case in cases
        ],
    }

    result = grade_candidate(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        ast_sha256(source.encode("utf-8")),
        seed=29,
        worker_runner=lambda _source, _cases: (
            json.dumps(report, separators=(",", ":")).encode("utf-8") + b"\n"
        ),
    )

    assert result["grader_status"] == "TEST_FAIL"
    assert result["cases"] == [
        {"case": case["name"], "candidate_error": "IndexError", "pass": False}
        for case in cases
    ]


def test_parent_oracle_accepts_numerically_correct_report_for_contract_source(
    tmp_path: Path,
):
    source = "def evaluate(x):\n    return x / 2\n"
    _candidate(tmp_path, source)
    cases = generate_cases(19)
    rows = []
    for case in cases:
        value, gradient = reference(case)
        rows.append(
            {
                "case": case["name"],
                "status": "EXECUTED",
                "logprobs": [value],
                "weight_grad": gradient,
                "finite": True,
            }
        )
    report = {
        "python_version": "3.9.6",
        "torch_version": "2.8.0",
        "rows": rows,
    }

    result = grade_candidate(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        ast_sha256(source.encode("utf-8")),
        seed=19,
        worker_runner=lambda _source, _cases: (
            json.dumps(report, separators=(",", ":")).encode("utf-8") + b"\n"
        ),
    )

    assert result["grader_status"] == "PASS"
    assert all(case["pass"] for case in result["cases"])
    assert result["worker_report_authenticated"] is False


def test_public_grader_entry_blocks_forger_before_runner_config_is_used(
    tmp_path: Path,
):
    forged = "import os\ndef evaluate(x):\n    os.write(1, b'fake')\n    return 123\n"
    _candidate(tmp_path, forged)
    worker_config = WorkerConfig(
        python=tmp_path / "missing-python",
        python_sha256="0" * 64,
        python_prefix=tmp_path / "missing-prefix",
        runtime_root=tmp_path / "missing-runtime",
        runtime_read_paths=(),
        worker=tmp_path / "missing-worker",
        worker_sha256="0" * 64,
        protected_paths=(),
        scratch_root=tmp_path / "missing-scratch",
    )

    result = grade_workspace(
        tmp_path,
        "trl/kernels/chunked_logprob.py",
        seed=31,
        config=worker_config,
    )

    assert result["grader_status"] == "TEST_FAIL"
    assert result["worker_started"] is False
    assert result["candidate_code_imported_by_parent"] is False
