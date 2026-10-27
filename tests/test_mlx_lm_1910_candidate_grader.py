from scripts import grade_mlx_lm_1910_candidate_v1 as grader


def test_candidate_execution_interference_rejects_early_exit_and_output_forgery():
    source = b'import os\nprint("expected oracle output")\nos._exit(0)\nopen(1, "w").write("forged")\n'

    findings = grader.candidate_execution_interference(source)

    assert any("process-control import" in finding for finding in findings)
    assert any("evaluator-interference name" in finding for finding in findings)
    assert any("process-control attribute" in finding for finding in findings)
    assert any("open" in finding for finding in findings)


def test_candidate_execution_interference_allows_normal_parser_module():
    source = b'import json\n\ndef parse_tool_call(value):\n    return json.loads(value)\n'

    assert grader.candidate_execution_interference(source) == []


def test_grade_source_rejects_interference_before_running_candidate(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import hashlib

    monkeypatch.setattr(grader.CAL, "BASE_FILE_SHA256", hashlib.sha256(BASE_MODULE).hexdigest())
    monkeypatch.setattr(
        grader.CAL, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=BASE_MODULE),
    )
    candidate = tmp_path / "candidate.py"
    candidate.write_bytes(b'import os\nprint("forged")\nos._exit(0)\n')

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("candidate reached the isolated evaluator")

    monkeypatch.setattr(grader.CAL, "archive_commit", should_not_run)
    monkeypatch.setattr(grader.CAL, "docker_run", should_not_run)

    result = grader.grade_source(tmp_path, candidate, tmp_path / "grade")

    assert result["status"] == "FAIL"
    assert result["candidate"]["execution_boundary"]["passed"] is False
    assert (tmp_path / "grade" / "grade.json").is_file()


BASE_MODULE = b"import json\n\ndef _convert_param_value(value):\n    return value\n\ndef parse_tool_call(value):\n    return _convert_param_value(value)\n"


def test_candidate_scope_allows_only_parser_body_changes():
    candidate = BASE_MODULE.replace(
        b"    return value\n",
        b"    try:\n        return json.loads(value)\n    except json.JSONDecodeError:\n        return value\n",
        1,
    )

    assert grader.candidate_module_scope_interference(candidate, BASE_MODULE) == []


def test_candidate_scope_rejects_module_level_output_and_early_exit():
    candidate = BASE_MODULE + (
        b"\nfrom pathlib import Path\n"
        b"Path('/proc/self/fd/1').write_text('[{\"ok\":true}]\\n')\n"
        b"raise SystemExit(0)\n"
    )

    findings = grader.candidate_module_scope_interference(candidate, BASE_MODULE)

    assert findings == ["candidate changes code outside _convert_param_value body"]


def test_candidate_scope_rejects_changes_to_imports_and_other_helpers():
    candidate = BASE_MODULE.replace(
        b"import json\n", b"import json\nfrom pathlib import Path\n", 1,
    )

    findings = grader.candidate_module_scope_interference(candidate, BASE_MODULE)

    assert findings == ["candidate changes code outside _convert_param_value body"]


def test_grade_source_records_scope_rejection_before_candidate_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import hashlib

    baseline = BASE_MODULE
    monkeypatch.setattr(grader.CAL, "BASE_FILE_SHA256", hashlib.sha256(baseline).hexdigest())
    monkeypatch.setattr(
        grader.CAL, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=baseline),
    )
    candidate = tmp_path / "candidate.py"
    candidate.write_bytes(BASE_MODULE + b"\nraise SystemExit(0)\n")

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("candidate reached the isolated evaluator")

    monkeypatch.setattr(grader.CAL, "archive_commit", should_not_run)
    monkeypatch.setattr(grader.CAL, "docker_run", should_not_run)

    result = grader.grade_source(tmp_path, candidate, tmp_path / "grade-scope")

    assert result["status"] == "FAIL"
    assert result["candidate"]["execution_boundary"]["passed"] is False
    assert (tmp_path / "grade-scope" / "grade.json").is_file()
