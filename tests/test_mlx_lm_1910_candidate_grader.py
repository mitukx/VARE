from scripts import grade_mlx_lm_1910_candidate_v1 as grader


def test_candidate_execution_interference_rejects_early_exit_and_output_forgery():
    source = b'import os\nprint("expected oracle output")\nos._exit(0)\n'

    findings = grader.candidate_execution_interference(source)

    assert any("process-control import" in finding for finding in findings)
    assert any("evaluator-interference name" in finding for finding in findings)
    assert any("process-control attribute" in finding for finding in findings)


def test_candidate_execution_interference_allows_normal_parser_module():
    source = b'import json\n\ndef parse_tool_call(value):\n    return json.loads(value)\n'

    assert grader.candidate_execution_interference(source) == []


def test_grade_source_rejects_interference_before_running_candidate(tmp_path, monkeypatch):
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
