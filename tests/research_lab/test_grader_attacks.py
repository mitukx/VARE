"""Grader-side attacks. These do not cover runner resource or network limits."""

from __future__ import annotations

import hashlib
from pathlib import Path

from vare.research_lab import evaluation as evaluation_mod
from vare.research_lab.evaluation import grade
from vare.research_lab.isolation import NOT_A_TRUSTED_SANDBOX
from vare.research_lab.spec import load_catalog
from vare.research_lab.workspace import IntegrityError, materialize, stage_subject
from vare.research_lab import paths


def _baseline(tmp_path: Path, task_id: str = "grad_accum_scale_v1") -> tuple[object, Path]:
    task = load_catalog()[task_id]
    workspace = tmp_path / "workspace"
    materialize(task, workspace)
    return task, workspace


def _assert_wrong_candidate_is_not_accepted(result: dict[str, object]) -> None:
    assert result["passed"] is False
    if result["classification"] == "isolation_unverified":
        assert result["correctness_evaluated"] is False
        assert result["correctness_passed"] is None
        return
    assert result["classification"] == "fail_correctness"
    assert result["correctness_evaluated"] is True
    assert result["correctness_passed"] is False


def test_metric_file_edit_does_not_pass(tmp_path: Path):
    # result.json forgery is covered by test_security.py. This case is the
    # separate metrics file the grader must also ignore.
    task, workspace = _baseline(tmp_path)
    (workspace / "metrics.json").write_text(
        '{"passed": true, "max_abs_error": 0}\n',
        encoding="utf-8",
    )
    result = grade(task, workspace)
    _assert_wrong_candidate_is_not_accepted(result)
    assert "not a trusted security sandbox" in NOT_A_TRUSTED_SANDBOX


def test_mock_substitution_is_scored_from_values(tmp_path: Path):
    task, workspace = _baseline(tmp_path)
    (workspace / "accum.py").write_text(
        "def accumulated_gradient(batches, params):\n"
        "    return {'passed': True, 'max_abs_error': 0.0}\n",
        encoding="utf-8",
    )
    result = grade(task, workspace)
    _assert_wrong_candidate_is_not_accepted(result)


def test_hidden_reference_read_does_not_pass(tmp_path: Path):
    task, workspace = _baseline(tmp_path)
    reference = Path(paths.__file__).resolve().parent / "reference.py"
    before = hashlib.sha256(reference.read_bytes()).hexdigest()
    (workspace / "accum.py").write_text(
        "def accumulated_gradient(batches, params):\n"
        "    from pathlib import Path\n"
        "    tried = [\n"
        "        Path('reference.py'),\n"
        "        Path('../reference.py'),\n"
        "        Path('/workspace/src/vare/research_lab/reference.py'),\n"
        "    ]\n"
        "    for path in tried:\n"
        "        try:\n"
        "            text = path.read_text(encoding='utf-8')\n"
        "        except OSError:\n"
        "            continue\n"
        "        if 'def accumulated_gradient' in text:\n"
        "            namespace = {}\n"
        "            exec(text, namespace, namespace)\n"
        "            return namespace['accumulated_gradient'](batches, params)\n"
        "    try:\n"
        "        from vare.research_lab.reference import accumulated_gradient as hidden\n"
        "    except Exception:\n"
        "        return list(params)\n"
        "    return hidden(batches, params)\n",
        encoding="utf-8",
    )
    staged = tmp_path / "subject"
    stage_subject(workspace, task.editable_files, staged)
    assert [path.name for path in staged.iterdir()] == ["accum.py"]
    result = grade(task, workspace)
    assert result["passed"] is False
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == before


def test_evaluator_modification_fails_closed(monkeypatch, tmp_path: Path):
    task, workspace = _baseline(tmp_path)
    sentinel = tmp_path / "grader_sentinel.py"
    sentinel.write_text("v1\n", encoding="utf-8")
    monkeypatch.setattr(evaluation_mod, "_grader_files", lambda: (sentinel,))
    real = evaluation_mod.launch_worker

    def tamper(subject, calls, limits):
        sentinel.write_text("tampered\n", encoding="utf-8")
        return real(subject, calls, limits)

    monkeypatch.setattr(evaluation_mod, "launch_worker", tamper)
    result = grade(task, workspace)
    assert result["passed"] is False
    assert result["classification"] == "evaluator_modified"
    assert result["correctness_evaluated"] is False


def test_symlink_into_grader_asset_is_rejected(tmp_path: Path):
    task, workspace = _baseline(tmp_path)
    reference = Path(paths.__file__).resolve().parent / "reference.py"
    target = workspace / "accum.py"
    target.unlink()
    target.symlink_to(reference)
    try:
        stage_subject(workspace, task.editable_files, tmp_path / "subject")
    except (IntegrityError, paths.PathViolation):
        pass
    else:
        raise AssertionError("symlink into a grader file was staged")
    result = grade(task, workspace)
    assert result["passed"] is False
    assert result["classification"] == "integrity"


def test_stale_source_hash_is_rejected(tmp_path: Path):
    task, workspace = _baseline(tmp_path)
    result = grade(task, workspace, expected_hashes={"accum.py": "0" * 64})
    assert result["passed"] is False
    assert result["classification"] == "stale_source"
    assert result["correctness_evaluated"] is False


def test_unverified_isolation_does_not_grade(monkeypatch, tmp_path: Path):
    task, workspace = _baseline(tmp_path)

    class Unverified:
        ok = False
        failures = ("backend_unavailable",)

    monkeypatch.setattr(
        "vare.research_lab.isolation.attested_backend",
        lambda: (None, Unverified()),
    )
    result = grade(task, workspace)
    assert result["passed"] is False
    assert result["classification"] == "isolation_unverified"
    assert result["correctness_evaluated"] is False
