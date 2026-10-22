from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from vare.mlxlm_1912_repo_task import (
    MlxLm1912TaskError,
    TARGET_PATH,
    _validate_patch,
    verify_baseline,
)


def valid_patch(path: str = TARGET_PATH) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "index 1234567..abcdef0 100644\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
    )


def test_patch_is_limited_to_pinned_target():
    _validate_patch(valid_patch())
    attacks = (
        valid_patch("../../grader/test.py"),
        valid_patch("/tmp/escape.py"),
        valid_patch() + valid_patch("mlx_lm/other.py"),
        valid_patch().replace("b/mlx_lm/sample_utils.py", "b/../../grader.py"),
        valid_patch().replace("index 1234567..abcdef0 100644", "new file mode 100644"),
    )
    for patch in attacks:
        with pytest.raises(MlxLm1912TaskError):
            _validate_patch(patch)


def test_patch_rejects_empty_binary_and_oversized_inputs():
    for patch in ("", "GIT binary patch", "x" * 256_001):
        with pytest.raises(MlxLm1912TaskError):
            _validate_patch(patch)


def test_baseline_source_symlink_is_rejected(tmp_path: Path, monkeypatch):
    root = tmp_path / "source"
    package = root / "mlx_lm"
    package.mkdir(parents=True)
    real = root / "real.py"
    real.write_text("safe")
    (package / "sample_utils.py").symlink_to(real)
    values = iter(("pinned\n", "", "H mlx_lm/sample_utils.py\n"))

    def fake_run(*_args, **_kwargs):
        return CompletedProcess([], 0, next(values), "")

    monkeypatch.setattr("vare.mlxlm_1912_repo_task.subprocess.run", fake_run)
    with pytest.raises(MlxLm1912TaskError):
        verify_baseline(root)

def test_prospective_v4_fixtures_defeat_constant_index_zero(monkeypatch, tmp_path: Path):
    import json
    import vare.mlxlm_1912_repo_task as repo_task

    source = tmp_path / "workspace"
    target = source / "mlx_lm/sample_utils.py"
    target.parent.mkdir(parents=True)
    target.write_text("def apply_top_p(*args): pass\n", encoding="utf-8")
    captured = []

    class FakeReport:
        passed = False
        cases = []

    def capture(_subject, calls, **_kwargs):
        captured.extend(calls)
        return FakeReport()

    monkeypatch.setattr(repo_task, "grade_calls", capture)
    repo_task.grade_workspace(source, image="unused")
    cases = {c.case_id: c for c in captured}
    shifted = cases["shifted_maximum_nonzero_values"]
    assert shifted.expected["kept_indices"] == [[1], [2]]
    assert shifted.expected["kept_values"] == [[-0.625], [-1.25]]
    assert all(
        value != 0.0 for values in shifted.expected["kept_values"] for value in values
    )
    assert cases["leading_maximum_nonzero_value"].expected["kept_values"] == [[-0.75]]
