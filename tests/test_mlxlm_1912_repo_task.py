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
