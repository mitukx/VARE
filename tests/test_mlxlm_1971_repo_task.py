from __future__ import annotations

from pathlib import Path

import pytest

from vare.mlxlm_1971_repo_task import (
    MlxLm1971TaskError,
    TARGET_PATH,
    validate_patch,
)


def _patch(path: str = TARGET_PATH) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
    )


def test_patch_is_limited_to_the_frozen_target():
    validate_patch(_patch())
    for attack in (
        _patch("../../grader.py"),
        _patch("mlx_lm/models/other.py"),
        _patch() + _patch("mlx_lm/models/other.py"),
        _patch().replace(f"b/{TARGET_PATH}", "b/../../grader.py"),
        _patch().replace("@@ -1 +1 @@", "new file mode 100644\n@@ -1 +1 @@"),
    ):
        with pytest.raises(MlxLm1971TaskError):
            validate_patch(attack)


def test_patch_rejects_empty_binary_and_oversized_inputs():
    for attack in ("", "GIT binary patch", "x" * 256_001):
        with pytest.raises(MlxLm1971TaskError):
            validate_patch(attack)


def test_candidate_workspace_rejects_symlinked_target(tmp_path: Path):
    from vare.mlxlm_1971_repo_task import grade_workspace

    workspace = tmp_path / "candidate"
    target = workspace / TARGET_PATH
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n")
    target.symlink_to(outside)
    with pytest.raises(MlxLm1971TaskError, match="unsafe"):
        grade_workspace(workspace)
