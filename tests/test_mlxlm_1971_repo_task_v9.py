"""Public v9 dispatch and locked evaluator coverage."""

from __future__ import annotations

from vare import mlxlm_1971_repo_task as public_task
from vare import mlxlm_1971_repo_task_v9 as task

import pytest


def test_v9_protocol_binds_all_transitive_evaluator_modules():
    protocol = task.verify_protocol_v9()
    assert protocol["protocol_id"] == "vare.mlsys-repository-task.mlxlm-1971.v9"
    assert {
        "src/vare/__init__.py",
        "src/vare/_public.py",
        "src/vare/attestation.py",
        "src/vare/types.py",
        "src/vare/evidence.py",
        "src/vare/container_isolation.py",
        "src/vare/isolated_candidate.py",
        "src/vare/mlxlm_1971_repo_task.py",
        "src/vare/mlxlm_1971_repo_task_v7.py",
        "src/vare/mlxlm_1971_repo_task_v9.py",
    } <= set(protocol["implementation_sha256"])


def test_public_entrypoints_are_versioned_v9_dispatchers():
    assert "mlxlm_1971_repo_task_v9" in public_task.export_candidate.__code__.co_names
    assert "mlxlm_1971_repo_task_v9" in public_task.grade_workspace.__code__.co_names


@pytest.mark.parametrize(
    ("image", "timeout_s", "memory_mb", "message"),
    [
        ("other-image", 30, 1536, "image differs"),
        (None, 31, 1536, "resource limits differ"),
        (None, 30, 2048, "resource limits differ"),
    ],
)
def test_v9_rejects_non_protocol_runtime_limits(
    tmp_path, image, timeout_s, memory_mb, message
):
    kwargs = {"timeout_s": timeout_s, "memory_mb": memory_mb}
    if image is not None:
        kwargs["image"] = image
    with pytest.raises(task.MlxLm1971TaskError, match=message):
        task.grade_workspace_v9(tmp_path, **kwargs)


def test_v9_rejects_symlinked_candidate_target(tmp_path):
    workspace = tmp_path / "candidate"
    target = workspace / "mlx_lm/models/mimo_v2_flash.py"
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("pass\n", encoding="utf-8")
    target.symlink_to(outside)
    with pytest.raises(task.MlxLm1971TaskError, match="symlink"):
        task.grade_workspace_v9(workspace)
