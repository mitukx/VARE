from __future__ import annotations

from pathlib import Path

import pytest

from vare.transformers_repo_task import (
    TASK_DESCRIPTION,
    TARGET_PATH,
    TransformersTaskError,
    _validate_diff,
    grade_workspace,
)
import vare.transformers_repo_task as task_module


def _workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path / "workspace"
    target = workspace / TARGET_PATH
    target.parent.mkdir(parents=True)
    target.write_text("candidate source\n", encoding="utf-8")
    (workspace / "TASK.md").write_bytes(TASK_DESCRIPTION.read_bytes())
    import hashlib

    manifest = {
        "TASK.md": hashlib.sha256((workspace / "TASK.md").read_bytes()).hexdigest(),
    }
    return workspace, manifest


def test_diff_accepts_only_single_allowlisted_target() -> None:
    _validate_diff(
        "diff --git a/src/transformers/tokenization_utils_base.py b/src/transformers/tokenization_utils_base.py\n"
        "--- a/src/transformers/tokenization_utils_base.py\n"
        "+++ b/src/transformers/tokenization_utils_base.py\n"
        "@@ -1 +1 @@\n-before\n+after\n"
    )


@pytest.mark.parametrize(
    "header",
    [
        "diff --git a/tests/test.py b/tests/test.py\n--- a/tests/test.py\n+++ b/tests/test.py\n",
        "diff --git a/../grader.py b/../grader.py\n--- a/../grader.py\n+++ b/../grader.py\n",
        "diff --git a/src/transformers/tokenization_utils_base.py b/src/transformers/tokenization_utils_base.py\n"
        "--- a/src/transformers/tokenization_utils_base.py\n+++ b/src/transformers/extra.py\n",
    ],
)
def test_diff_rejects_other_paths(header: str) -> None:
    with pytest.raises(TransformersTaskError):
        _validate_diff(header + "@@ -1 +1 @@\n-a\n+b\n")


def test_grader_requires_trusted_baseline_manifest(tmp_path: Path) -> None:
    workspace, _ = _workspace(tmp_path)
    with pytest.raises(TransformersTaskError, match="trusted baseline"):
        grade_workspace(
            workspace,
            image="localhost:5000/vare/transformers-cpu@sha256:"
            + "5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f",
        )


def test_grader_rejects_candidate_added_files_before_execution(tmp_path: Path) -> None:
    workspace, manifest = _workspace(tmp_path)
    (workspace / "tests").mkdir()
    (workspace / "tests" / "leak.py").write_text("secret = 1\n", encoding="utf-8")
    with pytest.raises(TransformersTaskError, match="file set"):
        grade_workspace(
            workspace,
            image="localhost:5000/vare/transformers-cpu@sha256:"
            + "5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f",
            baseline_files=manifest,
        )


def test_grader_rejects_mutated_task_statement(tmp_path: Path) -> None:
    workspace, manifest = _workspace(tmp_path)
    (workspace / "TASK.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(TransformersTaskError, match="changed a non-allowlisted"):
        grade_workspace(
            workspace,
            image="localhost:5000/vare/transformers-cpu@sha256:"
            + "5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f",
            baseline_files=manifest,
        )


def test_grader_rejects_image_not_in_frozen_protocol(tmp_path: Path) -> None:
    workspace, manifest = _workspace(tmp_path)
    with pytest.raises(TransformersTaskError, match="frozen task protocol"):
        grade_workspace(
            workspace,
            image="localhost:5000/vare/other@sha256:" + "f" * 64,
            baseline_files=manifest,
        )


@pytest.mark.parametrize(
    ("timeout_s", "memory_mb"),
    [(300.0, 1536), (30.0, 4096), (29.0, 1536), (30.0, 1024)],
)
def test_grader_rejects_limits_that_differ_from_frozen_protocol(
    tmp_path: Path, timeout_s: float, memory_mb: int
) -> None:
    workspace, manifest = _workspace(tmp_path)
    with pytest.raises(TransformersTaskError, match="limits differ"):
        grade_workspace(
            workspace,
            image="localhost:5000/vare/transformers-cpu@sha256:"
            "5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f",
            baseline_files=manifest,
            timeout_s=timeout_s,
            memory_mb=memory_mb,
        )


def test_grader_rejects_changed_locked_implementation(tmp_path: Path, monkeypatch) -> None:
    workspace, manifest = _workspace(tmp_path)
    original = task_module.sha256_file

    def altered(path: Path) -> str:
        if path == task_module.ROOT / "src/vare/isolated_candidate.py":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(task_module, "sha256_file", altered)
    with pytest.raises(TransformersTaskError, match="implementation hash mismatch"):
        grade_workspace(
            workspace,
            image="localhost:5000/vare/transformers-cpu@sha256:"
            "5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f",
            baseline_files=manifest,
        )


def test_grader_rejects_protocol_bytes_outside_protocol_lock(
    tmp_path: Path, monkeypatch
) -> None:
    workspace, manifest = _workspace(tmp_path)
    protocol = tmp_path / "protocol.json"
    lock = tmp_path / "protocol.lock.json"
    protocol.write_text('{"protocol_id":"changed"}\n', encoding="utf-8")
    lock.write_text(
        '{"protocol_id":"transformers-45852-isolated-eval-v2",'
        '"protocol_sha256":"' + "f" * 64 + '"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(task_module, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(task_module, "PROTOCOL_LOCK_PATH", lock)
    with pytest.raises(TransformersTaskError, match="protocol hash or identity"):
        grade_workspace(
            workspace,
            image="localhost:5000/vare/transformers-cpu@sha256:"
            "5461f5ec9d3022db72d49cf5bcd140c99521eba5b37747257b58a576d4e12e1f",
            baseline_files=manifest,
        )


def test_nonuniform_cases_distinguish_constant_and_mean_shortcuts() -> None:
    from vare.transformers_repo_task import _cases

    cases = {case.case_id: case for case in _cases()}
    assert cases["ordinary_variable_length_padded_batch"].expected["input_ids"] == [
        [1, 2, 1], [4, 2, 3]
    ]
    for name in ("torch_nonuniform_4d_mask", "numpy_nonuniform_4d_mask"):
        actual = cases[name].expected["values"]
        assert any(len(set(row)) > 1 for batch in actual for sample in batch for row in sample)
        flattened = [value for batch in actual for sample in batch for row in sample for value in row]
        assert len(set(flattened)) > 2


def test_inventory_rejects_symlink(tmp_path: Path) -> None:
    from vare.transformers_repo_task import sha256_inventory

    workspace, _ = _workspace(tmp_path)
    (workspace / "src/transformers/grader.py").symlink_to(workspace / "TASK.md")
    with pytest.raises(TransformersTaskError, match="symlinks"):
        sha256_inventory(workspace)
