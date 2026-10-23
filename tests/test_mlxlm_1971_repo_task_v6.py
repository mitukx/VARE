"""Source and workspace integrity regressions for MLX-LM #1971 v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from vare import mlxlm_1971_repo_task_v6 as task


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _make_source_repo(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "upstream"
    (root / "mlx_lm/models").mkdir(parents=True)
    files = {
        "mlx_lm/__init__.py": b"# pinned package\n",
        "mlx_lm/models/mimo_v2_flash.py": b"BASELINE = True\n",
        "mlx_lm/models/attention.py": b"def attention(): return 1\n",
    }
    for name, contents in files.items():
        (root / name).write_bytes(contents)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "test")
    _git(root, "add", "mlx_lm")
    _git(root, "commit", "-qm", "pinned baseline")
    commit = _git(root, "rev-parse", "HEAD")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "baseline_commit": commit,
                "file_count": len(files),
                "files": {name: _sha(data) for name, data in files.items()},
                "repository": "ml-explore/mlx-lm",
                "target_path": task.TARGET_PATH,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(task, "BASELINE_COMMIT", commit)
    monkeypatch.setattr(task, "BASELINE_SOURCE_SHA256", _sha(files[task.TARGET_PATH]))
    monkeypatch.setattr(task, "SOURCE_MANIFEST_PATH", manifest_path)
    return root


def test_source_verification_checks_committed_bytes_and_all_index_flags(
    tmp_path, monkeypatch
):
    root = _make_source_repo(tmp_path, monkeypatch)
    assert len(task.verify_pinned_source_tree(root)) == 3

    dependency = root / "mlx_lm/models/attention.py"
    _git(root, "update-index", "--skip-worktree", "mlx_lm/models/attention.py")
    dependency.write_text("def attention(): return 2\n", encoding="utf-8")
    assert not _git(root, "status", "--porcelain", "--untracked-files=all")
    with pytest.raises(
        ValueError, match="hidden index flag: mlx_lm/models/attention.py"
    ):
        task.verify_pinned_source_tree(root)


def test_v6_protocol_binds_manifest_and_exact_implementation():
    root = Path(__file__).resolve().parents[1]
    protocol_path = (
        root
        / "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v6.json"
    )
    lock_path = protocol_path.with_name("protocol-v6.lock.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == "vare.mlsys-repository-task.mlxlm-1971.v6"
    assert protocol["source_file_count"] == 201
    assert _sha(protocol_path.read_bytes()) == lock["protocol_sha256"]
    assert protocol["implementation_sha256"] == lock["implementation_sha256"]
    historical = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            "9f0993ac2c71a44bcfbabd2a1399fc12e5d6d777:src/vare/mlxlm_1971_repo_task.py",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout
    assert (
        _sha(historical)
        == protocol["implementation_sha256"]["src/vare/mlxlm_1971_repo_task.py"]
    )
    with pytest.raises(ValueError, match="v6 implementation hash mismatch"):
        task.verify_protocol_v6()


def test_source_verification_rejects_untracked_ignored_python_module(
    tmp_path, monkeypatch
):
    root = _make_source_repo(tmp_path, monkeypatch)
    with (root / ".git/info/exclude").open("a", encoding="utf-8") as stream:
        stream.write("/mlx_lm/models/injected.py\n")
    injected = root / "mlx_lm/models/injected.py"
    injected.write_text("import os\n", encoding="utf-8")
    assert not _git(root, "status", "--porcelain", "--untracked-files=all")
    with pytest.raises(ValueError, match="inventory differs from manifest"):
        task.verify_pinned_source_tree(root)


def _candidate(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    manifest = {
        "mlx_lm/__init__.py": _sha(b"# root package\n"),
        "mlx_lm/models/mimo_v2_flash.py": _sha(b"candidate target\n"),
        "mlx_lm/models/attention.py": _sha(b"def attention(): return 1\n"),
    }
    workspace = tmp_path / "workspace"
    package = workspace / "mlx_lm/models"
    package.mkdir(parents=True)
    (workspace / "mlx_lm/models/mimo_v2_flash.py").write_bytes(b"candidate target\n")
    (workspace / "mlx_lm/models/attention.py").write_bytes(
        b"def attention(): return 1\n"
    )
    return workspace, manifest


def test_candidate_workspace_allows_only_target_content_change(tmp_path):
    workspace, manifest = _candidate(tmp_path)
    task._candidate_inventory(workspace, manifest)
    (workspace / "mlx_lm/models/attention.py").write_text(
        "def attention(): return 9\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed a non-target dependency"):
        task._candidate_inventory(workspace, manifest)


def test_candidate_workspace_rejects_extra_module_and_symlink(tmp_path):
    workspace, manifest = _candidate(tmp_path)
    (workspace / "mlx_lm/sitecustomize.py").write_text("raise RuntimeError()\n")
    with pytest.raises(ValueError, match="file inventory differs"):
        task._candidate_inventory(workspace, manifest)
    (workspace / "mlx_lm/sitecustomize.py").unlink()
    outside = tmp_path / "outside.py"
    outside.write_text("# external\n")
    (workspace / "mlx_lm/models/mimo_v2_flash.py").unlink()
    (workspace / "mlx_lm/models/mimo_v2_flash.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        task._candidate_inventory(workspace, manifest)


def test_candidate_workspace_rejects_altered_imported_module_before_grading(tmp_path):
    workspace, manifest = _candidate(tmp_path)
    (workspace / "mlx_lm/models/attention.py").write_text(
        "def attention(): return 0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed a non-target dependency"):
        task._candidate_inventory(workspace, manifest)
