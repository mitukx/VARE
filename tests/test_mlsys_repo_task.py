from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import vare.mlsys_repo_task as repo_task
from vare.mlsys_repo_task import (
    RepositoryTaskError,
    _diff_paths,
    add_trusted_adapter,
    apply_candidate_patch,
)


def valid_diff(path: str = "mlx_lm/server.py") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "index 1234567..abcdef0 100644\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
    )


def test_patch_paths_are_confined_to_one_allowlisted_file():
    assert _diff_paths(valid_diff()) == ["mlx_lm/server.py", "mlx_lm/server.py"]
    attacks = [
        valid_diff("../../grader/hidden.py"),
        valid_diff("/tmp/escape.py"),
        valid_diff("mlx_lm/server.py") + valid_diff("mlx_lm/other.py"),
        valid_diff().replace("b/mlx_lm/server.py", "b/../../grader.py"),
        valid_diff().replace(
            "diff --git a/mlx_lm/server.py b/mlx_lm/server.py",
            "diff --git a/mlx_lm/server.py",
        ),
        valid_diff().replace("index 1234567..abcdef0 100644", "new file mode 100644"),
    ]
    for patch in attacks:
        with pytest.raises(RepositoryTaskError):
            _diff_paths(patch)


def test_trusted_adapter_cannot_be_supplied_by_candidate(tmp_path: Path):
    adapter = tmp_path / "vare_task_adapter.py"
    adapter.write_text("attacker content")
    with pytest.raises(RepositoryTaskError, match="already contains"):
        add_trusted_adapter(tmp_path)
    assert adapter.read_text() == "attacker content"


def test_trusted_adapter_uses_fixed_entrypoint_and_no_grader_cases(tmp_path: Path):
    add_trusted_adapter(tmp_path)
    source = (tmp_path / "vare_task_adapter.py").read_text()
    assert "process_message_content" in source
    assert "/subject/mlx_lm/server.py" in source
    assert "expected" not in source
    assert "grader" not in source


def test_patch_input_and_candidate_workspace_symlinks_fail_closed(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "source"
    (root / "mlx_lm").mkdir(parents=True)
    (root / "mlx_lm/server.py").write_text("baseline")
    patch = tmp_path / "patch.diff"
    patch.write_text(valid_diff())
    patch_link = tmp_path / "patch-link.diff"
    patch_link.symlink_to(patch)
    workspace = tmp_path / "workspace"
    workspace.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(repo_task, "verify_baseline", lambda _root: None)

    with pytest.raises(RepositoryTaskError, match="regular file"):
        apply_candidate_patch(root, patch_link, tmp_path / "fresh")
    with pytest.raises(RepositoryTaskError, match="already exists"):
        apply_candidate_patch(root, patch, workspace)


def test_pinned_source_target_symlink_fails_closed(tmp_path: Path, monkeypatch):
    root = tmp_path / "source"
    package = root / "mlx_lm"
    package.mkdir(parents=True)
    (root / "real.py").write_text("safe")
    (package / "server.py").symlink_to(root / "real.py")
    monkeypatch.setattr(repo_task, "BASELINE_COMMIT", "test-commit")
    from subprocess import CompletedProcess

    calls = iter(("test-commit\n", "", ""))

    def fake_run(*_args, **_kwargs):
        return CompletedProcess([], 0, next(calls), "")

    monkeypatch.setattr(repo_task.subprocess, "run", fake_run)
    with pytest.raises(RepositoryTaskError, match="symlink"):
        repo_task.verify_baseline(root)


def _git_baseline(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "git-source"
    target = root / repo_task.TARGET_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(b"pinned baseline bytes")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(root), "add", repo_task.TARGET_PATH], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "baseline"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(repo_task, "BASELINE_COMMIT", commit)
    monkeypatch.setattr(
        repo_task, "BASELINE_SERVER_SHA256", repo_task.sha256_file(target)
    )
    return root


def test_fixed_source_hash_is_independent_of_git_replace_refs(tmp_path: Path):
    repo = tmp_path / "fixed-source"
    target = repo / "mlx_lm/server.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"pinned fixed source\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "mlx_lm/server.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "pinned fix"], check=True)
    fixed_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    pinned_bytes = target.read_bytes()
    expected_sha256 = hashlib.sha256(pinned_bytes).hexdigest()

    target.write_bytes(b"attacker replacement source\n")
    subprocess.run(["git", "-C", str(repo), "add", "mlx_lm/server.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "replacement"], check=True)
    replacement = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repo), "replace", fixed_commit, replacement], check=True
    )

    replaced = subprocess.run(
        ["git", "-C", str(repo), "show", f"{fixed_commit}:mlx_lm/server.py"],
        check=True,
        capture_output=True,
    ).stdout
    assert replaced != pinned_bytes
    assert repo_task.read_pinned_source(
        repo, fixed_commit, "mlx_lm/server.py", expected_sha256
    ) == pinned_bytes
    with pytest.raises(RepositoryTaskError, match="SHA-256 mismatch"):
        repo_task.read_pinned_source(
            repo, fixed_commit, "mlx_lm/server.py", hashlib.sha256(replaced).hexdigest()
        )


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_baseline_rejects_git_index_flags_even_when_source_is_unchanged(
    tmp_path: Path,
    monkeypatch,
    index_flag: str,
):
    root = _git_baseline(tmp_path, monkeypatch)
    subprocess.run(
        ["git", "-C", str(root), "update-index", index_flag, repo_task.TARGET_PATH],
        check=True,
    )
    # Git status is clean for both flags, so the index metadata itself must be rejected.
    assert not subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    with pytest.raises(RepositoryTaskError, match="index flags"):
        repo_task.verify_baseline(root)


def test_baseline_rejects_target_hash_mismatch_even_when_git_status_is_clean(
    tmp_path: Path,
    monkeypatch,
):
    root = _git_baseline(tmp_path, monkeypatch)
    monkeypatch.setattr(repo_task, "BASELINE_SERVER_SHA256", "0" * 64)
    assert not subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    with pytest.raises(RepositoryTaskError, match="SHA-256"):
        repo_task.verify_baseline(root)


def _red_team_report() -> dict:
    expected_pass = {
        "grader_path_probe",
        "git_history_probe",
        "environment_probe",
        "network_probe",
        "subprocess_write_probe",
    }
    attempts = {
        name: {
            "audit_check_passed": True,
            "expected_grader_passed": name in expected_pass,
            "grader_passed": name in expected_pass,
        }
        for name in repo_task.RED_TEAM_PROBES
    }
    attempts["subprocess_write_probe"]["host_workspace_write"] = False
    return {
        "experiment_id": "mlsys-repository-task-mlxlm-1904-red-team-v1",
        "source_checkout": repo_task.BASELINE_COMMIT,
        "image": "python@sha256:example",
        "network": "none",
        "attempts": attempts,
        "decision": {"all_attacks_checked": True},
    }


def test_red_team_evidence_requires_complete_consistent_suite_and_pinned_bytes(
    tmp_path: Path,
):
    path = tmp_path / "red-team.json"
    raw = json.dumps(_red_team_report(), sort_keys=True).encode()
    path.write_bytes(raw)
    verified = repo_task.verify_red_team_evidence(
        path,
        image="python@sha256:example",
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert verified["probe_count"] == 8

    for mutate, match in [
        (lambda report: report["attempts"].pop("timeout"), "required suite"),
        (
            lambda report: report["attempts"]["timeout"].update(
                audit_check_passed=False
            ),
            "failed or is inconsistent",
        ),
        (
            lambda report: report["attempts"]["grader_path_probe"].update(
                grader_passed=False
            ),
            "failed or is inconsistent",
        ),
        (
            lambda report: report["attempts"]["subprocess_write_probe"].update(
                host_workspace_write=True
            ),
            "escaped",
        ),
    ]:
        report = _red_team_report()
        mutate(report)
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(RepositoryTaskError, match=match):
            repo_task.verify_red_team_evidence(path, image="python@sha256:example")


def test_red_team_evidence_missing_malformed_or_hash_tampered_fails_closed(
    tmp_path: Path,
):
    path = tmp_path / "red-team.json"
    with pytest.raises(RepositoryTaskError, match="missing"):
        repo_task.verify_red_team_evidence(path, image="python@sha256:example")
    path.write_text("not-json", encoding="utf-8")
    with pytest.raises(RepositoryTaskError, match="malformed"):
        repo_task.verify_red_team_evidence(path, image="python@sha256:example")
    path.write_text(json.dumps(_red_team_report()), encoding="utf-8")
    with pytest.raises(RepositoryTaskError, match="SHA-256"):
        repo_task.verify_red_team_evidence(
            path,
            image="python@sha256:example",
            expected_sha256="0" * 64,
        )


def test_protocol_implementation_lock_requires_all_exact_file_hashes(tmp_path: Path):
    locked = {}
    for relative in repo_task.LOCKED_IMPLEMENTATION_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        locked[relative] = hashlib.sha256(relative.encode()).hexdigest()
    protocol = {"implementation_sha256": locked}
    assert repo_task.verify_protocol_implementation(protocol, tmp_path) == locked

    altered = next(iter(repo_task.LOCKED_IMPLEMENTATION_FILES))
    (tmp_path / altered).write_text("changed bytes", encoding="utf-8")
    with pytest.raises(RepositoryTaskError, match="hash mismatch"):
        repo_task.verify_protocol_implementation(protocol, tmp_path)


def test_protocol_implementation_lock_fails_closed_on_missing_entries(
    tmp_path: Path,
):
    locked = {relative: "0" * 64 for relative in repo_task.LOCKED_IMPLEMENTATION_FILES}
    locked.pop(next(iter(repo_task.LOCKED_IMPLEMENTATION_FILES)))
    with pytest.raises(RepositoryTaskError, match="incomplete"):
        repo_task.verify_protocol_implementation(
            {"implementation_sha256": locked}, tmp_path
        )


@pytest.mark.parametrize("index_flag", ["--assume-unchanged", "--skip-worktree"])
def test_calibration_inputs_reject_hidden_git_index_flags(
    tmp_path: Path, index_flag: str
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
    )
    source = repo / "locked.py"
    source.write_text("pinned bytes\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "locked.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "pin"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "update-index", index_flag, "locked.py"],
        check=True,
    )
    source.write_text("altered but status-hidden\n", encoding="utf-8")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""
    with pytest.raises(RepositoryTaskError, match="index flag"):
        repo_task.verify_no_index_flags(repo, {"locked.py"})


def test_protocol_image_must_match_frozen_runtime():
    protocol = {"runtime": {"image": "python@sha256:" + "a" * 64}}
    repo_task.verify_protocol_image(protocol, "python@sha256:" + "a" * 64)
    with pytest.raises(RepositoryTaskError, match="frozen protocol image"):
        repo_task.verify_protocol_image(protocol, "python@sha256:" + "b" * 64)


def test_candidate_visible_task_prompt_must_match_frozen_protocol(tmp_path: Path):
    path = tmp_path / "task.md"
    path.write_text("frozen prompt\n", encoding="utf-8")
    protocol = {
        "task_contract": {"prompt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    }
    assert repo_task.verify_task_prompt(protocol, path) == protocol["task_contract"][
        "prompt_sha256"
    ]
    path.write_text("changed prompt\n", encoding="utf-8")
    with pytest.raises(RepositoryTaskError, match="prompt hash"):
        repo_task.verify_task_prompt(protocol, path)


def test_calibration_rejects_mutable_implementation_worktree():
    gates = {
        "baseline_failed": True,
        "known_fix_passed": True,
        "fixed_source_matches": True,
        "source_swap_rejected": True,
        "red_team_verified": True,
    }
    assert repo_task.calibration_accepts(**gates, implementation_worktree_clean=True)
    assert not repo_task.calibration_accepts(
        **gates, implementation_worktree_clean=False
    )
