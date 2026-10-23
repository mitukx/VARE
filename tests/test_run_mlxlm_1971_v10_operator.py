"""Source-only, no-model regression checks for the prospective v10 operator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from scripts import run_mlxlm_1971_v10_operator as v10


def _valid_child_report(action: str) -> dict[str, object]:
    protocol_path = v10.ROOT / v10.V9_PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if action == "grade":
        cases = [
            {
                "case_id": f"value-scale-{scale:g}-{phase}",
                "passed": False,
                "failure": "wrong_output",
                "wall_s": 0.1,
            }
            for scale in protocol["value_scales"]
            for phase in ("full", "cached")
        ]
        return {
            "task_id": protocol["task_id"],
            "protocol_id": protocol["protocol_id"],
            "passed": False,
            "cases": cases,
            "candidate_source_sha256": "a" * 64,
            "candidate_manifest_sha256": protocol["source_manifest_sha256"],
            "grader_source_sha256": protocol["implementation_sha256"][
                "src/vare/mlxlm_1971_repo_task_v9.py"
            ],
            "grader_adapter_sha256": protocol["implementation_sha256"][
                "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/grader_adapter_v4.py"
            ],
            "expected_output_sha256": protocol["expected_output_sha256"],
        }
    return {
        "task_id": protocol["task_id"],
        "baseline_commit": protocol["baseline_commit"],
        "candidate_patch_sha256": None,
        "candidate_source_sha256": "a" * 64,
        "fixed_source_match": False,
        "changed_files": [],
        "baseline_python_manifest_sha256": protocol["source_manifest_sha256"],
        "git_history_exported": False,
        "tests_exported": False,
        "license_exported": False,
        "task_prompt_sha256": protocol["task_prompt_sha256"],
    }


def _mock_git_bytes(revision: str, relative: str) -> bytes:
    if revision == v10.V9_COMMIT and relative == v10.V9_PROTOCOL_PATH:
        return (v10.ROOT / relative).read_bytes()
    return b"trusted historical operator"


def test_v10_protocol_binds_git_commit_and_frozen_v9_without_model() -> None:
    protocol = v10.verify_v10_protocol()
    assert protocol["operator_source_commit"] == v10.OPERATOR_COMMIT
    assert protocol["reference_v9_source_commit"] == v10.V9_COMMIT
    assert protocol["allowed_actions"] == ["export", "grade"]
    assert hashlib.sha256(v10.PROTOCOL.read_bytes()).hexdigest() == v10.PROTOCOL_SHA256


def test_v10_coherent_protocol_and_lock_edit_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = json.loads(v10.PROTOCOL.read_text(encoding="utf-8"))
    protocol["allowed_actions"] = ["grade", "export"]
    candidate = tmp_path / "protocol-v10.json"
    candidate.write_text(json.dumps(protocol), encoding="utf-8")
    lock = tmp_path / "protocol-v10.lock.json"
    lock.write_text(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    monkeypatch.setattr(v10, "PROTOCOL", candidate)
    monkeypatch.setattr(v10, "LOCK", lock)
    monkeypatch.setattr(
        v10, "_git_bytes",
        lambda *args: pytest.fail("coherent protocol tampering must fail before Git"),
    )
    with pytest.raises(ValueError, match="independently pinned bytes"):
        v10.verify_v10_protocol()


def test_v10_rejects_invalid_actions_before_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        v10, "verify_v10_protocol",
        lambda: pytest.fail("bad input must fail before protocol verification"),
    )
    with pytest.raises(ValueError, match="unsupported v10 operation"):
        v10.run_v10_operation("delete", tmp_path / "candidate")
    with pytest.raises(ValueError, match="export requires source"):
        v10.run_v10_operation("export", tmp_path / "candidate")
    with pytest.raises(ValueError, match="grade cannot receive"):
        v10.run_v10_operation("grade", tmp_path / "candidate", source_repo=tmp_path)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        v10.run_v10_operation("grade", symlink)


@pytest.mark.parametrize("action", ["grade", "export"])
def test_v10_dispatch_uses_pinned_source_and_disables_replace_refs(
    action: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    operations: list[list[str]] = []
    source = tmp_path / "upstream"
    source.mkdir()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        operations.append(command)
        assert kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
        if command[:1] == ["git"]:
            assert "core.hooksPath=/dev/null" in command
            assert "core.fsmonitor=false" in command
            assert kwargs["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
        if command[:1] == ["git"] and "worktree" in command:
            assert kwargs["check"] is True
            if "add" in command:
                checkout = Path(command[-2])
                for relative in v10.PINNED_OPERATOR_SOURCES:
                    checked = checkout / relative
                    checked.parent.mkdir(parents=True, exist_ok=True)
                    checked.write_bytes(b"trusted historical operator")
        if command[:1] == ["git"] and "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout="")
        if command[:3] == [sys.executable, "-I", "-c"]:
            request = json.loads(kwargs["input"])
            assert request["action"] == action
            if action == "export":
                assert request["source_repo"] == str(source)
            assert "run_pinned_v9_task" in command[3]
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(_valid_child_report(action))
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(v10, "verify_v10_protocol", lambda: {})
    monkeypatch.setattr(v10, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v10.subprocess, "run", fake_run)
    result = v10.run_v10_operation(
        action, tmp_path / "workspace",
        source_repo=source if action == "export" else None,
    )
    assert result == _valid_child_report(action)
    assert len(operations) == 4
    assert operations[0][-1] == v10.OPERATOR_COMMIT
    assert "status" in operations[1]
    assert operations[2][:3] == [sys.executable, "-I", "-c"]
    assert "remove" in operations[3]


def test_v10_forged_result_is_rejected_and_worktree_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "add" in command:
            checkout = Path(command[-2])
            for relative in v10.PINNED_OPERATOR_SOURCES:
                source = checkout / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"trusted historical operator")
        if command[:1] == ["git"] and "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout="")
        if command[:3] == [sys.executable, "-I", "-c"]:
            return subprocess.CompletedProcess(
                command, 0, stdout='{"passed": true}\\n{"passed": false}'
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(v10, "verify_v10_protocol", lambda: {})
    monkeypatch.setattr(v10, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v10.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="one JSON object"):
        v10.run_v10_operation("grade", tmp_path / "workspace")
    assert len(calls) == 4
    assert "remove" in calls[-1]


@pytest.mark.parametrize("action", ["grade", "export"])
def test_v10_rejects_incomplete_action_specific_evidence(
    action: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    source = tmp_path / "upstream"
    source.mkdir()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "add" in command:
            checkout = Path(command[-2])
            for relative in v10.PINNED_OPERATOR_SOURCES:
                source_file = checkout / relative
                source_file.parent.mkdir(parents=True, exist_ok=True)
                source_file.write_bytes(b"trusted historical operator")
        if command[:1] == ["git"] and "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout="")
        if command[:3] == [sys.executable, "-I", "-c"]:
            return subprocess.CompletedProcess(command, 0, stdout='{"passed": false}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(v10, "verify_v10_protocol", lambda: {})
    monkeypatch.setattr(v10, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v10.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="complete grade evidence|complete export evidence"):
        v10.run_v10_operation(
            action, tmp_path / "workspace",
            source_repo=source if action == "export" else None,
        )
    assert len(calls) == 4
    assert "remove" in calls[-1]


def test_v10_export_evidence_binds_the_private_patch_snapshot() -> None:
    protocol = json.loads(
        (v10.ROOT / v10.V9_PROTOCOL_PATH).read_text(encoding="utf-8")
    )
    report = _valid_child_report("export")
    report["candidate_patch_sha256"] = "b" * 64
    report["changed_files"] = [protocol["target_path"]]
    with pytest.raises(RuntimeError, match="export identity or provenance"):
        v10._validate_child_report("export", report, protocol, "a" * 64)


def test_v10_grade_evidence_binds_protocol_and_aggregate_case_results() -> None:
    protocol = json.loads(
        (v10.ROOT / v10.V9_PROTOCOL_PATH).read_text(encoding="utf-8")
    )
    report = _valid_child_report("grade")
    report["protocol_id"] = "vare.mlsys-repository-task.mlxlm-1971.v10"
    with pytest.raises(RuntimeError, match="grade identity"):
        v10._validate_child_report("grade", report, protocol)
    report = _valid_child_report("grade")
    report["passed"] = True
    with pytest.raises(RuntimeError, match="aggregate grade"):
        v10._validate_child_report("grade", report, protocol)


def test_v10_hook_created_import_package_is_rejected_before_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "add" in command:
            checkout = Path(command[-2])
            for relative in v10.PINNED_OPERATOR_SOURCES:
                source = checkout / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"trusted historical operator")
            injected = checkout / "scripts/__init__.py"
            injected.write_text("raise RuntimeError('hook ran')\n", encoding="utf-8")
        if command[:1] == ["git"] and "status" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout="?? scripts/__init__.py\n"
            )
        if command[:3] == [sys.executable, "-I", "-c"]:
            pytest.fail("Python child must not run with an injected package initializer")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(v10, "verify_v10_protocol", lambda: {})
    monkeypatch.setattr(v10, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v10.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="unexpected files"):
        v10.run_v10_operation("grade", tmp_path / "workspace")
    assert len(calls) == 3
    assert "remove" in calls[-1]


def test_v10_real_post_checkout_hook_is_disabled(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    trusted_v9 = json.loads(
        (v10.ROOT / v10.V9_PROTOCOL_PATH).read_text(encoding="utf-8")
    )
    expected_report = _valid_child_report("grade")
    sources = {
        v10.PINNED_OPERATOR_SOURCES[0]: (
            "def run_pinned_v9_task(action, workspace, **kwargs):\n"
            f"    return {expected_report!r}\n"
        ),
        v10.PINNED_OPERATOR_SOURCES[1]: "# pinned dependency\n",
        v10.PINNED_OPERATOR_SOURCES[2]: "# pinned dependency\n",
    }
    for relative, content in sources.items():
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "-c", "user.name=Test", "-c",
         "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    hook_dir = tmp_path / "hooks"
    hook_dir.mkdir()
    marker = tmp_path / "hook-ran"
    hook = hook_dir / "post-checkout"
    hook.write_text(f"#!/bin/sh\nprintf fired > {marker}\n", encoding="utf-8")
    hook.chmod(0o755)
    subprocess.run(
        ["git", "-C", str(repository), "config", "core.hooksPath", str(hook_dir)],
        check=True,
    )

    monkeypatch.setattr(v10, "ROOT", repository)
    monkeypatch.setattr(v10, "OPERATOR_COMMIT", commit)
    monkeypatch.setattr(v10, "verify_v10_protocol", lambda: {})
    monkeypatch.setattr(v10, "_trusted_v9_protocol", lambda: trusted_v9)
    report = v10.run_v10_operation("grade", tmp_path / "candidate")
    assert report == expected_report
    assert not marker.exists()


def test_v10_patch_snapshot_is_private_and_stable(tmp_path: Path) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_bytes(b"original patch")
    snapshot = v10._snapshot_patch(patch, tmp_path / "snapshot.patch")
    patch.write_bytes(b"replaced patch")
    assert snapshot.read_bytes() == b"original patch"
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o400


@pytest.mark.parametrize("field", ["source_repo", "patch_path"])
def test_v10_rejects_export_symlink_inputs_before_resolution(
    field: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = tmp_path / ("source" if field == "source_repo" else "patch")
    if field == "source_repo":
        real.mkdir()
    else:
        real.write_text("patch bytes", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=field == "source_repo")
    monkeypatch.setattr(
        v10, "verify_v10_protocol",
        lambda: pytest.fail("unsafe input must be rejected before protocol loading"),
    )
    kwargs = {field: alias}
    if field == "source_repo":
        kwargs["patch_path"] = None
    else:
        source = tmp_path / "source"
        source.mkdir()
        kwargs["source_repo"] = source
    with pytest.raises(ValueError, match="symlink"):
        v10.run_v10_operation("export", tmp_path / "workspace", **kwargs)


def test_v10_lock_has_independent_digest_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert hashlib.sha256(v10.LOCK.read_bytes()).hexdigest() == v10.LOCK_SHA256
    original = json.loads(v10.LOCK.read_text(encoding="utf-8"))
    original["adversarial_extra_field"] = "does not change protocol_sha256"
    changed = tmp_path / "protocol-v10.lock.json"
    changed.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(v10, "LOCK", changed)
    monkeypatch.setattr(
        v10, "_git_bytes",
        lambda *args: pytest.fail("lock tampering must fail before Git"),
    )
    with pytest.raises(ValueError, match="v10 lock differs"):
        v10.verify_v10_protocol()


@pytest.mark.parametrize("tampering", ["modified", "symlink"])
def test_v10_rejects_mutated_import_dependency_before_launch(
    tampering: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    stale = tmp_path / "malicious.py"
    stale.write_text("dangerous code", encoding="utf-8")
    dependency = v10.PINNED_OPERATOR_SOURCES[1]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "worktree" in command and "add" in command:
            checkout = Path(command[-2])
            for relative in v10.PINNED_OPERATOR_SOURCES:
                source = checkout / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                if relative == dependency and tampering == "symlink":
                    source.symlink_to(stale)
                else:
                    data = b"modified module" if (
                        relative == dependency and tampering == "modified"
                    ) else b"trusted historical operator"
                    source.write_bytes(data)
        if command[:3] == [sys.executable, "-I", "-c"]:
            pytest.fail("must not launch an untrusted Python dependency")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(v10, "verify_v10_protocol", lambda: {})
    monkeypatch.setattr(v10, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v10.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="untrusted operator source|differs from pinned Git"):
        v10.run_v10_operation("grade", tmp_path / "candidate")
    assert len(commands) == 3
    assert "remove" in commands[-1]
