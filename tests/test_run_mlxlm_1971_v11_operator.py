"""Source-only, no-model regression checks for the prospective v11 operator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from scripts import run_mlxlm_1971_v11_operator as v11


def _valid_child_report(action: str) -> dict[str, object]:
    protocol_path = v11.ROOT / v11.V9_PROTOCOL_PATH
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
    if revision == v11.V9_COMMIT and relative == v11.V9_PROTOCOL_PATH:
        return (v11.ROOT / relative).read_bytes()
    return b"trusted historical operator"


def test_v11_protocol_binds_git_commit_and_frozen_v9_without_model() -> None:
    protocol = v11.verify_v11_protocol()
    assert protocol["operator_source_commit"] == v11.OPERATOR_COMMIT
    assert protocol["supersedes_protocol_id"] == (
        "vare.mlsys-repository-task.mlxlm-1971.v10.operator"
    )
    assert protocol["reference_v9_source_commit"] == v11.V9_COMMIT
    assert protocol["sparse_checkout_paths"] == list(v11.SPARSE_PATHS)
    assert protocol["required_nested_v9_test_paths"] == [
        "tests/test_mlxlm_1971_calibration_v9.py",
        "tests/test_mlxlm_1971_repo_task_v9.py",
    ]
    assert protocol["allowed_actions"] == ["export", "grade"]
    assert hashlib.sha256(v11.PROTOCOL.read_bytes()).hexdigest() == v11.PROTOCOL_SHA256


def test_v11_coherent_protocol_and_lock_edit_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = json.loads(v11.PROTOCOL.read_text(encoding="utf-8"))
    protocol["allowed_actions"] = ["grade", "export"]
    candidate = tmp_path / "protocol-v11.json"
    candidate.write_text(json.dumps(protocol), encoding="utf-8")
    lock = tmp_path / "protocol-v11.lock.json"
    lock.write_text(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    monkeypatch.setattr(v11, "PROTOCOL", candidate)
    monkeypatch.setattr(v11, "LOCK", lock)
    monkeypatch.setattr(
        v11, "_git_bytes",
        lambda *args: pytest.fail("coherent protocol tampering must fail before Git"),
    )
    with pytest.raises(ValueError, match="independently pinned bytes"):
        v11.verify_v11_protocol()


def test_v11_rejects_invalid_actions_before_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        v11, "verify_v11_protocol",
        lambda: pytest.fail("bad input must fail before protocol verification"),
    )
    with pytest.raises(ValueError, match="unsupported v11 operation"):
        v11.run_v11_operation("delete", tmp_path / "candidate")
    with pytest.raises(ValueError, match="export requires source"):
        v11.run_v11_operation("export", tmp_path / "candidate")
    with pytest.raises(ValueError, match="grade cannot receive"):
        v11.run_v11_operation("grade", tmp_path / "candidate", source_repo=tmp_path)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        v11.run_v11_operation("grade", symlink)


@pytest.mark.parametrize("action", ["grade", "export"])
def test_v11_dispatch_uses_pinned_source_and_disables_replace_refs(
    action: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    operations: list[list[str]] = []
    source = tmp_path / "upstream"
    source.mkdir()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        operations.append(command)
        assert kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
        if command[:1] == ["git"]:
            assert kwargs["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
            assert kwargs["env"]["GIT_CONFIG_COUNT"] == "4"
            if "init" not in command:
                assert "core.hooksPath=/dev/null" in command
                assert "core.fsmonitor=false" in command
        if command[:1] == ["git"] and "worktree" in command:
            assert kwargs["check"] is True
            if "add" in command:
                checkout = Path(command[-2])
                for relative in v11.PINNED_OPERATOR_SOURCES:
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

    monkeypatch.setattr(v11, "verify_v11_protocol", lambda: {})
    monkeypatch.setattr(v11, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v11.subprocess, "run", fake_run)
    result = v11.run_v11_operation(
        action, tmp_path / "workspace",
        source_repo=source if action == "export" else None,
    )
    assert result == _valid_child_report(action)
    assert len(operations) == 10
    assert operations[3][-1] == v11.OPERATOR_COMMIT
    assert "--no-checkout" in operations[3]
    assert "status" in operations[7]
    assert operations[8][:3] == [sys.executable, "-I", "-c"]
    assert "remove" in operations[9]


def test_v11_forged_result_is_rejected_and_worktree_removed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "add" in command:
            checkout = Path(command[-2])
            for relative in v11.PINNED_OPERATOR_SOURCES:
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

    monkeypatch.setattr(v11, "verify_v11_protocol", lambda: {})
    monkeypatch.setattr(v11, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v11.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="one JSON object"):
        v11.run_v11_operation("grade", tmp_path / "workspace")
    assert len(calls) == 10
    assert "remove" in calls[-1]


@pytest.mark.parametrize("action", ["grade", "export"])
@pytest.mark.parametrize(
    "incomplete_report",
    [
        {"passed": False},
        {"passed": True},
        {"task_id": "mlxlm-1971-mimo-value-scale", "passed": True},
    ],
)
def test_v11_rejects_incomplete_action_specific_evidence(
    action: str, incomplete_report: dict[str, object],
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    source = tmp_path / "upstream"
    source.mkdir()

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "add" in command:
            checkout = Path(command[-2])
            for relative in v11.PINNED_OPERATOR_SOURCES:
                source_file = checkout / relative
                source_file.parent.mkdir(parents=True, exist_ok=True)
                source_file.write_bytes(b"trusted historical operator")
        if command[:1] == ["git"] and "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout="")
        if command[:3] == [sys.executable, "-I", "-c"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(incomplete_report)
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(v11, "verify_v11_protocol", lambda: {})
    monkeypatch.setattr(v11, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v11.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="complete grade evidence|complete export evidence"):
        v11.run_v11_operation(
            action, tmp_path / "workspace",
            source_repo=source if action == "export" else None,
        )
    assert len(calls) == 10
    assert "remove" in calls[-1]


def test_v11_export_evidence_binds_the_private_patch_snapshot() -> None:
    protocol = json.loads(
        (v11.ROOT / v11.V9_PROTOCOL_PATH).read_text(encoding="utf-8")
    )
    report = _valid_child_report("export")
    report["candidate_patch_sha256"] = "b" * 64
    report["changed_files"] = [protocol["target_path"]]
    with pytest.raises(RuntimeError, match="export identity or provenance"):
        v11._validate_child_report("export", report, protocol, "a" * 64)


def test_v11_grade_evidence_binds_protocol_and_aggregate_case_results() -> None:
    protocol = json.loads(
        (v11.ROOT / v11.V9_PROTOCOL_PATH).read_text(encoding="utf-8")
    )
    report = _valid_child_report("grade")
    report["protocol_id"] = "vare.mlsys-repository-task.mlxlm-1971.v11"
    with pytest.raises(RuntimeError, match="grade identity"):
        v11._validate_child_report("grade", report, protocol)
    report = _valid_child_report("grade")
    report["passed"] = True
    with pytest.raises(RuntimeError, match="aggregate grade"):
        v11._validate_child_report("grade", report, protocol)


def test_v11_hook_created_import_package_is_rejected_before_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "add" in command:
            checkout = Path(command[-2])
            for relative in v11.PINNED_OPERATOR_SOURCES:
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

    monkeypatch.setattr(v11, "verify_v11_protocol", lambda: {})
    monkeypatch.setattr(v11, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v11.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="unexpected files"):
        v11.run_v11_operation("grade", tmp_path / "workspace")
    assert len(calls) == 9
    assert "remove" in calls[-1]


def test_v11_real_post_checkout_hook_is_disabled(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    trusted_v9 = json.loads(
        (v11.ROOT / v11.V9_PROTOCOL_PATH).read_text(encoding="utf-8")
    )
    expected_report = _valid_child_report("grade")
    sources = {
        v11.PINNED_OPERATOR_SOURCES[0]: (
            "def run_pinned_v9_task(action, workspace, **kwargs):\n"
            f"    return {expected_report!r}\n"
        ),
        v11.PINNED_OPERATOR_SOURCES[1]: "# pinned dependency\n",
        v11.PINNED_OPERATOR_SOURCES[2]: "# pinned dependency\n",
    }
    for relative, content in sources.items():
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(content, encoding="utf-8")
    (repository / ".gitattributes").write_text(
        f"{v11.PINNED_OPERATOR_SOURCES[0]} filter=vare-marker\n",
        encoding="utf-8",
    )
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
    subprocess.run(
        ["git", "-C", str(repository), "config", "filter.vare-marker.smudge",
         f"printf fired > {marker}; cat"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "uploadpack.packObjectsHook",
         f"printf fired > {marker}"],
        check=True,
    )
    hook = hook_dir / "post-checkout"
    hook.write_text(f"#!/bin/sh\nprintf fired > {marker}\n", encoding="utf-8")
    hook.chmod(0o755)
    subprocess.run(
        ["git", "-C", str(repository), "config", "core.hooksPath", str(hook_dir)],
        check=True,
    )

    monkeypatch.setattr(v11, "ROOT", repository)
    monkeypatch.setattr(v11, "OPERATOR_COMMIT", commit)
    monkeypatch.setattr(v11, "verify_v11_protocol", lambda: {})
    monkeypatch.setattr(v11, "_trusted_v9_protocol", lambda: trusted_v9)
    report = v11.run_v11_operation("grade", tmp_path / "candidate")
    assert report == expected_report
    assert not marker.exists()


def test_v11_nested_v9_checkout_contains_attested_test_sources(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(repository)],
        env=v11.GIT_ENV, check=True,
    )
    required = (
        "tests/test_mlxlm_1971_calibration_v9.py",
        "tests/test_mlxlm_1971_repo_task_v9.py",
    )
    for relative in (
        "scripts/run_attested_mlxlm_1971_v9_calibration.py",
        "scripts/attest_mlxlm_1971_v9_current_git.py",
        "scripts/audit_mlxlm_1971_historical_implementation_hashes.py",
        "src/vare/__init__.py",
        "benchmarks/mlsys_repo_eval_v2/mlxlm_1971_mimo_value_scale/protocol-v9.json",
        *required,
    ):
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("trusted source\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repository), "add", "."],
        env=v11.GIT_ENV, check=True,
    )
    subprocess.run(
        ["git", "-c", "user.name=VARE test", "-c",
         "user.email=vare-test@example.invalid", "-C", str(repository),
         "commit", "--quiet", "-m", "operator fixture"],
        env=v11.GIT_ENV, check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        env=v11.GIT_ENV, check=True, capture_output=True, text=True,
    ).stdout.strip()
    isolated = tmp_path / "operator"
    nested = tmp_path / "nested-v9"
    subprocess.run(
        v11._git_command(
            repository, "worktree", "add", "--detach", "--no-checkout",
            str(isolated), commit,
        ),
        env=v11.GIT_ENV, check=True, capture_output=True,
    )
    v11._materialize_sparse_operator_checkout(isolated)
    subprocess.run(
        v11._git_command(
            isolated, "worktree", "add", "--detach", str(nested), commit,
        ),
        env=v11.GIT_ENV, check=True, capture_output=True,
    )
    try:
        for relative in required:
            assert (nested / relative).is_file()
        flags = subprocess.run(
            v11._git_command(nested, "ls-files", "-v", "--", *required),
            env=v11.GIT_ENV, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        assert len(flags) == len(required)
        assert all(row[0] != "S" for row in flags)
    finally:
        subprocess.run(
            v11._git_command(repository, "worktree", "remove", "--force", str(nested)),
            env=v11.GIT_ENV, check=True, capture_output=True,
        )
        subprocess.run(
            v11._git_command(repository, "worktree", "remove", "--force", str(isolated)),
            env=v11.GIT_ENV, check=True, capture_output=True,
        )


def test_v11_patch_snapshot_is_private_and_stable(tmp_path: Path) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_bytes(b"original patch")
    snapshot = v11._snapshot_patch(patch, tmp_path / "snapshot.patch")
    patch.write_bytes(b"replaced patch")
    assert snapshot.read_bytes() == b"original patch"
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o400


def test_v11_patch_fifo_is_rejected_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    fifo = tmp_path / "candidate.patch"
    os.mkfifo(fifo)
    snapshot = tmp_path / "snapshot.patch"
    code = (
        "from pathlib import Path; import sys; "
        "from scripts.run_mlxlm_1971_v11_operator import _snapshot_patch; "
        "_snapshot_patch(Path(sys.argv[1]), Path(sys.argv[2]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(fifo), str(snapshot)],
        cwd=v11.ROOT, capture_output=True, text=True, timeout=2,
    )
    assert result.returncode != 0
    assert "regular file" in result.stderr
    assert not snapshot.exists()


@pytest.mark.parametrize("field", ["PROTOCOL", "LOCK"])
def test_v11_protocol_fifo_is_rejected_without_blocking(
    field: str, tmp_path: Path
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    fifo = tmp_path / "protocol-input"
    os.mkfifo(fifo)
    code = (
        "from pathlib import Path; import sys; "
        "from scripts import run_mlxlm_1971_v11_operator as op; "
        "setattr(op, sys.argv[1], Path(sys.argv[2])); op.verify_v11_protocol()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, field, str(fifo)],
        cwd=v11.ROOT, capture_output=True, text=True, timeout=2,
    )
    assert result.returncode != 0
    assert "regular file" in result.stderr


@pytest.mark.parametrize("field", ["source_repo", "patch_path"])
def test_v11_rejects_export_symlink_inputs_before_resolution(
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
        v11, "verify_v11_protocol",
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
        v11.run_v11_operation("export", tmp_path / "workspace", **kwargs)


def test_v11_lock_has_independent_digest_anchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert hashlib.sha256(v11.LOCK.read_bytes()).hexdigest() == v11.LOCK_SHA256
    original = json.loads(v11.LOCK.read_text(encoding="utf-8"))
    original["adversarial_extra_field"] = "does not change protocol_sha256"
    changed = tmp_path / "protocol-v11.lock.json"
    changed.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(v11, "LOCK", changed)
    monkeypatch.setattr(
        v11, "_git_bytes",
        lambda *args: pytest.fail("lock tampering must fail before Git"),
    )
    with pytest.raises(ValueError, match="v11 lock differs"):
        v11.verify_v11_protocol()


@pytest.mark.parametrize("tampering", ["modified", "symlink"])
def test_v11_rejects_mutated_import_dependency_before_launch(
    tampering: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands: list[list[str]] = []
    stale = tmp_path / "malicious.py"
    stale.write_text("dangerous code", encoding="utf-8")
    dependency = v11.PINNED_OPERATOR_SOURCES[1]

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if "worktree" in command and "add" in command:
            checkout = Path(command[-2])
            for relative in v11.PINNED_OPERATOR_SOURCES:
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

    monkeypatch.setattr(v11, "verify_v11_protocol", lambda: {})
    monkeypatch.setattr(v11, "_git_bytes", _mock_git_bytes)
    monkeypatch.setattr(v11.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="untrusted operator source|differs from pinned Git"):
        v11.run_v11_operation("grade", tmp_path / "candidate")
    assert len(commands) == 9
    assert "remove" in commands[-1]
