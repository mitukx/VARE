from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from scripts import grade_transformers_phase_a_v1 as grader


TASK_ID = "transformers-49116-tokenizer-routing-v2"
TARGET = "src/transformers/models/auto/tokenization_auto.py"
IMAGE = "vare/transformers-phase-a-arm64:1@sha256:7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _candidate(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "workspace"
    (root / "src/transformers/models/auto").mkdir(parents=True)
    (root / "src/transformers/utils").mkdir(parents=True)
    files = {
        TARGET: b"baseline target\n",
        "src/transformers/utils/imported.py": b"trusted dependency\n",
        "test_public.py": b"immutable smoke test\n",
    }
    file_records = {}
    for name, data in files.items():
        path = root / name
        path.write_bytes(data)
        path.chmod(0o644)
        file_records[name] = {"mode": 0o644, "sha256": _sha(data)}
    directory_records = {}
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts)):
        path.chmod(0o755)
        directory_records[path.relative_to(root).as_posix()] = 0o755
    manifest = {
        "task_id": TASK_ID,
        "source_revision": "fixture-revision",
        "workspace_snapshot_sha256": "0" * 64,
        "files": file_records,
        "directories": directory_records,
    }
    return root, manifest


def test_frozen_protocol_and_complete_baseline_manifests_are_hash_pinned():
    protocol, lock, protocol_sha = grader._load_protocol()
    assert protocol_sha == lock["protocol_sha256"]
    assert protocol["protocol_id"] == "vare-coding-agent-phase-a-integrity-v6"
    assert grader._check_image(IMAGE, lock) == IMAGE
    for task_id in grader.TASKS:
        manifest, digest = grader._load_manifest(task_id, lock)
        assert digest == lock["baseline_manifests"][task_id]["sha256"]
        assert len(manifest["files"]) > 3000
        assert "test_public.py" in manifest["files"]


def test_running_grader_must_match_the_frozen_implementation_hash(tmp_path: Path, monkeypatch):
    protocol = {
        "protocol_id": "vare-coding-agent-phase-a-integrity-v6",
        "container_image": IMAGE,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(__import__("json").dumps(protocol))
    lock = {
        "protocol_id": protocol["protocol_id"],
        "container_image": IMAGE,
        "protocol_sha256": _sha(protocol_path.read_bytes()),
        "implementation_sha256": {
            "scripts/grade_transformers_phase_a_v1.py": "0" * 64,
        },
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(__import__("json").dumps(lock))
    monkeypatch.setattr(grader, "PROTOCOL", protocol_path)
    monkeypatch.setattr(grader, "PROTOCOL_LOCK", lock_path)
    with pytest.raises(ValueError, match="running grader implementation"):
        grader._load_protocol()


def test_one_target_change_is_captured_for_clean_staging(tmp_path: Path):
    root, manifest = _candidate(tmp_path)
    (root / TARGET).write_bytes(b"candidate target\n")
    captured, file_count, directory_count = grader._scan_candidate(root, manifest, TARGET)
    assert captured[TARGET] == b"candidate target\n"
    assert file_count == 3
    assert directory_count == len(manifest["directories"])

    # Files introduced after capture never enter the staged grading subject.
    (root / "sitecustomize.py").write_text("raise RuntimeError('late hook')\n")
    subject = tmp_path / "subject"
    subject.mkdir()
    grader._stage_verified_snapshot(subject, manifest, captured)
    assert not (subject / "sitecustomize.py").exists()
    assert (subject / TARGET).read_bytes() == b"candidate target\n"


@pytest.mark.parametrize(
    "attack",
    [
        "sitecustomize",
        "imported_module",
        "malicious_symlink",
        "target_symlink",
        "hardlink",
        "forged_grade",
        "hidden_git",
        "missing_file",
        "extra_executable",
        "extra_directory",
    ],
)
def test_candidate_workspace_tampering_fails_closed(tmp_path: Path, attack: str):
    root, manifest = _candidate(tmp_path)
    if attack == "sitecustomize":
        (root / "sitecustomize.py").write_text("pass\n")
    elif attack == "imported_module":
        (root / "src/transformers/utils/imported.py").write_text("forged dependency\n")
    elif attack == "malicious_symlink":
        (root / "src/transformers/utils/evil.py").symlink_to("imported.py")
    elif attack == "target_symlink":
        target = root / TARGET
        target.unlink()
        target.symlink_to(root / "src/transformers/utils/imported.py")
    elif attack == "hardlink":
        external = tmp_path / "external-source.py"
        external.write_text("outside workspace\n")
        (root / TARGET).unlink()
        os.link(external, root / TARGET)
    elif attack == "forged_grade":
        (root / "grade-result.json").write_text('{"passed":true}\n')
    elif attack == "hidden_git":
        (root / ".git").mkdir()
        (root / ".git/config").write_text("[core]\n")
    elif attack == "missing_file":
        (root / "test_public.py").unlink()
    elif attack == "extra_executable":
        extra = root / "startup.py"
        extra.write_text("pass\n")
        extra.chmod(0o755)
    elif attack == "extra_directory":
        (root / "src/transformers/sitecustomize").mkdir()

    with pytest.raises(ValueError):
        grader._scan_candidate(root, manifest, TARGET)


def test_mode_change_is_rejected_even_when_source_bytes_match(tmp_path: Path):
    root, manifest = _candidate(tmp_path)
    (root / TARGET).chmod(0o755)
    with pytest.raises(ValueError, match="mode differs"):
        grader._scan_candidate(root, manifest, TARGET)


def test_grade_rejects_tampered_workspace_before_calling_candidate(monkeypatch, tmp_path: Path):
    root, manifest = _candidate(tmp_path)
    (root / "sitecustomize.py").write_text("raise SystemExit(0)\n")
    protocol = {
        "protocol_id": "fixture",
        "tasks": {
            TASK_ID: {
                "target_path": TARGET,
                "source_revision": "fixture-revision",
                "baseline_manifest_sha256": "2" * 64,
                "grader_adapter_sha256": "unused",
            }
        },
    }
    lock = {"container_image": IMAGE, "baseline_manifests": {TASK_ID: {}}}
    monkeypatch.setattr(grader, "_load_protocol", lambda: (protocol, lock, "1" * 64))
    monkeypatch.setattr(grader, "_load_manifest", lambda _task, _lock: (manifest, "2" * 64))

    def forbidden_candidate_call(*_args, **_kwargs):
        pytest.fail("candidate code reached the grader before workspace validation")

    monkeypatch.setattr(grader, "grade_calls", forbidden_candidate_call)
    with pytest.raises(ValueError, match="unpinned file"):
        grader.grade(TASK_ID, root, IMAGE)


@pytest.mark.parametrize(
    "attack",
    [
        "sitecustomize",
        "imported_module",
        "malicious_symlink",
        "target_symlink",
        "hardlink",
        "forged_grade",
        "hidden_git",
        "missing_file",
        "extra_executable",
        "extra_directory",
    ],
)
def test_full_grade_path_rejects_red_team_workspace_before_execution(
    monkeypatch, tmp_path: Path, attack: str
):
    root, manifest = _candidate(tmp_path)
    if attack == "sitecustomize":
        (root / "sitecustomize.py").write_text("raise SystemExit(0)\n")
    elif attack == "imported_module":
        (root / "src/transformers/utils/imported.py").write_text("forged dependency\n")
    elif attack == "malicious_symlink":
        (root / "src/transformers/utils/evil.py").symlink_to("imported.py")
    elif attack == "target_symlink":
        target = root / TARGET
        target.unlink()
        target.symlink_to(root / "src/transformers/utils/imported.py")
    elif attack == "hardlink":
        external = tmp_path / "external-source.py"
        external.write_text("outside workspace\n")
        (root / TARGET).unlink()
        os.link(external, root / TARGET)
    elif attack == "forged_grade":
        (root / "grade-result.json").write_text('{"passed":true}\n')
    elif attack == "hidden_git":
        (root / ".git").mkdir()
        (root / ".git/config").write_text("[core]\n")
    elif attack == "missing_file":
        (root / "test_public.py").unlink()
    elif attack == "extra_executable":
        extra = root / "startup.py"
        extra.write_text("pass\n")
        extra.chmod(0o755)
    elif attack == "extra_directory":
        (root / "src/transformers/sitecustomize").mkdir()

    protocol = {
        "protocol_id": "fixture",
        "tasks": {
            TASK_ID: {
                "target_path": TARGET,
                "source_revision": "fixture-revision",
                "baseline_manifest_sha256": "2" * 64,
                "grader_adapter_sha256": "unused",
            }
        },
    }
    lock = {"container_image": IMAGE, "baseline_manifests": {TASK_ID: {}}}
    monkeypatch.setattr(grader, "_load_protocol", lambda: (protocol, lock, "1" * 64))
    monkeypatch.setattr(grader, "_load_manifest", lambda _task, _lock: (manifest, "2" * 64))

    def forbidden_candidate_call(*_args, **_kwargs):
        pytest.fail(f"{attack} reached candidate grading before workspace validation")

    monkeypatch.setattr(grader, "grade_calls", forbidden_candidate_call)
    with pytest.raises(ValueError):
        grader.grade(TASK_ID, root, IMAGE)


def test_other_digest_pinned_image_is_rejected_before_candidate_call(monkeypatch):
    lock = {"container_image": IMAGE}
    with pytest.raises(ValueError, match="exactly match"):
        grader._check_image(
            "vare/transformers-phase-a-arm64:other@sha256:" + "0" * 64,
            lock,
        )


def test_full_grade_path_rejects_alternate_image_before_candidate_execution(monkeypatch, tmp_path: Path):
    lock = {"container_image": IMAGE}
    monkeypatch.setattr(grader, "_load_protocol", lambda: ({}, lock, "1" * 64))

    def forbidden_candidate_call(*_args, **_kwargs):
        pytest.fail("alternate image reached candidate grading")

    monkeypatch.setattr(grader, "grade_calls", forbidden_candidate_call)
    alternate = "vare/transformers-phase-a-arm64:other@sha256:" + "0" * 64
    with pytest.raises(ValueError, match="exactly match"):
        grader.grade(TASK_ID, tmp_path / "workspace", alternate)


def test_frozen_image_inspection_records_manifest_and_config_digest(monkeypatch):
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout='["vare/transformers-phase-a-arm64@sha256:'
        '7c8ba69d9269b92bb9d164c54b734da6d56532ef2b455f177d14cab7c348c25b"] '
        'sha256:abcdef\n',
        stderr="",
    )
    monkeypatch.setattr(grader.subprocess, "run", lambda *_args, **_kwargs: completed)
    assert grader._inspect_frozen_image(IMAGE) == "sha256:abcdef"

    bad = subprocess.CompletedProcess(args=[], returncode=0,
                                       stdout='["vare/other@sha256:' + "0" * 64 + '"] sha256:abcdef',
                                       stderr="")
    monkeypatch.setattr(grader.subprocess, "run", lambda *_args, **_kwargs: bad)
    with pytest.raises(ValueError, match="manifest digest differs"):
        grader._inspect_frozen_image(IMAGE)


def test_candidate_root_symlink_is_rejected(tmp_path: Path):
    root, _manifest = _candidate(tmp_path)
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="root must not be a symlink"):
        grader.grade(TASK_ID, alias, IMAGE)
