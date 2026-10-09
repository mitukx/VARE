import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import validate_rvl_grpo_midstep_revision_guard as guard


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _repo(tmp_path):
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "VARE test")
    _git(repo, "config", "user.email", "vare-test@example.invalid")
    (repo / "source.py").write_text("value = 1\n")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-q", "-m", "source")
    return repo


def test_exact_clean_checkout_is_accepted(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    root, head = guard.require_clean_revision(repo, expected)
    assert root == repo.resolve()
    assert head == expected


def test_nonmatching_commit_is_rejected(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(guard.RevisionGuardError, match="revision mismatch"):
        guard.require_clean_revision(repo, "0" * 40)


def test_dirty_checkout_is_rejected(tmp_path):
    repo = _repo(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    (repo / "untracked.py").write_text("unexpected = True\n")
    with pytest.raises(guard.RevisionGuardError, match="changes"):
        guard.require_clean_revision(repo, expected)


def _source_pair(tmp_path):
    vare = tmp_path / "vare"
    vare.mkdir()
    _git(vare, "init", "-q")
    _git(vare, "config", "user.name", "VARE test")
    _git(vare, "config", "user.email", "vare-test@example.invalid")
    for relative in (
        "scripts/validate_rvl_grpo_midstep_fault.py",
        "protocols/rvl_grpo_midstep_fault_v1.json",
    ):
        target = vare / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parents[1] / relative, target)
    _git(vare, "add", ".")
    _git(vare, "commit", "-q", "-m", "locked VARE source")

    rvl_root = tmp_path / "rvl"
    rvl = rvl_root / "src/rvl_systems"
    rvl.mkdir(parents=True)
    _git(rvl_root, "init", "-q")
    _git(rvl_root, "config", "user.name", "RVL test")
    _git(rvl_root, "config", "user.email", "rvl-test@example.invalid")
    (rvl / "trainer.py").write_text("value = 1\n")
    _git(rvl_root, "add", ".")
    _git(rvl_root, "commit", "-q", "-m", "locked RVL source")
    return vare, rvl


def _patch_revisions(monkeypatch, vare, rvl):
    monkeypatch.setattr(guard, "VARE_COMMIT", _git(vare, "rev-parse", "HEAD"))
    monkeypatch.setattr(
        guard, "RVL_COMMIT", _git(rvl.parent.parent, "rev-parse", "HEAD")
    )


@pytest.mark.parametrize("dirty", ["vare", "rvl"])
def test_dirty_checkout_is_rejected_before_v1_invocation(tmp_path, monkeypatch, dirty):
    vare, rvl = _source_pair(tmp_path)
    _patch_revisions(monkeypatch, vare, rvl)
    dirty_root = vare if dirty == "vare" else rvl
    (dirty_root / "unexpected.py").write_text("unexpected = True\n")

    original_run = subprocess.run
    v1_calls = []

    def spy_run(args, *positional, **kwargs):
        if args and args[0] == sys.executable:
            v1_calls.append(args)
            raise AssertionError("v1 validator must not run for dirty source")
        return original_run(args, *positional, **kwargs)

    monkeypatch.setattr(guard.subprocess, "run", spy_run)
    with pytest.raises(guard.RevisionGuardError, match="changes"):
        guard.run(vare, rvl)
    assert v1_calls == []


@pytest.mark.parametrize("wrong", ["vare", "rvl"])
def test_wrong_revision_is_rejected_before_v1_invocation(tmp_path, monkeypatch, wrong):
    vare, rvl = _source_pair(tmp_path)
    _patch_revisions(monkeypatch, vare, rvl)
    wrong_root = vare if wrong == "vare" else rvl.parent.parent
    (wrong_root / "extra.py").write_text("value = 2\n")
    _git(wrong_root, "add", "extra.py")
    _git(wrong_root, "commit", "-q", "-m", "different revision")

    original_run = subprocess.run
    v1_calls = []

    def spy_run(args, *positional, **kwargs):
        if args and args[0] == sys.executable:
            v1_calls.append(args)
            raise AssertionError("v1 validator must not run for wrong revision")
        return original_run(args, *positional, **kwargs)

    monkeypatch.setattr(guard.subprocess, "run", spy_run)
    with pytest.raises(guard.RevisionGuardError, match="revision mismatch"):
        guard.run(vare, rvl)
    assert v1_calls == []


def test_exact_clean_sources_delegate_to_v1_and_require_all_checks(
    tmp_path, monkeypatch
):
    vare, rvl = _source_pair(tmp_path)
    _patch_revisions(monkeypatch, vare, rvl)
    checks = {f"check_{index}": True for index in range(12)}
    original_run = subprocess.run
    v1_calls = []

    def spy_run(args, *positional, **kwargs):
        if args and args[0] == sys.executable:
            v1_calls.append(args)
            return subprocess.CompletedProcess(
                args, 0, json.dumps({"status": "pass", "checks": checks}), ""
            )
        return original_run(args, *positional, **kwargs)

    monkeypatch.setattr(guard.subprocess, "run", spy_run)
    result = guard.run(vare, rvl)
    assert len(v1_calls) == 1
    assert result["checks"] == checks
    assert result["revision_guard"]["both_checkouts_clean"] is True
