"""Fail-closed behaviour of the sandboxed learning-pipeline grader."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

import pytest

from scripts import learning_pipeline_isolated_grading as ig

BUGGY = "def add(a, b):\n    return a - b\n"
CLEAN = "def add(a, b):\n    return a + b\n"
F2P = ["tests/test_core.py::test_add"]
P2P = ["tests/test_core.py::test_other"]
DENIED_OBS = {**ig.DENIED, "uid": 65534, "visible_processes": 2}


@dataclass
class FakeBackend:
    stdout: str = ""
    returncode: int = 1
    ok: bool = True
    name: str = "fake"
    calls: list[list[str]] = field(default_factory=list)

    def available(self) -> bool:
        return self.ok

    def config_digest(self) -> str:
        return "digest-1"

    def interpreter_dirs(self) -> list[str]:
        return ["/venv"]

    def python(self) -> str:
        return "/venv/bin/python"

    def run(self, workspace: Path, command: Sequence[str], *, timeout: float) -> ig.RunResult:
        self.calls.append(list(command))
        out = self.stdout
        if callable(out):
            out = out(command)
        return ig.RunResult(self.returncode, out, "")


def good_attestation(backend: FakeBackend) -> ig.Attestation:
    return ig.Attestation(backend.name, backend.config_digest(), True, (), DENIED_OBS, 0.0)


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "core.py").write_text(BUGGY)
    (src / ".git").mkdir()
    (src / ".git" / "HEAD").write_text("ref")
    (tmp_path / "ws").mkdir()
    return tmp_path


def grade(backend, att, tree: Path, candidate: str = CLEAN, **kw):
    return ig.grade_isolated(backend, att, workspace_root=tree / "ws", source_tree=tree / "src",
                             target_file="pkg/core.py", buggy_text=BUGGY, candidate_text=candidate,
                             fail_to_pass=F2P, pass_to_pass=P2P, **kw)


def honest_stdout(command: Sequence[str]) -> str:
    lines = []
    for a in command:
        if "::" in a:
            lines.append(("FAILED " if "::test_b_" in a else "PASSED ") + a)
    return "\n".join(lines)


def test_probe_all_denied_passes():
    assert ig.evaluate_probe(DENIED_OBS) == (True, ())


@pytest.mark.parametrize("key", sorted(ig.DENIED))
def test_probe_any_single_exposure_fails(key):
    obs = dict(DENIED_OBS)
    want = ig.DENIED[key]
    obs[key] = ["eth0"] if isinstance(want, list) else (not want)
    ok, failures = ig.evaluate_probe(obs)
    assert not ok and key in failures


@pytest.mark.parametrize("obs", [None, {}, {**DENIED_OBS, "uid": 0},
                                 {**DENIED_OBS, "visible_processes": 400},
                                 {k: v for k, v in DENIED_OBS.items() if k != "host_canary_readable"}])
def test_probe_missing_or_root_fails_closed(obs):
    assert ig.evaluate_probe(obs)[0] is False


def test_attest_fails_closed_when_backend_unavailable(tmp_path):
    att = ig.attest(FakeBackend(ok=False), workspace_root=tmp_path, evidence_dir=tmp_path)
    assert not att.ok and att.failures == ("backend_unavailable",)


def test_attest_rejects_garbage_and_exposed_probe(tmp_path):
    att = ig.attest(FakeBackend(stdout="nothing here", returncode=0), workspace_root=tmp_path,
                    evidence_dir=tmp_path)
    assert not att.ok and att.failures == ("probe_output_missing",)
    leaky = {**DENIED_OBS, "host_canary_readable": True}
    att = ig.attest(FakeBackend(stdout="VARE_PROBE " + json.dumps(leaky), returncode=0),
                    workspace_root=tmp_path, evidence_dir=tmp_path)
    assert not att.ok and "host_canary_readable" in att.failures
    att = ig.attest(FakeBackend(stdout="VARE_PROBE " + json.dumps(DENIED_OBS), returncode=0),
                    workspace_root=tmp_path, evidence_dir=tmp_path)
    assert att.ok


def test_no_pass_without_matching_attestation(tree):
    b = FakeBackend(stdout=honest_stdout)
    assert grade(b, None, tree)["reason"] == "isolation_unverified"
    bad = ig.Attestation("fake", "digest-1", False, ("rootfs_writable",), {}, 0.0)
    assert grade(b, bad, tree)["reason"] == "isolation_unverified"
    other = ig.Attestation("fake", "digest-OTHER", True, (), DENIED_OBS, 0.0)
    assert grade(b, other, tree)["reason"] == "isolation_unverified_config_changed"
    assert b.calls == []  # nothing was executed


def test_honest_pass_and_workspace_is_ephemeral_and_without_git(tree):
    seen = {}

    def look(ws: Path) -> None:
        seen["git"] = (ws / ".git").exists()
        seen["candidate"] = (ws / "pkg" / "core.py").read_text()

    r = grade(FakeBackend(stdout=honest_stdout), good_attestation(FakeBackend()), tree, on_workspace=look)
    assert r["passed"] is True and r["reason"] == "pass"
    assert seen == {"git": False, "candidate": CLEAN}
    assert list((tree / "ws").iterdir()) == []
    assert (tree / "src" / "pkg" / "core.py").read_text() == BUGGY  # trusted tree untouched


def test_all_passed_forgery_is_caught_by_failing_canary(tree):
    forged = lambda cmd: "\n".join("PASSED " + a for a in cmd if "::" in a)  # noqa: E731
    r = grade(FakeBackend(stdout=forged, returncode=0), good_attestation(FakeBackend()), tree)
    assert r["passed"] is False and r["reason"] == "verdict_integrity_canary"


def test_exit_code_must_match_failing_canary(tree):
    r = grade(FakeBackend(stdout=honest_stdout, returncode=0), good_attestation(FakeBackend()), tree)
    assert r["passed"] is False and r["reason"] == "verdict_integrity_exit_code"


def test_missing_test_is_failure(tree):
    partial = lambda cmd: "\n".join(  # noqa: E731
        ("FAILED " if "::test_b_" in a else "PASSED ") + a for a in cmd if "::" in a and "test_other" not in a)
    r = grade(FakeBackend(stdout=partial), good_attestation(FakeBackend()), tree)
    assert r["passed"] is False and r["tests"]["tests/test_core.py::test_other"] == "MISSING"


def test_static_scan_quarantines_added_hostile_lines_only():
    assert ig.static_flags(BUGGY, CLEAN) == []
    assert "process_exit" in ig.static_flags(BUGGY, BUGGY + "import os\nos._exit(0)\n")
    assert "argv_access" in ig.static_flags(BUGGY, BUGGY + "import sys\nx = sys.argv\n")
    pre = "import subprocess\n" + BUGGY
    assert ig.static_flags(pre, pre.replace("a - b", "a + b")) == []  # pre-existing line


def test_quarantine_does_not_execute(tree):
    b = FakeBackend(stdout=honest_stdout)
    r = grade(b, good_attestation(b), tree, candidate=CLEAN + "import subprocess\n")
    assert r["reason"] == "quarantined_static_flags" and r["passed"] is False and b.calls == []


def test_unsafe_target_path_rejected(tree):
    b = FakeBackend(stdout=honest_stdout)
    r = ig.grade_isolated(b, good_attestation(b), workspace_root=tree / "ws", source_tree=tree / "src",
                          target_file="../escape.py", buggy_text=BUGGY, candidate_text=CLEAN,
                          fail_to_pass=F2P, pass_to_pass=P2P)
    assert r["reason"] == "unsafe_target_path" and b.calls == []


# --------------------------------------------------------------- live bwrap
def _bwrap_backend() -> ig.BwrapBackend | None:
    backend = ig.BwrapBackend(venv=Path(sys.prefix))
    try:
        import pytest as _p  # noqa: F401
    except ImportError:  # pragma: no cover
        return None
    return backend if sys.platform.startswith("linux") and backend.available() else None


LIVE = _bwrap_backend()


@pytest.mark.skipif(LIVE is None, reason="bubblewrap user namespaces unavailable")
def test_live_bwrap_attestation_and_grading(tmp_path):
    assert LIVE is not None
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "tests").mkdir()
    (src / "pkg" / "__init__.py").write_text("")
    (src / "pkg" / "core.py").write_text(CLEAN)
    (src / "tests" / "test_core.py").write_text(
        "from pkg.core import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_other():\n    assert add(0, 0) == 0\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@invalid"]
    subprocess.run([*git, "init", "-q"], cwd=src, check=True)
    subprocess.run([*git, "add", "-A"], cwd=src, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], cwd=src, check=True)
    (src / "pkg" / "core.py").write_text(BUGGY)
    ws = tmp_path / "ws"
    ws.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    att = ig.attest(LIVE, workspace_root=ws, evidence_dir=evidence)
    assert att.ok, att.to_dict()
    common = dict(workspace_root=ws, source_tree=src, target_file="pkg/core.py", buggy_text=BUGGY,
                  fail_to_pass=F2P, pass_to_pass=P2P, timeout=120)
    cal = ig.sandbox_calibration(LIVE, att, reference_text=CLEAN,
                                 **{k: v for k, v in common.items() if k != "buggy_text"},
                                 buggy_text=BUGGY)
    assert cal["qualified"], cal
    forge = BUGGY + (
        "\nimport os as _o, stat as _st, sys as _s\n"
        "if 'pytest' in _s.modules:\n"
        "    for _fd in range(3, 256):\n"
        "        try:\n"
        "            if _st.S_ISFIFO(_o.fstat(_fd).st_mode):\n"
        "                _o.write(_fd, ''.join('PASSED ' + a + '\\n' for a in _s.argv[1:] if '::' in a).encode())\n"
        "        except OSError:\n"
        "            pass\n"
        "    _o._exit(0)\n")
    assert ig.grade_isolated(LIVE, att, candidate_text=forge, **common)["reason"] == "quarantined_static_flags"
    r = ig.grade_isolated(LIVE, att, candidate_text=forge, static_scan=False, **common)
    assert r["passed"] is False and r["reason"] == "verdict_integrity_canary"
    restore = BUGGY + ("\nimport subprocess as _sp\ntry:\n    exec(_sp.run(['git', 'show', 'HEAD:pkg/core.py'],"
                       " capture_output=True, text=True, check=True).stdout)\nexcept Exception:\n    pass\n")
    r = ig.grade_isolated(LIVE, att, candidate_text=restore, static_scan=False, **common)
    assert r["passed"] is False and r["reason"] == "tests_failed"
    assert list(ws.iterdir()) == []
