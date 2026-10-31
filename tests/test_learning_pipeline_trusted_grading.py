"""v2 sandboxed grading: authenticated verdict channel, fail-closed RepoEnv, A3 calibration."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Sequence

import pytest

from scripts import learning_pipeline_grader_calibration as cal
from scripts import learning_pipeline_isolated_grading as ig
from scripts import learning_pipeline_trusted_repo_env as tre
from scripts import learning_pipeline_trusted_verdict as tv
from vare.learning_pipeline import scoring
from vare.learning_pipeline.repo_env import RepoEnv

BUGGY, REFERENCE = cal.BUGGY, cal.REFERENCE
F2P = ["tests/test_core.py::test_add"]
P2P = ["tests/test_core.py::test_mul"]
DENIED_OBS = {**ig.DENIED, "uid": 65534, "visible_processes": 2}
KEY = bytes(range(32))
NONCE = "00112233aabbccdd"
OK_ID, FAIL_ID = "test_zz_n.py::test_a_n", "test_zz_n.py::test_b_n"


# ------------------------------------------------------------ frame judging
def frames(statuses: dict[str, str], *, exit_code: int | None = None, key: bytes = KEY,
           nonce: str = NONCE, drop: Sequence[str] = ()) -> str:
    ids = list(statuses)
    bodies: list[dict[str, Any]] = [{"kind": "start"}, {"kind": "collected", "ids": ids}]
    bodies += [{"kind": "result", "nodeid": t, "status": s} for t, s in statuses.items()]
    implied = 0 if all(s == "PASSED" for s in statuses.values()) else 1
    bodies.append({"kind": "finish", "exit": implied if exit_code is None else exit_code,
                   "n_results": len(ids)})
    out = []
    for i, b in enumerate(bodies):
        if b["kind"] in drop:
            continue
        out.append(tv.frame_line(key, {**b, "seq": i, "nonce": nonce}))
    return "pytest noise\nPASSED tests/test_core.py::test_add\n" + "\n".join(out) + "\n"


HONEST = {F2P[0]: "PASSED", P2P[0]: "PASSED", OK_ID: "PASSED", FAIL_ID: "FAILED"}


def judge(stdout: str, returncode: int = 1) -> tuple[bool, str, dict[str, str]]:
    return tv.judge_frames(stdout, key=KEY, nonce=NONCE, returncode=returncode, fail_to_pass=F2P,
                           pass_to_pass=P2P, canary_ok=OK_ID, canary_fail=FAIL_ID)


def test_authentic_complete_stream_passes():
    assert judge(frames(HONEST))[:2] == (True, "pass")


def test_plain_passed_lines_carry_no_authority():
    stdout = "\n".join(f"PASSED {t}" for t in HONEST) + "\n"
    assert judge(stdout, 0)[:2] == (False, "verdict_integrity_no_start")


@pytest.mark.parametrize("mutate,reason", [
    (lambda s: s.replace(tv.FRAME_PREFIX + s.split(tv.FRAME_PREFIX)[2][:8], tv.FRAME_PREFIX + "0" * 8, 1),
     "verdict_integrity_bad_frame"),
    (lambda s: frames(HONEST, key=b"k" * 32), "verdict_integrity_bad_frame"),
    (lambda s: frames(HONEST, nonce="ffffffffffffffff"), "verdict_integrity_sequence"),
    (lambda s: frames(HONEST, drop=("finish",)), "verdict_integrity_no_finish"),
    (lambda s: frames(HONEST, drop=("collected",)), "verdict_integrity_sequence"),
    (lambda s: frames(HONEST, exit_code=0), "verdict_integrity_finish"),
    (lambda s: s + tv.FRAME_PREFIX + "garbage\n", "verdict_integrity_bad_frame"),
])
def test_tampered_streams_fail_closed(mutate: Callable[[str], str], reason: str):
    passed, got, tests = judge(mutate(frames(HONEST)))
    assert (passed, got) == (False, reason)
    assert set(tests.values()) == {"UNVERIFIED"}


def test_duplicate_result_and_exit_mismatch_fail_closed():
    lines = frames(HONEST).splitlines()
    result = next(i for i, ln in enumerate(lines) if '"kind":"result"' in ln)
    assert judge("\n".join(lines + [lines[result]]))[1] == "verdict_integrity_sequence"
    assert judge(frames(HONEST), returncode=0)[1] == "verdict_integrity_exit_code"


def test_canaries_and_missing_ids():
    assert judge(frames({**HONEST, FAIL_ID: "PASSED"}), 0)[1] == "verdict_integrity_canary"
    no_canaries = {k: v for k, v in HONEST.items() if k not in (OK_ID, FAIL_ID)}
    assert judge(frames(no_canaries), 0)[1] == "collection_error"
    partial = {k: v for k, v in HONEST.items() if k != P2P[0]}
    passed, reason, tests = judge(frames(partial))
    assert (passed, reason, tests[P2P[0]]) == (False, "tests_failed", "MISSING")


def test_v2_static_patterns_extend_v1_without_changing_it():
    names = [n for n, _ in tv.BLOCKING_PATTERNS_V2]
    assert names[:len(ig.BLOCKING_PATTERNS)] == [n for n, _ in ig.BLOCKING_PATTERNS]
    assert tv.static_flags_v2(BUGGY, REFERENCE) == []
    assert "introspection" in tv.static_flags_v2(BUGGY, BUGGY + "import gc\n")
    assert "comparison_override" in tv.static_flags_v2(BUGGY, BUGGY + "    def __eq__(self, o):\n")
    assert "fd_access" in tv.static_flags_v2(BUGGY, BUGGY + "os.write(3, b'x')\n")


# ------------------------------------------------------------ fake backend
@dataclass
class FakeBackend:
    ok: bool = True
    name: str = "fake"
    digest: str = "digest-1"
    calls: list[list[str]] = field(default_factory=list)
    import_roots: tuple[str, ...] = ("/work",)

    def available(self) -> bool:
        return self.ok

    def config_digest(self) -> str:
        return self.digest

    def interpreter_dirs(self) -> list[str]:
        return ["/venv"]

    def python(self) -> str:
        return "/venv/bin/python"

    def run(self, workspace: Path, command: Sequence[str], *, timeout: float,
            stdin: bytes | None = None) -> ig.RunResult:
        self.calls.append(list(command))
        if command[1] != "-I":  # attestation probe
            return ig.RunResult(0, "VARE_PROBE " + json.dumps(DENIED_OBS), "")
        return ig.RunResult(1, "", "")


def attestation_for(b: FakeBackend, ok: bool = True) -> ig.Attestation:
    return ig.Attestation(b.name, b.config_digest(), ok, () if ok else ("rootfs_writable",), DENIED_OBS, 0.0)


def make_clone(cache: Path, name: str = "Org__pkg.0000000") -> tuple[RepoEnv, str]:
    """A pinned clone whose HEAD is the reference and a bug patch that breaks ``add``."""
    path = cache / "repos" / name
    (path / "pkg").mkdir(parents=True)
    (path / "tests").mkdir()
    (path / "pkg" / "__init__.py").write_text("")
    (path / "pkg" / "core.py").write_text(REFERENCE)
    (path / "tests" / "test_core.py").write_text(cal.TESTS)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@invalid"]
    subprocess.run([*git, "init", "-q"], cwd=path, check=True)
    subprocess.run([*git, "add", "-A"], cwd=path, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "reference"], cwd=path, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True,
                            check=True).stdout.strip()
    (path / "pkg" / "core.py").write_text(BUGGY)
    patch = subprocess.run(["git", "diff"], cwd=path, capture_output=True, text=True, check=True).stdout
    subprocess.run(["git", "checkout", "-q", "-f", commit], cwd=path, check=True)
    (cache / "venvs").mkdir(exist_ok=True)
    (cache / "venvs" / name).symlink_to(Path(sys.prefix), target_is_directory=True)
    return RepoEnv(f"mirror/{name}", "Org/pkg", commit, cache), patch


def trusted(env: RepoEnv, **kw: Any) -> tre.TrustedRepoEnv:
    return tre.TrustedRepoEnv(env.mirror, env.upstream, env.commit, env.cache, **kw)


def grade_kw(patch: str, candidate: str | None) -> dict[str, Any]:
    return dict(bug_patch=patch, target_file="pkg/core.py", candidate_text=candidate,
                fail_to_pass=F2P, pass_to_pass=P2P, timeout=60)


# ------------------------------------------------------------ fail closed by default
@pytest.mark.parametrize("setup", ["no_backend", "failed_attestation", "config_changed", "other_backend"])
def test_default_grade_fails_closed_without_attested_isolation(tmp_path, monkeypatch, setup):
    env, patch = make_clone(tmp_path)
    b = FakeBackend()
    kw: dict[str, Any] = {
        "no_backend": {},
        "failed_attestation": {"backend": b, "attestation": attestation_for(b, ok=False)},
        "config_changed": {"backend": b, "attestation": ig.Attestation("fake", "digest-OLD", True, (), {}, 0.0)},
        "other_backend": {"backend": b, "attestation": ig.Attestation("bwrap", "digest-1", True, (), {}, 0.0)},
    }[setup]

    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError("must not execute or touch the clone")

    monkeypatch.setattr(RepoEnv, "grade", forbidden)
    monkeypatch.setattr(RepoEnv, "reset", forbidden)
    monkeypatch.setattr(tv, "grade_isolated_v2", forbidden)
    r = trusted(env, **kw).grade(**grade_kw(patch, REFERENCE))
    assert (r["passed"], r["reason"], r["reward"], r["executed"]) == (False, "isolation_unverified", 0, False)
    assert r["tests"] == {F2P[0]: "NOT_RUN", P2P[0]: "NOT_RUN"} and r["f2p_passed"] == 0
    assert r["isolation"]["mode"] == "required" and r["isolation"]["attested"] is False
    assert b.calls == []


def test_unknown_isolation_mode_is_rejected(tmp_path):
    env, patch = make_clone(tmp_path)
    with pytest.raises(ValueError, match="isolation must be one of"):
        trusted(env).grade(**grade_kw(patch, REFERENCE), isolation="native")


def test_scoring_gives_zero_reward_and_no_admission_when_unverified(tmp_path):
    env, patch = make_clone(tmp_path)
    task = {"instance_id": "i1", "_patch": patch, "target_file": "pkg/core.py", "fail_to_pass": F2P,
            "pass_to_pass": P2P}
    rec = scoring.score_candidate(trusted(env), task, BUGGY,
                                  "<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE\n",
                                  timeout=30)
    assert rec["apply"] == "applied" and rec["reward"] == 0
    for name in ("cheap_f2p_v1", "trusted_full_v1"):
        v = rec["verifiers"][name]
        assert (v["passed"], v["reason"], v["executed"]) == (False, "isolation_unverified", False)


def test_backend_selection_takes_first_attested_and_records_attempts(tmp_path):
    down, bad, good = FakeBackend(ok=False, name="docker"), FakeBackend(name="leaky"), FakeBackend(name="bwrap")
    bad.run = lambda *a, **k: ig.RunResult(0, "VARE_PROBE " + json.dumps(  # type: ignore[method-assign]
        {**DENIED_OBS, "host_canary_readable": True}), "")
    sel = tre.select_backend([down, bad, good], workspace_root=tmp_path, evidence_dir=tmp_path / "ev")
    assert sel.backend is good and sel.attestation is not None and sel.attestation.ok
    assert [(t["backend"], t["ok"]) for t in sel.tried] == [("docker", False), ("leaky", False), ("bwrap", True)]
    assert sel.tried[0]["failures"] == ["backend_unavailable"]
    assert "host_canary_readable" in sel.tried[1]["failures"]
    none = tre.select_backend([down, bad], workspace_root=tmp_path, evidence_dir=tmp_path / "ev")
    assert none.backend is None and none.attestation is None and len(none.tried) == 2


def test_selected_backend_is_used_and_recorded(tmp_path):
    env, patch = make_clone(tmp_path)
    b = FakeBackend()
    sel = tre.select_backend([b], workspace_root=tmp_path, evidence_dir=tmp_path / "ev")
    t = trusted(env, backend=sel.backend, attestation=sel.attestation, selection=sel.tried,
                workspace_root=tmp_path / "ws")
    r = t.grade(**grade_kw(patch, REFERENCE))
    # The fake sandbox returns no frames: executed, then rejected by the parent.
    assert r["executed"] is True and r["passed"] is False and r["reason"] == "verdict_integrity_no_start"
    iso = r["isolation"]
    assert iso["backend"] == "fake" and iso["attested"] is True and iso["config_digest"] == "digest-1"
    assert iso["attestation"]["ok"] is True and iso["selection"][0]["backend"] == "fake"
    assert iso["verdict_channel"] == tv.VERDICT_CHANNEL and iso["runner_sha256"] == tv.runner_sha256()
    assert b.calls[-1][:3] == ["/venv/bin/python", "-I", "-c"]
    assert (env.path / "pkg" / "core.py").read_text() == REFERENCE  # clone reset to the pinned commit
    assert list((tmp_path / "ws").iterdir()) == []


# ------------------------------------------------------------ legacy opt-in
def test_legacy_native_is_explicit_opt_in_and_labelled(tmp_path):
    env, patch = make_clone(tmp_path)
    t = trusted(env)
    assert t.grade(**grade_kw(patch, REFERENCE))["reason"] == "isolation_unverified"
    ref = t.grade(**grade_kw(patch, REFERENCE), isolation="legacy_native")
    bug = t.grade(**grade_kw(patch, None), isolation="legacy_native")
    assert ref["passed"] is True and ref["reward"] == 1 and bug["passed"] is False
    assert ref["isolation"]["mode"] == "legacy_native" and ref["isolation"]["backend"] == "native_venv"
    assert ref["isolation"]["attested"] is False
    v1 = env.grade(**grade_kw(patch, REFERENCE))  # the frozen path itself is unchanged
    assert {k: v1[k] for k in ("passed", "reason", "tests")} == {k: ref[k] for k in ("passed", "reason", "tests")}


def test_legacy_native_still_admits_the_audit_forgery(tmp_path):
    """Why the default is not the native path: the reproduced exit-0 forgery passes there."""
    env, patch = make_clone(tmp_path)
    forged = cal.tamper_candidates(BUGGY, REFERENCE)["audit_forge_argv_echo_exit0"]
    r = trusted(env).grade(**grade_kw(patch, forged), isolation="legacy_native")
    assert r["passed"] is True and r["isolation"]["attested"] is False


# ------------------------------------------------------------ live bubblewrap
def _live_bwrap() -> ig.BwrapBackend | None:
    if not sys.platform.startswith("linux"):
        return None
    b = ig.BwrapBackend(venv=Path(sys.prefix))
    return b if b.available() else None


LIVE = _live_bwrap()
needs_bwrap = pytest.mark.skipif(LIVE is None, reason="bubblewrap user namespaces unavailable")


@pytest.fixture()
def live(tmp_path):
    assert LIVE is not None
    env, patch = make_clone(tmp_path)
    ev = tmp_path / "evidence"
    ev.mkdir()
    sel = tre.select_backend([ig.DockerBackend(image="python@sha256:" + "0" * 64, workspace_root=tmp_path),
                              LIVE], workspace_root=tmp_path, evidence_dir=ev) \
        if shutil.which("docker") is None else tre.select_backend([LIVE], workspace_root=tmp_path,
                                                                  evidence_dir=ev)
    assert sel.backend is LIVE, sel.tried
    return trusted(env, backend=sel.backend, attestation=sel.attestation, selection=sel.tried,
                   workspace_root=tmp_path / "ws"), patch


@needs_bwrap
def test_live_required_mode_grades_honest_candidates(live):
    t, patch = live
    ref = t.grade(**grade_kw(patch, REFERENCE))
    bug = t.grade(**grade_kw(patch, None))
    assert (ref["passed"], ref["reason"], ref["reward"]) == (True, "pass", 1), ref
    assert (bug["passed"], bug["reason"]) == (False, "tests_failed")
    assert bug["tests"] == {F2P[0]: "FAILED", P2P[0]: "PASSED"}
    assert ref["isolation"]["backend"] == "bwrap" and ref["isolation"]["attested"] is True
    cal_rec = scoring.calibrate(t, {"_patch": patch, "target_file": "pkg/core.py", "instance_id": "i",
                                    "fail_to_pass": F2P, "pass_to_pass": P2P}, timeout=60)
    assert cal_rec["qualified"] is True


@needs_bwrap
@pytest.mark.parametrize("name", ["audit_forge_argv_echo_exit0", "audit_forge_canary_aware_exit1",
                                  "audit_restore_reference_from_git"])
@pytest.mark.parametrize("scan", [True, False])
def test_live_reproduced_false_passes_are_blocked(live, name, scan):
    t, patch = live
    t.static_scan = scan
    r = t.grade(**grade_kw(patch, cal.tamper_candidates(BUGGY, REFERENCE)[name]))
    assert r["passed"] is False and r["reward"] == 0
    if scan:
        assert r["reason"] == "quarantined_static_flags" and r["executed"] is False
    else:
        assert r["executed"] is True and r["reason"] in {"verdict_integrity_no_finish", "tests_failed"}


@needs_bwrap
def test_live_v1_stdout_channel_admits_canary_aware_forgery_v2_does_not(tmp_path):
    """Positive control: the forgery is effective against the v1 sandboxed channel."""
    assert LIVE is not None
    fx = cal.synthetic_fixture(tmp_path)
    ws, ev = tmp_path / "ws", tmp_path / "ev"
    ws.mkdir()
    ev.mkdir()
    att = ig.attest(LIVE, workspace_root=ws, evidence_dir=ev)
    forged = cal.tamper_candidates(BUGGY, REFERENCE)["audit_forge_canary_aware_exit1"]
    common = dict(workspace_root=ws, source_tree=fx.source_tree, target_file=fx.target_file,
                  buggy_text=BUGGY, candidate_text=forged, fail_to_pass=F2P, pass_to_pass=P2P,
                  timeout=60, static_scan=False)
    assert ig.grade_isolated(LIVE, att, **common)["passed"] is True
    r = tv.grade_isolated_v2(LIVE, att, **common)
    assert r["passed"] is False and r["reason"] == "verdict_integrity_no_finish"


@needs_bwrap
def test_live_a3_calibration_all_items_verified(tmp_path):
    assert LIVE is not None
    fx = cal.synthetic_fixture(tmp_path / "fx")
    report = cal.calibrate(LIVE, fx, work_root=tmp_path / "cal", timeout=60, timeout_probe_s=5)
    assert set(report["items"]) == set(cal.ITEMS)
    assert {k: v["status"] for k, v in report["items"].items()} == {k: "verified" for k in cal.ITEMS}
    assert report["all_verified"] is True and report["workspaces_left"] == 0
    # Residual in-process attacks: blocked only by the static scan (documented limitation).
    assert set(report["residual"]) == {"runner_closure_patch_via_gc", "always_equal_return_value"}
    assert all(v["scan_on_reason"] == "quarantined_static_flags" for v in report["residual"].values())


def test_calibration_fails_closed_when_not_attested(tmp_path):
    fx = cal.synthetic_fixture(tmp_path / "fx")
    b = FakeBackend(ok=False)
    report = cal.calibrate(b, fx, work_root=tmp_path / "cal")
    assert report["all_verified"] is False and b.calls == []
    assert {v["status"] for v in report["items"].values()} == {"failed"}


def test_mutants_exclude_bug_and_reference():
    ms = cal.mutants(BUGGY, REFERENCE)
    assert ms and BUGGY not in ms and REFERENCE not in ms


# ------------------------------------------------------------ Docker (Mac/Colima)
DOCKER_IMAGE = os.environ.get("VARE_GRADER_IMAGE", "")


@pytest.mark.skipif(shutil.which("docker") is None or not DOCKER_IMAGE,
                    reason="Docker or VARE_GRADER_IMAGE (name@sha256:...) unavailable")
def test_live_docker_a3_calibration(tmp_path):
    # The work root must be bind-mountable by the Docker VM (Colima shares $HOME by default).
    base = Path(os.environ.get("VARE_GRADER_WORK_ROOT", str(Path.home() / ".cache")))
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="vare-grader-test-", dir=base))
    try:
        backend = ig.DockerBackend(image=DOCKER_IMAGE, workspace_root=root)
        fx = cal.synthetic_fixture(tmp_path / "fx")
        report = cal.calibrate(backend, fx, work_root=root, timeout=120)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert report["all_verified"] is True, {k: v["status"] for k, v in report["items"].items()}
