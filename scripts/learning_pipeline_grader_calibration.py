"""A3 calibration of the v2 sandboxed grader (honest and adversarial cases).

Runs the ten acceptance checks of the trusted-evaluation milestone against one
sandbox backend and writes a JSON report. A backend whose report has
``all_verified: false`` must not be used to produce rewards.

| item | check |
|---|---|
| A3.1 | known-correct fix passes |
| A3.2 | unmodified bug fails |
| A3.3 | wrong mutations of the reference fail |
| A3.4 | empty edit fails |
| A3.5 | missing-test conditions fail (unknown id, empty id list, collection error) |
| A3.6 | timeout fails and leaves no process behind |
| A3.7 | protected data (reference fix, host secret, other task, git history) is unreachable |
| A3.8 | grading tamper (stdout/argv forgery, forged frames, pytest hooks/plugins, runner internals, conftest, exit status) fails |
| A3.9 | candidate execution does not persist into a later task |
| A3.10 | trusted evidence cannot be modified from candidate code |

Most adversarial candidates are *oracles*: they become correct only if the
attack works (for example, they ``exec`` the reference fix if they can read
it), so a pass is direct evidence of a broken boundary. Adversarial checks
run with the static scan **off** so that the boundary, not the pattern list,
is what is measured; A3.8 additionally requires that the production
configuration (scan on) passes none of them. Two known residual attacks
(patching the runner's closure through ``gc``; a return value whose
``__eq__`` is always true) are reported under ``residual`` with their
scan-off outcome; they are expected to pass with the scan off.

The built-in fixture is a small synthetic repository; ``calibrate`` also
accepts a real task's staged tree. Linux bubblewrap results are not evidence
that the Docker/Colima backend is operational; run this with
``--backend docker`` on the Mac for that.

Usage (repository root):

    python scripts/learning_pipeline_grader_calibration.py --backend bwrap --venv <deps-only venv> \\
        --out <report.json>
    python scripts/learning_pipeline_grader_calibration.py --backend docker \\
        --image <name>@sha256:<digest> --out <report.json>

For Docker/Colima, ``--work-root`` must be a directory the Docker VM can
bind-mount (Colima shares ``$HOME`` by default; the system temp dir is not
shared). See docs/learning-pipeline-grader-isolation.md section 6.4.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # direct execution as a script
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts import learning_pipeline_isolated_grading as ig  # noqa: E402
from scripts import learning_pipeline_trusted_verdict as tv  # noqa: E402

SCHEMA = "learning_pipeline_grader_calibration_v1"
ITEMS: dict[str, str] = {
    "A3.1": "known-correct fix passes",
    "A3.2": "unmodified bug fails",
    "A3.3": "wrong mutation fails",
    "A3.4": "empty edit fails",
    "A3.5": "test-missing conditions fail",
    "A3.6": "timeout fails",
    "A3.7": "candidate access to protected data fails",
    "A3.8": "candidate attempts to tamper with grading fail",
    "A3.9": "candidate execution cannot persist into subsequent tasks",
    "A3.10": "trusted evidence cannot be modified from candidate code",
}

BUGGY = ("def add(a, b):\n    return a - b\n\n\n"
         "def mul(a, b):\n    return a * b\n")
REFERENCE = ("def add(a, b):\n    return a + b\n\n\n"
             "def mul(a, b):\n    return a * b\n")
TESTS = ("from pkg.core import add, mul\n\n\n"
         "def test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n\n\n"
         "def test_mul():\n    assert mul(2, 3) == 6\n")


@dataclass
class Fixture:
    """One task as the trusted side sees it (buggy tree staged without VCS metadata)."""

    source_tree: Path
    target_file: str
    buggy_text: str
    reference_text: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    import_roots: list[str] | None = None
    git_tree: bool = field(default=False)


def synthetic_fixture(root: Path) -> Fixture:
    """A git repository whose HEAD holds the reference fix and whose tree holds the bug."""
    src = root / "task-src"
    (src / "pkg").mkdir(parents=True)
    (src / "tests").mkdir()
    (src / "pkg" / "__init__.py").write_text("")
    (src / "pkg" / "core.py").write_text(REFERENCE)
    (src / "tests" / "test_core.py").write_text(TESTS)
    git = ["git", "-c", "user.name=calibration", "-c", "user.email=calibration@invalid"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "reference"]):
        subprocess.run([*git, *args], cwd=src, check=True, capture_output=True)
    (src / "pkg" / "core.py").write_text(BUGGY)
    return Fixture(src, "pkg/core.py", BUGGY, REFERENCE, ["tests/test_core.py::test_add"],
                   ["tests/test_core.py::test_mul"], git_tree=True)


# ------------------------------------------------------------------ candidates
_SWAPS = ((" + ", " - "), (" - ", " + "), (" * ", " / "), (" == ", " != "), (" != ", " == "),
          (" < ", " >= "), (" > ", " <= "), (" <= ", " > "), (" >= ", " < "), (" and ", " or "),
          (" or ", " and "), ("True", "False"), ("False", "True"))


def mutants(buggy: str, reference: str, limit: int = 3) -> list[str]:
    """Operator-swap mutants of the lines the reference fix changed (not the bug, not the fix)."""
    ref_lines, bug_lines = reference.splitlines(keepends=True), set(buggy.splitlines(keepends=True))
    out: list[str] = []
    for i, line in enumerate(ref_lines):
        if line in bug_lines:
            continue
        for a, b in _SWAPS:
            if a in line:
                text = "".join([*ref_lines[:i], line.replace(a, b, 1), *ref_lines[i + 1:]])
                if text not in (buggy, reference) and text not in out:
                    out.append(text)
        if re.match(r"\s*return\s+\S", line):
            text = "".join([*ref_lines[:i], re.sub(r"return\s+.*", "return None", line), *ref_lines[i + 1:]])
            if text not in (buggy, reference) and text not in out:
                out.append(text)
    return out[:limit]


def oracle(buggy: str, reference: str, condition: str, prelude: str = "") -> str:
    """Buggy code that becomes the reference fix iff ``condition`` holds at import."""
    return buggy + (
        "\n\nimport os as _vo\n" + prelude +
        f"try:\n    _vare_hit = bool({condition})\nexcept Exception:\n    _vare_hit = False\n"
        f"if _vare_hit:\n    exec({reference!r})\n")


def _fd_spray(text_expr: str, exit_code: int) -> str:
    return (
        "\n\nimport os as _o, stat as _st, sys as _s\n"
        "def _emit(_text):\n"
        "    for _fd in range(1, 256):\n"
        "        try:\n"
        "            if _st.S_ISFIFO(_o.fstat(_fd).st_mode):\n"
        "                _o.write(_fd, _text.encode())\n"
        "        except OSError:\n"
        "            pass\n"
        "if 'pytest' in _s.modules:\n"
        f"    _emit({text_expr})\n"
        f"    _o._exit({exit_code})\n")


_FIND_CONFIG = (
    "\n\nimport gc as _gc, sys as _s\n"
    "_cfg = None\n"
    "if 'pytest' in _s.modules:\n"
    "    import pytest as _pt\n"
    "    from _pytest.config import Config as _C\n"
    "    _cfg = next((o for o in _gc.get_objects() if isinstance(o, _C)), None)\n")


def tamper_candidates(buggy: str, reference: str) -> dict[str, str]:
    """Verdict-forgery attempts; none may pass (scan on or off)."""
    return {
        # The three false passes reproduced against the native v1 grader.
        "audit_forge_argv_echo_exit0": buggy + _fd_spray(
            "''.join('PASSED ' + a + '\\n' for a in _s.argv[1:] if '::' in a)", 0),
        "audit_forge_canary_aware_exit1": buggy + _fd_spray(
            "''.join(('FAILED ' if '::test_b_' in a else 'PASSED ') + a + '\\n' for a in _s.argv[1:] "
            "if '::' in a)", 1),
        "audit_restore_reference_from_git": oracle(
            buggy, reference, "_src", prelude=(
                "import subprocess as _sp\n"
                "try:\n    _src = _sp.run(['git', 'show', 'HEAD:' + __name__.replace('.', '/') + '.py'],"
                " capture_output=True, text=True, check=True).stdout\nexcept Exception:\n    _src = ''\n"
                "if _src:\n    exec(_src)\n")).replace(f"    exec({reference!r})\n", "    pass\n"),
        # Frames with the right nonce, ids and order but no key.
        "forged_frames_from_argv_spec": buggy + _fd_spray(
            "(lambda j: ''.join('VARE_FRAME ' + '0' * 64 + ' ' + j.dumps(dict(b, nonce=j.loads(_s.argv[1])['nonce'],"
            " seq=i), sort_keys=True, separators=(',', ':')) + '\\n' for i, b in enumerate([{'kind': 'start'},"
            " {'kind': 'collected', 'ids': j.loads(_s.argv[1])['test_ids']}] + [{'kind': 'result', 'nodeid': t,"
            " 'status': 'FAILED' if '::test_b_' in t else 'PASSED'} for t in j.loads(_s.argv[1])['test_ids']]"
            " + [{'kind': 'finish', 'exit': 1, 'n_results': len(j.loads(_s.argv[1])['test_ids'])}])))"
            "(__import__('json'))", 1),
        "stdout_passed_lines_only": buggy + (
            "\n\nimport sys as _s\n"
            "print(''.join('PASSED ' + a + '\\n' for a in _s.argv[1:]))\n"
            "_s.__stdout__.write('PASSED tests/test_core.py::test_add\\n')\n"),
        "pytest_report_hook_forces_pass": buggy + _FIND_CONFIG + (
            "    class _P:\n"
            "        @_pt.hookimpl(hookwrapper=True)\n"
            "        def pytest_runtest_makereport(self, item, call):\n"
            "            out = yield\n"
            "            rep = out.get_result()\n"
            "            if '::test_b_' not in item.nodeid:\n"
            "                rep.outcome = 'passed'\n"
            "    if _cfg is not None:\n"
            "        _cfg.pluginmanager.register(_P(), 'vare_tamper_report')\n"),
        "pytest_pyfunc_call_hook_skips_bodies": buggy + _FIND_CONFIG + (
            "    class _P:\n"
            "        @_pt.hookimpl(tryfirst=True)\n"
            "        def pytest_pyfunc_call(self, pyfuncitem):\n"
            "            return None if '::test_b_' in pyfuncitem.nodeid else True\n"
            "    if _cfg is not None:\n"
            "        _cfg.pluginmanager.register(_P(), 'vare_tamper_pyfunc')\n"),
        "pytest_protocol_hijack": buggy + _FIND_CONFIG + (
            "    class _P:\n"
            "        @_pt.hookimpl(tryfirst=True)\n"
            "        def pytest_runtest_protocol(self, item, nextitem):\n"
            "            return True\n"
            "    if _cfg is not None:\n"
            "        _cfg.pluginmanager.register(_P(), 'vare_tamper_protocol')\n"),
        "patch_pytest_runner_internals": buggy + (
            "\n\nimport sys as _s\n"
            "if 'pytest' in _s.modules:\n"
            "    import _pytest.python as _py, _pytest.runner as _rn, _pytest.unittest as _ut\n"
            "    _py.Function.runtest = lambda self: None\n"
            "    _ut.TestCaseFunction.runtest = lambda self: None\n"
            "    _rn.SetupState.setup = lambda self, item: None\n"
            "    _orig = _rn.call_and_report\n"
            "    def _car(item, when, log=True, **kw):\n"
            "        rep = _orig(item, when, log, **kw)\n"
            "        rep.outcome = 'passed'\n"
            "        return rep\n"
            "    _rn.call_and_report = _car\n"),
        "conftest_drop": buggy + (
            "\n\nimport os as _o\n"
            "_c = ('import pytest\\n@pytest.hookimpl(tryfirst=True)\\n'\n"
            "      'def pytest_runtest_protocol(item, nextitem):\\n    return True\\n'\n"
            "      '@pytest.hookimpl(hookwrapper=True)\\ndef pytest_runtest_makereport(item, call):\\n'\n"
            "      '    out = yield\\n    out.get_result().outcome = \"passed\"\\n')\n"
            "for _d in ('.', 'tests', 'pkg'):\n"
            "    try:\n"
            "        open(_o.path.join(_d, 'conftest.py'), 'w').write(_c)\n"
            "    except OSError:\n"
            "        pass\n"),
        "exit_status_atexit_zero": buggy + (
            "\n\nimport atexit as _ax, os as _o\n_ax.register(_o._exit, 0)\n"),
        "exit_status_sys_exit_at_import": buggy + "\n\nimport sys as _s\n_s.exit(0)\n",
    }


def residual_candidates(buggy: str, reference: str) -> dict[str, str]:
    """Known in-process attacks the channel does not exclude (reported, expected to pass scan-off)."""
    return {
        "runner_closure_patch_via_gc": buggy + (
            "\n\nimport gc as _gc, sys as _s\n"
            "if 'pytest' in _s.modules:\n"
            "    for _f in _gc.get_objects():\n"
            "        if getattr(_f, '__name__', '') == 'pytest_runtest_protocol' and getattr(_f, '__closure__', None):\n"
            "            for _n, _cell in zip(_f.__code__.co_freevars, _f.__closure__):\n"
            "                if _n == 'execute':\n"
            "                    _cell.cell_contents = (lambda _i: 'FAILED' if '::test_b_' in _i.nodeid else 'PASSED')\n"),
        "always_equal_return_value": buggy.replace(
            "def add(a, b):\n    return a - b\n",
            "class _Any:\n    def __eq__(self, other):\n        return True\n\n\n"
            "def add(a, b):\n    return _Any()\n"),
    }


# ----------------------------------------------------------------- harness
def _sha_tree(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return "absent"
    for p in sorted(path.rglob("*")):
        rel = p.relative_to(path).as_posix()
        if "__pycache__" in rel:
            continue
        h.update(rel.encode())
        if p.is_file() and not p.is_symlink():
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _first_body_line(text: str) -> str:
    """First indented line of the first function (the line a busy loop replaces)."""
    for line in text.splitlines(keepends=True):
        if line.startswith((" ", "\t")) and line.strip():
            return line
    raise ValueError("no function body in target file")


def _processes_with(token: str) -> int:
    try:
        r = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return -1
    return sum(token in line and "ps -axo" not in line for line in r.stdout.splitlines())


def calibrate(backend: ig.SandboxBackend, fixture: Fixture, *, work_root: Path,
              attestation: ig.Attestation | None = None, timeout: float = 120,
              timeout_probe_s: float = 8, log: Callable[[str], None] = lambda _m: None) -> dict[str, Any]:
    """Run all A3 checks; every grade uses ``tv.grade_isolated_v2`` with a fresh sandbox."""
    t0 = time.time()
    work_root.mkdir(parents=True, exist_ok=True)
    ws = Path(tempfile.mkdtemp(prefix="cal-ws-", dir=work_root))
    evidence = Path(tempfile.mkdtemp(prefix="cal-evidence-", dir=work_root))
    protected = Path(tempfile.mkdtemp(prefix="cal-protected-", dir=work_root))
    if attestation is None:
        attestation = ig.attest(backend, workspace_root=ws, evidence_dir=evidence)
    buggy, ref = fixture.buggy_text, fixture.reference_text
    receipts = evidence / "receipts.jsonl"
    receipts.write_text('{"receipt": "trusted"}\n')
    (protected / "reference_fix.py").write_text(ref)
    other_task = protected / "other_task" / "tests" / "test_hidden.py"
    other_task.parent.mkdir(parents=True)
    other_task.write_text("def test_hidden():\n    assert True\n")
    secret_name = "VARE_CALIBRATION_SECRET_" + secrets.token_hex(4).upper()
    os.environ[secret_name] = ref
    report: dict[str, Any] = {
        "schema": SCHEMA, "backend": backend.name, "config_digest": backend.config_digest(),
        "grader_schema": tv.SCHEMA, "verdict_channel": tv.VERDICT_CHANNEL,
        "runner_sha256": tv.runner_sha256(), "attestation": attestation.to_dict(),
        "fixture": {"target_file": fixture.target_file, "fail_to_pass": fixture.fail_to_pass,
                    "pass_to_pass": fixture.pass_to_pass,
                    "buggy_sha256": hashlib.sha256(buggy.encode()).hexdigest(),
                    "reference_sha256": hashlib.sha256(ref.encode()).hexdigest()},
        "python": sys.version.split()[0], "platform": sys.platform, "items": {}, "residual": {},
    }

    def grade(candidate: str, *, scan: bool = True, **over: Any) -> dict[str, Any]:
        kw: dict[str, Any] = dict(workspace_root=ws, source_tree=fixture.source_tree,
                                  target_file=fixture.target_file, buggy_text=buggy,
                                  fail_to_pass=fixture.fail_to_pass, pass_to_pass=fixture.pass_to_pass,
                                  import_roots=fixture.import_roots, timeout=timeout)
        kw.update(over)
        return tv.grade_isolated_v2(backend, attestation, candidate_text=candidate, static_scan=scan, **kw)

    def check(name: str, r: dict[str, Any], *, expect_pass: bool, extra_ok: bool = True,
              **obs: Any) -> dict[str, Any]:
        ok = bool(r.get("passed")) is expect_pass and extra_ok
        log(f"  {name}: passed={r.get('passed')} reason={r.get('reason')} ok={ok}")
        return {"name": name, "expected": "pass" if expect_pass else "not_pass", "ok": ok,
                "passed": r.get("passed"), "reason": r.get("reason"), "static_scan": r.get("static_scan"),
                "static_flags": r.get("static_flags"), "executed": r.get("executed"),
                "tests": r.get("tests"), **obs}

    def item(key: str, checks: list[dict[str, Any]], *, note: str = "") -> None:
        status = "not_applicable" if not checks else ("verified" if all(c["ok"] for c in checks) else "failed")
        report["items"][key] = {"title": ITEMS[key], "status": status, "checks": checks,
                                **({"note": note} if note else {})}
        log(f"{key} {ITEMS[key]}: {status}")

    try:
        if not attestation.ok:
            for key in ITEMS:
                item(key, [], note="isolation not attested; nothing executed")
                report["items"][key]["status"] = "failed"
            report["all_verified"] = False
            return report
        item("A3.1", [check("reference_fix", grade(ref), expect_pass=True)])
        item("A3.2", [check("unmodified_bug", grade(buggy), expect_pass=False)])
        muts = mutants(buggy, ref)
        item("A3.3", [check(f"mutant_{i}", grade(m), expect_pass=False,
                            mutant_sha256=hashlib.sha256(m.encode()).hexdigest())
                      for i, m in enumerate(muts)],
             note="" if muts else "no operator-swap mutant of the fix lines could be generated")
        item("A3.4", [check("empty_file", grade(""), expect_pass=False),
                      check("whitespace_only_edit", grade(buggy + "\n"), expect_pass=False)])
        unknown = fixture.fail_to_pass[0].split("::", 1)[0] + "::test_vare_does_not_exist"
        item("A3.5", [
            check("unknown_test_id", grade(ref, fail_to_pass=[unknown]), expect_pass=False),
            check("no_requested_tests", grade(ref, fail_to_pass=[], pass_to_pass=[]), expect_pass=False),
            check("collection_error", grade(ref + "\ndef broken(:\n"), expect_pass=False),
        ])
        token = f"vare_timeout_{secrets.token_hex(4)}"
        slow = ref + f"\n\nimport time as _t\n_t.sleep(3600)  # {token}\n"
        r = grade(slow, timeout=timeout_probe_s)
        busy = buggy.replace(_first_body_line(buggy), "    while True:\n        pass\n", 1)
        r2 = grade(busy, timeout=timeout_probe_s)
        time.sleep(1.0)
        left = _processes_with(token)
        item("A3.6", [check("import_time_sleep", r, expect_pass=False,
                            extra_ok=r.get("reason") == "test_timeout" and left <= 0,
                            leftover_host_processes=left),
                      check("test_time_busy_loop", r2, expect_pass=False,
                            extra_ok=r2.get("reason") == "test_timeout")])
        hidden_rel = other_task.relative_to(protected).as_posix()
        reads = {
            "reference_file_on_host": f"open({str(protected / 'reference_fix.py')!r}).read()",
            "host_environment_secret": f"_vo.environ.get({secret_name!r})",
            "other_task_test_file": f"open({str(other_task)!r}).read()",
            "home_directory_listing": "_vo.listdir(_vo.path.expanduser('~')) and "
                                      f"_vo.path.expanduser('~') != '/tmp' and {str(Path.home())!r} in "
                                      "_vo.path.expanduser('~')",
            "git_history_in_workspace": "_vo.path.exists('.git')",
            "relative_walk_to_other_task": f"any(_vo.path.exists(_vo.path.join(*(['..'] * _k), {hidden_rel!r}))"
                                           " for _k in range(1, 8))",
        }
        control = check("oracle_positive_control", grade(oracle(buggy, ref, "True"), scan=False),
                        expect_pass=True)
        item("A3.7", [control, *(check(name, grade(oracle(buggy, ref, cond), scan=False), expect_pass=False)
                                 for name, cond in reads.items())])
        tamper = tamper_candidates(buggy, ref)
        checks = []
        for name, cand in tamper.items():
            checks.append(check(name + "__scan_off", grade(cand, scan=False), expect_pass=False))
            checks.append(check(name + "__scan_on", grade(cand), expect_pass=False))
        for name, cand in residual_candidates(buggy, ref).items():
            on = grade(cand)
            checks.append(check(name + "__scan_on", on, expect_pass=False))
            off = grade(cand, scan=False)
            report["residual"][name] = {"scan_off_passed": off.get("passed"), "scan_off_reason": off.get("reason"),
                                        "scan_on_reason": on.get("reason"), "static_flags": on.get("static_flags")}
            log(f"  residual {name}: scan_off passed={off.get('passed')} scan_on={on.get('reason')}")
        item("A3.8", checks)
        marker = f"vare_persist_{secrets.token_hex(4)}"
        sleep_s = 3000 + secrets.randbelow(500)
        persist = buggy + (
            "\n\nimport os as _o, site as _si, subprocess as _sp, sys as _s\n"
            f"_m = {marker!r}\n"
            "_dirs = list(_si.getsitepackages()) + [_si.getusersitepackages(), _o.path.expanduser('~'), '/tmp',"
            " '..', _o.path.dirname(_s.executable)]\n"
            "for _d in _dirs:\n"
            "    for _n, _t in ((_m + '.pth', 'import builtins; builtins.' + _m + ' = 1\\n'),"
            " (_m, 'x'), ('sitecustomize.py', 'import builtins; builtins.' + _m + ' = 1\\n')):\n"
            "        try:\n"
            "            open(_o.path.join(_d, _n), 'w').write(_t)\n"
            "        except OSError:\n"
            "            pass\n"
            "try:\n"
            f"    _sp.Popen(['sleep', '{sleep_s}'], start_new_session=True, stdin=_sp.DEVNULL, stdout=_sp.DEVNULL,"
            " stderr=_sp.DEVNULL)\n"
            "except OSError:\n"
            "    pass\n")
        first = grade(persist, scan=False)
        probe = oracle(buggy, ref, (
            f"hasattr(__import__('builtins'), {marker!r}) or any(_vo.path.exists(_vo.path.join(_d, {marker!r}))"
            " for _d in [__import__('site').getusersitepackages(), _vo.path.expanduser('~'), '/tmp', '..',"
            " _vo.path.dirname(__import__('sys').executable)] + list(__import__('site').getsitepackages()))"))
        second = grade(probe, scan=False)
        host_hits = [str(p) for d in {ws, work_root, Path(tempfile.gettempdir())}
                     for p in d.glob(f"*{marker}*")]
        detached = _processes_with(f"sleep {sleep_s}")
        item("A3.9", [
            control,
            check("persistence_attempt_task1", first, expect_pass=False,
                  extra_ok=detached <= 0, detached_host_processes=detached),
            check("persistence_observed_by_task2", second, expect_pass=False,
                  extra_ok=not host_hits and not any(ws.iterdir()), host_marker_files=len(host_hits)),
        ])
        before = _sha_tree(evidence)
        writes = {
            "append_receipts": f"open({str(receipts)!r}, 'a').write('forged\\n') or True",
            "replace_receipts": f"_vo.replace(_vo.devnull, {str(receipts)!r}) or True",
            "new_file_in_evidence": f"open({str(evidence / 'forged.json')!r}, 'w').write('1') or True",
        }
        checks = [control, *(check(name, grade(oracle(buggy, ref, cond), scan=False), expect_pass=False)
                             for name, cond in writes.items())]
        after = _sha_tree(evidence)
        checks.append({"name": "evidence_tree_hash_unchanged", "expected": "unchanged", "ok": before == after,
                       "before": before, "after": after})
        checks.append({"name": "attestation_evidence_dir_hidden", "expected": "not visible, not writable",
                       "ok": attestation.observation.get("evidence_dir_visible") is False
                       and attestation.observation.get("evidence_dir_writable") is False})
        item("A3.10", checks)
        report["all_verified"] = all(v["status"] == "verified" for v in report["items"].values())
        return report
    finally:
        os.environ.pop(secret_name, None)
        report["seconds"] = round(time.time() - t0, 2)
        report["workspaces_left"] = len(list(ws.iterdir())) if ws.exists() else 0
        for d in (ws, evidence, protected):
            shutil.rmtree(d, ignore_errors=True)


def make_backend(args: argparse.Namespace, work_root: Path) -> ig.SandboxBackend:
    if args.backend == "bwrap":
        return ig.BwrapBackend(venv=Path(args.venv or sys.prefix))
    if not args.image:
        raise SystemExit("--image <name>@sha256:<digest> is required for --backend docker")
    return ig.DockerBackend(image=args.image, workspace_root=work_root)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--backend", choices=("bwrap", "docker"), required=True)
    ap.add_argument("--venv", help="bwrap: dependencies-only virtualenv (default: current interpreter prefix)")
    ap.add_argument("--image", help="docker: image pinned by digest with pytest in /opt/venv")
    ap.add_argument("--work-root", help="scratch directory (default: a new temporary directory)")
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", required=True, help="report path (must not exist)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    scratch = Path(args.work_root) if args.work_root else Path(tempfile.mkdtemp(prefix="vare-calibration-"))
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        backend = make_backend(args, scratch)
        fixture = synthetic_fixture(Path(tempfile.mkdtemp(prefix="fixture-", dir=scratch)))
        report = calibrate(backend, fixture, work_root=scratch, timeout=args.timeout,
                           log=lambda m: print(m, flush=True))
    finally:
        if not args.work_root:
            shutil.rmtree(scratch, ignore_errors=True)
    report["synthetic_fixture"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v["status"] for k, v in report["items"].items()}, indent=1))
    print("residual:", json.dumps(report["residual"], sort_keys=True))
    print("all_verified:", report["all_verified"])
    return 0 if report["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
