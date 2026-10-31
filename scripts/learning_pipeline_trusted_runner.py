"""In-sandbox test executor whose results are judged by a trusted parent.

This file is executed *inside* the sandbox as ``python -I -c <source> <spec>``
by ``learning_pipeline_trusted_verdict``. It must stay self-contained (standard
library plus pytest) and must not be imported by the grader process.

Order of operations, chosen so that every security-relevant step happens
before any candidate-controlled code is imported:

1. read the per-call frame key from standard input (one hex line), then close
   standard input; the key is never placed in argv, the environment or a file;
2. keep a private duplicate of the original standard output as the frame
   channel and point file descriptor 1 at standard error, so pytest output and
   anything candidate code prints never share the channel's descriptor number;
3. capture references to the pytest internals used for execution
   (``SetupState.setup``/``teardown_exact``, ``TestCaseFunction.runtest``,
   skip/xfail mark evaluation) and to ``hmac``/``os.write``;
4. emit an authenticated ``start`` frame, then run pytest with a plugin that
   replaces ``pytest_runtest_protocol``: each item's outcome is decided here by
   calling the test function directly (no ``pytest_pyfunc_call``,
   ``pytest_runtest_makereport`` or report objects are consulted), and one
   authenticated ``result`` frame is written per item;
5. emit a ``finish`` frame carrying the exit code this runner will return.

Frames are ``VARE_FRAME <hmac-sha256 hex> <canonical json>``. The parent
rejects the whole grade if any frame fails authentication, is out of
sequence, repeats a test id, or disagrees with the process exit code.

Limitation: candidate code still runs in this interpreter. Code written to
search the interpreter's memory for the key, or to replace test functions
through interpreter hooks before they are collected, is not excluded by this
design (see docs/learning-pipeline-grader-isolation.md, section 6).
"""
import hashlib
import hmac
import json
import os
import sys


def _main():
    spec = json.loads(sys.argv[1])
    key = bytes.fromhex(sys.stdin.readline().strip())
    if len(key) < 16:
        os._exit(97)
    sys.stdin.close()
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    channel = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    write, new_mac, sha256, dumps = os.write, hmac.new, hashlib.sha256, json.dumps
    nonce = spec["nonce"]
    seq = [0]

    def emit(kind, body):
        payload = dumps(dict(body, kind=kind, seq=seq[0], nonce=nonce), sort_keys=True,
                        separators=(",", ":"))
        mac = new_mac(key, payload.encode(), sha256).hexdigest()
        data = ("VARE_FRAME " + mac + " " + payload + "\n").encode()
        while data:
            data = data[write(channel, data):]
        seq[0] += 1

    for root in reversed(spec["import_roots"]):
        sys.path.insert(0, root)
    emit("start", {"python": sys.version.split()[0]})

    import pytest
    from _pytest.python import Function
    from _pytest.runner import SetupState
    from _pytest.skipping import evaluate_skip_marks, evaluate_xfail_marks
    from _pytest.unittest import TestCaseFunction

    setup, teardown_exact = SetupState.setup, SetupState.teardown_exact
    unittest_runtest = TestCaseFunction.runtest
    BaseExc, isinstance_, getattr_, type_ = BaseException, isinstance, getattr, type
    statuses = []

    def execute(item):
        if evaluate_skip_marks(item) is not None:
            return "SKIPPED"
        if evaluate_xfail_marks(item) is not None:
            return "XFAIL_MARKED"
        setup(item.session._setupstate, item)
        if isinstance_(item, TestCaseFunction):
            if getattr_(item.obj, "__unittest_expecting_failure__", False):
                return "XFAIL_MARKED"
            unittest_runtest(item)
            return "FAILED" if getattr_(item, "_excinfo", None) else "PASSED"
        if type_(item) is not Function:
            return "UNSUPPORTED_ITEM"
        names = item._fixtureinfo.argnames
        result = item.obj(**{name: item.funcargs[name] for name in names})
        if result is not None and (hasattr(result, "__await__") or hasattr(result, "send")):
            close = getattr_(result, "close", None)
            if close is not None:
                close()
            return "UNSUPPORTED_ITEM"
        return "PASSED"

    wanted = list(dict.fromkeys(spec["test_ids"]))
    wanted_set = frozenset(wanted)
    # Collect whole files and select the requested ids here, so that a missing id
    # is reported as absent instead of aborting the session (and the canaries).
    files = [f for f in dict.fromkeys(t.split("::", 1)[0] for t in wanted) if os.path.isfile(f)]

    class TrustedExecution:
        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, session, config, items):
            items[:] = [item for item in items if item.nodeid in wanted_set]

        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_finish(self, session):
            emit("collected", {"ids": [item.nodeid for item in session.items]})

        @pytest.hookimpl(tryfirst=True)
        def pytest_runtest_protocol(self, item, nextitem):
            try:
                status = execute(item)
            except BaseExc as exc:  # noqa: B902 - any exception, incl. SystemExit, is a failure
                status = "SKIPPED" if type_(exc).__name__ == "Skipped" else "FAILED"
            try:
                teardown_exact(item.session._setupstate, nextitem)
            except BaseExc:
                if status == "PASSED":
                    status = "ERROR"
            statuses.append(status)
            emit("result", {"nodeid": item.nodeid, "status": status})
            return True

    pytest_exit = pytest.main(["-q", "-p", "no:cacheprovider", "--continue-on-collection-errors",
                               *files], plugins=[TrustedExecution()])
    code = 0 if statuses and all(s == "PASSED" for s in statuses) else 1
    emit("finish", {"exit": code, "pytest_exit": int(pytest_exit), "n_results": len(statuses)})
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _main()
