"""Native (no Docker) repository environments and the independent grader.

One clean clone and one virtualenv per upstream repository live under the
cache directory. Grading is sequential: reset the clone to its pinned commit,
apply the bug patch, overwrite only the task's target source file with the
candidate text, run the selected tests, and reset again. The evaluated model
never has tool or filesystem access; it only returns text, so the tests, the
reference fix and the grader are outside its reach by construction.

A grade fails closed: patch-apply errors, timeouts, missing test ids,
collection errors or any non-PASSED selected test all score 0.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Mapping, Sequence

# Extra test-time requirements discovered during box calibration. Kept small
# and explicit; anything not listed is installed only via `pip install -e .`.
GRADER_PYTEST = "pytest==8.3.5"

EXTRA_TEST_DEPS: dict[str, tuple[str, ...]] = {
    "Suor/funcy": ("whatever==0.7",),
    "msiemens/tinydb": ("PyYAML==6.0.3", "pytest-cov==7.1.0"),
    "arrow-py/arrow": ("pytz", "simplejson", "python-dateutil", "pytest-mock", "dateparser"),
    "spulec/freezegun": ("python-dateutil",),
    "jd/tenacity": ("tornado", "typeguard"),
    "theskumar/python-dotenv": ("sh",),
    "cool-RR/PySnooper": ("python-toolbox",),
}

_RESULT = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S.*?)(?:\s+-\s.*)?$")


def _run(cmd: Sequence[str], cwd: Path | None = None, timeout: float = 600, env=None):
    return subprocess.run(
        list(cmd), cwd=cwd, timeout=timeout, capture_output=True, text=True, env=env
    )


@dataclass
class RepoEnv:
    mirror: str  # e.g. "swesmith/Suor__funcy.207a7810"
    upstream: str  # e.g. "Suor/funcy"
    commit: str  # full sha of the mirror commit pinned in the manifest
    cache: Path

    @property
    def name(self) -> str:
        return self.mirror.split("/", 1)[1]

    @property
    def path(self) -> Path:
        return self.cache / "repos" / self.name

    @property
    def venv(self) -> Path:
        return self.cache / "venvs" / self.name

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    def materialize(self, python_exe: str = sys.executable) -> dict[str, object]:
        t0 = time.time()
        if not (self.path / ".git").exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            r = _run(["git", "clone", "-q", f"https://github.com/{self.mirror}", str(self.path)])
            if r.returncode:
                raise RuntimeError(f"clone failed for {self.mirror}: {r.stderr[-500:]}")
        _run(["git", "fetch", "-q", "origin", self.commit], cwd=self.path)
        self.reset()
        head = _run(["git", "rev-parse", "HEAD"], cwd=self.path).stdout.strip()
        if head != self.commit:
            raise RuntimeError(f"{self.mirror}: HEAD {head} != pinned {self.commit}")
        if not self.python.exists():
            r = _run([python_exe, "-m", "venv", str(self.venv)])
            if r.returncode:
                raise RuntimeError(f"venv failed: {r.stderr[-500:]}")
            deps = [GRADER_PYTEST] + list(EXTRA_TEST_DEPS.get(self.upstream, ()))
            r = _run([str(self.python), "-m", "pip", "-q", "install", "-e", str(self.path), *deps],
                     timeout=900)
            if r.returncode:
                raise RuntimeError(f"install failed for {self.mirror}: {r.stderr[-800:]}")
        return {"mirror": self.mirror, "commit": head, "seconds": round(time.time() - t0, 2)}

    def reset(self) -> None:
        _run(["git", "checkout", "-q", "-f", self.commit], cwd=self.path)
        # -fd (not -x): keeps ignored build products such as setuptools-scm
        # version files and egg-info that the editable install relies on.
        _run(["git", "clean", "-q", "-fd"], cwd=self.path)

    def read(self, rel: str) -> str:
        return (self.path / rel).read_text(encoding="utf-8")

    def buggy_file_text(self, bug_patch: str, rel: str) -> str:
        self.reset()
        try:
            self._apply(bug_patch)
            return self.read(rel)
        finally:
            self.reset()

    def _apply(self, patch: str) -> None:
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=self.path,
                           input=patch, capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise RuntimeError(f"bug patch did not apply: {r.stderr[-300:]}")

    def grade(
        self,
        *,
        bug_patch: str,
        target_file: str,
        candidate_text: str | None,
        fail_to_pass: Sequence[str],
        pass_to_pass: Sequence[str],
        timeout: float = 180,
    ) -> dict[str, object]:
        """Grade one candidate. ``candidate_text=None`` grades the unrepaired bug."""
        t0 = time.time()
        self.reset()
        try:
            try:
                self._apply(bug_patch)
            except RuntimeError as exc:
                return _fail("bug_patch_apply_error", t0, detail=str(exc))
            if candidate_text is not None:
                (self.path / target_file).write_text(candidate_text, encoding="utf-8")
            changed = _run(["git", "diff", "--name-only"], cwd=self.path).stdout.split()
            if candidate_text is not None and any(p != target_file for p in changed):
                return _fail("protected_file_changed", t0)
            ids = list(dict.fromkeys([*fail_to_pass, *pass_to_pass]))
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTEST_ADDOPTS="")
            env.pop("PYTHONPATH", None)
            cmd = [str(self.python), "-m", "pytest", "-rA", "-q", "-p", "no:cacheprovider",
                   "--no-header", *ids]  # repo addopts are honored (e.g. -p no:logging)
            t_test = time.time()
            try:
                r = _run(cmd, cwd=self.path, timeout=timeout, env=env)
            except subprocess.TimeoutExpired:
                return _fail("test_timeout", t0, tests={t: "TIMEOUT" for t in ids})
            pytest_seconds = round(time.time() - t_test, 3)
            status = parse_pytest_results(r.stdout)
            f2p = {t: status.get(t, "MISSING") for t in fail_to_pass}
            p2p = {t: status.get(t, "MISSING") for t in pass_to_pass}
            ok = all(v == "PASSED" for v in f2p.values()) and all(v == "PASSED" for v in p2p.values())
            return {
                "passed": bool(ok),
                "reason": "pass" if ok else "tests_failed",
                "f2p_passed": sum(v == "PASSED" for v in f2p.values()),
                "f2p_total": len(f2p),
                "p2p_passed": sum(v == "PASSED" for v in p2p.values()),
                "p2p_total": len(p2p),
                "missing": sorted(t for t, v in {**f2p, **p2p}.items() if v == "MISSING")[:10],
                "pytest_exit": r.returncode,
                "stdout_sha256": hashlib.sha256(r.stdout.encode()).hexdigest(),
                "seconds": round(time.time() - t0, 3),
                "pytest_seconds": pytest_seconds,
                # Per-test status for every requested id (PASSED/FAILED/ERROR/
                # SKIPPED/XFAIL/XPASS/MISSING), for offline verifier replay.
                "tests": {**f2p, **p2p},
            }
        finally:
            self.reset()


def _fail(reason: str, t0: float, **extra: object) -> dict[str, object]:
    return {"passed": False, "reason": reason, "seconds": round(time.time() - t0, 3), **extra}


def parse_pytest_results(stdout: str) -> dict[str, str]:
    """Parse ``pytest -rA`` short-summary lines into {nodeid: STATUS}."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        m = _RESULT.match(line.strip())
        if m:
            status, nodeid = m.group(1), m.group(2).strip()
            # An ERROR or FAILED line wins over an earlier PASSED for the same id.
            if out.get(nodeid) in ("FAILED", "ERROR"):
                continue
            out[nodeid] = status
    return out


def remote_head(mirror: str) -> str:
    r = _run(["git", "ls-remote", f"https://github.com/{mirror}", "HEAD"], timeout=60)
    if r.returncode or not r.stdout.strip():
        raise RuntimeError(f"ls-remote failed for {mirror}")
    return r.stdout.split()[0]


def env_for(task: Mapping[str, object], cache: Path) -> RepoEnv:
    return RepoEnv(str(task["mirror"]), str(task["upstream_repo"]), str(task["mirror_commit"]), cache)


def dump_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
