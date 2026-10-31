"""Sandboxed, fail-closed grading for learning-pipeline candidates (proposal).

Not used by the frozen v1 protocol. Kept outside ``src/vare`` so that the
hash-frozen ``vare.learning_pipeline`` package and the versioned continuation
source inventory stay unchanged. ``repo_env.RepoEnv.grade`` (v1) writes the
candidate into a host clone and runs pytest as the operator, with the
operator's environment, network, filesystem and process table. The candidate
file is imported by the tests, so candidate code runs with all of that
authority (see docs/learning-pipeline-grader-isolation.md).

This module grades one candidate in a fresh sandbox per call:

* the trusted side stages a copy of the buggy tree (no ``.git``, no other
  task, no reference fix) into an ephemeral workspace and writes the
  candidate file there; the candidate text is hashed before execution;
* the sandbox has no network, a cleared environment, a non-root uid, a
  read-only interpreter/dependency tree and no view of host paths except the
  workspace; the workspace is deleted afterwards;
* a pass is reported only if (1) an isolation attestation for the *same*
  sandbox configuration exists and every negative probe in it was denied,
  (2) two per-call canary tests (one must pass, one must fail, random names)
  have their expected outcomes, (3) the pytest exit code is the one implied by
  the expected failing canary, (4) every requested test id is PASSED, and
  (5) the candidate's added lines do not match the blocking static patterns.
  Anything else, including a missing sandbox tool, is ``passed: False``.

Residual risk (not removed by this module): candidate code runs inside the
pytest process, so code written specifically to forge pytest results can in
principle defeat any in-sandbox check. Canaries, the exit-code cross-check and
the static scan detect the simple forgeries reproduced in the audit; they are
not a proof of verdict integrity. Isolation probes check the configured
sandbox only; they do not certify the kernel, the container runtime or the VM.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

from vare.learning_pipeline.repo_env import parse_pytest_results

SCHEMA = "learning_pipeline_isolated_grade_v1"
ENV_ALLOWLIST = frozenset({"PATH", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED",
                           "PYTHONPATH", "TMPDIR", "PWD", "SHLVL", "_", "HOSTNAME", "LC_CTYPE",
                           "TERM", "GPG_KEY", "PYTHON_VERSION", "PYTHON_SHA256"})
MAX_VISIBLE_PROCESSES = 16

# Executed inside the sandbox. Reports what candidate code could reach; every
# field must have the denied value for the attestation to pass.
PROBE_SCRIPT = r'''
import json, os, socket, sys
spec = json.loads(sys.argv[1])
out = {}
def can_read(p):
    try:
        with open(p, "rb") as f:
            f.read(1)
        return True
    except OSError:
        return False
def can_write_dir(d):
    p = os.path.join(d, ".vare_probe_%d" % os.getpid())
    try:
        with open(p, "w") as f:
            f.write("x")
        os.unlink(p)
        return True
    except OSError:
        return False
out["uid"] = os.getuid()
out["env_canary_visible"] = spec["env_canary_name"] in os.environ
out["env_unexpected_keys"] = sorted(k for k in os.environ if k not in spec["env_allowlist"])
out["host_canary_readable"] = can_read(spec["host_canary_path"])
out["evidence_dir_visible"] = os.path.exists(spec["evidence_dir"])
out["evidence_dir_writable"] = can_write_dir(spec["evidence_dir"])
out["interpreter_tree_writable"] = any(can_write_dir(d) for d in spec["interpreter_dirs"])
out["rootfs_writable"] = can_write_dir("/") or can_write_dir("/usr")
out["workspace_writable"] = can_write_dir(os.getcwd())
try:
    with open("/proc/net/route") as f:
        rows = [ln.split() for ln in f.read().splitlines()[1:]]
    out["default_route_present"] = any(len(r) > 1 and r[1] == "00000000" for r in rows)
except OSError:
    out["default_route_present"] = False
try:
    with open("/proc/net/dev") as f:
        names = [ln.split(":")[0].strip() for ln in f.read().splitlines()[2:]]
    out["non_loopback_interfaces"] = sorted(n for n in names if n and n != "lo")
except OSError:
    out["non_loopback_interfaces"] = []
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", int(spec["host_loopback_port"])))
    s.sendall(spec["host_loopback_token"].encode())
    out["host_loopback_connect"] = True
except OSError:
    out["host_loopback_connect"] = False
finally:
    s.close()
out["docker_socket_present"] = any(os.path.exists(p) for p in
                                   ("/var/run/docker.sock", "/run/docker.sock"))
try:
    out["visible_processes"] = sum(1 for n in os.listdir("/proc") if n.isdigit())
except OSError:
    out["visible_processes"] = 0
print("VARE_PROBE " + json.dumps(out, sort_keys=True))
'''

DENIED: dict[str, Any] = {
    "env_canary_visible": False,
    "env_unexpected_keys": [],
    "host_canary_readable": False,
    "evidence_dir_visible": False,
    "evidence_dir_writable": False,
    "interpreter_tree_writable": False,
    "rootfs_writable": False,
    "default_route_present": False,
    "non_loopback_interfaces": [],
    "host_loopback_connect": False,
    "docker_socket_present": False,
    "workspace_writable": True,  # required for grading at all
}

# Added lines matching these are not graded (quarantined, passed False).
BLOCKING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("process_exit", r"\bos\._exit\b|\bsys\.exit\b|\bos\.kill\b|\bos\.abort\b"),
    ("argv_access", r"\bsys\.argv\b|\bsys\.orig_argv\b"),
    ("test_runner_access", r"\b_?pytest\b|\bunittest\b|\bconftest\b|\bPASSED\b"),
    ("process_spawn", r"\bsubprocess\b|\bos\.(?:system|popen|fork|exec\w*|spawn\w*)\b"),
    ("network", r"\bsocket\b|\burllib\b|\bhttp\.client\b|\brequests\b"),
    ("native_code", r"\bctypes\b|\bcffi\b"),
    ("dynamic_import", r"\bimportlib\b|__import__\s*\(|\bsys\.modules\b|\bbuiltins\b"),
    ("interpreter_hooks", r"\bsys\.settrace\b|\bsys\.setprofile\b|\batexit\b|\bsitecustomize\b|\.pth\b"),
)


class IsolationUnavailable(RuntimeError):
    """The sandbox tool is missing or refused to start (fail closed)."""


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str


class SandboxBackend(Protocol):
    name: str

    def available(self) -> bool: ...

    def config_digest(self) -> str: ...

    def interpreter_dirs(self) -> list[str]: ...

    def run(self, workspace: Path, command: Sequence[str], *, timeout: float) -> RunResult: ...

    def python(self) -> str: ...


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@dataclass
class BwrapBackend:
    """Linux user-namespace sandbox (bubblewrap). Used for tests on Linux hosts.

    ``venv`` holds the dependencies (pytest, test extras) but not the repository
    itself; it is bind-mounted read-only at its own path. The workspace is the
    only writable bind, at ``/work``.
    """

    venv: Path
    bwrap: str = "bwrap"
    import_roots: tuple[str, ...] = ("/work",)
    name: str = "bwrap"

    def available(self) -> bool:
        exe = shutil.which(self.bwrap)
        if not exe or not (self.venv / "bin" / "python").exists():
            return False
        try:
            r = subprocess.run([exe, "--unshare-all", "--ro-bind", "/", "/", "true"],
                               capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0

    def python(self) -> str:
        return str(self.venv / "bin" / "python")

    def interpreter_dirs(self) -> list[str]:
        return [str(self.venv), str(self.venv / "bin"), *sorted(
            str(p) for p in (self.venv / "lib").glob("python*/site-packages"))]

    def _system_binds(self) -> list[str]:
        args = ["--ro-bind", "/usr", "/usr"]
        for top in ("/bin", "/lib", "/lib64", "/lib32", "/sbin"):
            p = Path(top)
            if p.is_symlink():
                args += ["--symlink", os.readlink(p), top]
            elif p.exists():
                args += ["--ro-bind", top, top]
        for f in ("/etc/ld.so.cache", "/etc/localtime"):
            args += ["--ro-bind-try", f, f]
        return args

    def _argv(self, workspace: Path, command: Sequence[str]) -> list[str]:
        return [
            self.bwrap, "--unshare-all", "--die-with-parent", "--new-session",
            "--uid", "65534", "--gid", "65534", "--hostname", "sandbox",
            "--clearenv", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
            "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
            "--setenv", "PYTHONHASHSEED", "0", "--setenv", "PYTHONPATH", ":".join(self.import_roots),
            # tmpfs first: later binds may live below /tmp on the host.
            "--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev",
            *self._system_binds(),
            "--ro-bind", str(self.venv), str(self.venv),
            "--bind", str(workspace), "/work", "--chdir", "/work",
            "--remount-ro", "/",
            "--", *command,
        ]

    def config_digest(self) -> str:
        return _digest({"backend": self.name, "argv": self._argv(Path("/WORKSPACE"), ["CMD"])})

    def run(self, workspace: Path, command: Sequence[str], *, timeout: float) -> RunResult:
        try:
            r = subprocess.run(self._argv(workspace, command), capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL, start_new_session=True)
        except FileNotFoundError as exc:
            raise IsolationUnavailable("bwrap not found") from exc
        return RunResult(r.returncode, r.stdout, r.stderr)


@dataclass
class DockerBackend:
    """Docker/Colima backend built on ``vare.container_isolation``.

    The image must be pinned by digest and contain the task's dependencies in
    ``/opt/venv`` (not the repository; the workspace is mounted at
    ``/workspace``). Flags come from ``DockerIsolationRunner``: no network,
    read-only root, all capabilities dropped, no-new-privileges, uid 65534,
    pids/memory/cpu limits, ``--rm``.
    """

    image: str
    workspace_root: Path
    memory_mb: int = 2048
    cpus: float = 2.0
    pids_limit: int = 256
    import_roots: tuple[str, ...] = ("/workspace",)
    venv_python: str = "/opt/venv/bin/python"
    name: str = "docker"

    def _limits(self, timeout: float) -> Any:
        from vare.container_isolation import IsolationLimits

        return IsolationLimits(image=self.image, timeout_s=timeout, memory_mb=self.memory_mb,
                               cpus=self.cpus, pids_limit=self.pids_limit, max_output_bytes=500_000)

    def available(self) -> bool:
        if not shutil.which("docker"):
            return False
        try:
            r = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0

    def python(self) -> str:
        return self.venv_python

    def interpreter_dirs(self) -> list[str]:
        return ["/opt/venv", "/opt/venv/bin", "/usr/local/lib"]

    def _command(self, command: Sequence[str]) -> list[str]:
        return ["env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "HOME=/tmp", "LANG=C.UTF-8",
                "PYTHONDONTWRITEBYTECODE=1", "PYTHONHASHSEED=0",
                "PYTHONPATH=" + ":".join(self.import_roots), *command]

    def config_digest(self) -> str:
        return _digest({"backend": self.name, "image": self.image, "memory_mb": self.memory_mb,
                        "cpus": self.cpus, "pids_limit": self.pids_limit,
                        "import_roots": list(self.import_roots)})

    def run(self, workspace: Path, command: Sequence[str], *, timeout: float) -> RunResult:
        from vare.container_isolation import DockerIsolationRunner, IsolationError

        runner = DockerIsolationRunner()
        limits = self._limits(timeout)
        argv = runner.candidate_argv(limits, name=f"vare-grade-{secrets.token_hex(12)}",
                                     allowed_workspace_root=self.workspace_root, workspace=workspace,
                                     command=self._command(command))
        try:
            res = asyncio.run(runner.execute(argv, limits))
        except IsolationError as exc:
            raise IsolationUnavailable(str(exc)) from exc
        return RunResult(res.returncode, res.stdout, res.stderr)


@dataclass(frozen=True)
class Attestation:
    backend: str
    config_digest: str
    ok: bool
    failures: tuple[str, ...]
    observation: Mapping[str, Any]
    created_unix: float

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "config_digest": self.config_digest, "ok": self.ok,
                "failures": list(self.failures), "observation": dict(self.observation),
                "created_unix": self.created_unix}


def evaluate_probe(obs: Mapping[str, Any] | None) -> tuple[bool, tuple[str, ...]]:
    """Every expected key present with the denied value, else fail (closed)."""
    if not isinstance(obs, Mapping):
        return False, ("probe_output_missing",)
    failures = [k for k, want in DENIED.items() if k not in obs or obs[k] != want]
    uid = obs.get("uid")
    if type(uid) is not int or uid == 0:
        failures.append("uid")
    vp = obs.get("visible_processes")
    if type(vp) is not int or not 0 < vp <= MAX_VISIBLE_PROCESSES:
        failures.append("visible_processes")
    return not failures, tuple(failures)


def _parse_probe(stdout: str) -> dict[str, Any] | None:
    lines = [ln for ln in stdout.splitlines() if ln.startswith("VARE_PROBE ")]
    if len(lines) != 1:
        return None
    try:
        obs = json.loads(lines[0][len("VARE_PROBE "):])
    except json.JSONDecodeError:
        return None
    return obs if isinstance(obs, dict) else None


def attest(backend: SandboxBackend, *, workspace_root: Path, evidence_dir: Path,
           timeout: float = 60) -> Attestation:
    """Run the negative probes in the sandbox configuration used for grading."""
    digest = backend.config_digest()

    def failed(reason: str, obs: Mapping[str, Any] | None = None) -> Attestation:
        return Attestation(backend.name, digest, False, (reason,), dict(obs or {}), time.time())

    if not backend.available():
        return failed("backend_unavailable")
    env_name = "VARE_HOST_ENV_CANARY_" + secrets.token_hex(4).upper()
    os.environ[env_name] = secrets.token_hex(8)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.2)
    workspace = Path(tempfile.mkdtemp(prefix="probe-", dir=workspace_root))
    canary_dir = Path(tempfile.mkdtemp(prefix="host-canary-"))
    try:
        canary = canary_dir / "canary.txt"
        canary.write_text(secrets.token_hex(16))
        spec = {"env_canary_name": env_name, "env_allowlist": sorted(ENV_ALLOWLIST),
                "host_canary_path": str(canary), "evidence_dir": str(evidence_dir),
                "interpreter_dirs": backend.interpreter_dirs(),
                "host_loopback_port": listener.getsockname()[1],
                "host_loopback_token": secrets.token_hex(8)}
        try:
            r = backend.run(workspace, [backend.python(), "-c", PROBE_SCRIPT, json.dumps(spec)],
                            timeout=timeout)
        except (IsolationUnavailable, subprocess.TimeoutExpired, OSError) as exc:
            return failed(f"probe_not_run:{type(exc).__name__}")
        obs = _parse_probe(r.stdout)
        if r.returncode != 0 or obs is None:
            return failed("probe_output_missing", {"returncode": r.returncode,
                                                   "stderr_tail": r.stderr[-300:]})
        try:  # Independent host-side check: did anything reach the listener?
            conn, _ = listener.accept()
            conn.close()
            obs["host_loopback_connect"] = True
        except OSError:
            pass
        ok, failures = evaluate_probe(obs)
        return Attestation(backend.name, digest, ok, failures, obs, time.time())
    finally:
        os.environ.pop(env_name, None)
        listener.close()
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(canary_dir, ignore_errors=True)


def static_flags(buggy_text: str, candidate_text: str) -> list[str]:
    """Blocking pattern names found in lines the candidate added."""
    added = [ln[1:] for ln in difflib.unified_diff(buggy_text.splitlines(), candidate_text.splitlines(),
                                                    lineterm="", n=0)
             if ln.startswith("+") and not ln.startswith("+++")]
    text = "\n".join(added)
    return [name for name, pat in BLOCKING_PATTERNS if re.search(pat, text)]


def _canary_source(ok_name: str, fail_name: str) -> str:
    return (f"def {ok_name}():\n    assert 1 + 1 == 2\n\n\n"
            f"def {fail_name}():\n    assert 1 + 1 == 3\n")


def stage_tree(source_tree: Path, dest: Path) -> None:
    """Copy a trusted, already-prepared buggy tree without VCS metadata."""
    shutil.copytree(source_tree, dest, symlinks=True, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))


def judge(status: Mapping[str, str], *, returncode: int, fail_to_pass: Sequence[str],
          pass_to_pass: Sequence[str], canary_ok: str, canary_fail: str) -> tuple[bool, str]:
    """Verdict from parsed pytest statuses; any inconsistency is a failure."""
    if status.get(canary_ok) != "PASSED" or status.get(canary_fail) != "FAILED":
        return False, "verdict_integrity_canary"
    if returncode != 1:  # exactly one canary must fail, so pytest exit code is 1
        return False, "verdict_integrity_exit_code"
    ids = list(dict.fromkeys([*fail_to_pass, *pass_to_pass]))
    if not ids:
        return False, "no_tests"
    if all(status.get(t) == "PASSED" for t in ids):
        return True, "pass"
    return False, "tests_failed"


def grade_isolated(
    backend: SandboxBackend,
    attestation: Attestation | None,
    *,
    workspace_root: Path,
    source_tree: Path,
    target_file: str,
    buggy_text: str,
    candidate_text: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    timeout: float = 180,
    static_scan: bool = True,
    on_workspace: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    """Grade one candidate; ``passed`` is True only when every check holds.

    ``static_scan=False`` exists only for containment audits that must execute
    hostile probes; production grading keeps the default.
    """
    t0 = time.time()
    rec: dict[str, Any] = {
        "schema": SCHEMA, "backend": backend.name, "config_digest": backend.config_digest(),
        "candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(), "passed": False,
    }

    def done(reason: str, **extra: Any) -> dict[str, Any]:
        rec.update(reason=reason, seconds=round(time.time() - t0, 3), **extra)
        return rec

    if attestation is None or not attestation.ok:
        return done("isolation_unverified")
    if attestation.config_digest != rec["config_digest"] or attestation.backend != backend.name:
        return done("isolation_unverified_config_changed")
    flags = static_flags(buggy_text, candidate_text)
    rec["static_flags"] = flags
    if flags and static_scan:
        return done("quarantined_static_flags", static_flags=flags)
    rel = Path(target_file)
    if rel.is_absolute() or ".." in rel.parts:
        return done("unsafe_target_path")
    workspace = Path(tempfile.mkdtemp(prefix="grade-", dir=workspace_root))
    try:
        stage_tree(source_tree, workspace)
        (workspace / rel).write_text(candidate_text, encoding="utf-8")
        nonce = secrets.token_hex(6)
        canary_file = f"test_zz_{nonce}.py"
        canary_ok, canary_fail = f"test_a_{nonce}", f"test_b_{nonce}"
        (workspace / canary_file).write_text(_canary_source(canary_ok, canary_fail))
        if on_workspace is not None:
            on_workspace(workspace)
        ids = list(dict.fromkeys([*fail_to_pass, *pass_to_pass]))
        ok_id, fail_id = f"{canary_file}::{canary_ok}", f"{canary_file}::{canary_fail}"
        cmd = [backend.python(), "-m", "pytest", "-rA", "-q", "-p", "no:cacheprovider",
               "--no-header", *ids, ok_id, fail_id]
        try:
            r = backend.run(workspace, cmd, timeout=timeout)
        except subprocess.TimeoutExpired:
            return done("test_timeout")
        except IsolationUnavailable as exc:
            return done("isolation_unverified", detail=str(exc)[:300])
        status = parse_pytest_results(r.stdout)
        passed, reason = judge(status, returncode=r.returncode, fail_to_pass=fail_to_pass,
                               pass_to_pass=pass_to_pass, canary_ok=ok_id, canary_fail=fail_id)
        rec["passed"] = passed
        return done(reason, pytest_exit=r.returncode,
                    tests={t: status.get(t, "MISSING") for t in ids},
                    stdout_sha256=hashlib.sha256(r.stdout.encode()).hexdigest())
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def sandbox_calibration(
    backend: SandboxBackend,
    attestation: Attestation | None,
    *,
    workspace_root: Path,
    source_tree: Path,
    target_file: str,
    buggy_text: str,
    reference_text: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    timeout: float = 180,
) -> dict[str, Any]:
    """Re-run v1 calibration inside the sandbox before admitting a task.

    The reference fix must pass and the unrepaired bug must fail. This is the
    check that the sandbox imports the workspace copy (and not some other
    installed copy of the package); a task that does not calibrate is not
    graded in the sandbox.
    """
    common: dict[str, Any] = dict(workspace_root=workspace_root, source_tree=source_tree,
                                  target_file=target_file, buggy_text=buggy_text,
                                  fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass,
                                  timeout=timeout)
    ref = grade_isolated(backend, attestation, candidate_text=reference_text, **common)
    # The unrepaired file is graded as an (unchanged) candidate.
    base = grade_isolated(backend, attestation, candidate_text=buggy_text, **common)
    base_tests = base.get("tests") or {}
    bug_fails = base.get("reason") == "tests_failed" and any(
        base_tests.get(t) != "PASSED" for t in fail_to_pass)
    p2p_ok = all(base_tests.get(t) == "PASSED" for t in pass_to_pass)
    return {"qualified": bool(ref["passed"] and bug_fails and p2p_ok),
            "reference": ref, "baseline": base}
