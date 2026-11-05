"""Fail-closed bubblewrap adapter for ResearchGym grading.

The argv, user id, and mount flags come from ``BwrapBackend`` in
``scripts/learning_pipeline_isolated_grading.py``. A grade runs only after
that backend's negative probe is denied for the same configuration digest.

This is not a trusted security sandbox. The probe checks the configured
namespace and mounts. It does not certify the kernel, the host runtime, or a
candidate that shares the worker interpreter. If bubblewrap is missing or the
probe is not fully denied, grading stops. There is no fallback to a same-user
process group.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from vare.environments.spec import ResourceLimits

from .execute import (
    CLASS_MALFORMED,
    CLASS_NONZERO,
    CLASS_OK,
    CLASS_OUTPUT_LIMIT,
    CLASS_TIMEOUT,
    Execution,
    parse_worker_stdout,
)
from .execute import _WORKER as _WORKER_PROGRAM

ISOLATION_UNVERIFIED = "isolation_unverified"
NOT_A_TRUSTED_SANDBOX = (
    "Bubblewrap namespace and mount flags reused from the learning-pipeline "
    "grader. This is not a trusted security sandbox: a passed probe does not "
    "certify the kernel, the host runtime, or a candidate sharing the worker "
    "interpreter. A missing backend or a probe that is not fully denied fails "
    "closed and does not fall back to a same-user process group."
)

_CACHE: dict[str, Any] = {}


def _module() -> Any:
    path = Path(__file__).resolve().parents[3] / "scripts" / "learning_pipeline_isolated_grading.py"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    cached = _CACHE.get("module")
    if cached is not None and _CACHE.get("module_path") == str(path):
        return cached
    spec = importlib.util.spec_from_file_location("vare_researchgym_bwrap_backend", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _CACHE["module"] = module
    _CACHE["module_path"] = str(path)
    return module


def _stdlib_venv() -> Path:
    root = Path(tempfile.gettempdir()) / "vare-researchgym-stdlib-venv"
    bindir = root / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    link = bindir / "python"
    target = Path(sys.executable).resolve()
    if link.is_symlink():
        try:
            current = link.resolve()
        except OSError:
            current = None
        if current != target:
            link.unlink()
            link.symlink_to(target)
    elif link.exists():
        link.unlink()
        link.symlink_to(target)
    else:
        link.symlink_to(target)
    return root


def attested_backend() -> tuple[Any, Any]:
    """Return ``(backend, attestation)``. The attestation may be unverified."""
    module = _module()
    backend = module.BwrapBackend(venv=_stdlib_venv())
    try:
        digest = str(backend.config_digest())
    except Exception:
        digest = ""
    if _CACHE.get("digest") == digest and _CACHE.get("attestation") is not None:
        return backend, _CACHE["attestation"]
    workspace = Path(_CACHE.get("probe_workspace") or tempfile.mkdtemp(prefix="vare-rg-probe-"))
    evidence = Path(_CACHE.get("probe_evidence") or tempfile.mkdtemp(prefix="vare-rg-evidence-"))
    _CACHE["probe_workspace"] = str(workspace)
    _CACHE["probe_evidence"] = str(evidence)
    try:
        attestation = module.attest(
            backend,
            workspace_root=workspace,
            evidence_dir=evidence,
            timeout=30,
        )
    except Exception as exc:
        attestation = module.Attestation(
            getattr(backend, "name", "bwrap"),
            digest,
            False,
            (f"probe_not_run:{type(exc).__name__}",),
            {},
            0.0,
        )
    _CACHE["digest"] = digest
    _CACHE["attestation"] = attestation
    return backend, attestation


def isolation_summary() -> dict[str, Any]:
    """Machine-readable description of the gate. Never claims a sandbox."""
    try:
        _backend, attestation = attested_backend()
        ok = bool(getattr(attestation, "ok", False))
        failures = tuple(getattr(attestation, "failures", ()) or ())
    except Exception as exc:
        ok = False
        failures = (type(exc).__name__,)
    return {
        "backend": "bwrap" if ok else "unverified",
        "attestation_ok": ok,
        "not_a_trusted_sandbox": True,
        "failures": list(failures),
        "note": NOT_A_TRUSTED_SANDBOX,
    }


def _unverified(reason: str) -> Execution:
    return Execution(
        classification=ISOLATION_UNVERIFIED,
        returncode=0,
        wall_s=0.0,
        stdout="",
        stderr=reason,
        timed_out=False,
        output_limited=False,
        peak_rss_kb=None,
        payload=None,
        memory_limit_applied=False,
    )


def _kill(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def launch_worker(
    subject: Path,
    calls: list[dict[str, Any]],
    limits: ResourceLimits,
) -> Execution:
    """Run the trusted worker program inside the attested bubblewrap argv.

    The subject is bind-mounted at ``/work``. The host path is not passed in,
    so the worker does not need the repository mount. On an unverified probe
    this returns without starting the candidate.
    """
    import json
    import time

    try:
        backend, attestation = attested_backend()
    except Exception as exc:
        return _unverified(type(exc).__name__)
    if not bool(getattr(attestation, "ok", False)):
        failures = getattr(attestation, "failures", ()) or ()
        detail = ",".join(str(item) for item in failures) or "isolation_unverified"
        return _unverified(detail)
    try:
        request = json.dumps({"subject": "/work", "calls": calls}, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        return _unverified("worker_request")
    builder = getattr(backend, "_argv", None)
    python = getattr(backend, "python", None)
    if not callable(builder) or not callable(python):
        return _unverified("backend_argv")
    try:
        argv = [str(part) for part in builder(subject, [str(python()), "-c", _WORKER_PROGRAM])]
    except Exception as exc:
        return _unverified(type(exc).__name__)
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        return _unverified(type(exc).__name__)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(request, timeout=limits.wall_time_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(proc)
        stdout, stderr = proc.communicate()
    wall = time.perf_counter() - started
    limited = len(stdout) > limits.max_output_bytes or len(stderr) > limits.max_output_bytes
    if limited:
        stdout = stdout[: limits.max_output_bytes]
        stderr = stderr[: limits.max_output_bytes]
    text_out = stdout.decode("utf-8", errors="replace")
    text_err = stderr.decode("utf-8", errors="replace")
    if timed_out:
        classification = CLASS_TIMEOUT
        payload = None
    elif limited:
        classification = CLASS_OUTPUT_LIMIT
        payload = None
    elif proc.returncode not in (0, None):
        classification = CLASS_NONZERO
        payload = None
    else:
        try:
            payload = parse_worker_stdout(text_out)
            classification = CLASS_OK
        except ValueError:
            payload = None
            classification = CLASS_MALFORMED
    return Execution(
        classification=classification,
        returncode=124 if timed_out else int(proc.returncode or 0),
        wall_s=wall,
        stdout=text_out,
        stderr=text_err,
        timed_out=timed_out,
        output_limited=limited,
        peak_rss_kb=None,
        payload=payload,
        memory_limit_applied=False,
    )
