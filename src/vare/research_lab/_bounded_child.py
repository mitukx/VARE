"""Trusted launcher placed between the process-group guard and a candidate.

The candidate command is executed with ``exec``. It is not imported. Limits
applied here are resource controls on the same user, not a sandbox.
"""
from __future__ import annotations

import json
import os
import signal
import sys


_NETWORK_TOKEN = "research_lab_network_isolation_unavailable"
_EXEC_TOKEN = "research_lab_exec_failed"


def _pdeathsig() -> None:
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            return
    except (OSError, AttributeError):
        return


def _deny_network() -> None:
    host_uid = os.getuid()
    host_gid = os.getgid()
    clone_newuser = getattr(os, "CLONE_NEWUSER", 0x10000000)
    clone_newnet = getattr(os, "CLONE_NEWNET", 0x40000000)
    try:
        os.unshare(clone_newuser | clone_newnet)
        with open("/proc/self/setgroups", "w", encoding="ascii") as handle:
            handle.write("deny")
        with open("/proc/self/uid_map", "w", encoding="ascii") as handle:
            handle.write(f"{host_uid} {host_uid} 1\n")
        with open("/proc/self/gid_map", "w", encoding="ascii") as handle:
            handle.write(f"{host_gid} {host_gid} 1\n")
    except OSError:
        print(_NETWORK_TOKEN, file=sys.stderr, flush=True)
        raise SystemExit(126)


def _user_process_count(uid: int) -> int:
    total = 0
    proc = "/proc"
    try:
        names = os.listdir(proc)
    except OSError:
        return 1
    for name in names:
        if not name.isdigit():
            continue
        try:
            if os.stat(os.path.join(proc, name)).st_uid == uid:
                total += 1
        except OSError:
            continue
    return max(1, total)


def _apply_limits(limits: dict) -> None:
    import resource

    cpu = limits.get("cpu_time_s")
    if isinstance(cpu, int) and cpu > 0:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    file_bytes = limits.get("max_file_bytes")
    if isinstance(file_bytes, int) and file_bytes > 0:
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    children = limits.get("max_child_processes")
    if children == 0:
        # RLIMIT_NPROC counts processes of this real uid. Pinning it to the
        # current count makes a further fork fail. It is not a sandbox.
        count = _user_process_count(os.getuid())
        resource.setrlimit(resource.RLIMIT_NPROC, (count, count))
    elif isinstance(children, int) and children > 0:
        count = _user_process_count(os.getuid()) + children
        resource.setrlimit(resource.RLIMIT_NPROC, (count, count))
    memory_mb = limits.get("address_space_mb")
    if isinstance(memory_mb, int) and memory_mb > 0:
        cap = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ValueError, OSError):
            pass


def main() -> None:
    if "--" not in sys.argv:
        print(_EXEC_TOKEN, file=sys.stderr, flush=True)
        raise SystemExit(127)
    separator = sys.argv.index("--")
    command = sys.argv[separator + 1 :]
    if not command:
        print(_EXEC_TOKEN, file=sys.stderr, flush=True)
        raise SystemExit(127)
    raw = os.environ.pop("VARE_RESEARCH_LAB_LIMITS", "")
    try:
        limits = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        print(_EXEC_TOKEN, file=sys.stderr, flush=True)
        raise SystemExit(127)
    if not isinstance(limits, dict):
        print(_EXEC_TOKEN, file=sys.stderr, flush=True)
        raise SystemExit(127)
    _pdeathsig()
    if limits.get("network") == "deny":
        _deny_network()
    try:
        _apply_limits(limits)
    except (ValueError, OSError):
        print(_EXEC_TOKEN, file=sys.stderr, flush=True)
        raise SystemExit(127)
    env = os.environ.copy()
    env.pop("VARE_RESEARCH_LAB_LIMITS", None)
    try:
        os.execvpe(command[0], command, env)
    except OSError:
        print(_EXEC_TOKEN, file=sys.stderr, flush=True)
        raise SystemExit(127)


if __name__ == "__main__":
    main()
