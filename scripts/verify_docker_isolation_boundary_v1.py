#!/usr/bin/env python3
"""Exercise VARE's Docker isolation boundary with disposable synthetic probes.

This is not a security certification and must never be pointed at real secrets.
The only disclosed canary is generated inside the new output directory.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any

from vare.container_isolation import (
    DockerIsolationRunner,
    IsolationError,
    IsolationLimits,
)
from vare.separated_grading import CANARY_PROBE_SOURCE, build_confidentiality_probe_argv


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/isolation_boundary_verification_v1.lock.json"
IMAGE = "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258"
CONTEXT = "colima"
REPLICATES = 3
MEMORY_MB = 256
CPUS = 0.5
PIDS_LIMIT = 12
UID = GID = 65534


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _command(*args: str, context: str = CONTEXT) -> str:
    result = subprocess.run(
        ["docker", "--context", context, *args],
        check=True,
        text=True,
        capture_output=True,
        timeout=20,
    )
    return result.stdout.strip()


def _container_names(context: str) -> set[str]:
    raw = _command("ps", "-a", "--format", "{{.Names}}", context=context)
    return {line for line in raw.splitlines() if line}


def _limits(image: str = IMAGE, *, timeout: float = 15.0,
            output: int = 16_000) -> IsolationLimits:
    return IsolationLimits(
        image=image,
        timeout_s=timeout,
        memory_mb=MEMORY_MB,
        cpus=CPUS,
        pids_limit=PIDS_LIMIT,
        uid=UID,
        gid=GID,
        max_output_bytes=output,
    )


def _probe_code(host_canary: Path) -> str:
    payload = json.dumps(str(host_canary))
    return f'''import errno,json,os,socket,subprocess,sys
from pathlib import Path
p=Path({payload})
readable=False
try: readable=p.read_text()=="vare-synthetic-host-canary-v1"
except OSError: pass
writeable=False
try: p.write_text("modified"); writeable=True
except OSError: pass
network=False
try:
 s=socket.create_connection(("203.0.113.1",65000),timeout=0.5); s.close(); network=True
except OSError: pass
children=[]; eagain=False
try:
 for _ in range(32):
  try: children.append(subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
  except OSError as exc:
   eagain=exc.errno==errno.EAGAIN; break
finally:
 for child in children:
  try: child.kill()
  except OSError: pass
 for child in children:
  try: child.wait(timeout=1)
  except Exception: pass
def read(path):
 try: return Path(path).read_text().strip()
 except OSError: return None
status=Path("/proc/self/status").read_text()
fields={{line.split(":",1)[0]:line.split(":",1)[1].strip() for line in status.splitlines() if ":" in line}}
try: Path("/workspace/probe-write.txt").write_text("workspace writable") ; workspace_write=True
except OSError: workspace_write=False
try: Path("/probe-rootfs-write.txt").write_text("must fail"); rootfs_write=True
except OSError: rootfs_write=False
print(json.dumps({{"host_canary_readable":readable,"host_canary_writable":writeable,"network_reachable":network,"uid":os.getuid(),"docker_socket_present":Path("/var/run/docker.sock").exists(),"cap_eff":fields.get("CapEff"),"no_new_privs":fields.get("NoNewPrivs"),"rootfs_writable":rootfs_write,"workspace_writable":workspace_write,"pids_max":read("/sys/fs/cgroup/pids.max"),"memory_max":read("/sys/fs/cgroup/memory.max"),"cpu_max":read("/sys/fs/cgroup/cpu.max"),"pids_limit_enforced":eagain,"spawned_children":len(children),"default_route_present":any(line.split()[1]=="00000000" for line in Path("/proc/net/route").read_text().splitlines()[1:])}},sort_keys=True))'''


async def _candidate_probe(
    runner: DockerIsolationRunner,
    root: Path,
    workspace: Path,
    host_canary: Path,
    *,
    repetitions: int,
) -> list[dict[str, Any]]:
    observations = []
    code = _probe_code(host_canary)
    for index in range(repetitions):
        before = _container_names(CONTEXT)
        result = await runner.run_candidate(
            _limits(),
            allowed_workspace_root=root,
            workspace=workspace,
            command=["python", "-c", code],
        )
        after = _container_names(CONTEXT)
        parsed = json.loads(result.stdout)
        observations.append({
            "replicate": index + 1,
            "returncode": result.returncode,
            "observation": parsed,
            "new_managed_containers_after": sorted(
                name for name in after - before if name.startswith("vare-agent-")
            ),
            "stdout_sha256": _sha256(result.stdout.encode()),
            "stderr_sha256": _sha256(result.stderr.encode()),
        })
    return observations


async def _observe_active_container(
    runner: DockerIsolationRunner,
    root: Path,
    workspace: Path,
) -> dict[str, Any]:
    before = _container_names(CONTEXT)
    run = asyncio.create_task(runner.run_candidate(
        _limits(timeout=10), allowed_workspace_root=root, workspace=workspace,
        command=["python", "-c", "import time; time.sleep(3)"]
    ))
    name = None
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not run.done():
        current = _container_names(CONTEXT)
        new = sorted(n for n in current - before if n.startswith("vare-agent-"))
        if new:
            name = new[0]
            break
        await asyncio.sleep(0.05)
    if name is None:
        await run
        raise RuntimeError("could not observe live managed container")
    inspected = json.loads(_command("inspect", name))
    code = inspected[0]
    host = code["HostConfig"]
    mounts = [
        {"destination": item["Destination"], "rw": item["RW"], "type": item["Type"]}
        for item in code.get("Mounts", [])
    ]
    result = {
        "name": name,
        "running_when_inspected": code["State"]["Running"],
        "image_id": code["Image"],
        "configured_image": code["Config"]["Image"],
        "user": code["Config"].get("User"),
        "network_mode": host.get("NetworkMode"),
        "readonly_rootfs": host.get("ReadonlyRootfs"),
        "cap_drop": host.get("CapDrop"),
        "security_opt": host.get("SecurityOpt"),
        "memory_bytes": host.get("Memory"),
        "nano_cpus": host.get("NanoCpus"),
        "pids_limit": host.get("PidsLimit"),
        "mounts": mounts,
    }
    completed = await run
    after = _container_names(CONTEXT)
    result["returncode"] = completed.returncode
    result["left_after_normal_exit"] = sorted(after - before)
    return result


async def _lifecycle_probe(
    runner: DockerIsolationRunner,
    root: Path,
    workspace: Path,
    *,
    repetitions: int,
) -> dict[str, list[dict[str, Any]]]:
    cases: dict[str, list[dict[str, Any]]] = {"timeout": [], "cancellation": [], "output_overflow": []}
    for index in range(repetitions):
        before = _container_names(CONTEXT)
        timed_out = False
        try:
            await runner.run_candidate(
                _limits(timeout=1.0), allowed_workspace_root=root, workspace=workspace,
                command=["python", "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); time.sleep(30)"],
            )
        except IsolationError as exc:
            timed_out = "wall-time" in str(exc)
            timeout_error = str(exc)
        after = _container_names(CONTEXT)
        cases["timeout"].append({
            "replicate": index + 1,
            "expected_timeout": timed_out,
            "error": timeout_error if timed_out else "unexpected successful completion",
            "new_containers_after": sorted(after - before),
        })

        before = _container_names(CONTEXT)
        task = asyncio.create_task(runner.run_candidate(
            _limits(timeout=10), allowed_workspace_root=root, workspace=workspace,
            command=["python", "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); time.sleep(30)"],
        ))
        await asyncio.sleep(0.5)
        task.cancel()
        cancelled = False
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True
        after = _container_names(CONTEXT)
        cases["cancellation"].append({
            "replicate": index + 1,
            "cancelled": cancelled,
            "new_containers_after": sorted(after - before),
        })

        before = _container_names(CONTEXT)
        overflowed = False
        try:
            await runner.run_candidate(
                _limits(timeout=10, output=1024),
                allowed_workspace_root=root,
                workspace=workspace,
                command=["python", "-c", "print('X'*1048576)"],
            )
        except IsolationError as exc:
            overflowed = "output byte limit" in str(exc)
            overflow_error = str(exc)
        after = _container_names(CONTEXT)
        cases["output_overflow"].append({
            "replicate": index + 1,
            "overflowed": overflowed,
            "error": overflow_error if overflowed else "unexpected successful completion",
            "new_containers_after": sorted(after - before),
        })
    return cases


async def _grader_probe(
    runner: DockerIsolationRunner,
    workspace_root: Path,
    workspace: Path,
    grader_root: Path,
    canary: str,
    *,
    image: str,
) -> dict[str, Any]:
    """Read the historical canary path from the candidate sandbox only.

    The canary file stays on the operator side. The candidate argv is the
    separated workspace mount, so ``/grader/synthetic-hidden-canary.txt`` is
    not present in that container. The canary value is not passed in argv.
    """
    if canary and canary in CANARY_PROBE_SOURCE:
        raise RuntimeError("probe source must not embed the canary value")
    grader_resolved = grader_root.resolve()
    workspace_resolved = workspace.resolve()
    if (grader_resolved == workspace_resolved or grader_resolved in workspace_resolved.parents
            or workspace_resolved in grader_resolved.parents):
        raise RuntimeError("canary directory overlaps the candidate workspace")
    limits = _limits(image=image)
    name = "vare-agent-" + secrets.token_hex(12)
    argv = build_confidentiality_probe_argv(
        runner, limits, name=name, allowed_workspace_root=workspace_root, workspace=workspace,
    )
    if canary and canary in "\n".join(argv):
        raise RuntimeError("canary value leaked into the candidate argv")
    result = await runner.execute(argv, limits)
    mounts = [argv[index + 1] for index, arg in enumerate(argv[:-1]) if arg == "--mount"]
    return {
        "returncode": result.returncode,
        "canary_visible": result.stdout.strip() == canary,
        "stdout": result.stdout,
        "stdout_sha256": _sha256(result.stdout.encode()),
        "mount_flags": mounts,
        "network_mode_none": "--network=none" in argv,
        "readonly_rootfs": "--read-only" in argv,
        "readonly_mounts": all(item.endswith(",readonly") for item in mounts),
    }


def _environment(context: str, image: str) -> dict[str, Any]:
    info = _command("info", "--format", "{{json .SecurityOptions}} {{.OSType}} {{.Architecture}} {{.MemTotal}}", context=context)
    inspect = json.loads(_command("image", "inspect", image, context=context))[0]
    colima = subprocess.run(
        ["colima", "status", "--json"], check=True, text=True,
        capture_output=True, timeout=20,
    )
    return {
        "docker_context": context,
        "docker_security_options": info.split("linux ")[0].strip(),
        "docker_info": info,
        "image_reference": image,
        "image_id": inspect["Id"],
        "image_repo_digests": inspect.get("RepoDigests", []),
        "colima": json.loads(colima.stdout),
        "python": sys.version,
        "platform": sys.platform,
    }


def _verify_protocol(protocol_path: Path) -> dict[str, Any]:
    raw = protocol_path.read_bytes()
    protocol = json.loads(raw)
    for relative, expected in protocol["implementation_files"].items():
        actual = _sha256((ROOT / relative).read_bytes())
        if actual != expected:
            raise RuntimeError(f"protocol source hash mismatch: {relative}")
    return {"protocol": protocol, "sha256": _sha256(raw), "raw": raw}


async def _run(args: argparse.Namespace) -> None:
    out = args.output.expanduser().resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing evidence directory: {out}")
    if args.context != CONTEXT:
        raise RuntimeError("protocol fixes the Colima VM context; use a new protocol for another daemon")
    os.environ["DOCKER_CONTEXT"] = args.context
    out.mkdir(parents=True)
    protocol_path = args.protocol.expanduser().resolve()
    locked = _verify_protocol(protocol_path)
    protocol_bytes = locked["raw"]
    (out / "protocol.lock.json").write_bytes(protocol_bytes)
    if args.image != locked["protocol"]["runtime"]["container_image"]:
        raise RuntimeError("image differs from frozen protocol")
    stage: dict[str, Any] = {"schema_version": 1, "status": "running"}

    def checkpoint(name: str, value: Any) -> None:
        stage[name] = value
        (out / "run-progress.json").write_text(
            json.dumps(stage, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )

    workspace_root = out / "candidate-root"
    workspace = workspace_root / "workspace"
    grader_root = out / "grader-root"
    workspace.mkdir(parents=True)
    grader_root.mkdir(parents=True)
    host_canary = out / "synthetic-host-canary.txt"
    host_canary.write_text("vare-synthetic-host-canary-v1", encoding="utf-8")

    environment = _environment(args.context, args.image)
    checkpoint("environment", environment)
    if args.image != IMAGE:
        raise RuntimeError("protocol pins the official Python image digest; use a new protocol to change it")
    security = environment["docker_security_options"]
    colima = environment["colima"]
    if ("seccomp" not in security or colima.get("driver") != "macOS Virtualization.Framework"
            or colima.get("runtime") != "docker" or colima.get("kubernetes")):
        raise RuntimeError("Docker VM security preconditions did not pass")

    runner = DockerIsolationRunner()
    active = await _observe_active_container(runner, workspace_root, workspace)
    checkpoint("active_container", active)
    candidate = await _candidate_probe(
        runner, workspace_root, workspace, host_canary, repetitions=REPLICATES,
    )
    checkpoint("candidate_probe", candidate)
    lifecycle = await _lifecycle_probe(
        runner, workspace_root, workspace, repetitions=REPLICATES,
    )
    checkpoint("lifecycle", lifecycle)
    grader = []
    for replicate in range(REPLICATES):
        current_canary = "vare-synthetic-grader-canary-" + secrets.token_hex(16)
        (grader_root / "synthetic-hidden-canary.txt").write_text(current_canary, encoding="utf-8")
        grader.append(await _grader_probe(
            runner, workspace_root, workspace, grader_root, current_canary,
            image=args.image,
        ))
        grader[-1]["replicate"] = replicate + 1
        checkpoint("grader_probe", grader)
    evidence = {
        "schema_version": 1,
        "protocol": "repository-isolation-boundary-v1",
        "environment": environment,
        "limits": {
            "memory_mb": MEMORY_MB, "cpus": CPUS,
            "pids_limit": PIDS_LIMIT, "uid": UID, "gid": GID,
            "replicates": REPLICATES,
        },
        "host_canary": {
            "path": str(host_canary), "sha256": _sha256(host_canary.read_bytes()),
            "unchanged": host_canary.read_text() == "vare-synthetic-host-canary-v1",
        },
        "active_container": active,
        "candidate_probe": candidate,
        "lifecycle": lifecycle,
        "grader_probe": grader,
        "claim_limit": "E0 runtime configuration and synthetic boundary evidence only; not a hostile-code sandbox certification or model/coding capability result",
    }
    encoded = json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    (out / "observations.json").write_bytes(encoded)
    (out / "manifest.json").write_text(json.dumps({
        "observations_sha256": _sha256(encoded),
        "protocol_sha256": _sha256(protocol_bytes),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()),
        "implementation_sha256": locked["protocol"]["implementation_files"],
    }, sort_keys=True, indent=2) + "\n")
    checkpoint("status", "complete")
    print(f"evidence written to {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", default=CONTEXT)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except Exception as exc:
        out = args.output.expanduser().resolve()
        if out.is_dir():
            (out / "failure.json").write_text(json.dumps({
                "exception": type(exc).__name__,
                "message": str(exc),
            }, sort_keys=True, indent=2) + "\n")
        print(f"verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
