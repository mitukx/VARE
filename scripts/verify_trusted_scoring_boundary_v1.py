#!/usr/bin/env python3
"""Run a synthetic real-Docker test of scorer/candidate filesystem separation."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOL = ROOT / "protocols/trusted_scoring_boundary_v1.lock.json"
CONTEXT = "colima"
IMAGE = "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _run(argv: list[str], *, timeout: float = 15.0) -> str:
    proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout, check=False)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {argv[0]} {proc.stderr[-500:]}")
    return proc.stdout.strip()


def _docker(*argv: str) -> str:
    return _run(["docker", "--context", CONTEXT, *argv])


def _check_lock() -> dict[str, Any]:
    raw = PROTOCOL.read_bytes()
    lock = json.loads(raw)
    runtime = lock.get("runtime", {})
    limits = lock.get("limits", {})
    acceptance = lock.get("acceptance", {})
    if lock.get("protocol_id") != "trusted-scoring-boundary-v1":
        raise RuntimeError("unexpected protocol identity")
    if runtime != {
        "python": ".".join(map(str, sys.version_info[:3])),
        "docker_context": CONTEXT,
        "colima_driver": "macOS Virtualization.Framework",
        "container_runtime": "docker",
        "seccomp_required": True,
        "network": "none",
        "image": IMAGE,
    }:
        raise RuntimeError("runtime pins do not match this runner/environment")
    if limits != {
        "memory_mb": 256, "cpus": 0.5, "pids_limit": 12,
        "timeout_s": 15, "max_output_bytes": 4096,
        "max_input_bytes": 64000, "cases_positive": 2, "cases_negative": 1,
    }:
        raise RuntimeError("resource limits do not match this runner")
    if acceptance != {
        "candidate_mounts_exactly_readonly_workspace": True,
        "synthetic_canary_readable": False,
        "positive_control_passes_all_cases": True,
        "candidate_forged_score_rejected": True,
        "all_managed_containers_cleaned": True,
        "live_untrusted_repository_tests_eligible": False,
    }:
        raise RuntimeError("acceptance criteria do not match this runner")
    for relative, expected in lock["implementation_sha256"].items():
        if _sha((ROOT / relative).read_bytes()) != expected:
            raise RuntimeError(f"source hash mismatch: {relative}")
    if _sha(Path(__file__).read_bytes()) != lock["runner_script_sha256"]:
        raise RuntimeError("runner script hash mismatch")
    return {"lock": lock, "raw": raw}


def _environment() -> dict[str, Any]:
    os.environ["DOCKER_CONTEXT"] = CONTEXT
    context = _run(["docker", "context", "show"])
    if context != CONTEXT:
        raise RuntimeError(f"unexpected Docker context: {context}")
    colima = json.loads(_run(["colima", "status", "--json"]))
    info = json.loads(_docker("info", "--format", "{{json .SecurityOptions}}"))
    if (colima.get("driver") != "macOS Virtualization.Framework"
            or colima.get("runtime") != "docker" or colima.get("kubernetes")
            or not any("seccomp" in item for item in info)):
        raise RuntimeError("Colima or seccomp preconditions failed")
    image = json.loads(_docker("image", "inspect", IMAGE))[0]
    if image.get("Id") != IMAGE.split("@", 1)[1]:
        raise RuntimeError("digest-pinned Python image is not present locally")
    return {
        "docker_context": context,
        "docker_client_server": _docker("version", "--format", "{{.Client.Version}}/{{.Server.Version}}"),
        "docker_security_options": info,
        "colima": colima,
        "image_reference": IMAGE,
        "image_id": image["Id"],
        "python": sys.version,
        "platform": platform.platform(),
    }


async def _run_probe(out: Path) -> dict[str, Any]:
    # The synthetic canary stays on the operator side and outside the one mount.
    candidate_root = out / "candidate-root"
    workspace = candidate_root / "workspace"
    grader_root = out / "operator-private-grader"
    workspace.mkdir(parents=True)
    grader_root.mkdir(parents=True)
    canary = grader_root / "synthetic-hidden-canary.txt"
    canary.write_text(secrets.token_hex(32), encoding="utf-8")
    solver = workspace / "solve.py"
    solver.write_text(
        "import hashlib, json, pathlib, sys\n"
        "request = json.loads(sys.stdin.readline())\n"
        "try: canary = hashlib.sha256(pathlib.Path('/grader/synthetic-hidden-canary.txt').read_bytes()).hexdigest()\n"
        "except OSError: canary = None\n"
        "print(json.dumps({'value': request['n'] ** 2, 'grader_canary_sha256': canary}, sort_keys=True))\n",
        encoding="utf-8",
    )
    wrong = workspace / "wrong.py"
    wrong.write_text(
        "import json\n"
        "print(json.dumps({'passed': True, 'score': 1.0}, sort_keys=True))\n",
        encoding="utf-8",
    )

    from vare.container_isolation import DockerIsolationRunner, IsolationLimits
    from vare.trusted_scoring import JSONIOCase, TrustedJSONScorer

    runner = DockerIsolationRunner()
    limits = IsolationLimits(IMAGE, timeout_s=15, memory_mb=256, cpus=0.5,
                             pids_limit=12, max_output_bytes=4096)
    layout = runner.candidate_argv(
        limits, name="vare-agent-" + "a" * 24,
        allowed_workspace_root=candidate_root, workspace=workspace,
        command=("python", "/workspace/solve.py"),
        readonly_workspace=True, interactive=True,
    )
    mounts = [layout[i + 1] for i, value in enumerate(layout[:-1]) if value == "--mount"]
    if mounts != [f"type=bind,source={workspace},target=/workspace,readonly"]:
        raise RuntimeError("candidate mount set is not exactly the readonly workspace")
    if "/grader" in " ".join(layout) or str(grader_root) in " ".join(layout):
        raise RuntimeError("grader path appears in candidate launch configuration")

    scorer = TrustedJSONScorer(
        runner=runner, limits=limits, allowed_workspace_root=candidate_root,
    )
    cases = (
        JSONIOCase("square-2", {"n": 2}, {"value": 4, "grader_canary_sha256": None}),
        JSONIOCase("square-7", {"n": 7}, {"value": 49, "grader_canary_sha256": None}),
    )
    leak_probe = await runner.run_candidate_io(
        limits, allowed_workspace_root=candidate_root, workspace=workspace,
        command=("python", "/workspace/solve.py"),
        stdin_data=b'{"n":2}\n',
    )
    try:
        leak_probe_response = json.loads(leak_probe.stdout)
        candidate_canary_sha256 = leak_probe_response.get("grader_canary_sha256")
        candidate_can_read_canary = candidate_canary_sha256 is not None
    except (json.JSONDecodeError, AttributeError):
        leak_probe_response = None
        candidate_canary_sha256 = None
        candidate_can_read_canary = None
    baseline = set(_docker("ps", "-a", "--filter", "name=vare-agent-", "--format", "{{.Names}}").splitlines())
    positive = await scorer.evaluate(
        task_id="synthetic-json-square-v1", policy_id="candidate-positive",
        policy_version=0, workspace=workspace,
        command=("python", "/workspace/solve.py"), cases=cases,
    )
    # This contestant claims success but returns no task answer; the trusted
    # scorer must compare its output to the operator-held expected JSON.
    negative = await scorer.evaluate(
        task_id="synthetic-json-square-v1", policy_id="candidate-forged-score",
        policy_version=0, workspace=workspace,
        command=("python", "/workspace/wrong.py"),
        cases=(JSONIOCase("square-2", {"n": 2}, cases[0].expected),),
    )
    after = set(_docker("ps", "-a", "--filter", "name=vare-agent-", "--format", "{{.Names}}").splitlines())
    operator_canary_sha256 = _sha(canary.read_bytes())
    canary.unlink()
    grader_root.rmdir()
    return {
        "schema_version": 1,
        "positive_control": {
            "passed": positive.passed,
            "score": positive.score,
            "case_results": [{
                "case_id": case.case_id, "passed": case.passed,
                "returncode": case.returncode, "wall_s": case.wall_s,
                "response_sha256": case.response_sha256, "failure": case.failure,
            } for case in positive.cases],
            "case_set_sha256": positive.case_set_sha256,
        },
        "forged_score_control": {
            "passed": negative.passed,
            "score": negative.score,
            "case_results": [{
                "case_id": case.case_id, "passed": case.passed,
                "returncode": case.returncode, "wall_s": case.wall_s,
                "response_sha256": case.response_sha256, "failure": case.failure,
            } for case in negative.cases],
        },
        "candidate_launch_mounts": mounts,
        "operator_canary_sha256": operator_canary_sha256,
        "canary_probe": {
            "returncode": leak_probe.returncode,
            "response": leak_probe_response,
            "candidate_canary_sha256": candidate_canary_sha256,
            "candidate_can_read_canary": candidate_can_read_canary,
        },
        "managed_container_names_before": sorted(baseline),
        "managed_container_names_after": sorted(after),
        "no_new_candidate_containers": after == baseline,
    }


async def _main(args: argparse.Namespace) -> int:
    out = args.output.expanduser().resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite evidence directory: {out}")
    locked = _check_lock()
    os.environ["DOCKER_CONTEXT"] = CONTEXT
    out.mkdir(parents=True)
    (out / "protocol.lock.json").write_bytes(locked["raw"])
    try:
        env = _environment()
        result = await _run_probe(out)
        evidence = {
            "protocol_id": locked["lock"]["protocol_id"],
            "environment": env,
            "limits": locked["lock"]["limits"],
            "probe": result,
            "claim_limit": "Synthetic JSON-I/O scorer boundary only; no real repository task, confidential held-out data, model capability, or general sandbox certification.",
        }
        raw = json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        (out / "observations.json").write_bytes(raw)
        manifest = {
            "observations_sha256": _sha(raw),
            "protocol_sha256": _sha(locked["raw"]),
            "runner_script_sha256": _sha(Path(__file__).read_bytes()),
            "checkout_revision": _run(["git", "rev-parse", "HEAD"]),
            "working_tree_dirty": bool(_run(["git", "status", "--porcelain"])),
            "implementation_sha256": locked["lock"]["implementation_sha256"],
        }
        (out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        print(f"evidence written to {out}")
        return 0
    except Exception as exc:
        private_canary = out / "operator-private-grader/synthetic-hidden-canary.txt"
        private_canary.unlink(missing_ok=True)
        try:
            private_canary.parent.rmdir()
        except OSError:
            pass
        (out / "failure.json").write_text(json.dumps({
            "exception": type(exc).__name__, "message": str(exc),
        }, sort_keys=True, indent=2) + "\n")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        return asyncio.run(_main(args))
    except Exception as exc:
        print(f"trusted scoring verification failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
