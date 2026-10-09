#!/usr/bin/env python3
"""Reproduce the frozen TRL #7249 DataLoaderDispatcher integration check."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/trl_pr7249_dispatcher_v3.lock.json"
RUNNER = ROOT / "scripts/replay_trl_pr7249_dispatcher_v3.py"
UPSTREAM = "https://github.com/huggingface/trl.git"
SOURCE_PATH = "trl/experimental/async_grpo/async_grpo_trainer.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: dict) -> str:
    clean = {k: v for k, v in value.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def call(command: list[str], *, cwd: Path | None = None, env=None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if canonical_sha(protocol) != protocol["sha256"]:
        raise RuntimeError("protocol hash mismatch")
    if sha256(RUNNER) != protocol["runner_sha256"] or sha256(Path(__file__)) != protocol["wrapper_sha256"]:
        raise RuntimeError("runner or wrapper hash mismatch")
    import accelerate, torch, transformers
    runtime = {"python": sys.version.split()[0], "torch": torch.__version__,
               "transformers": transformers.__version__, "accelerate": accelerate.__version__}
    expected_runtime = {key: protocol["runtime"][key] for key in runtime}
    if runtime != expected_runtime:
        raise RuntimeError(f"runtime mismatch: actual={runtime}, expected={expected_runtime}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output must be new: {output}")
    output.mkdir(parents=True)
    source_info = {}
    checkouts = output / "source-checkouts"
    checkouts.mkdir()
    for arm in ("base", "candidate"):
        ref = protocol["sources"][arm]["ref"]
        repo = checkouts / arm
        repo.mkdir()
        call(["git", "init", "--quiet", str(repo)])
        call(["git", "-C", str(repo), "remote", "add", "origin", UPSTREAM])
        call(["git", "-C", str(repo), "fetch", "--quiet", "--depth=1", "origin", ref])
        call(["git", "-C", str(repo), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
        commit = call(["git", "-C", str(repo), "rev-parse", "HEAD"])
        dirty = call(["git", "-C", str(repo), "status", "--porcelain"])
        source_hash = sha256(repo / SOURCE_PATH)
        expected = protocol["sources"][arm]
        if commit != expected["commit"] or dirty or source_hash != expected["source_sha256"]:
            raise RuntimeError(f"{arm} checkout mismatch commit={commit}, dirty={bool(dirty)}, hash={source_hash}")
        source_info[arm] = {"commit": commit, "source_sha256": source_hash, "clean": True}

    started = time.monotonic()
    runs = {}
    for arm in ("base", "candidate"):
        run_dir = output / arm
        port_sock = socket.socket()
        port_sock.bind(("127.0.0.1", 0))
        port = port_sock.getsockname()[1]
        port_sock.close()
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
               "--rdzv_backend=static", "--master_addr=127.0.0.1", f"--master_port={port}",
               "--max_restarts=0", str(RUNNER), "--repo", str(checkouts / arm), "--arm", arm,
               "--output", str(run_dir)]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        t0 = time.monotonic()
        result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=protocol["resources"]["max_wall_seconds_per_arm"], check=False)
        log = output / f"{arm}-launcher.log"
        log.write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(f"{arm} runner failed ({result.returncode}); see {log}")
        summary = json.loads((run_dir / "summary.json").read_text())
        if summary["source_commit"] != protocol["sources"][arm]["commit"]:
            raise RuntimeError(f"{arm} result source mismatch")
        runs[arm] = {"returncode": result.returncode, "seconds": time.monotonic() - t0,
                     "log_sha256": sha256(log), "summary_sha256": sha256(run_dir / "summary.json"),
                     "summary": summary}

    threshold = protocol["acceptance"]["candidate_max_error_lte"]
    candidate = runs["candidate"]["summary"]
    base = runs["base"]["summary"]
    candidate_errors = [e for r in candidate["rank_results"] for key in ("loss_abs_errors", "gradient_abs_errors") for e in r[key]]
    candidate_errors += [r["final_parameter_abs_error"] for r in candidate["rank_results"]]
    if max(candidate_errors) > threshold or not candidate["rank_state_equal"]:
        raise RuntimeError(f"candidate did not match pooled oracle: {max(candidate_errors)}")
    if candidate["rank_results"][0]["global_step"] != 2:
        raise RuntimeError("candidate did not finish both optimizer updates")
    base_primary = max(e for r in base["rank_results"] for e in r["gradient_abs_errors"][:1])
    if base_primary < protocol["acceptance"]["base_first_window_gradient_error_gte"]:
        raise RuntimeError(f"base primary defect not reproduced: {base_primary}")
    manifest = {
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol["sha256"],
        "runner_sha256": sha256(RUNNER), "wrapper_sha256": sha256(Path(__file__)),
        "runtime": runtime, "source_checkouts": source_info,
        "runs": {arm: {k: v for k, v in item.items() if k != "summary"} for arm, item in runs.items()},
        "acceptance": {"candidate_max_error": max(candidate_errors), "base_primary_gradient_error": base_primary,
                       "candidate": "pass", "base_reproduction": "pass"},
        "total_seconds": time.monotonic() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
