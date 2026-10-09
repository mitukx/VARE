#!/usr/bin/env python3
"""Reproduce the frozen TRL #7249 Trainer/DDP audit from clean upstream checkouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/trl_async_accumulation_normalization_pr7249_async_distillation_ddp_v4.lock.json"
UPSTREAM = "https://github.com/huggingface/trl.git"
RUNNER = ROOT / "scripts/replay_trl_pr7249_async_distillation_ddp_v4.py"
SOURCE_PATH = "trl/experimental/async_distillation/async_distillation_trainer.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    content = {key: val for key, val in value.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True, stderr=subprocess.STDOUT).strip()


def load_and_check_protocol() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if canonical_sha256(protocol) != protocol.get("sha256"):
        raise RuntimeError("protocol digest mismatch")
    if sha256(RUNNER) != protocol["source"]["runner_sha256"]:
        raise RuntimeError("frozen experiment runner digest mismatch")
    if sha256(Path(__file__).resolve()) != protocol["source"]["wrapper_sha256"]:
        raise RuntimeError("frozen clean-checkout wrapper digest mismatch")
    return protocol


def check_runtime(protocol: dict) -> dict:
    import accelerate
    import torch
    import transformers

    actual = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
    }
    expected = {name: protocol["runtime"][name] for name in actual}
    if actual != expected:
        raise RuntimeError(f"runtime mismatch: actual={actual}, expected={expected}")
    if not torch.distributed.is_gloo_available():
        raise RuntimeError("this PyTorch build has no Gloo backend")
    return actual


def checkout_source(path: Path, ref: str, expected_commit: str, expected_file_hash: str) -> dict:
    path.mkdir(parents=True)
    run(["git", "init", "--quiet", str(path)])
    run(["git", "-C", str(path), "remote", "add", "origin", UPSTREAM])
    run(["git", "-C", str(path), "fetch", "--quiet", "--depth=1", "origin", ref])
    run(["git", "-C", str(path), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
    commit = run(["git", "-C", str(path), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(path), "status", "--porcelain"])
    source_hash = sha256(path / SOURCE_PATH)
    if commit != expected_commit or dirty or source_hash != expected_file_hash:
        raise RuntimeError(
            f"checkout mismatch at {path}: commit={commit}, dirty={bool(dirty)}, source_hash={source_hash}"
        )
    return {"commit": commit, "source_file_sha256": source_hash, "clean": True}


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def run_arm(protocol: dict, arm: str, source: Path, output: Path, extra_pythonpath: list[str]) -> dict:
    source_record = protocol["source"]
    commit = source_record["base_revision"] if arm == "base" else source_record["candidate_head"]
    port = free_local_port()
    log_path = output.parent / f"{arm}-launcher.log"
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2",
        "--rdzv_backend=static", "--master_addr=127.0.0.1", f"--master_port={port}",
        "--max_restarts=0", str(RUNNER), "--repo", str(source), "--arm", arm, "--output", str(output),
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    pythonpath = [*extra_pythonpath]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    if pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=protocol["resources"]["wall_seconds_per_arm_cap"],
            check=False,
        )
        log_path.write_text(result.stdout, encoding="utf-8")
        print(result.stdout, end="")
        if result.returncode:
            raise RuntimeError(f"{arm} arm failed with exit code {result.returncode}; see {log_path}")
    except subprocess.TimeoutExpired as exc:
        output_text = exc.stdout or ""
        if isinstance(output_text, bytes):
            output_text = output_text.decode(errors="replace")
        log_path.write_text(output_text, encoding="utf-8")
        raise RuntimeError(f"{arm} arm exceeded its frozen wall-time cap; see {log_path}") from exc

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["arm"] != arm or summary["source_commit"] != commit:
        raise RuntimeError(f"{arm} arm summary provenance mismatch")
    if time.monotonic() - started > protocol["resources"]["wall_seconds_per_arm_cap"]:
        raise RuntimeError(f"{arm} arm exceeded its frozen wall-time cap")
    return {
        "arm": arm,
        "commit": commit,
        "source_file_sha256": source_record["base_source_sha256"] if arm == "base" else source_record["candidate_source_sha256"],
        "exit_code": result.returncode,
        "wall_seconds": time.monotonic() - started,
        "launcher_log": str(log_path.relative_to(output.parents[1])),
        "launcher_log_sha256": sha256(log_path),
        "summary": str(summary_path.relative_to(output.parents[1])),
        "summary_sha256": sha256(summary_path),
        "case_count": len(summary["cases"]),
    }


def validate_summaries(protocol: dict, output: Path) -> dict:
    summaries = {
        arm: json.loads((output / "runs" / arm / "summary.json").read_text(encoding="utf-8"))
        for arm in ("base", "candidate")
    }
    for arm, summary in summaries.items():
        if summary["protocol_id"] != protocol["protocol_id"]:
            raise RuntimeError(f"{arm} arm protocol ID mismatch")
        for case_name, case in summary["cases"].items():
            if not case["finite"] or not case["rank_state_equal"]:
                raise RuntimeError(f"{arm} {case_name} has non-finite values or rank disagreement")
            for row in case["rank_results"]:
                if row["global_step"] != 1:
                    raise RuntimeError(f"{arm} {case_name} did not complete exactly one optimizer step")
                expected = protocol["fixture_layout"][case_name]
                if row["local_tokens_by_rank_per_microbatch"] != expected["local_tokens_by_rank_per_microbatch"]:
                    raise RuntimeError(f"{arm} {case_name} local token layout differs from the lock")
                if row["global_tokens_per_microbatch"] != expected["global_tokens_per_microbatch"]:
                    raise RuntimeError(f"{arm} {case_name} global token layout differs from the lock")
    candidate_gate = protocol["acceptance"]["candidate_max_abs_pooled_loss_gradient_parameter_error_each_case_lte"]
    for case in summaries["candidate"]["cases"].values():
        errors = [case[key] for key in ("max_loss_abs_error", "max_gradient_abs_error", "max_sgd_parameter_abs_error")]
        if max(errors) > candidate_gate:
            raise RuntimeError(f"candidate acceptance failure: errors={errors}")
    base_control_gate = protocol["acceptance"]["base_equal_token_and_short_final_window_max_abs_error_lte"]
    for name in ("equal_token_control", "short_final_window"):
        case = summaries["base"]["cases"][name]
        errors = [case[key] for key in ("max_loss_abs_error", "max_gradient_abs_error", "max_sgd_parameter_abs_error")]
        if max(errors) > base_control_gate:
            raise RuntimeError(f"base control acceptance failure: {name} errors={errors}")
    base_defect_gate = protocol["acceptance"]["base_unequal_and_zero_local_gradient_or_parameter_error_gte"]
    for name in ("unequal_rank_and_microbatch_counts", "zero_local_tokens_positive_global"):
        case = summaries["base"]["cases"][name]
        errors = [case[key] for key in ("max_gradient_abs_error", "max_sgd_parameter_abs_error")]
        if max(errors) < base_defect_gate:
            raise RuntimeError(f"base defect reproduction missing: {name} errors={errors}")
    return {"base_and_candidate_gates": "pass", "cases_per_arm": 4}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="A new directory to hold logs, summaries, and manifest.")
    parser.add_argument(
        "--pythonpath-extra", action="append", default=[],
        help="Optional dependency overlay path to add to child PYTHONPATH; can be repeated.",
    )
    args = parser.parse_args()
    protocol = load_and_check_protocol()
    runtime = check_runtime(protocol)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory must be new: {output}")
    output.mkdir(parents=True)
    (output / "runs").mkdir()

    started = time.monotonic()
    source_record = protocol["source"]
    with tempfile.TemporaryDirectory(prefix="vare-trl-pr7249-v4-") as temporary:
        temp = Path(temporary)
        base = checkout_source(
            temp / "base", source_record["base_revision"], source_record["base_revision"], source_record["base_source_sha256"]
        )
        candidate = checkout_source(
            temp / "candidate", "refs/pull/7249/head", source_record["candidate_head"], source_record["candidate_source_sha256"]
        )
        base_run = run_arm(protocol, "base", temp / "base", output / "runs/base", args.pythonpath_extra)
        candidate_run = run_arm(protocol, "candidate", temp / "candidate", output / "runs/candidate", args.pythonpath_extra)

    gates = validate_summaries(protocol, output)
    manifest = {
        "run_id": "trl-async-distillation-pr7249-full-trainer-ddp-v4",
        "protocol_sha256": protocol["sha256"],
        "experiment_runner_sha256": source_record["runner_sha256"],
        "reproduction_wrapper_sha256": sha256(Path(__file__).resolve()),
        "runtime": runtime,
        "source_checkouts": {"base": base, "candidate": candidate},
        "runs": {"base": base_run, "candidate": candidate_run},
        "acceptance": gates,
        "total_wall_seconds": time.monotonic() - started,
        "interpretation": "Frozen synthetic AsyncDistillation trainer-correctness run; not outside human reproduction or capability evidence.",
    }
    manifest_path = output / "reproduction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
