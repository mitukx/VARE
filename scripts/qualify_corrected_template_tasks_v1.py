#!/usr/bin/env python3
"""Run pre-registered baseline/fix/mutant controls for narrow task contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def durable(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def monitor(command: list[str], *, wall_s: float, rss_limit: int,
            swap_growth_limit: int) -> dict:
    start = time.monotonic()
    swap_start = psutil.swap_memory().used
    samples = []
    abort = None
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    while process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            rss = sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)]
                      if item.is_running())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss = 0
        swap_used = psutil.swap_memory().used
        samples.append({"elapsed_s": round(time.monotonic() - start, 3),
                        "process_tree_rss_bytes": rss, "swap_used_bytes": swap_used,
                        "available_memory_bytes": psutil.virtual_memory().available})
        if time.monotonic() - start > wall_s:
            abort = "wall_limit"
        elif rss > rss_limit:
            abort = "rss_limit"
        elif swap_used - swap_start > swap_growth_limit:
            abort = "host_swap_growth_limit"
        if abort:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(0.1)
    stdout, stderr = process.communicate()
    return {"command": command, "returncode": process.returncode,
            "elapsed_s": round(time.monotonic() - start, 3), "abort": abort,
            "stdout": stdout, "stderr": stderr,
            "peak_sampled_rss_bytes": max((row["process_tree_rss_bytes"] for row in samples), default=0),
            "swap_used_start_bytes": swap_start,
            "swap_used_end_bytes": psutil.swap_memory().used, "samples": samples}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != args.protocol_sha256:
        raise ValueError("operator-supplied qualification protocol hash mismatch")
    protocol = json.loads(protocol_bytes)
    for relative, expected in protocol["source_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"qualification source drift: {relative}")
    for source in protocol.get("upstream_sources", []):
        result = subprocess.run(
            ["git", "--no-replace-objects", "-C", source["repository"], "show",
             f"{source['revision']}:{source['path']}"],
            capture_output=True, check=True, timeout=30,
        )
        observed = hashlib.sha256(result.stdout).hexdigest()
        if observed != source["sha256"]:
            raise ValueError(f"pinned upstream source drift: {source['revision']}:{source['path']}")
    assets = ROOT / protocol["asset_root"]
    for relative, expected in protocol["asset_hashes"].items():
        if sha(assets / relative) != expected:
            raise ValueError(f"qualification asset drift: {relative}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    outcomes = []
    for task in protocol["tasks"]:
        for variant in protocol["variants"]:
            key = {"baseline": "candidate_projection_asset",
                   "known-fix": "known_fix_asset",
                   "incorrect-mutant": "incorrect_mutant_asset"}[variant]
            source = assets / task[key]
            grading_dir = output / task["id"]
            grading_dir.mkdir(exist_ok=True)
            raw = monitor(
                [sys.executable, "-I", str(ROOT / protocol["grader"]),
                 "--task", task["id"], "--source", str(source)],
                wall_s=protocol["qualification_resources"]["wall_seconds"],
                rss_limit=protocol["qualification_resources"]["rss_limit_bytes"],
                swap_growth_limit=protocol["qualification_resources"]["swap_growth_limit_bytes"],
            )
            try:
                observation = json.loads(raw["stdout"]) if raw["returncode"] == 0 else None
            except json.JSONDecodeError:
                observation = None
            expected_success = variant == "known-fix"
            passed_gate = (
                raw["abort"] is None and isinstance(observation, dict)
                and observation.get("task") == task["id"]
                and observation.get("admitted") is True
                and observation.get("success") is expected_success
            )
            record = {"task_id": task["id"], "variant": variant,
                      "source_sha256": sha(source), "expected_success": expected_success,
                      "qualification_gate_passed": passed_gate,
                      "observation": observation, "process": raw}
            durable(grading_dir / f"{variant}.json", record)
            outcomes.append({"task_id": task["id"], "variant": variant,
                             "expected_success": expected_success,
                             "observed_success": observation.get("success") if observation else None,
                             "qualification_gate_passed": passed_gate})
            print(json.dumps(outcomes[-1], sort_keys=True), flush=True)
            if raw["abort"] is not None:
                break
        if outcomes and outcomes[-1]["qualification_gate_passed"] is False:
            # Preserve all planned controls for the current task, but do not
            # continue into the next task after a failed resource/setup gate.
            current = [row for row in outcomes if row["task_id"] == task["id"]]
            if any(row["observed_success"] is None for row in current):
                break
    passed = len(outcomes) == len(protocol["tasks"]) * len(protocol["variants"])
    passed = passed and all(row["qualification_gate_passed"] for row in outcomes)
    durable(output / "summary.json", {"protocol_sha256": args.protocol_sha256,
                                      "outcomes": outcomes,
                                      "qualification_passed": passed,
                                      "model_inference": "NOT_RUN"})
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
