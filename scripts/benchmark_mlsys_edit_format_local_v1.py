#!/usr/bin/env python3
"""Measure the vetted scripted edit-format control through the black-box runner."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "mlsys_edit_format_local_v1"))

from vare.container_isolation import DockerIsolationRunner  # noqa: E402
from vare.isolated_candidate import FunctionCall, grade_calls, paired_difference_ci  # noqa: E402
from vare.learning_pipeline import edit_format as baseline_module  # noqa: E402
import workload  # noqa: E402

ASSET = ROOT / "benchmarks" / "mlsys_edit_format_local_v1"
SOURCE = ROOT / "src" / "vare" / "learning_pipeline" / "edit_format.py"
PROTOCOL = ROOT / "protocols" / "mlsys_edit_format_local_benchmark_v1.json"
DEFAULT_OUT = ROOT / "results" / "mlsys-edit-format-local-v1" / "control-benchmark.json"
BASELINE_SHA256 = "4f0ee215d5769b42df3aa83128e947b7259234141678856fba74d98bdc7c1f59"
PATCH_SHA256 = "339451b7bba7b261a3576e5a81d63961de08f14c7a846d3c8719eaafdb340fa1"
IMAGE = "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258"
RUNNER_SOURCE = (
    "import edit_format\n"
    "def run_case(original, completion):\n"
    "    o = edit_format.apply_blocks(original, edit_format.parse_blocks(completion))\n"
    "    return {'ok': o.ok, 'reason': o.reason, 'has_new_text': o.new_text is not None, "
    "'new_text': o.new_text if o.new_text is not None else ''}\n"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_subject(path: Path, source: str) -> Path:
    path.mkdir()
    (path / "edit_format.py").write_text(source, encoding="utf-8")
    (path / "candidate_runner.py").write_text(RUNNER_SOURCE, encoding="utf-8")
    return path


def build_calls(seed: int, count: int) -> list[FunctionCall]:
    cases = workload.make_cases(seed, count, profile="benchmark")
    calls = []
    for index, (original, completion) in enumerate(cases):
        outcome = baseline_module.apply_blocks(original, baseline_module.parse_blocks(completion))
        calls.append(FunctionCall(
            f"case-{index:03d}", "candidate_runner", "run_case", [original, completion],
            {"ok": outcome.ok, "reason": outcome.reason,
             "has_new_text": outcome.new_text is not None,
             "new_text": outcome.new_text if outcome.new_text is not None else ""},
        ))
    return calls


def timed_session(subject: Path, calls: list[FunctionCall], runner: DockerIsolationRunner) -> dict[str, object]:
    started = time.perf_counter()
    report = grade_calls(
        subject, calls, task_kind="correctness", image=IMAGE, timeout_s=90,
        runner=runner, preserve_session=True, discard_candidate_stdout=True,
    )
    outer_wall = time.perf_counter() - started
    if not report.passed or any(case.wall_s is None for case in report.cases):
        failures = [case.failure for case in report.cases if not case.passed]
        raise RuntimeError(f"black-box timing batch failed closed ({len(failures)} cases)")
    samples = [float(case.wall_s) for case in report.cases if case.wall_s is not None]
    return {
        "case_count": len(samples),
        "mean_request_response_s": statistics.mean(samples),
        "median_request_response_s": statistics.median(samples),
        "sum_request_response_s": sum(samples),
        "outer_wall_s": outer_wall,
        "samples_s": samples,
        "output_pass": True,
    }


def _ci(values: list[float]) -> dict[str, object]:
    mean, low, high, t_critical, df = paired_difference_ci(values)
    return {"mean": mean, "ci95": [low, high], "t_critical": t_critical, "df": df}


def append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    progress_path = args.out.with_suffix(".progress.jsonl")
    append_event(progress_path, {"run_id": run_id, "event": "start"})
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if sha256(SOURCE) != BASELINE_SHA256:
        raise SystemExit("baseline source hash mismatch")
    if sha256(ASSET / "workload.py") != protocol["frozen_sha256"]["workload"]:
        raise SystemExit("workload source hash mismatch")
    if sha256(ASSET / "known_good_control.patch") != PATCH_SHA256:
        raise SystemExit("known-good control patch hash mismatch")
    if os.environ.get("DOCKER_CONTEXT") != "colima":
        raise SystemExit("set DOCKER_CONTEXT=colima")

    baseline_source = SOURCE.read_text(encoding="utf-8")
    temp_root = ROOT / "work"
    temp_root.mkdir(exist_ok=True)
    records: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="mlsys-edit-format-v1-", dir=temp_root) as temp:
        tmp = Path(temp)
        base = write_subject(tmp / "baseline", baseline_source)
        candidate = write_subject(tmp / "known-good", baseline_source)
        patch = (ASSET / "known_good_control.patch").read_bytes()
        applied = subprocess.run(["patch", "-p1", "-d", str(candidate)], input=patch,
                                 capture_output=True, check=True)
        records["control_patch_apply_stdout_sha256"] = hashlib.sha256(applied.stdout).hexdigest()
        candidate_sha = sha256(candidate / "edit_format.py")
        if candidate_sha != protocol["control"]["candidate_source_sha256"]:
            raise SystemExit("known-good control source hash mismatch")

        runner = DockerIsolationRunner()
        aa_pairs = []
        for pair_index, seed in enumerate(protocol["measurement"]["aa_seeds"]):
            calls = build_calls(int(seed), int(protocol["measurement"]["aa_cases_per_session"]))
            first = timed_session(base, calls, runner)
            second = timed_session(base, calls, runner)
            aa_pairs.append({"seed": int(seed), "baseline_first": first, "baseline_second": second,
                             "difference_s": first["mean_request_response_s"]
                             - second["mean_request_response_s"]})
            append_event(progress_path, {
                "run_id": run_id, "event": "aa_pair_complete", "pair_index": pair_index,
                "seed": int(seed), "difference_s": aa_pairs[-1]["difference_s"],
                "output_pass": first["output_pass"] and second["output_pass"],
            })

        control_pairs = []
        for pair_index, seed in enumerate(protocol["measurement"]["paired_seeds"]):
            calls = build_calls(int(seed), int(protocol["measurement"]["cases_per_session"]))
            order = protocol["measurement"]["order"][pair_index]
            runs: dict[str, dict[str, object]] = {}
            for role in order:
                subject = base if role == "baseline" else candidate
                runs[role] = timed_session(subject, calls, runner)
            base_mean = float(runs["baseline"]["mean_request_response_s"])
            candidate_mean = float(runs["known_good"]["mean_request_response_s"])
            control_pairs.append({
                "seed": int(seed), "order": order, "baseline": runs["baseline"],
                "known_good": runs["known_good"], "difference_s": base_mean - candidate_mean,
                "speedup": base_mean / candidate_mean,
            })
            append_event(progress_path, {
                "run_id": run_id, "event": "control_pair_complete", "pair_index": pair_index,
                "seed": int(seed), "order": order, "speedup": control_pairs[-1]["speedup"],
                "output_pass": runs["baseline"]["output_pass"] and runs["known_good"]["output_pass"],
            })

    aa_ci = _ci([float(pair["difference_s"]) for pair in aa_pairs])
    aa_valid = aa_ci["ci95"][0] <= 0 <= aa_ci["ci95"][1]
    diffs = [float(pair["difference_s"]) for pair in control_pairs]
    log_ratios = [math.log(float(pair["speedup"])) for pair in control_pairs]
    diff_ci = _ci(diffs)
    log_ci = _ci(log_ratios)
    speedup = math.exp(float(log_ci["mean"]))
    speedup_ci = [math.exp(float(value)) for value in log_ci["ci95"]]
    accepted = bool(aa_valid and speedup_ci[0] > 1.0 and speedup >= 1.05)
    report = {
        "schema": "vare.mlsys_edit_format_local_control_result.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_sha256": sha256(PROTOCOL),
        "progress_log_sha256": sha256(progress_path),
        "environment": {
            "system": platform.system(), "machine": platform.machine(),
            "python": platform.python_version(), "docker_context": os.environ["DOCKER_CONTEXT"],
            "image": IMAGE,
        },
        "source_sha256": {
            "baseline": sha256(SOURCE), "workload": sha256(ASSET / "workload.py"),
            "control_patch": sha256(ASSET / "known_good_control.patch"),
            "candidate": candidate_sha,
        },
        "aa_noise_floor": {"pairs": aa_pairs, "difference_ci95": aa_ci, "valid": aa_valid},
        "paired_measurements": control_pairs,
        "primary": {
            "metric": "geometric mean paired speedup from warm-container request-response intervals",
            "speedup": speedup, "speedup_ci95": speedup_ci,
            "mean_paired_difference_s": diff_ci,
            "accepted_by_frozen_rule": accepted,
        },
        "analysis_limits": [
            "Generated source/completion workloads are synthetic and public; this is a real VARE code-path microbenchmark, not end-to-end model throughput.",
            "The two model proposals failed correctness and were not timed.",
            "Per-call request-response timing excludes container startup and model generation; outer wall is retained per session.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({"output": str(args.out), "speedup": speedup,
                      "speedup_ci95": speedup_ci, "aa_valid": aa_valid,
                      "accepted": accepted}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
