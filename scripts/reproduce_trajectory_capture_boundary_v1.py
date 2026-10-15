"""CPU comparison of trajectory policy parity, capture closure and promotion."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
LOCK = "protocols/trajectory_capture_boundary_v1.lock.json"
TEST = "tests/test_trajectory_capture_boundary.py"
SOURCES = ["src/vare/trajectory_dispatch.py", "src/vare/trajectory_safety.py"]


def require(value, detail):
    if not value:
        raise ValueError(detail)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def worker(protocol, trace_output):
    import resource
    import vare.trajectory_dispatch as dispatch
    import vare.trajectory_safety as safety

    with tempfile.TemporaryDirectory(prefix="vare-trajectory-cost-") as folder:
        path = Path(folder) / "trace.jsonl"
        policy = safety.TrajectoryPolicy("capture-cost-v1", ("noop",))
        recorder = safety.TrustedTrajectoryRecorder(path, run_id="cost-v1", policy_id="candidate")
        dispatcher = dispatch.AuditedDispatcher(recorder=recorder, policy=policy)
        calls = protocol["calls_per_worker"]

        async def noop():
            return 1

        async def run():
            for _ in range(calls):
                require(await dispatcher.execute("noop", noop) == 1, "tool result changed")
            dispatcher.close(expected_event_count=2 * calls)

        tracemalloc.start()
        started = time.perf_counter()
        asyncio.run(run())
        dispatch_s = time.perf_counter() - started
        started = time.perf_counter()
        audit = safety.audit_trajectory(path, run_id="cost-v1", policy_id="candidate", policy=policy)
        audit_s = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        require(audit.status == "pass" and audit.event_count == calls * 2, "clean audit failed")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        trace_raw = path.read_bytes()
        if trace_output is not None:
            trace_output.parent.mkdir(parents=True, exist_ok=True)
            trace_output.write_bytes(trace_raw)
        return {"dispatch_wall_s": dispatch_s, "audit_wall_s": audit_s,
                "throughput_tools_s": calls / dispatch_s, "python_peak_bytes": peak,
                "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                "event_count": audit.event_count, "audit_status": audit.status,
                "calls": calls, "scope": "noop tool timing; real child execution tested separately",
                "trace_sha256": hashlib.sha256(trace_raw).hexdigest(),
                "ledger_head_sha256": audit.ledger_head_sha256,
                "source_sha256": {name: sha(module.__file__) for name, module in
                                  zip(SOURCES, [dispatch, safety], strict=True)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--trace-output", type=Path)
    args = parser.parse_args()
    protocol = json.loads((ROOT / LOCK).read_text())
    if args.worker:
        print(json.dumps(worker(protocol, args.trace_output), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), "retain earlier runs; output must be empty")
    for name, digest in protocol["protected_sha256"].items():
        require(sha(ROOT / name) == digest, f"frozen asset changed: {name}")
    revision = protocol["baseline_revision"]
    if subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=ROOT,
                      stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["git", "fetch", "--filter=blob:none", "origin", revision],
                       cwd=ROOT, check=True, timeout=120)
    tests, measurements = {}, []
    with tempfile.TemporaryDirectory(prefix="vare-trajectory-compare-") as folder:
        baseline = Path(folder) / "baseline"
        paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "src"],
                                        cwd=ROOT, text=True).splitlines()
        for name in paths:
            dest = baseline / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(subprocess.check_output(["git", "show", f"{revision}:{name}"],
                                                    cwd=ROOT, timeout=30))
        trees = {"baseline": baseline, "candidate": ROOT}
        for label, tree in trees.items():
            env = {**os.environ, "PYTHONPATH": str(tree / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                   "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                   "VARE_TRAJECTORY_BOUNDARY_EVIDENCE_DIR": str(out / label)}
            with (out / f"{label}-pytest.log").open("w") as log:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
                    "-o", f"pythonpath={tree / 'src'}", TEST,
                    f"--junitxml={out / (label + '-pytest.xml')}",
                ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=protocol["test_timeout_s"])
            cases = ET.parse(out / f"{label}-pytest.xml").findall(".//testcase")
            tests[label] = {"exit_code": result.returncode,
                "cases": {c.attrib["name"]: "error" if c.find("error") is not None else
                          "fail" if c.find("failure") is not None else
                          "skip" if c.find("skipped") is not None else "pass" for c in cases},
                "source_sha256": {name: sha(tree / name) for name in SOURCES}}
        for repetition in range(protocol["pairs"]):
            order = ["baseline", "candidate"] if repetition % 2 == 0 else ["candidate", "baseline"]
            for label in order:
                env = {**os.environ, "PYTHONPATH": str(trees[label] / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                                         "--trace-output", str(out / f"cost-traces/{label}-{repetition}.jsonl")],
                                        cwd=folder, env=env, capture_output=True, text=True,
                                        timeout=protocol["measurement_timeout_s"])
                if result.returncode:
                    write(out / f"worker-failure-{label}-{repetition}.json",
                          {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
                    raise RuntimeError("measurement worker failed; see retained diagnostics")
                measurements.append({**json.loads(result.stdout), "label": label, "repetition": repetition})
    summary = {label: {f"median_{key}": statistics.median(r[key] for r in measurements if r["label"] == label)
                      for key in ["dispatch_wall_s", "audit_wall_s", "throughput_tools_s",
                                  "python_peak_bytes", "process_peak_rss_bytes"]}
               for label in ["baseline", "candidate"]}
    write(out / "test-results.json", tests)
    write(out / "measurements.json", measurements)
    write(out / "summary.json", summary)
    write(out / "manifest.json", {
        "baseline_revision": revision,
        "candidate_parent_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu"},
        "source_sha256": {name: sha(ROOT / name) for name in [*SOURCES, TEST, LOCK,
                            "scripts/reproduce_trajectory_capture_boundary_v1.py"]},
        "artifact_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    })
    # Keep raw records before checking the frozen acceptance rule.
    for label, record in tests.items():
        counts = Counter(record["cases"].values())
        require({k: counts[k] for k in ["pass", "fail", "error", "skip"]} == protocol["expected"][label],
                (label, dict(counts)))
        require(record["exit_code"] == (1 if label == "baseline" else 0), "unexpected pytest status")
    for row in measurements:
        require(row["source_sha256"] == tests[row["label"]]["source_sha256"], "worker source differs from graded source")
    old = json.loads((out / "baseline/command-promotion.json").read_text())
    new = json.loads((out / "candidate/command-promotion.json").read_text())
    require(old["early_closed"] and old["early_audit_passed"] and old["final_promotion_accepted"],
            "baseline did not reproduce premature certification")
    require(not new["early_closed"] and not new["early_audit_passed"]
            and new["command_wrapper_error"] is None and new["final_promotion_accepted"],
            "candidate failed ownership/promotion control")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
