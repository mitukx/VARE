"""Pinned-source CPU IO comparison and existing-loop recovery, no model inference."""
from __future__ import annotations

import argparse
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
PROTOCOL = "protocols/checkpoint_read_bounds_v1.lock.json"
TESTS = ["tests/test_checkpoint_read_bounds.py", "tests/test_checkpoint_io_recovery.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def measure(case, protocol):
    # Import only after the parent binds PYTHONPATH to a pinned source tree.
    import resource
    import vare.checkpoint_artifacts as backend

    with tempfile.TemporaryDirectory(prefix="vare-checkpoint-measure-") as folder:
        root = Path(folder)
        store = backend.TrustedCheckpointStore(root / "store", max_bytes=128 * 1024**2)
        source = root / "source"
        raw = b"v" * protocol["valid_bytes"] if case == "valid" else b"checkpoint-io-v1"
        source.write_bytes(raw)
        digest = store.register_file("p0", source)
        obj = store.objects / digest
        real_fstat = os.fstat
        fired = False
        payload = b"x" * protocol["growth_bytes"] if case == "growth" else b""
        if case == "oversized":
            with obj.open("wb") as out:
                out.truncate(protocol["replacement_bytes"])
        elif case == "growth":
            def grow(fd):
                nonlocal fired
                info = real_fstat(fd)
                if not fired and info.st_ino == obj.stat().st_ino:
                    fired = True
                    with obj.open("ab") as out:
                        out.write(payload)
                return info
            os.fstat = grow
        count = protocol["valid_reads"] if case == "valid" else 1
        total = 0
        rejection = None
        tracemalloc.start()
        started = time.perf_counter()
        try:
            for _ in range(count):
                value, observed = store.read_verified(
                    "p0", max_read_bytes=protocol["valid_bytes"] if case == "valid" else 64,
                )
                require(value == raw and observed == digest, 'checkpoint evidence check failed')
                total += len(value)
        except backend.CheckpointIntegrityError as exc:
            rejection = str(exc)
        finally:
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            os.fstat = real_fstat
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "case": case, "backend_sha256": sha(backend.__file__),
            "wall_s": elapsed, "python_peak_bytes": peak,
            "process_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "verified_bytes": total, "reads": count,
            "throughput_mib_s": total / 1024**2 / elapsed,
            "rejection": rejection, "growth_injected": fired,
        }


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=["valid", "oversized", "growth"])
    args = parser.parse_args()
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    if args.worker:
        print(json.dumps(measure(args.worker, protocol), allow_nan=False))
        return
    if args.output is None:
        parser.error("--output required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("output must be new or empty; retain earlier attempts")
    for path, expected in protocol["protected_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"frozen asset changed: {path}")
    revision = protocol["baseline_revision"]
    if subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=ROOT,
                      stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["git", "fetch", "--filter=blob:none", "origin", revision],
                       cwd=ROOT, check=True, timeout=120)
    raw_results = []
    test_results = {}
    with tempfile.TemporaryDirectory(prefix="vare-checkpoint-comparison-") as temp:
        baseline = Path(temp) / "baseline"
        baseline.mkdir()
        # git archive may eagerly hydrate unrelated large blobs in a partial
        # clone. Read only the source paths required by this comparison.
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision, "src"], cwd=ROOT,
        ).decode().splitlines()
        for name in paths:
            dest = baseline / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(subprocess.check_output(
                ["git", "show", f"{revision}:{name}"], cwd=ROOT, timeout=30,
            ))
        sources = {"baseline": baseline, "candidate": ROOT}
        for label, source in sources.items():
            env = {**os.environ, "PYTHONPATH": str(source / "src"),
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                   "VARE_CHECKPOINT_IO_EVIDENCE_DIR": str(out / label)}
            with (out / f"{label}-pytest.log").open("w") as log:
                run = subprocess.run([
                    sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
                    "-o", f"pythonpath={source / 'src'}", *TESTS,
                    f"--junitxml={out / (label + '-pytest.xml')}",
                ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=45)
            cases = ET.parse(out / f"{label}-pytest.xml").findall(".//testcase")
            test_results[label] = {
                "exit_code": run.returncode,
                "cases": {c.attrib["name"]: ("error" if c.find("error") is not None else
                         "fail" if c.find("failure") is not None else
                         "skip" if c.find("skipped") is not None else "pass") for c in cases},
                "backend_sha256": sha(source / "src/vare/checkpoint_artifacts.py"),
            }
        for repetition in range(protocol["pairs"]):
            order = ["baseline", "candidate"] if repetition % 2 == 0 else ["candidate", "baseline"]
            for case in ["valid", "oversized", "growth"]:
                for label in order:
                    env = {**os.environ, "PYTHONPATH": str(sources[label] / "src"),
                           "PYTHONDONTWRITEBYTECODE": "1"}
                    started = time.perf_counter()
                    run = subprocess.run(
                        [sys.executable, str(Path(__file__).resolve()), "--worker", case],
                        cwd=temp, env=env, capture_output=True, text=True, timeout=10, check=True,
                    )
                    record = {**json.loads(run.stdout), "label": label, "repetition": repetition,
                              "subprocess_wall_s": time.perf_counter() - started}
                    raw_results.append(record)
                    write(out / f"{label}-{case}-{repetition}.json", record)
    # Persist before checking acceptance so every failed comparison survives.
    write(out / "test-results.json", test_results)
    summary = {}
    for case in ["valid", "oversized", "growth"]:
        summary[case] = {}
        for label in ["baseline", "candidate"]:
            rows = [r for r in raw_results if r["case"] == case and r["label"] == label]
            summary[case][label] = {
                f"median_{key}": statistics.median(r[key] for r in rows)
                for key in ["wall_s", "python_peak_bytes", "process_peak_rss_bytes", "throughput_mib_s"]
            }
    write(out / "summary.json", summary)
    write(out / "manifest.json", {
        "baseline_revision": revision,
        "candidate_parent_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_sha256": {p: sha(ROOT / p) for p in [PROTOCOL, *TESTS,
                          "src/vare/checkpoint_artifacts.py", "scripts/reproduce_checkpoint_read_bounds_v1.py"]},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu"},
        "artifact_sha256": {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
        "scope": "same-host IO correctness/resources and scripted protocol recovery; no model learning",
    })
    expected = protocol["expected_test_counts"]
    for label, counts in expected.items():
        actual = {verdict: list(test_results[label]["cases"].values()).count(verdict)
                  for verdict in ["pass", "fail", "error", "skip"]}
        require(actual == counts, (label, actual, counts))
        require(test_results[label]["exit_code"] == (1 if counts["fail"] else 0), 'checkpoint evidence check failed')
    for row in raw_results:
        require(row["backend_sha256"] == test_results[row["label"]]["backend_sha256"], 'checkpoint evidence check failed')
        if row["case"] == "valid":
            require(row["rejection"] is None and row["verified_bytes"] == protocol["valid_bytes"] * protocol["valid_reads"], 'checkpoint evidence check failed')
        else:
            require(row["rejection"] is not None, 'checkpoint evidence check failed')
            if row["label"] == "candidate":
                require(row["python_peak_bytes"] < protocol["invalid_allocation_ceiling_bytes"], 'checkpoint evidence check failed')
            if row["case"] == "growth":
                require(row["growth_injected"], 'checkpoint evidence check failed')
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
