"""Compare whole-round ownership and restart faults against pinned main."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "protocols/runtime_round_ownership_v1.lock.json"
TEST = "tests/test_runtime_round_ownership.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("output must be empty")
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    assert hashlib.sha256((ROOT / TEST).read_bytes()).hexdigest() == protocol["test_sha256"]
    revision = protocol["baseline_revision"]
    if subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=ROOT,
                      stderr=subprocess.DEVNULL).returncode:
        subprocess.run(["git", "fetch", "--quiet", "--filter=blob:none", "origin", revision],
                       cwd=ROOT, check=True, timeout=120)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    results = {}
    with tempfile.TemporaryDirectory(prefix="vare-runtime-round-") as scratch:
        baseline = Path(scratch)
        archive_bytes = subprocess.check_output(["git", "archive", revision, "src"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as archive:
            archive.extractall(baseline, filter="data")
        for label, source in (("baseline", baseline), ("candidate", ROOT)):
            with (out / (label + "-pytest.log")).open("w") as log:
                run = subprocess.run([
                    sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
                    "-o", "pythonpath=" + str(source / "src"), TEST,
                    "--junitxml=" + str(out / (label + "-pytest.xml")),
                ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=45)
            cases = ET.parse(out / (label + "-pytest.xml")).findall(".//testcase")
            failures = [c.find("failure") for c in cases if c.find("failure") is not None]
            record = {"exit_code": run.returncode, "tests": len(cases),
                      "failures": len(failures),
                      "errors": sum(c.find("error") is not None for c in cases),
                      "skips": sum(c.find("skipped") is not None for c in cases),
                      "engine_sha256": hashlib.sha256((source / "src/vare/engine.py").read_bytes()).hexdigest()}
            record["case_results"] = {c.attrib["name"]: "failure" if c.find("failure") is not None else "pass" for c in cases}
            results[label] = record
            assert {k: record[k] for k in ("exit_code", "tests", "failures", "errors", "skips")} == {
                "exit_code": 1 if label == "baseline" else 0, "tests": 10,
                "failures": 8 if label == "baseline" else 0, "errors": 0, "skips": 0,
            }, record
            if label == "baseline":
                for case in cases:
                    failure = case.find("failure")
                    if case.attrib["name"].startswith("test_success_control"):
                        assert failure is None
                    else:
                        assert failure is not None
                        expected = ("second loop overlapped a live runtime transaction"
                                    if case.attrib["name"].startswith("test_two_loops")
                                    else "unresolved runtime allowed another hook access")
                        assert expected in (failure.text or ""), case.attrib["name"]
    sources = [PROTOCOL, TEST, "src/vare/engine.py", "src/vare/runtime.py", "src/vare/promotion_journal.py",
               "src/vare/interfaces.py", "tests/test_checkpoint_bound_promotion.py",
               "tests/test_promotion_publication_recovery.py",
               "scripts/reproduce_runtime_round_ownership_v1.py"]
    manifest = {
        "protocol_sha256": hashlib.sha256((ROOT / PROTOCOL).read_bytes()).hexdigest(),
        "baseline_revision": revision,
        "candidate_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "pytest": importlib.metadata.version("pytest"), "device": "cpu"},
        "results": results,
        "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
