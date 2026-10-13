"""Reproduce task-lifetime failure on pinned source and compare the correction."""
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

BASELINE = "79046a09768476909b89ff64faa77c512ecb8ced"
ROOT = Path(__file__).resolve().parents[1]
TEST = "tests/test_engine_evaluation_lifecycle.py"
PROBE = "reproducers/engine_evaluation_lifecycle_v1.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("output must be empty")
    assert sys.version_info >= (3, 12)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    summary = {}
    with tempfile.TemporaryDirectory(prefix="vare-engine-") as scratch:
        baseline = Path(scratch)
        if subprocess.run(["git", "cat-file", "-e", BASELINE + "^{commit}"], cwd=ROOT,
                          stderr=subprocess.DEVNULL).returncode:
            subprocess.run(["git", "fetch", "--quiet", "--filter=blob:none", "origin", BASELINE], cwd=ROOT, check=True)
        data = subprocess.check_output(["git", "archive", BASELINE, "src"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(baseline, filter="data")
        for label, source in (("baseline", baseline), ("fixed", ROOT)):
            with (out / (label + "-pytest.log")).open("w") as log:
                run = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
                    "-o", "pythonpath=" + str(source / "src"),
                    "--junitxml=" + str(out / (label + "-pytest.xml")), TEST],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            cases = ET.parse(out / (label + "-pytest.xml")).findall(".//testcase")
            record = {"exit_code": run.returncode, "tests": len(cases),
                      "failures": sum(t.find("failure") is not None for t in cases),
                      "errors": sum(t.find("error") is not None for t in cases),
                      "skips": sum(t.find("skipped") is not None for t in cases)}
            assert record == {"exit_code": 1 if label == "baseline" else 0, "tests": 13,
                              "failures": 12 if label == "baseline" else 0, "errors": 0, "skips": 0}, record
            subprocess.run([sys.executable, str(ROOT / PROBE), "--source", str(source),
                            "--output", str(out / (label + "-probe.json"))], cwd=ROOT, env=env, check=True, timeout=10)
            probe = json.loads((out / (label + "-probe.json")).read_text())
            assert probe["round_error"] == "incumbent evaluator failure"
            assert probe["evaluation_live_after_round_return"] == (label == "baseline")
            assert probe["discard_after_evaluation_finish"] == (label == "fixed")
            assert probe["candidate_error"] == ("KeyError" if label == "baseline" else "CancelledError")
            assert not probe["round_lock_held_after_return"] and not probe["candidate_snapshot_exists_after_return"]
            summary[label] = record
    sources = ["src/vare/engine.py", TEST, PROBE, "scripts/reproduce_engine_evaluation_lifecycle_v1.py"]
    manifest = {"baseline": BASELINE,
                "candidate_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())},
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "pytest": importlib.metadata.version("pytest"), "execution": "asyncio, CPU only"},
                "results": summary}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
