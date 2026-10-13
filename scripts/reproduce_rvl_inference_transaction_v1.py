"""One entry point comparing pinned main, alias-only PR, and current adapter."""
from __future__ import annotations

import argparse
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET

BASE = "b8dc1baccaadfa7f699549c254a751c2b9bc0d95"
ALIAS = "5f2d2e61d2ea8f771d99315757550091053bad46"
RVL = "27ebf7fe239d97504eeb960bf1407de220584dac"
ROOT = Path(__file__).resolve().parents[1]
TESTS = ["tests/test_rvl_grpo_torch_state.py", "tests/test_rvl_grpo_inference_cancellation.py"]


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def source(revision, destination):
    if subprocess.run(["git", "cat-file", "-e", revision + "^{commit}"], cwd=ROOT,
                      stderr=subprocess.DEVNULL).returncode:
        run(["git", "fetch", "--quiet", "--filter=blob:none", "origin", revision], cwd=ROOT)
    data = git("archive", revision, "src")
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(destination, filter="data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        parser.error("output must be an empty directory")
    import torch
    import transformers
    import pytest
    assert sys.version_info >= (3, 12)
    started = time.monotonic()
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    summaries = {}
    with tempfile.TemporaryDirectory(prefix="vare-inference-") as scratch:
        scratch = Path(scratch)
        rvl = scratch / "rvl"
        run(["git", "init", "-q", str(rvl)])
        run(["git", "remote", "add", "origin", "https://github.com/mitukx/Recursive-Verification-Lag.git"], cwd=rvl)
        run(["git", "fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", RVL], cwd=rvl)
        run(["git", "sparse-checkout", "init", "--cone"], cwd=rvl)
        run(["git", "sparse-checkout", "set", "src"], cwd=rvl)
        run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=rvl)
        assert git("rev-parse", "HEAD", cwd=rvl).decode().strip() == RVL
        for label, revision in (("baseline", BASE), ("alias_only", ALIAS), ("fixed", None)):
            selected = ROOT if revision is None else scratch / label
            if revision:
                source(revision, selected)
            with (out / (label + "-pytest.log")).open("w") as log:
                proc = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
                    "-o", "pythonpath=" + str(selected / "src"),
                    "--junitxml=" + str(out / (label + "-pytest.xml")), *TESTS],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            xml = ET.parse(out / (label + "-pytest.xml"))
            cases = xml.findall(".//testcase")
            summary = {"exit_code": proc.returncode, "tests": len(cases),
                       "failures": sum(c.find("failure") is not None for c in cases),
                       "errors": sum(c.find("error") is not None for c in cases),
                       "skips": sum(c.find("skipped") is not None for c in cases)}
            assert summary == {"exit_code": 0 if label == "fixed" else 1, "tests": 14,
                               "failures": {"baseline": 12, "alias_only": 10, "fixed": 0}[label],
                               "errors": 0, "skips": 0}, summary
            with (out / (label + "-hf.log")).open("w") as log:
                run([sys.executable, str(ROOT / "reproducers/rvl_inference_transaction_v1.py"),
                     "--vare-source", str(selected), "--rvl-source", str(rvl),
                     "--output", str(out / (label + "-hf.json"))],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            probe = json.loads((out / (label + "-hf.json")).read_text())
            for name, case in probe["standard"].items():
                assert case["error"] == ("score fixture failure after real generation"
                                         if name == "evaluate_score_failure" else None)
                assert case["state"]["model_preserved"] and case["state"]["optimizer_preserved"]
                assert case["state"]["module_modes_preserved"] == (label == "fixed")
                assert case["state"]["cpu_rng_preserved"] == (label == "fixed")
            for key in ("cancelled_call_pending_until_worker_finishes", "no_next_generation_overlap", "lock_held_after_cancel"):
                assert probe["cancellation"][key] == (label == "fixed"), probe
            if label == "fixed":
                assert all(probe["cancellation"]["final_state"].values()), probe
            summaries[label] = summary
        files = ["src/vare/integrations/rvl_grpo.py", *TESTS,
                 "reproducers/rvl_inference_transaction_v1.py", "scripts/reproduce_rvl_inference_transaction_v1.py"]
        manifest = {"baseline": BASE, "alias_only": ALIAS, "rvl": RVL,
                    "candidate_parent": git("rev-parse", "HEAD").decode().strip(),
                    "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
                    "rvl_source_sha256": {p: hashlib.sha256((rvl / p).read_bytes()).hexdigest()
                        for p in ("src/rvl_systems/hf_backend.py", "src/rvl_systems/hf_trainer.py")},
                    "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__, "pytest": pytest.__version__,
                        "device": "cpu", "torch_threads_in_hf_probe": 1},
                    "installed_distributions": dict(sorted(
                        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                        if d.metadata["Name"])),
                    "pytest": summaries, "wall_seconds": time.monotonic() - started}
    manifest["artifact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
