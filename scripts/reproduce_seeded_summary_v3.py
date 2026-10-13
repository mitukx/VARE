"""Compare pinned legacy analysis and corrected seed-aware analysis, CPU only."""
import argparse
import hashlib
from importlib.metadata import version
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
BASELINE = "3eb6de55251406ed6c8f77eeef34817f41fc5155"
CLI_BASELINE = "b918e6ea0e69954c3081fa63e770428785cc7840"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output must be empty")
    assert sys.version_info >= (3, 12)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    with tempfile.TemporaryDirectory(prefix="vare-seeded-summary-") as temporary:
        baseline = Path(temporary)
        if subprocess.run(["git", "cat-file", "-e", BASELINE + "^{commit}"], cwd=ROOT,
                          stderr=subprocess.DEVNULL).returncode:
            subprocess.run(["git", "fetch", "--quiet", "--filter=blob:none", "origin", BASELINE], cwd=ROOT, check=True)
        blob = subprocess.check_output(["git", "archive", BASELINE, "src"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            archive.extractall(baseline, filter="data")
        for label, source, implementation in (("baseline", baseline, "legacy"), ("fixed", ROOT, "seeded")):
            with (output / (label + ".log")).open("w") as log:
                proc = subprocess.run([sys.executable, str(ROOT / "reproducers/seeded_summary_v3.py"),
                    "--source", str(source), "--implementation", implementation, "--output", str(output / label)],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            assert proc.returncode == (1 if label == "baseline" else 0), proc.returncode
            if label == "baseline":
                assert "pooled outputs do not estimate mean fitted-run" in (output / "baseline.log").read_text()
        old = json.loads((output / "baseline/summary.json").read_text())
        new = json.loads((output / "fixed/summary.json").read_text())
        for field in ("paired_bootstrap", "decision", "decision_split", "by_seed_confirm_final", "promotes_weights"):
            assert old[field] == new[field], field
        assert (output / "baseline/synthetic-progress.jsonl").read_bytes() == (output / "fixed/synthetic-progress.jsonl").read_bytes()
        cli_baseline = baseline / "cli-baseline"
        cli_baseline.mkdir()
        if subprocess.run(["git", "cat-file", "-e", CLI_BASELINE + "^{commit}"], cwd=ROOT,
                          stderr=subprocess.DEVNULL).returncode:
            subprocess.run(["git", "fetch", "--quiet", "--filter=blob:none", "origin", CLI_BASELINE], cwd=ROOT, check=True)
        blob = subprocess.check_output(["git", "archive", CLI_BASELINE, "src", "scripts", "protocols"], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            archive.extractall(cli_baseline, filter="data")
        for label, source in (("cli-baseline", cli_baseline), ("cli-fixed", ROOT)):
            with (output / (label + ".log")).open("w") as log:
                proc = subprocess.run([sys.executable, str(ROOT / "reproducers/seeded_summary_progress_snapshot.py"),
                    "--source", str(source), "--output", str(output / label)],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            assert proc.returncode == (1 if label == "cli-baseline" else 0), proc.returncode
            if label == "cli-baseline":
                assert "summary metrics and digest must refer to the same captured bytes" in (output / "cli-baseline.log").read_text()
        with (output / "pytest.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider",
                "--junitxml=" + str(output / "pytest.xml"), "tests/test_seeded_summary_v3.py"],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30, check=True)
        cases = ET.parse(output / "pytest.xml").findall(".//testcase")
        assert len(cases) == 26 and all(not any(t.find(tag) is not None for tag in ("failure", "error", "skipped")) for t in cases)
    paths = ["src/vare/format_vs_correctness_v2.py", "src/vare/format_vs_correctness_seeded_summary_v3.py",
             "scripts/run_format_vs_correctness_comparison_v2.py", "scripts/summarize_format_vs_correctness_seeded_v3.py",
             "scripts/reproduce_seeded_summary_v3.py", "reproducers/seeded_summary_v3.py",
             "reproducers/seeded_summary_progress_snapshot.py",
             "reproducers/seeded_summary_fixture_v1.py", "tests/test_seeded_summary_v3.py",
             "protocols/format_vs_correctness_comparison_v2.json", "protocols/format_vs_correctness_seeded_summary_v3.json"]
    manifest = {"baseline": BASELINE, "cli_baseline": CLI_BASELINE,
                "candidate_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
                "artifact_sha256": {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(output.rglob("*")) if p.is_file()},
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "pytest": version("pytest"), "execution": "CPU synthetic output analysis"},
                "baseline_expected_failure": True, "cli_baseline_expected_failure": True,
                "fixed_cases_passed": len(cases), "primary_fields_preserved": True}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"baseline": "expected estimand failure", "fixed": "pass", "tests": len(cases), "primary_fields_preserved": True}))


if __name__ == "__main__":
    main()
