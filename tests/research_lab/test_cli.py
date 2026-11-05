import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "vare", *args],
        cwd=cwd or ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_lists_research():
    result = _run([])
    assert result.returncode == 0
    assert "research" in result.stdout


def test_cli_run_status_report_and_resume(tmp_path: Path):
    listed = _run(["research", "list"])
    assert listed.returncode == 0
    catalog = json.loads(listed.stdout)
    assert len(catalog["tasks"]) == 3
    run_root = tmp_path / "runs"
    base = [
        "research", "run",
        "--task", "descent_sign_debug_v1",
        "--run-id", "cli-descent",
        "--run-root", str(run_root),
        "--proposer", "diagnostic",
        "--max-actions", "1",
    ]
    started = _run(base)
    assert started.returncode == 0, started.stderr
    first = json.loads(started.stdout)
    assert first["actions"] == 1
    status = _run([
        "research", "status",
        "--task", "descent_sign_debug_v1",
        "--run-id", "cli-descent",
        "--run-root", str(run_root),
    ])
    assert status.returncode == 0
    assert json.loads(status.stdout)["last_hypothesis_id"] == "baseline"
    resumed = _run([
        "research", "resume",
        "--task", "descent_sign_debug_v1",
        "--run-id", "cli-descent",
        "--run-root", str(run_root),
    ])
    assert resumed.returncode == 0, resumed.stderr
    summary = json.loads(resumed.stdout)
    assert summary["verified"] is True
    assert summary["success_class"] == "diagnostic_policy"
    report = _run([
        "research", "report",
        "--task", "descent_sign_debug_v1",
        "--run-id", "cli-descent",
        "--run-root", str(run_root),
    ])
    assert json.loads(report.stdout)["journal_sha256"] == summary["journal_sha256"]
    journal = (run_root / "cli-descent" / "journal.jsonl").read_text(encoding="utf-8")
    fresh = _run([
        "research", "evaluate",
        "--task", "descent_sign_debug_v1",
        "--run-id", "cli-descent",
        "--run-root", str(run_root),
    ])
    assert fresh.returncode == 0, fresh.stderr
    assert json.loads(fresh.stdout)["passed"] is True
    source = run_root / "cli-descent" / "workspace" / "experiment.py"
    source.write_text(source.read_text(encoding="utf-8") + "\n# stale\n", encoding="utf-8")
    evaluated = _run([
        "research", "evaluate",
        "--task", "descent_sign_debug_v1",
        "--run-id", "cli-descent",
        "--run-root", str(run_root),
    ])
    assert evaluated.returncode == 1
    assert json.loads(evaluated.stdout)["classification"] == "stale_source"
    assert (run_root / "cli-descent" / "journal.jsonl").read_text(encoding="utf-8") == journal
