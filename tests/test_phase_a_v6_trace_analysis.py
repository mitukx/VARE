"""Reproducibility checks for the additive v6 trace reanalysis."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from scripts import analyze_phase_a_v6_traces


def _make_run(root: Path) -> Path:
    run_dir = root / "run-2"
    trace_dir = run_dir / "transformers-demo" / "greedy-0"
    trace_dir.mkdir(parents=True)
    (run_dir / "protocol-v1.json").write_text(
        json.dumps({"protocol_id": "fixture-v1"}), encoding="utf-8"
    )
    (run_dir / "run-manifest.json").write_text(
        json.dumps({"source_repo_head": "0123456789abcdef"}), encoding="utf-8"
    )
    (trace_dir / "agent.json").write_text(
        json.dumps(
            {
                "task_id": "demo-task",
                "seed": 7,
                "transcript": [],
                "tool_turns": 1,
                "new_tokens": 12,
                "elapsed_seconds": 0.5,
                "stop_reason": "turn_budget_exhausted",
                "no_tool_retries": 0,
                "format_parse_failure_count": 0,
                "edit_applied": False,
                "public_test_passed": None,
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "independent-grade.json").write_text(
        json.dumps({"passed": False, "failure": "fixture_failure"}),
        encoding="utf-8",
    )
    return run_dir


def _generate(run_dir: Path, output_dir: Path, monkeypatch: Any) -> tuple[bytes, bytes]:
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_phase_a_v6_traces.py",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert analyze_phase_a_v6_traces.main() == 0
    return (
        (output_dir / "analysis-v1.json").read_bytes(),
        (output_dir / "analysis-v1.md").read_bytes(),
    )


def test_analysis_paths_and_hashes_are_stable_across_checkout_roots(
    tmp_path: Path, monkeypatch: Any
) -> None:
    first = _make_run(tmp_path / "checkout-a")
    second_root = tmp_path / "checkout-b"
    second_root.mkdir()
    second = second_root / "run-2"
    shutil.copytree(first, second)

    first_json, first_markdown = _generate(
        first, tmp_path / "analysis-a", monkeypatch
    )
    second_json, second_markdown = _generate(
        second, tmp_path / "analysis-b", monkeypatch
    )

    assert first_json == second_json
    assert first_markdown == second_markdown
    report = json.loads(first_json)
    assert report["rollouts"][0]["trace_path"] == (
        "transformers-demo/greedy-0/agent.json"
    )

