from __future__ import annotations

from pathlib import Path

from scripts import grade_transformers_fresh_dev_v5 as grader


def test_candidate_runner_passes_protocol_cpu_and_pid_limits(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_one_call(*args, **kwargs):
        seen.update(kwargs)
        return {"output_ids": [[1, 2, 3, 4, 5, 6]]}, 0.1, None

    monkeypatch.setattr(grader, "_one_call", fake_one_call)
    result = grader._run_candidate(
        grader.TASK_IDS[0], tmp_path, grader.IMAGE,
        {"limits": {"timeout_s": 120, "memory_mb": 1900, "cpus": 1, "pids": 32}},
    )

    assert result["failure"] is None
    assert seen["cpus"] == 1.0
    assert seen["pids_limit"] == 32
