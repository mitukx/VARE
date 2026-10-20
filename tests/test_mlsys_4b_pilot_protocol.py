"""Regression checks for the preregistered 4B MLX coding-agent pilot."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_4b_pilot_uses_json_task_scoring_and_no_heldout_tasks():
    protocol = json.loads(
        (ROOT / "protocols/mlsys_coding_agent_4b_mlx_pilot_v6.json").read_text()
    )
    assert protocol["execution"]["scoring_command"] == ["python", "/workspace/solve.py"]
    assert protocol["models"]["primary"]["backend"] == "mlx"
    assert protocol["inference"]["edit_format"] == "whole_file"
    assert protocol["budgets"]["max_new_tokens_per_turn"] == 768
    assert protocol["pack"]["train"] == [
        "mlsys-v1-grad-accum-tail",
        "mlsys-v1-paired-shuffle-loader",
        "mlsys-v1-range-sums",
        "mlsys-v1-row-log-softmax",
        "mlsys-v1-warmup-cosine",
    ]
    assert protocol["pack"]["heldout_not_used"] == [
        "mlsys-v1-aging-scheduler",
        "mlsys-v1-causal-attention",
        "mlsys-v1-dataset-standardize",
        "mlsys-v1-global-grad-clip",
    ]


def test_truncated_whole_file_call_gets_actionable_retry():
    from scripts.run_local_agent import no_tool_retry_reminder

    reminder = no_tool_retry_reminder(
        '<tool_call>{"name":"write_file"',
        edit_format="whole_file", allowed_paths=["grad_accum.py"], token_limit=768,
    )
    assert "cut off" in reminder
    assert "Do not reread" in reminder
    assert "write_file" in reminder
