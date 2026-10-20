"""Regression checks for the preregistered 4B MLX coding-agent pilot."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_4b_pilot_uses_json_task_scoring_and_no_heldout_tasks():
    protocol = json.loads(
        (ROOT / "protocols/mlsys_coding_agent_4b_mlx_pilot_v5.json").read_text()
    )
    assert protocol["execution"]["scoring_command"] == ["python", "/workspace/solve.py"]
    assert protocol["models"]["primary"]["backend"] == "mlx"
    assert protocol["inference"]["edit_format"] == "unified_diff"
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
