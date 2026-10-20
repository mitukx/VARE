from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_mlx_lora_update_smoke_v1.py"
SPEC = importlib.util.spec_from_file_location("mlx_lora_smoke", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_training_step_parser_accepts_mlx_terminal_progress_format():
    log = "\x1b[1miter   train_loss\x1b[0m\r     1    1.28 ▼   25\r     2    1.10 ▼  380\r    10    0.80 ▼ 410\n"
    assert MODULE.completed_training_steps(log) == [1, 2, 10]


def test_training_step_parser_rejects_missing_or_duplicate_steps():
    assert MODULE.completed_training_steps(" 1 1.2\n 3 0.8\n") == [1, 3]
    assert MODULE.completed_training_steps(" 1 1.2\n 1 0.9\n") == [1, 1]
