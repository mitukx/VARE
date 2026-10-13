from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vare.heldout_exact_integer import load_task_pack
from vare.update_cost_smoke import (
    HELDOUT_PACK_SHA256,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    accept_smoke_record,
    assert_train_example_is_outside_the_heldout_pack,
    train_example,
    validate_protocol,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/heldout_exact_integer_update_cost_smoke_v1.json"
PACK = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
RUNNER = ROOT / "scripts/run_exact_integer_update_cost_smoke_v1.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _accepted_record() -> dict[str, object]:
    return {
        "heldout_pack_used_for_update": False,
        "loss": 1.5,
        "optimizer_steps": 1,
        "reload_matches_saved": True,
        "resources": {"peak_memory_bytes": 1024, "wall_seconds": 0.1},
        "trainable_parameters_changed": True,
    }


def test_smoke_protocol_bytes_and_train_example_stay_outside_the_pack():
    assert _sha(PROTOCOL) == PROTOCOL_SHA256
    assert _sha(PACK) == HELDOUT_PACK_SHA256
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    example = assert_train_example_is_outside_the_heldout_pack(PACK.read_text(encoding="utf-8"))
    assert example == train_example()
    assert example["seed"] == 1029953
    assert example["expression"] == "23544 + 19091 - 10702"
    assert example["completion"] == "#### 31933"
    assert example["value"] == 31933
    assert "31933" not in str(example["prompt"])
    expressions = {row["expression"] for row in load_task_pack(PACK.read_text(encoding="utf-8"))}
    assert example["expression"] not in expressions
    assert protocol["scores_heldout_pack"] is False
    assert protocol["optimizer_steps"] == 1


def test_smoke_protocol_rejects_a_second_update():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["optimizer_steps"] = 2
    try:
        validate_protocol(protocol)
    except ValueError as exc:
        assert "one optimizer update" in str(exc)
    else:
        raise AssertionError("two updates were accepted")


def test_accept_smoke_record_fails_closed():
    assert accept_smoke_record(_accepted_record())["accepted"] is True
    cases = {
        "optimizer_steps": 2,
        "loss": float("nan"),
        "trainable_parameters_changed": False,
        "reload_matches_saved": False,
        "heldout_pack_used_for_update": True,
    }
    for key, value in cases.items():
        record = _accepted_record()
        record[key] = value
        assert accept_smoke_record(record)["accepted"] is False
    record = _accepted_record()
    record["resources"] = {"peak_memory_bytes": 0, "wall_seconds": -1}
    decision = accept_smoke_record(record)
    assert decision["accepted"] is False
    assert decision["protocol_id"] == PROTOCOL_ID
    assert "missing_peak_memory" in decision["reasons"]
    assert "missing_wall_seconds" in decision["reasons"]


def test_runner_cannot_score_the_pack_or_change_the_step_count():
    text = RUNNER.read_text(encoding="utf-8")
    assert "iters=1" in text
    assert "val_dataset=None" in text
    assert "training_callback=capture" in text
    assert "snapshot_download" in text
    assert "--iters" not in text
    assert "grade_output" not in text
    assert "run_gate" not in text
    assert text.count("tasks.jsonl") == 1
