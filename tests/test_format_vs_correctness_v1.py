from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from vare.format_vs_correctness import PROTOCOL_ID, PROTOCOL_SHA256, comparison_refusal

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/format_vs_correctness_comparison_v1.json"
SCRIPT = ROOT / "scripts/format_vs_correctness_comparison_v1.py"
SPLITS = ROOT / "benchmarks/heldout/integer_seed_split_v1/splits.json"


def test_five_arm_protocol_is_refused_and_does_not_freeze_a_step_count():
    blob = PROTOCOL.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == PROTOCOL_SHA256
    protocol = json.loads(blob.decode("utf-8"))
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["authorized"] is False
    assert protocol["status"] == "refused"
    assert protocol["optimizer_steps"] == "not_frozen"
    assert protocol["uses_confirm_seeds_for_training"] is False
    assert protocol["minimum_seeds"] == 5
    assert hashlib.sha256(SPLITS.read_bytes()).hexdigest() == protocol["confirm_split_sha256"]
    body = comparison_refusal()
    assert body["authorized"] is False
    assert body["status"] == "refused"
    assert "correctness_reward_rl minus format_sft" in str(body["primary_contrast"])


def test_five_arm_entry_point_exits_2():
    proc = subprocess.run([sys.executable, str(SCRIPT)], check=False, capture_output=True, text=True)
    assert proc.returncode == 2
    body = json.loads(proc.stdout)
    assert body["protocol_id"] == PROTOCOL_ID
    assert body["status"] == "refused"
    assert body["authorized"] is False
    text = SCRIPT.read_text(encoding="utf-8")
    assert "smoke_result" not in text
    assert "train(" not in text
