import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_cpu_generated_arithmetic_feasibility_v1 as audit_v1
import generate_compositional_arithmetic_feasibility_pilot as generator
import run_cpu_generated_arithmetic_feasibility_v1 as runner


def test_pilot_data_matches_frozen_generator_and_hash():
    rows_path = ROOT / "data/cpu-arithmetic-feasibility-v1/pilot.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    assert rows == generator.build_rows()
    assert len(rows) == 64
    assert len({row["prompt"] for row in rows}) == 64
    for row in rows:
        assert audit_v1.evaluate_oracle(row["expression"]) == row["oracle_answer"]


def test_strict_integer_parser_agrees_between_runner_and_auditor():
    for text, expected in (("FINAL: 12", 12), (" final: -4\n", -4),
                           ("FINAL: 1,200", None), ("FINAL: 4\nwork", None),
                           ("answer: 2", None), ("FINAL: 3 4", None)):
        assert runner.parse_answer(text) == expected
        assert audit_v1.strict_parse(text) == expected


def test_oracle_rejects_non_integer_or_unsupported_expressions():
    with pytest.raises(ValueError):
        audit_v1.evaluate_oracle("__import__('os').system('false')")
    with pytest.raises(ValueError):
        audit_v1.evaluate_oracle("1 / 0")


def test_protocol_lock_and_asset_hashes_are_consistent():
    spec, protocol_hash = runner.load_spec()
    lock = json.loads((ROOT / "protocols/cpu_generated_arithmetic_feasibility_v1.lock.json").read_text())
    assert lock["sha256"] == protocol_hash
    assert spec["data"]["pilot_rows_sha256"] == audit_v1.sha256_file(ROOT / spec["data"]["pilot_rows_path"])
    assert spec["data"]["generator_sha256"] == audit_v1.sha256_file(ROOT / spec["data"]["generator_path"])
