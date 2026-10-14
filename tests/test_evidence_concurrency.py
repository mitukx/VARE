"""CPU-only durability and competing-writer checks for the evidence ledger."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from vare.evidence import HashChainLedger


def _write_records(args: tuple[str, int, int]) -> None:
    path, writer, count = args
    ledger = HashChainLedger(path)
    for i in range(count):
        ledger.append("concurrent", {"writer": writer, "index": i})


def test_two_stale_instances_append_using_current_head(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = HashChainLedger(path)
    second = HashChainLedger(path)
    first.append("one", {"x": 1})
    second.append("two", {"x": 2})
    first.append("three", {"x": 3})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["seq"] for row in rows] == [0, 1, 2]
    assert HashChainLedger(path).seq == 3


def test_competing_processes_preserve_every_record(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(_write_records, [(str(path), writer, 12) for writer in range(4)]))
    recovered = HashChainLedger(path)
    assert recovered.seq == 48
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted((row["payload"]["writer"], row["payload"]["index"]) for row in rows) == [
        (writer, index) for writer in range(4) for index in range(12)
    ]


def test_incomplete_tail_is_not_silently_accepted(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = HashChainLedger(path)
    ledger.append("valid", {"x": 1})
    with path.open("a") as handle:
        handle.write('{"seq":1')
    with pytest.raises(ValueError, match="incomplete"):
        ledger.append("another", {"x": 2})
    with pytest.raises(ValueError, match="incomplete"):
        HashChainLedger(path)


def test_sequence_tampering_fails_closed_even_after_rehash(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = HashChainLedger(path)
    ledger.append("valid", {"x": 1})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["seq"] = 77
    from vare.evidence import sha256_json
    rows[0]["sha256"] = sha256_json({k: v for k, v in rows[0].items() if k != "sha256"})
    path.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="sequence"):
        HashChainLedger(path)
