from __future__ import annotations

import json

import pytest

from scripts.run_agent_edit_interface_systems_v1 import (
    ResourceMonitor,
    TASKS,
    _apply_known_fix,
    changed_opcodes,
    verify_event_chain,
)


class WordCounter:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


@pytest.mark.parametrize("interface", ["lease_bound_lines_v1", "legacy_structured_replace"])
def test_deterministic_controller_applies_multiple_known_hunks(tmp_path, monkeypatch, interface):
    target_name = "src/module.py"
    monkeypatch.setitem(TASKS["mlxlm-2004-jamba-moe-routing"], "target", target_name)
    target = tmp_path / target_name
    target.parent.mkdir()
    base = "alpha = 1\nbeta = 2\ngamma = 3\ndelta = 4\n"
    fixed = "alpha = 10\nbeta = 2\ngamma = 30\ndelta = 4\n"
    target.write_text(base, encoding="utf-8")

    record = _apply_known_fix(
        tmp_path, "mlxlm-2004-jamba-moe-routing", interface,
        fixed, WordCounter(),
    )

    assert target.read_text(encoding="utf-8") == fixed
    assert record["operation_count"] == 2
    assert record["fixed_sha256"] == record["expected_fixed_sha256"]
    assert record["serialized_tool_call_tokens"] > 0
    if interface == "lease_bound_lines_v1":
        assert len(record["lease_audit"]) == 2
        assert all(item["consumed"] for item in record["lease_audit"])


def test_real_task_diff_opcode_counts_are_stable():
    assert len(changed_opcodes("before = 1\n", "before = 2\n")) == 1
    changes = changed_opcodes("a\nb\nc\nd\ne\n", "a\nB\nc\nD\ne\n")
    assert [(item[0], item[1], item[2]) for item in changes] == [
        ("replace", 1, 2), ("replace", 3, 4),
    ]


def test_event_chain_verifier_detects_tampering(tmp_path):
    path = tmp_path / "events.jsonl"
    from scripts.run_local_agent import DurableEventJournal

    journal = DurableEventJournal(path)
    journal.append("edit_committed", path="src.py")
    journal.close()
    count, last_hash = verify_event_chain(path)
    assert count == 2
    assert last_hash == json.loads(path.read_text().splitlines()[-1])["event_sha256"]

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["event"]["path"] = "tampered.py"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(RuntimeError, match="event journal event hash mismatch"):
        verify_event_chain(path)


def test_resource_monitor_durably_samples_system_and_fails_closed_on_preflight(tmp_path):
    limits = {
        "sample_interval_seconds": 0.01,
        "minimum_available_memory_bytes": 1,
        "max_swap_growth_bytes": 2**63 - 1,
        "minimum_free_swap_bytes": 1,
        "minimum_disk_free_bytes": 1,
        "max_process_tree_rss_bytes": 2**63 - 1,
        "maximum_wall_seconds": 2**31 - 1,
    }
    monitor = ResourceMonitor(tmp_path, tmp_path, limits)
    monitor.start()
    import time
    time.sleep(0.02)
    summary = monitor.close()
    rows = [json.loads(line) for line in (tmp_path / "resources.jsonl").read_text().splitlines()]
    assert summary["sample_count"] >= 3
    assert rows[0]["phase"] == "preflight"
    assert rows[-1]["phase"] == "final"
    assert summary["resource_sample_sha256"]
    assert summary["abort_reason"] is None

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    impossible_limits = {**limits, "minimum_available_memory_bytes": 2**63 - 1}
    with pytest.raises(RuntimeError, match="available memory is below"):
        ResourceMonitor(blocked, blocked, impossible_limits)
    assert (blocked / "resources.jsonl").is_file()
