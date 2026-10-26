from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from scripts import run_agent_edit_interface_systems_v1 as systems
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


def test_execute_cell_records_one_terminal_completion_event(tmp_path, monkeypatch):
    task_id = "mlxlm-2004-jamba-moe-routing"
    target_name = "src/module.py"
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "test"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    source_target = source / target_name
    source_target.parent.mkdir()
    base_text = "def value():\n    return 1\n"
    fixed_text = "def value():\n    return 2\n"
    source_target.write_text(base_text, encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", target_name], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    source_target.write_text(fixed_text, encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", target_name], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fix"], check=True)
    fix = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    workspace = tmp_path / "workspace"
    subprocess.run(["git", "clone", "-q", str(source), str(workspace)], check=True)
    subprocess.run(["git", "-C", str(workspace), "checkout", "-q", "--detach", base], check=True)
    monkeypatch.setitem(TASKS[task_id], "base", base)
    monkeypatch.setitem(TASKS[task_id], "fix", fix)
    monkeypatch.setitem(TASKS[task_id], "target", target_name)
    monkeypatch.setattr(systems, "_grade", lambda *_args: {
        "command": [], "network_denied": True, "grader_sha256": "0" * 64,
        "returncode": 0, "passed": True, "stdout": "", "stderr": "",
        "stdout_sha256": systems.sha256_bytes(b""),
        "stderr_sha256": systems.sha256_bytes(b""), "elapsed_seconds": 0.0,
    })

    class NoAbort:
        abort_reason = None

        def check(self):
            return None

    output = tmp_path / "cell"
    output.mkdir()
    systems.execute_cell(
        source_repo=source, workspace=workspace, output=output, task_id=task_id,
        interface="lease_bound_lines_v1", tokenizer=WordCounter(),
        venv_python=Path("/usr/bin/python3"), queue_submitted_at=time.monotonic(),
        monitor=NoAbort(),
    )
    records = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    events = [row["event"] for row in records]
    assert sum(event["event_type"] == "known_fix_controller_complete" for event in events) == 1
    assert sum(event["event_type"] == "deterministic_tool_action_committed" for event in events) == 1
