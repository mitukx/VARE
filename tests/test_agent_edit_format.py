"""Edit-format parsing stays separate from task success."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from vare.agent_edit_format import (
    EditFormatError,
    ParseFailure,
    apply_parsed_edit,
    classify_outcome,
    inference_arms,
    parse_submission,
    pass_at_k,
    per_task_reward_variance,
)


DIFF = """--- a/solve.py
+++ b/solve.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


def _repo(tmp_path: Path, source: str) -> Path:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "solve.py").write_text(source, encoding="utf-8")
    return tmp_path


def test_unified_diff_accepts_prose_and_a_fence() -> None:
    text = "I will patch the file.\n```diff\n" + DIFF + "```\nThanks.\n"
    parsed = parse_submission(text, edit_format="unified_diff")
    assert not isinstance(parsed, ParseFailure)
    assert parsed.paths == ("solve.py",)
    assert "return a + b" in (parsed.diff_text or "")


def test_unified_diff_rejects_a_whole_file_submission() -> None:
    text = "```python path=solve.py\ndef add(a, b):\n    return a + b\n```\n"
    parsed = parse_submission(text, edit_format="unified_diff")
    assert isinstance(parsed, ParseFailure)
    assert parsed.reason == "wrong_format"


def test_whole_file_accepts_a_fenced_path_and_json() -> None:
    fenced = parse_submission(
        "```python path=solve.py\ndef add(a, b):\n    return a + b\n```\n",
        edit_format="whole_file",
    )
    assert not isinstance(fenced, ParseFailure)
    assert fenced.files[0].content == "def add(a, b):\n    return a + b\n"
    packed = parse_submission(
        '{"path": "solve.py", "content": "def add(a, b):\\n    return a + b\\n"}',
        edit_format="whole_file",
    )
    assert not isinstance(packed, ParseFailure)
    assert packed.paths == ("solve.py",)


def test_whole_file_rejects_a_unified_diff() -> None:
    parsed = parse_submission(DIFF, edit_format="whole_file")
    assert isinstance(parsed, ParseFailure)
    assert parsed.reason == "wrong_format"


def test_plain_prose_is_not_an_edit_attempt() -> None:
    parsed = parse_submission("I am not sure what to change.", edit_format="unified_diff")
    assert isinstance(parsed, ParseFailure)
    assert parsed.reason == "no_edit"


def test_broken_diff_is_a_parse_failure() -> None:
    parsed = parse_submission("--- a/solve.py\n+++ b/solve.py\nnot a hunk\n", edit_format="unified_diff")
    assert isinstance(parsed, ParseFailure)
    assert parsed.reason == "malformed_diff"


def test_apply_unified_diff_and_whole_file(tmp_path: Path) -> None:
    original = "def add(a, b):\n    return a - b\n"
    _repo(tmp_path, original)
    parsed = parse_submission(DIFF, edit_format="unified_diff")
    assert not isinstance(parsed, ParseFailure)
    applied = apply_parsed_edit(tmp_path, parsed, {"solve.py"})
    assert applied.applied is True
    assert "return a + b" in (tmp_path / "solve.py").read_text(encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    (other / "solve.py").write_text(original, encoding="utf-8")
    whole = parse_submission(
        "FILE: solve.py\n```python\ndef add(a, b):\n    return 1\n```\n",
        edit_format="whole_file",
    )
    assert not isinstance(whole, ParseFailure)
    written = apply_parsed_edit(other, whole, {"solve.py"})
    assert written.applied is True
    assert (other / "solve.py").read_text(encoding="utf-8") == "def add(a, b):\n    return 1\n"


def test_apply_rejects_paths_outside_the_allowlist(tmp_path: Path) -> None:
    _repo(tmp_path, "x = 1\n")
    (tmp_path / "secret.txt").write_text("nope\n", encoding="utf-8")
    parsed = parse_submission(
        "```python path=secret.txt\npwned\n```\n",
        edit_format="whole_file",
    )
    assert not isinstance(parsed, ParseFailure)
    result = apply_parsed_edit(tmp_path, parsed, {"solve.py"})
    assert result.applied is False
    assert (tmp_path / "secret.txt").read_text(encoding="utf-8") == "nope\n"


def test_outcome_taxonomy_keeps_parse_failure_off_the_task_score() -> None:
    assert classify_outcome(
        task_passed=False, stop_reason="no_tool_call_retries_exhausted",
        source_changed=False, edit_attempted=False, edit_applied=False,
        format_parse_failure=None, timed_out=False,
    ) == "no_tool_call"
    assert classify_outcome(
        task_passed=False, stop_reason="malformed_edit",
        source_changed=False, edit_attempted=True, edit_applied=False,
        format_parse_failure="malformed_diff", timed_out=False,
    ) == "malformed_edit"
    assert classify_outcome(
        task_passed=False, stop_reason="agent_finished",
        source_changed=True, edit_attempted=True, edit_applied=True,
        format_parse_failure=None, timed_out=False,
    ) == "edit_applied_but_wrong"
    assert classify_outcome(
        task_passed=False, stop_reason="wall_time_exhausted",
        source_changed=False, edit_attempted=False, edit_applied=False,
        format_parse_failure=None, timed_out=True,
    ) == "timeout"
    assert classify_outcome(
        task_passed=True, stop_reason="agent_finished",
        source_changed=True, edit_attempted=True, edit_applied=True,
        format_parse_failure=None, timed_out=False,
    ) == "pass"


def test_per_task_reward_variance_and_pass_at_k() -> None:
    rows = [
        {"task_id": "a", "reward": 0.0, "task_passed": False},
        {"task_id": "a", "reward": 1.0, "task_passed": True},
        {"task_id": "b", "reward": 0.0, "task_passed": False},
        {"task_id": "b", "task_passed": False},
    ]
    variance = per_task_reward_variance(rows)
    assert variance["a"]["n"] == 2
    assert variance["a"]["mean"] == 0.5
    assert variance["a"]["sample_variance"] == 0.5
    assert variance["b"]["n"] == 1
    assert variance["b"]["sample_variance"] is None
    assert variance["b"]["missing_reward"] == 1
    summary = pass_at_k(rows)
    assert summary["tasks_with_any_success"] == 1
    assert summary["pass_at_k"] == 0.5


def test_v2_protocol_locks_one_format_and_the_sampling_arm() -> None:
    protocol = json.loads(
        (Path(__file__).resolve().parents[1] / "protocols" / "separated_coding_eval_cpu_v2.lock.json").read_text(
            encoding="utf-8",
        )
    )
    assert protocol["status"] == "preregistered; not run"
    assert "secondary_if_feasible" not in protocol["models"]
    assert set(protocol["pack"]["train"]).isdisjoint(protocol["pack"]["heldout_not_used"])
    arms = inference_arms(protocol)
    assert [arm["name"] for arm in arms] == ["greedy", "sample_pass8"]
    assert {arm["edit_format"] for arm in arms} == {"unified_diff"}
    assert arms[0]["seeds"] == [101]
    assert arms[0]["do_sample"] is False
    assert arms[0]["temperature"] == 0
    assert arms[1]["seeds"] == [101, 102, 103, 104, 105, 106, 107, 108]
    assert arms[1]["temperature"] == 0.7
    assert arms[1]["top_p"] == 0.95
    assert arms[1]["top_k"] == 20
    command = protocol["local_measurement"]["command"]
    assert "protocols/separated_coding_eval_cpu_v2.lock.json" in command
    assert "$HOME/vare-mlsys-c8c0488" in command
    assert "$HOME/vare-eval/separated-coding-eval-cpu-v2" in command
    assert "/tmp" not in command


def test_inference_arms_share_one_edit_format() -> None:
    protocol = {
        "inference": {
            "edit_format": "whole_file",
            "arms": [
                {"name": "greedy", "temperature": 0, "do_sample": False, "seeds": [101]},
                {
                    "name": "sample_pass8", "temperature": 0.7, "do_sample": True,
                    "top_p": 0.95, "top_k": 20, "seeds": [101, 102],
                },
            ],
        },
    }
    arms = inference_arms(protocol)
    assert [arm["edit_format"] for arm in arms] == ["whole_file", "whole_file"]
    assert arms[0]["do_sample"] is False
    assert arms[1]["seeds"] == [101, 102]
    assert arms[1]["top_p"] == 0.95
    with pytest.raises(EditFormatError):
        inference_arms({"inference": {"edit_format": "mixed", "arms": [{"name": "a", "seeds": [1]}]}})
