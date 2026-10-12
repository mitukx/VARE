from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from vare.heldout_exact_integer import (
    FINAL_LINE,
    PROTOCOL_ID_V2,
    grade_output,
    load_task_pack,
    parse_final_integer,
    sample_seed,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/heldout-exact-integer-base-gate-v1/m1-mlx-5d36336"
RESULT_V2 = ROOT / "results/heldout-exact-integer-base-gate-v2/m1-mlx-89f671c"
PACK = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
PROTOCOL_V1 = ROOT / "benchmarks/heldout/exact_integer_v1/protocol.json"
PROTOCOL_V2 = ROOT / "benchmarks/heldout/exact_integer_v2/protocol.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_protocol_and_result_bytes_stay_frozen():
    assert _sha(PROTOCOL_V1) == "21c959a86423505a68fa53969837b883989857072a599110e08ee21ee5bf5276"
    assert _sha(PACK) == "189b5e482b0f8c2b40ff36e0e1a84827699f348c11981a0ff912ce67323fd7fc"
    manifest = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    assert _sha(RESULT / "manifest.json") == "349d93ca5f3863d3e3c6bb91ba6370bb5b9eb316e90bdd9d5f2724e1b5a10b5e"
    for name, spec in manifest["files"].items():
        blob = (RESULT / name).read_bytes()
        assert len(blob) == spec["bytes"] < 1_048_576
        assert hashlib.sha256(blob).hexdigest() == spec["sha256"]
    assert manifest["status"] == "stop_zero"
    assert manifest["passed_tasks"] == 0
    assert manifest["final_line_parser_matches"] == 3
    assert manifest["graded_correct_samples"] == 0


def test_retained_v1_samples_match_the_frozen_grader():
    tasks = {row["task_id"]: row for row in load_task_pack(PACK.read_text(encoding="utf-8"))}
    rows = [json.loads(line) for line in (RESULT / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 256
    parsed = 0
    correct = 0
    for row in rows:
        task = tasks[row["task_id"]]
        assert grade_output(task["expression"], row["output"]) is row["correct"]
        correct += int(row["correct"])
        parsed += parse_final_integer(row["output"]) is not None
        task_index = int(row["task_id"].rsplit("-", 1)[1])
        assert row["seed"] == sample_seed(task_index, int(row["sample_index"]))
    assert parsed == 3
    assert correct == 0
    summary = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "stop_zero"
    assert summary["wilson95_low"] == 0.0
    assert summary["wilson95_high"] == 0.05662405979280534


def test_v2_protocol_is_frozen_and_reuses_the_v1_pack():
    protocol = json.loads(PROTOCOL_V2.read_text(encoding="utf-8"))
    assert _sha(PROTOCOL_V2) == "8c52c9cdff4403c15f43e4c617fc0e082e3a2a2a71ae78719fd5495d2534b85c"
    assert protocol["protocol_id"] == PROTOCOL_ID_V2
    assert protocol["frozen_before_model_output"] is True
    assert protocol["tasks_sha256"] == _sha(PACK)
    assert protocol["answer_extraction"]["regex"] == FINAL_LINE.pattern
    assert protocol["v1_decision"] == "stop_zero"
    assert protocol["models"]["mlx"]["revision"] == "4f83f8f146fdf28b512a06562b671d7af4fab457"
    assert protocol["models"]["transformers"]["revision"] == "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def test_v2_mlx_result_bytes_stay_frozen():
    manifest = json.loads((RESULT_V2 / "manifest.json").read_text(encoding="utf-8"))
    assert _sha(RESULT_V2 / "manifest.json") == "ee582b61fc2ca21d45bacb829c728bf47118d987f302b6f0f2eb8aeb82571278"
    for name, spec in manifest["files"].items():
        blob = (RESULT_V2 / name).read_bytes()
        assert len(blob) == spec["bytes"] < 1_048_576
        assert hashlib.sha256(blob).hexdigest() == spec["sha256"]
    assert manifest["status"] == "nondegenerate_base_signal"
    assert manifest["passed_tasks"] == 29
    assert manifest["pass_rate"] == 0.453125
    assert manifest["wilson95_low"] == 0.3372944778061167
    assert manifest["wilson95_high"] == 0.5742640277994588
    assert manifest["final_line_parser_matches"] == 250
    assert manifest["graded_correct_samples"] == 109
    assert manifest["code_revision"] == "89f671c00e7ed84fda1fc873e8814e0ff17bef2c"


def test_retained_v2_samples_match_the_frozen_grader():
    tasks = {row["task_id"]: row for row in load_task_pack(PACK.read_text(encoding="utf-8"))}
    rows = [json.loads(line) for line in (RESULT_V2 / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 256
    parsed = 0
    correct = 0
    by_task: dict[str, list[bool]] = {}
    for row in rows:
        task = tasks[row["task_id"]]
        got = grade_output(task["expression"], row["output"])
        assert got is row["correct"]
        correct += int(got)
        parsed += parse_final_integer(row["output"]) is not None
        by_task.setdefault(row["task_id"], []).append(got)
        task_index = int(row["task_id"].rsplit("-", 1)[1])
        assert row["seed"] == sample_seed(task_index, int(row["sample_index"]))
    assert parsed == 250
    assert correct == 109
    assert sum(any(flags) for flags in by_task.values()) == 29
    summary = json.loads((RESULT_V2 / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "nondegenerate_base_signal"
    assert summary["model_id"] == "mlx-community/Qwen2.5-3B-Instruct-4bit"


def test_v2_update_comparison_is_refused():
    script = ROOT / "scripts/heldout_exact_integer_update_comparison_v2.py"
    proc = subprocess.run([sys.executable, str(script)], check=False, capture_output=True, text=True)
    assert proc.returncode == 2
    body = json.loads(proc.stdout)
    assert body["protocol_id"] == PROTOCOL_ID_V2
    assert body["status"] == "refused"
