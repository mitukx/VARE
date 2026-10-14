from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vare.format_vs_correctness_v2 import (
    confirm_eval_tasks,
    contrast_decision,
    iter_eval_units,
    iter_train_units,
    pack_eval_tasks,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/format-vs-correctness-comparison-v2/m1-mlx-transcribed/operator_report.json"
PROTOCOL = ROOT / "protocols/format_vs_correctness_comparison_v2.json"
PACK = ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"


def test_transcribed_mlx_result_is_the_null_interval() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["artifact_status"] == "transcribed_operator_report"
    assert payload["original_summary_json_in_repo"] is False
    assert payload["original_progress_jsonl_in_repo"] is False
    assert payload["summary_json_sha256"] is None
    assert payload["progress_jsonl_sha256"] is None
    assert payload["code_revision"] is None
    assert payload["decision"] == "interval_includes_zero"
    assert payload["promotes_weights"] is False
    assert payload["authorizes_more_steps"] is False
    rates = payload["confirm_pass_at_1"]
    assert rates == {
        "base": 0.2523,
        "format_sft": 0.2594,
        "format_reward_rl": 0.2625,
        "correctness_reward_rl": 0.2641,
        "random_reward_rl": 0.2625,
    }
    bootstrap = payload["paired_bootstrap_correctness_minus_format_sft"]
    assert bootstrap == {"mean": 0.0047, "low": -0.0008, "high": 0.0102, "n": 64}
    assert contrast_decision(bootstrap["low"], bootstrap["high"]) == "interval_includes_zero"
    assert payload["mcnemar_discordant_pairs"] == 0
    assert payload["format_compliance_at_base"] == 1.0
    assert payload["format_arms_saturated"] is True
    assert payload["reported_rows"] == {"eval": 17920, "train": 320}
    confirm = confirm_eval_tasks()
    pack = pack_eval_tasks(PACK.read_text(encoding="utf-8"))
    assert len(iter_train_units()) == 320
    assert len(iter_eval_units(confirm, pack)) == 17920
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert digest == "8214a9c784035f47b20d9e44fc6bbbd3cef45f821d86d6eb2f69936cc25eb6b4"
    assert payload["protocol_sha256"] == digest
    assert not (ROOT / "results/format-vs-correctness-comparison-v2/m1-mlx-transcribed/progress.jsonl").exists()
    assert not (ROOT / "results/format-vs-correctness-comparison-v2/m1-mlx-transcribed/summary.json").exists()
