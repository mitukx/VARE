"""Lock and hash checks for the frozen Development repair protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_local_agent import tools_for

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "protocols/development_repository_repair_campaign_v1.json"
LOCK_PATH = ROOT / "protocols/development_repository_repair_campaign_v1.lock.json"
EXPECTED_SHA256 = (
    "9239406621a3ac68a6977cf341e4debd35278735a91783626ff764c87210c354"
)
INCLUDED = (
    "transformers-49197",
    "transformers-49842",
    "transformers-49900",
    "transformers-49703",
    "trl-7913",
    "mlx-lm-1777",
)
TAXONOMY = (
    "poor_decision",
    "missing_context",
    "tool_schema",
    "parser_error",
    "budget",
    "infrastructure",
    "incorrect_patch",
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()


def load_pair() -> tuple[dict, dict]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return protocol, lock


def test_lock_sha256_matches_canonical_protocol():
    protocol, lock = load_pair()
    recorded = lock.pop("sha256")
    digest = hashlib.sha256(canonical(protocol)).hexdigest()
    assert recorded == digest == EXPECTED_SHA256
    assert canonical(lock) == canonical(protocol)


def test_tampered_protocol_does_not_match_the_lock():
    protocol, lock = load_pair()
    protocol["decoding"]["attempts_per_family"] = 3
    digest = hashlib.sha256(canonical(protocol)).hexdigest()
    assert digest != lock["sha256"]


def test_campaign_is_frozen_development_only():
    protocol, _lock = load_pair()
    assert protocol["protocol_id"] == "development-repository-repair-campaign-v1"
    assert protocol["status"] == "frozen_before_inference"
    assert protocol["frozen_before_model_execution"] is True
    assert protocol["inference_authorized"] is False
    assert protocol["fresh_task_evidence"] is False
    assert protocol["held_out_tasks_used"] is False
    assert protocol["held_out_task_ids"] == []
    assert protocol["runner"]["require_split_flag"] == "must not be passed"
    assert protocol["runner"]["do_not_modify_runner"] is True
    output = protocol["output"]["directory"]
    assert output == "results/development-repository-repair-campaign-v1"
    assert not (ROOT / output).exists()
    assert protocol["reporting"]["synthetic_control_success_counts_as_model_success"] is False
    assert protocol["reporting"]["one_family_success_is_feasibility_milestone_only"] is True
    assert protocol["reporting"]["unique_family_success_reported_separately"] is True
    assert protocol["reporting"]["attempt_level_success_reported_separately"] is True


def test_model_budgets_and_fail_closed_limits():
    protocol, _lock = load_pair()
    model = protocol["model"]
    assert model["repository"] == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    assert model["revision"] == "019cc73c45c770444708a6dd8690c66243cc5c80"
    assert model["device"] == "Apple Metal through MLX"
    assert model["weights_must_already_be_cached"] is True
    assert model["download_permitted"] is False
    tokenizer = model["tokenizer"]
    assert tokenizer["tokenizer_json_sha256"] == (
        "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf"
    )
    assert model["snapshot_file_sha256"]["tokenizer.json"] == (
        tokenizer["tokenizer_json_sha256"]
    )
    assert model["snapshot_file_sha256"]["model.safetensors"] == (
        "56a3d94706833f753e6c6b47ea57af8ef638cd8cc1d74eca1142ba640b26060e"
    )
    decoding = protocol["decoding"]
    assert decoding["do_sample"] is False
    assert decoding["temperature"] == 0
    assert decoding["seeds"] == [101, 211]
    assert decoding["attempts_per_family"] == 2
    budgets = protocol["budgets"]
    assert budgets["max_context_tokens"] == 8192
    assert budgets["max_new_tokens_per_turn"] == 512
    assert budgets["max_total_new_tokens"] == 4096
    assert budgets["max_turns"] == 12
    assert budgets["wall_seconds"] == 900
    assert budgets["process_lifetime_seconds"] == 900
    limits = protocol["resource_limits"]
    assert limits["max_mlx_peak_memory_bytes"] == 8589934592
    assert limits["max_process_rss_bytes"] == 10737418240
    assert limits["max_swap_growth_bytes"] == 33554432
    assert limits["minimum_available_memory_bytes"] == 2147483648
    assert limits["minimum_free_swap_bytes"] == 536870912
    assert "29efdbab55a161237ab1e432a3abaf6c7ae2b477" in (
        model["selection_justification"]["rejected_cached_checkpoint"]
    )


def test_frozen_tool_allowlist_remains_valid_as_runner_grows():
    protocol, _lock = load_pair()
    names = [
        item["function"]["name"]
        for item in tools_for("unified_diff", public_tests=True)
    ]
    # The campaign is hash-locked and not authorized for inference. The runner's
    # later exact-text editing tool must not silently expand this frozen contract.
    assert "replace_text" in names
    assert protocol["tool_schema"]["allowlist"] == [
        name for name in names if name != "replace_text"
    ]
    steps = [item["step"] for item in protocol["tool_schema"]["required_progression"]]
    assert steps == [
        "inspect",
        "search",
        "read",
        "edit",
        "test_or_syntax_check",
        "inspect_diff",
        "repair_or_finish",
    ]
    assert protocol["runner"]["diff_function"] == "candidate_source_diff"
    assert protocol["runner"]["edit_format"] == "unified_diff"
    assert protocol["failure_taxonomy"] == list(TAXONOMY)
    required = {
        "model_repository",
        "model_revision",
        "tokenizer_sha256",
        "generated_token_count",
        "tool_calls",
        "inspections",
        "applied_edits",
        "rejected_edits",
        "public_test_runs",
        "candidate_diff",
        "candidate_diff_sha256",
        "grader_decision",
        "wall_seconds",
        "mlx_peak_memory_bytes",
        "process_rss_bytes",
        "swap_growth_bytes",
        "failure_reason",
        "prior_exposure",
    }
    assert required <= set(protocol["attempt_record_fields"])


def test_included_families_are_pinned_and_not_fresh():
    protocol, _lock = load_pair()
    families = protocol["families"]
    assert [family["family_id"] for family in families] == list(INCLUDED)
    scheduled = set()
    for family in families:
        assert family["fresh"] is False
        assert "not fresh" in family["prior_exposure"]
        controls = family["controls"]
        assert controls["baseline"] == "FAIL"
        assert controls["known_fix"] == "PASS"
        assert controls["incorrect_mutant"] == "FAIL"
        for key in (
            "upstream_baseline_commit",
            "workspace_baseline_commit",
            "fix_commit",
        ):
            assert len(family[key]) == 40
            assert family[key] != "HEAD"
        for key in ("baseline_source_sha256", "grader_sha256"):
            assert len(family[key]) == 64
        scheduled.add(family["task_id"])
    assert protocol["families"][1]["counted_once"] is True
    excluded_ids = {row["task_id"] for row in protocol["excluded_families"]}
    duplicate_ids = {
        row["task_id"] for row in protocol["duplicate_rows_not_separate_families"]
    }
    assert "transformers-video-short-fps-min-frame-20261027-v1" in excluded_ids
    assert protocol["excluded_families"][0]["reason"] == "no_immutable_fix_commit"
    assert excluded_ids.isdisjoint(scheduled)
    assert duplicate_ids.isdisjoint(scheduled)
    assert "transformers-musicgen-melody-cached-position-20261027-v1" in duplicate_ids
    assert (
        "transformers-49842-musicgen-melody-long-cache-position-v1" in duplicate_ids
    )
