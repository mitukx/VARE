"""Validate the prospective contract against the actual runner tool builder."""

import hashlib
import json
from pathlib import Path
from scripts.run_local_agent import tools_for

ROOT = Path(__file__).parents[1]


def test_v1_is_preserved_and_v2_is_hash_bound():
    p = ROOT / "protocols/development_repository_repair_campaign_v2.json"
    data = json.loads(p.read_text())
    old = data["supersedes_for_future_execution"]
    assert (
        hashlib.sha256((ROOT / old["path"]).read_bytes()).hexdigest() == old["sha256"]
    )
    assert not old["historical_protocol_modified"]
    assert (
        p.with_suffix(".json.sha256").read_text().split()[0]
        == hashlib.sha256(p.read_bytes()).hexdigest()
    )
    assert not data["inference_authorized"]
    assert not data["fresh_task_evidence"]


def test_each_family_exposes_only_executable_validation_tools():
    data = json.loads(
        (ROOT / "protocols/development_repository_repair_campaign_v2.json").read_text()
    )
    for family in data["families"]:
        contract = data["family_tool_contracts"][family["family_id"]]
        available = contract["public_tests"]
        actual = [
            t["function"]["name"]
            for t in tools_for("unified_diff", public_tests=bool(available))
        ]
        assert contract["allowlist"] == actual
        assert ("run_public_tests" in actual) == (available is True)
        assert contract["public_test_container_required"] == (available is True)
        if available is None:
            assert (
                contract["execution_status"] == "BLOCKED_PUBLIC_TEST_METADATA_UNKNOWN"
            )
        elif available:
            assert len(contract["public_test_sha256"]) == 64
        else:
            assert contract["validation_tools"] == ["check_syntax"]
    trl = data["family_tool_contracts"]["trl-7913"]
    assert trl["public_tests"] is False
    assert "run_public_tests" not in trl["allowlist"]
