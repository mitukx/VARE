from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from discover_mlsys_fresh_development_candidates_v4 import load_frozen_protocol, screen_candidate


def _protocol():
    return {
        "discovery": {
            "merged_date_start": "2026-10-31",
            "merged_date_end": "2026-10-31",
            "snapshot_cutoff_utc": "2026-10-31T03:35:14Z",
            "model_training_cutoff_proxy": "2025-08-09",
        },
        "coarse_screen": {
            "maximum_changed_lines": 250,
            "maximum_changed_files": 8,
            "specialized_title_markers": ["gpu", "distributed"],
        },
    }


def _candidate():
    return {
        "repository": "example/project",
        "number": 12,
        "title": "Fix metric cache reset",
        "state": "MERGED",
        "merged_at": "2026-10-31T02:00:00Z",
        "merge_commit_sha": "a" * 40,
        "merge_base_sha": "b" * 40,
        "changed_paths_complete": True,
        "changed_paths": ["src/cache.py", "tests/test_cache.py"],
        "additions": 4,
        "deletions": 2,
        "changed_file_count": 2,
    }


def test_precollection_addendum_binds_the_corrected_collector():
    protocol, addendum, addendum_2 = load_frozen_protocol()

    assert protocol["protocol_id"] == "mlsys-fresh-development-candidate-discovery-v4"
    assert addendum["protocol_id"] == "mlsys-fresh-development-candidate-discovery-v4-addendum-1"
    assert addendum_2["protocol_id"] == "mlsys-fresh-development-candidate-discovery-v4-addendum-2"
    assert addendum["pre_collection_failure"]["metadata_queries_started"] is False
    assert addendum_2["pre_collection_failure"]["metadata_queries_started"] is False


def test_graphql_merged_state_can_enter_metadata_shortlist():
    result = screen_candidate(_candidate(), protocol=_protocol(), prior=set())

    assert result["coarse_screen"] == "provisional_for_manual_review"
    assert result["qualification_status"] == "UNQUALIFIED"
    assert result["split_assignment"] == "UNASSIGNED"


def test_rest_closed_state_is_not_mistaken_for_graphql_merged_state():
    row = _candidate()
    row["state"] = "CLOSED"

    result = screen_candidate(row, protocol=_protocol(), prior=set())

    assert result["coarse_screen"] == "excluded_from_shortlist"
    assert "not_merged_or_missing_merge_commit" in result["screen_reasons"]


def test_snapshot_cutoff_and_exact_exposure_fail_closed():
    row = _candidate()
    row["merged_at"] = "2026-10-31T04:00:00Z"

    after_cutoff = screen_candidate(row, protocol=_protocol(), prior=set())
    already_exposed = screen_candidate(
        _candidate(), protocol=_protocol(), prior={("example/project", 12)},
    )

    assert "merged_after_frozen_snapshot_cutoff" in after_cutoff["screen_reasons"]
    assert "exact_identity_already_in_development_train_or_prior_queue" in already_exposed["screen_reasons"]
