from __future__ import annotations

import pytest

import scripts.discover_mlsys_fresh_development_candidates_v3 as discovery
from scripts.discover_mlsys_fresh_development_candidates_v3 import screen_candidate


def candidate(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repository": "ml-explore/mlx-lm",
        "number": 9999,
        "title": "Fix a narrow cache regression",
        "state": "CLOSED",
        "merged_at": "2026-10-12T12:00:00Z",
        "merge_commit_sha": "a" * 40,
        "merge_base_sha": "b" * 40,
        "changed_paths_complete": True,
        "changed_paths": ["mlx_lm/cache.py", "tests/test_cache.py"],
        "additions": 15,
        "deletions": 3,
        "changed_file_count": 2,
    }
    row.update(updates)
    return row


def screen(row: dict[str, object], *, prior: set[tuple[str, int]] | None = None) -> dict[str, object]:
    return screen_candidate(
        row,
        date_start="2026-10-01",
        date_end="2026-10-30",
        cutoff_proxy="2025-08-09",
        prior_identities=prior or set(),
    )


def test_metadata_candidate_remains_unqualified_and_unassigned() -> None:
    result = screen(candidate())

    assert result["coarse_screen"] == "provisional_for_manual_review"
    assert result["qualification_status"] == "UNQUALIFIED"
    assert result["split_assignment"] == "UNASSIGNED"
    assert result["task_family_overlap"] == "NOT_ASSESSED"


def test_exact_prior_identity_is_not_reintroduced() -> None:
    result = screen(candidate(), prior={("ml-explore/mlx-lm", 9999)})

    assert result["coarse_screen"] == "excluded_from_shortlist"
    assert "exact_identity_already_in_vare_candidate_or_exposure_audit" in result["screen_reasons"]


def test_missing_tests_old_dates_and_large_diffs_are_excluded() -> None:
    result = screen(candidate(
        merged_at="2025-08-09T12:00:00Z",
        changed_paths=["mlx_lm/cache.py"],
        additions=251,
    ))

    assert result["coarse_screen"] == "excluded_from_shortlist"
    assert {
        "merged_outside_frozen_date_window",
        "not_after_model_training_cutoff_proxy",
        "no_test_path_in_merged_pr_diff",
        "diff_exceeds_250_changed_lines",
    } <= set(result["screen_reasons"])


def test_specialized_title_marker_is_retained_as_excluded_metadata() -> None:
    result = screen(candidate(title="Add distributed runtime support"))

    assert result["coarse_screen"] == "excluded_from_shortlist"
    assert "title_flags_specialized_or_feature_scope_for_manual_review" in result["screen_reasons"]


def test_search_paginates_and_keeps_only_pull_request_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        {"data": {"search": {
            "issueCount": 2,
            "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
            "nodes": [{"number": 1, "title": "one"}],
        }}},
        {"data": {"search": {
            "issueCount": 2,
            "pageInfo": {"hasNextPage": False, "endCursor": "cursor-2"},
            "nodes": [{"number": 2, "title": "two"}],
        }}},
    ]
    seen: list[dict[str, str | int]] = []

    def fake_graphql(query: str, variables: dict[str, str | int]) -> dict[str, object]:
        assert "body" not in query
        assert "patch" not in query
        seen.append(variables)
        return responses.pop(0)

    monkeypatch.setattr(discovery, "run_gh_graphql", fake_graphql)
    rows = discovery.search_repo_month("owner/repo", "2026-10-01", "2026-10-30")

    assert [row["number"] for row in rows] == [1, 2]
    assert seen[0] == {"searchText": "repo:owner/repo is:pr is:merged merged:2026-10-01..2026-10-30"}
    assert seen[1]["cursor"] == "cursor-1"


def test_search_rejects_incomplete_result_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        discovery,
        "run_gh_graphql",
        lambda *_: {"data": {"search": {
            "issueCount": 1001,
            "pageInfo": {"hasNextPage": True, "endCursor": "cursor"},
            "nodes": [],
        }}},
    )

    with pytest.raises(ValueError, match="1,000-result ceiling"):
        discovery.search_repo_month("owner/repo", "2026-10-01", "2026-10-30")


def test_frozen_search_windows_cover_the_entire_source_month() -> None:
    protocol = discovery.load_protocol()
    windows = protocol["discovery"]["search_windows"]

    assert windows[0] == {"start": "2026-10-01", "end": "2026-10-07"}
    assert windows[-1] == {"start": "2026-10-29", "end": "2026-10-30"}
