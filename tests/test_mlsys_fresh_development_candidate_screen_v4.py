from __future__ import annotations

import scripts.screen_mlsys_fresh_development_candidate_pool_v4 as screening


def candidate(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repository": "ml-explore/mlx-lm",
        "number": 9999,
        "title": "Fix a narrow cache regression",
        "state": "MERGED",
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
    return screening.screen_candidate(
        row,
        date_start="2026-10-01",
        date_end="2026-10-30",
        cutoff_proxy="2025-08-09",
        prior=prior or set(),
    )


def test_graphql_merged_state_passes_the_metadata_screen_only() -> None:
    result = screen(candidate())

    assert result["coarse_screen"] == "provisional_for_manual_review"
    assert result["qualification_status"] == "UNQUALIFIED"
    assert result["split_assignment"] == "UNASSIGNED"
    assert result["task_family_overlap"] == "NOT_ASSESSED"


def test_prior_identity_and_rest_closed_state_are_not_reintroduced() -> None:
    prior_result = screen(candidate(), prior={("ml-explore/mlx-lm", 9999)})
    rest_state_result = screen(candidate(state="CLOSED"))

    assert "exact_identity_already_in_vare_candidate_or_exposure_audit" in prior_result["screen_reasons"]
    assert "not_merged_or_missing_merge_commit" in rest_state_result["screen_reasons"]


def test_original_coarse_filters_are_preserved() -> None:
    result = screen(candidate(
        changed_paths=["mlx_lm/cache.py"],
        additions=251,
        title="Add distributed runtime support",
    ))

    assert {
        "no_test_path_in_merged_pr_diff",
        "diff_exceeds_250_changed_lines",
        "title_flags_specialized_or_feature_scope_for_manual_review",
    } <= set(result["screen_reasons"])
