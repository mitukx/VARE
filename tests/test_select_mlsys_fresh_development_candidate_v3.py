from scripts.select_mlsys_fresh_development_candidate_v3 import (
    eligible_rows,
    rank_and_exclude,
)


def row(number: int, **updates: object) -> dict[str, object]:
    base: dict[str, object] = {
        "repository": "example/project",
        "number": number,
        "coarse_screen": "provisional_for_manual_review",
        "screen_reasons": [],
        "changed_paths_complete": True,
        "changed_paths": ["src/a.py", "tests/test_a.py"],
        "test_paths": ["tests/test_a.py"],
        "state": "MERGED",
        "qualification_status": "UNQUALIFIED",
        "split_assignment": "UNASSIGNED",
        "task_family_overlap": "NOT_ASSESSED",
    }
    base.update(updates)
    return base


def test_eligibility_requires_all_frozen_metadata_gates() -> None:
    candidates = [
        row(1),
        row(2, screen_reasons=["specialized"]),
        row(3, changed_paths_complete=False),
        row(4, split_assignment="DEVELOPMENT"),
        row(5, test_paths=["tests/outside.py"]),
    ]

    assert eligible_rows({"candidates": candidates}) == [candidates[0]]


def test_rank_is_deterministic_and_explicit_exclusions_are_exact() -> None:
    candidates = [row(1), row(2, repository="other/project")]

    ranked = rank_and_exclude(candidates, {("example/project", 1)}, "frozen-domain")

    assert len(ranked) == 1
    assert ranked[0]["repository"] == "other/project"
    assert ranked[0]["selection_rank_sha256"] == (
        "9b9c2480f61f9ca5d469ab82243a0c8ed580329190b5c9d6e39468fc4d1fa63e"
    )


def test_empty_inventory_stays_empty_after_exclusions() -> None:
    assert rank_and_exclude([], set(), "frozen-domain") == []
