from __future__ import annotations

import pytest

from scripts.sample_mlsys_fresh_development_candidate_metadata_v1 import select_sample


def candidate(repo: str, number: int, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repository": repo,
        "number": number,
        "title": f"Fix {number}",
        "coarse_screen": "provisional_for_manual_review",
        "qualification_status": "UNQUALIFIED",
        "split_assignment": "UNASSIGNED",
        "task_family_overlap": "NOT_ASSESSED",
    }
    row.update(updates)
    return row


def protocol() -> dict[str, object]:
    return {
        "repositories": ["owner/a", "owner/b", "owner/c"],
        "selection": {"sha256_seed": "fixed-test-seed", "sample_size_per_repository": 2},
    }


def test_sample_is_deterministic_stratified_and_unqualified() -> None:
    rows = [candidate("owner/a", n) for n in range(1, 5)]
    rows.extend(candidate("owner/b", n) for n in range(10, 12))
    rows.append(candidate("owner/c", 20, coarse_screen="excluded_from_shortlist"))
    source = {"candidates": rows}

    first, counts = select_sample(source, protocol())
    second, _ = select_sample({"candidates": list(reversed(rows))}, protocol())

    assert counts == {"owner/a": 2, "owner/b": 2, "owner/c": 0}
    assert [(row["repository"], row["number"]) for row in first] == [
        (row["repository"], row["number"]) for row in second
    ]
    assert all(row["qualification_status"] == "UNQUALIFIED" for row in first)
    assert all(row["split_assignment"] == "UNASSIGNED" for row in first)


def test_sampler_rejects_rows_promoted_by_source_screen() -> None:
    source = {"candidates": [candidate("owner/a", 1, qualification_status="QUALIFIED")]}

    with pytest.raises(ValueError, match="promoted a metadata candidate"):
        select_sample(source, protocol())
