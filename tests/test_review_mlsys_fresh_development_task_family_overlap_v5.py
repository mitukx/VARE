from scripts.review_mlsys_fresh_development_task_family_overlap_v5 import (
    deduplicate,
    disposition_records,
    nested_split_records,
    normalize_split,
    queue_records,
)


class HeldoutIdentityMustNotBeRead(dict):
    def get(self, key: str, default: object = None) -> object:
        if key in {"repository", "pull_request", "task_id", "task_family", "title"}:
            raise AssertionError(f"Held-Out identity field was read: {key}")
        return super().get(key, default)


def test_split_normalization_accepts_only_development_and_train() -> None:
    assert normalize_split("Development") == "development"
    assert normalize_split("train") == "train"
    assert normalize_split("held-out") is None
    assert normalize_split(None) is None


def test_queue_records_discards_heldout_before_identity_extraction() -> None:
    heldout = HeldoutIdentityMustNotBeRead(
        historical_split="heldout", qualification_status="historical_heldout_reserved",
    )
    development = {
        "historical_split": "development",
        "repository": "example/project",
        "pull_request": 2,
        "task_family": "safe-family",
    }

    records = queue_records({"candidates": [heldout, development]}, "fixture")

    assert len(records) == 1
    assert records[0]["task_family"] == "safe-family"


def test_development_train_dispositions_only() -> None:
    records = disposition_records(
        {
            "dispositions": [
                {"original_split": "train", "task_id": "train-1", "task_family": "a"},
                {"original_split": "heldout", "task_id": "must-not-appear"},
            ],
        },
        "fixture",
    )

    assert len(records) == 1
    assert records[0]["task_id"] == "train-1"


def test_disposition_source_pr_is_normalized_after_split_filter() -> None:
    records = disposition_records(
        {
            "dispositions": [
                {
                    "original_split": "development",
                    "source_pr": "https://github.com/example/project/pull/12",
                    "task_id": "dev-12",
                    "task_family": "safe-family",
                },
            ],
        },
        "fixture",
    )

    assert records[0]["repository"] == "example/project"
    assert records[0]["pull_request"] == 12


def test_nested_heldout_parent_stops_before_descendant_identity() -> None:
    heldout = HeldoutIdentityMustNotBeRead(
        split="heldout",
        nested={"task_id": "must-not-appear"},
    )

    assert nested_split_records(heldout, "fixture") == []


def test_nested_development_record_inherits_parent_split() -> None:
    records = nested_split_records(
        {
            "split_assignment": {"split": "development"},
            "task": {"task_id": "dev-1", "task_family": "safe-family"},
        },
        "fixture",
    )

    assert len(records) == 1
    assert records[0]["task_id"] == "dev-1"


def test_duplicate_records_merge_metadata_without_repeating_a_task() -> None:
    rows = [
        {"split": "development", "repository": "example/project", "pull_request": 4, "task_id": "task-4", "source": "ledger-a"},
        {"split": "development", "repository": "example/project", "pull_request": 4, "task_id": "task-4", "task_family": "family-4", "source": "ledger-b"},
    ]

    unique = deduplicate(rows)

    assert len(unique) == 1
    assert unique[0]["task_family"] == "family-4"
    assert set(unique[0]["duplicate_sources"]) == {"ledger-a", "ledger-b"}
