from scripts.select_mlsys_fresh_development_candidate_v2 import (
    exclude_identities,
    frozen_screen_path,
)


def test_frozen_screen_path_uses_bound_source_scan_location() -> None:
    protocol = {"source_scan": {"screen_path": "results/frozen-screen.json"}}

    assert frozen_screen_path(protocol) == "results/frozen-screen.json"


def test_exclude_identities_preserves_order_and_other_candidates() -> None:
    rows = [
        {"repository": "vllm-project/vllm", "number": 64675},
        {"repository": "vllm-project/vllm", "number": 64575},
        {"repository": "ray-project/ray", "number": 67563},
    ]

    remaining = exclude_identities(rows, {("vllm-project/vllm", 64675)})

    assert remaining == rows[1:]


def test_exclude_identities_does_not_match_different_repository_or_number() -> None:
    rows = [
        {"repository": "vllm-project/vllm", "number": 64675},
        {"repository": "ray-project/ray", "number": 64675},
    ]

    remaining = exclude_identities(rows, {("other/repo", 64675)})

    assert remaining == rows
