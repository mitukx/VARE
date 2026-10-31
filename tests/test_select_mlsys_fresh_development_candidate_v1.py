from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from select_mlsys_fresh_development_candidate_v1 import eligible_rows, load_protocol


def _protocol():
    return {
        "selection_rules": {
            "excluded_repository_owners": ["huggingface", "ml-explore"],
            "excluded_path_markers": [
                "cuda", "rocm", "metal", "gpu", "kernel", "triton", "amd",
            ],
            "excluded_title_markers": ["rocm", "amd", "kimi", "eplb", "feature", "perf"],
            "maximum_changed_files": 5,
            "maximum_changed_lines": 200,
            "rank_domain": "vare-fresh-development-selection-v1",
        },
    }


def _row(repo, number, paths, *, owner_status="provisional_for_manual_review"):
    test_paths = [path for path in paths if "test" in path.lower()]
    return {
        "repository": repo,
        "number": number,
        "title": "Fix behavior",
        "coarse_screen": owner_status,
        "qualification_status": "UNQUALIFIED",
        "split_assignment": "UNASSIGNED",
        "task_family_overlap": "NOT_ASSESSED",
        "changed_paths": paths,
        "test_paths": test_paths,
        "changed_file_count": len(paths),
        "additions": 10,
        "deletions": 5,
    }


def test_selection_is_deterministic_and_keeps_candidate_unassigned():
    screen = {
        "candidates": [
            _row("vllm-project/vllm", 20, ["tests/test_b.py", "vllm/b.py"]),
            _row("vllm-project/vllm", 10, ["tests/test_a.py", "vllm/a.py"]),
        ],
    }

    first = eligible_rows(screen, _protocol())
    second = eligible_rows(screen, _protocol())

    assert first == second
    assert len(first) == 2
    assert first[0]["selection_rank_sha256"] < first[1]["selection_rank_sha256"]
    assert first[0]["split_assignment"] == "UNASSIGNED"


def test_selector_protocol_is_bound_to_the_completed_metadata_scan():
    protocol, addendum = load_protocol()

    assert protocol["protocol_id"] == "mlsys-fresh-development-candidate-selection-v1"
    assert protocol["selection_rules"]["selection_count"] == 1
    assert protocol["authorization"]["source_or_test_content_fetch"] is False
    assert addendum["protocol_id"] == "mlsys-fresh-development-candidate-selection-v1-addendum-1"


def test_selection_excludes_prior_owners_and_hardware_paths():
    screen = {
        "candidates": [
            _row("ml-explore/mlx", 1, ["tests/test_conv.py", "mlx/backend/cuda/conv.cpp"]),
            _row("vllm-project/vllm", 2, ["tests/test_rocm.py", "vllm/rocm/ops.py"]),
            _row("huggingface/trl", 3, ["tests/test_x.py", "trl/x.py"]),
            _row("vllm-project/vllm", 4, ["tests/test_feature.py", "vllm/config.py"]),
        ],
    }
    screen["candidates"][-1]["title"] = "Feature: EPLB setting"

    assert eligible_rows(screen, _protocol()) == []
