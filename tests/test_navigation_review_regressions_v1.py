"""Behavioral regressions for the two independently reported navigation bugs."""
import json
from pathlib import Path

from vare import source_navigation_v1 as module
from vare.source_navigation_v1 import SourceNavigatorV1


def test_large_inventory_does_not_displace_a_useful_hit(tmp_path: Path):
    paths = set()
    for i in range(128):
        relative = f"src/{'long_name_' * 10}{i:03d}.py"
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("needle = 1\n" if i == 71 else "other = 0\n")
        paths.add(relative)
    navigator = SourceNavigatorV1(tmp_path, paths)
    result = navigator.search("needle")
    assert result["status"] == "ok"
    assert len(result["results"]) == 1
    assert result["results"][0]["path"].endswith("071.py")
    assert result["indexed_path_count"] == 128
    assert result["indexed_paths_truncated"] is True
    assert result["output_bytes"] == len(json.dumps(result, ensure_ascii=False).encode())
    assert result["output_bytes"] <= 8192


def test_failed_rebuild_is_retried_after_source_repair(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("needle = 1\n")
    navigator = SourceNavigatorV1(tmp_path, {"source.py"})
    monkeypatch.setattr(module, "MAX_FILE_SOURCE_LINES", 3)
    source.write_text("bad\n" * 4)
    assert navigator.search("needle")["status"] == "source_index_limit_exceeded"
    source.write_text("needle = 2\n")
    recovered = navigator.search("needle")
    assert recovered["status"] == "ok"
    assert recovered["index_rebuilt"] is True
    assert recovered["results"][0]["context"] == "needle = 2"
    assert navigator.search("needle")["index_rebuilt"] is False


def test_failed_initial_read_can_recover(tmp_path: Path):
    navigator = SourceNavigatorV1(tmp_path, {"source.py"})
    assert navigator.search("needle")["status"] == "source_read_failed"
    (tmp_path / "source.py").write_text("needle = 3\n")
    assert navigator.search("needle")["status"] == "ok"


def test_empty_allowlist_remains_an_error(tmp_path: Path):
    assert SourceNavigatorV1(tmp_path, set()).search("needle")["status"] == (
        "no_candidate_visible_source_files"
    )
