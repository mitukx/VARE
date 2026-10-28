from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from vare.agent_edit_v2 import (
    SourceEditLeaseStoreV1,
    read_source_lines_v1,
    replace_python_source_lines_transactional_v1,
    replace_source_lines_v1,
)


def _source(root: Path, text: str = "def value():\n    return 1\n") -> tuple[Path, str]:
    path = root / "src" / "value.py"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path, "src/value.py"


def test_read_and_replace_are_bound_to_the_exact_file_preimage(tmp_path: Path) -> None:
    source, rel = _source(tmp_path)
    read = read_source_lines_v1(tmp_path, rel, 1, 2, {rel})

    assert read["status"] == "read_complete"
    assert read["content"] == "def value():\n    return 1\n"
    result = replace_source_lines_v1(
        tmp_path, rel, read["sha256"], 2, 2, "    return 2\n", {rel},
    )

    assert result["applied"] is True
    assert result["status"] == "edit_applied"
    assert result["before_sha256"] == read["sha256"]
    assert result["after_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.read_text(encoding="utf-8") == "def value():\n    return 2\n"


def test_replayed_edit_is_rejected_as_stale(tmp_path: Path) -> None:
    _source(tmp_path)
    expected = hashlib.sha256((tmp_path / "src/value.py").read_bytes()).hexdigest()
    first = replace_source_lines_v1(
        tmp_path, "src/value.py", expected, 2, 2, "    return 2\n", {"src/value.py"},
    )
    replay = replace_source_lines_v1(
        tmp_path, "src/value.py", expected, 2, 2, "    return 2\n", {"src/value.py"},
    )

    assert first["applied"] is True
    assert replay["applied"] is False
    assert replay["error"] == "stale_preimage"


@pytest.mark.parametrize("raw_path", ["../value.py", "/src/value.py", "src\\value.py", "./src/value.py"])
def test_path_aliases_and_traversal_are_rejected(tmp_path: Path, raw_path: str) -> None:
    _source(tmp_path)
    result = read_source_lines_v1(tmp_path, raw_path, 1, 1, {"src/value.py"})
    assert result["status"] == "valid_call_rejected_by_policy"
    assert result["error"] == "path_not_allowlisted"


def test_file_symlink_and_parent_symlink_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("secret = True\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "value.py").symlink_to(outside)
    file_link = read_source_lines_v1(workspace, "src/value.py", 1, 1, {"src/value.py"})
    assert file_link["status"] == "valid_call_rejected_by_policy"

    parent_workspace = tmp_path / "parent-workspace"
    parent_workspace.mkdir()
    (parent_workspace / "src").symlink_to(outside.parent, target_is_directory=True)
    parent_link = read_source_lines_v1(parent_workspace, "src/value.py", 1, 1, {"src/value.py"})
    assert parent_link["status"] == "valid_call_rejected_by_policy"


@pytest.mark.parametrize("data", [b"\xef\xbb\xbfvalue = 1\n", b"value = 1\x00\n", b"value = 1\r\n", b"\xff\xfe"])
def test_unsafe_or_unsupported_source_encodings_are_rejected(tmp_path: Path, data: bytes) -> None:
    source, rel = _source(tmp_path)
    source.write_bytes(data)
    result = read_source_lines_v1(tmp_path, rel, 1, 1, {rel})
    assert result["status"] == "valid_call_rejected_by_policy"


def test_changed_preimage_is_rejected_without_partial_write(tmp_path: Path) -> None:
    source, rel = _source(tmp_path)
    stale = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("def value():\n    return 7\n", encoding="utf-8")
    before = source.read_bytes()
    result = replace_source_lines_v1(
        tmp_path, rel, stale, 2, 2, "    return 2\n", {rel},
    )

    assert result["applied"] is False
    assert result["error"] == "stale_preimage"
    assert source.read_bytes() == before
    assert not list(source.parent.glob("*.vare-*.tmp"))


@pytest.mark.parametrize("start,end", [(0, 1), (2, 1), (1, 121), (True, 1)])
def test_invalid_ranges_are_rejected(tmp_path: Path, start: object, end: object) -> None:
    _source(tmp_path)
    result = read_source_lines_v1(tmp_path, "src/value.py", start, end, {"src/value.py"})
    assert result["status"] == "valid_call_rejected_by_policy"


def test_atomic_replace_failure_leaves_source_unchanged_and_cleans_temp(tmp_path: Path, monkeypatch) -> None:
    source, rel = _source(tmp_path)
    before = source.read_bytes()
    expected = hashlib.sha256(before).hexdigest()

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected rename failure")

    monkeypatch.setattr("vare.agent_edit_v2.os.replace", fail_replace)
    result = replace_source_lines_v1(
        tmp_path, rel, expected, 2, 2, "    return 2\n", {rel},
    )

    assert result["applied"] is False
    assert source.read_bytes() == before
    assert not list(source.parent.glob(".*.vare-*.tmp"))


def test_unchanged_replacement_is_rejected(tmp_path: Path) -> None:
    source, rel = _source(tmp_path)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    result = replace_source_lines_v1(
        tmp_path, rel, expected, 2, 2, "    return 1\n", {rel},
    )
    assert result["applied"] is False
    assert result["error"] == "no_change"


def test_edit_lease_hides_long_preimage_hash_and_is_single_use(tmp_path: Path) -> None:
    source, rel = _source(tmp_path)
    store = SourceEditLeaseStoreV1(tmp_path, {rel}, token_factory=lambda: "1a2b3c4d5e6f")
    observed = store.read_lines(rel, 2, 2)
    assert observed["lease"] == "1a2b3c4d5e6f"
    assert "sha256" not in observed
    applied = store.replace_lines(observed["lease"], "    return 9\n")
    replay = store.replace_lines(observed["lease"], "    return 10\n")

    assert applied["status"] == "edit_applied"
    assert source.read_text(encoding="utf-8") == "def value():\n    return 9\n"
    assert replay["error"] == "unknown_or_consumed_lease"
    record = store.audit_records()[0]
    assert record["expected_sha256"] == applied["before_sha256"]
    assert record["consumed"] is True
    assert record["outcome"] == "edit_applied"


def test_edit_lease_rejects_source_changes_after_read(tmp_path: Path) -> None:
    source, rel = _source(tmp_path)
    store = SourceEditLeaseStoreV1(tmp_path, {rel}, token_factory=lambda: "0123456789ab")
    observed = store.read_lines(rel, 2, 2)
    source.write_text("def value():\n    return 42\n", encoding="utf-8")
    before = source.read_bytes()
    result = store.replace_lines(observed["lease"], "    return 2\n")

    assert result["status"] == "valid_call_rejected_by_policy"
    assert result["error"] == "stale_preimage"
    assert source.read_bytes() == before


def test_transactional_python_edit_rejects_syntax_without_mutating_source(tmp_path: Path) -> None:
    source, rel = _source(tmp_path)
    before = source.read_bytes()
    digest = hashlib.sha256(before).hexdigest()

    result = replace_python_source_lines_transactional_v1(
        tmp_path, rel, digest, 2, 2, "    return (\n", {rel},
    )

    assert result["applied"] is False
    assert result["status"] == "candidate_rejected_syntax"
    assert result["syntax_valid"] is False
    assert result["raw_replacement_sha256"] == hashlib.sha256(b"    return (\n").hexdigest()
    assert source.read_bytes() == before
    assert not list(source.parent.glob(".*.vare-*.tmp"))


@pytest.mark.parametrize(
    ("raw", "normalization", "expected"),
    [
        ("    return 2\n", "none", "    return 2\n"),
        ("```python\n    return 3\n```\n", "strip_single_python_fence", "    return 3\n"),
        ("    return 4", "append_final_newline", "    return 4\n"),
        ("```py\n    return 5\n```", "strip_single_python_fence_append_final_newline", "    return 5\n"),
    ],
)
def test_transactional_python_edit_requires_and_records_explicit_format_transform(
        tmp_path: Path, raw: str, normalization: str, expected: str) -> None:
    source, rel = _source(tmp_path)
    store = SourceEditLeaseStoreV1(tmp_path, {rel}, token_factory=lambda: "0123456789ab")
    observed = store.read_lines(rel, 2, 2)

    result = store.replace_lines_transactional(observed["lease"], raw, normalization)

    assert result["applied"] is True
    assert result["syntax_valid"] is True
    assert source.read_text(encoding="utf-8") == f"def value():\n{expected}"
    assert result["raw_replacement_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert result["normalized_replacement_sha256"] == hashlib.sha256(expected.encode()).hexdigest()
    assert result["normalization"] == normalization
    record = store.audit_records()[0]
    assert record["raw_replacement_sha256"] == result["raw_replacement_sha256"]
    assert record["normalized_replacement_sha256"] == result["normalized_replacement_sha256"]


def test_transactional_python_edit_does_not_strip_fences_without_explicit_selection(tmp_path: Path) -> None:
    source, rel = _source(tmp_path)
    before = source.read_bytes()
    store = SourceEditLeaseStoreV1(tmp_path, {rel}, token_factory=lambda: "0123456789ab")
    observed = store.read_lines(rel, 2, 2)

    result = store.replace_lines_transactional(observed["lease"], "```python\n    return 2\n```\n", "none")

    assert result["applied"] is False
    assert result["status"] == "candidate_rejected_syntax"
    assert source.read_bytes() == before


def test_transactional_python_edit_rejects_inexact_fence_transform_without_write(tmp_path: Path) -> None:
    source, rel = _source(tmp_path)
    before = source.read_bytes()
    digest = hashlib.sha256(before).hexdigest()

    result = replace_python_source_lines_transactional_v1(
        tmp_path, rel, digest, 2, 2, "```python\n    return 2\n", {rel},
        normalization="strip_single_python_fence",
    )

    assert result["applied"] is False
    assert result["error"] == "python_fence_not_exact"
    assert source.read_bytes() == before
