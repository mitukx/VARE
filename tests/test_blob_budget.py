from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from vare.blob_budget import (
    THRESHOLD_BYTES,
    format_allowlist,
    list_tracked_blobs,
    parse_allowlist,
    violations,
)

REPO = Path(__file__).resolve().parents[1]
ALLOWLIST = REPO / "protocols/tracked_large_blob_allowlist_v1.txt"
PINNED_SHA256 = "946a940250e2610a3ccc0e9bb8507f2f523395dd1b774c283a0da58a0e3ffd34"


def test_parse_allowlist_rejects_malformed_rows():
    with pytest.raises(ValueError, match="malformed"):
        parse_allowlist("not-a-row", threshold=1)
    with pytest.raises(ValueError, match="below"):
        parse_allowlist(f"1 {'a' * 40} results/a.bin", threshold=2)
    with pytest.raises(ValueError, match="unsafe"):
        parse_allowlist(f"2 {'b' * 40} ../secret.bin", threshold=2)


def test_violations_accept_an_exact_census_and_reject_drift():
    oid = "a" * 40
    other = "b" * 40
    tracked = [("keep.txt", oid, 8), ("big.bin", other, 64)]
    allowlist = [(64, other, "big.bin")]
    assert violations(tracked, allowlist, threshold=32) == []

    added = tracked + [("new.bin", "c" * 40, 40)]
    assert any("not allowlisted" in item for item in violations(added, allowlist, threshold=32))

    replaced = [("keep.txt", oid, 8), ("big.bin", "d" * 40, 64)]
    messages = violations(replaced, allowlist, threshold=32)
    assert any("not allowlisted" in item for item in messages)
    assert any("missing from the tree" in item for item in messages)

    deleted = [("keep.txt", oid, 8)]
    assert any("missing from the tree" in item for item in violations(deleted, allowlist, threshold=32))


def test_git_index_reports_size_and_format_roundtrip(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "small.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "big.bin").write_bytes(b"x" * 64)
    subprocess.run(["git", "add", "small.txt", "big.bin"], cwd=tmp_path, check=True, capture_output=True)
    tracked = list_tracked_blobs(tmp_path)
    rendered = format_allowlist(tracked, threshold=32)
    parsed = parse_allowlist(rendered, threshold=32)
    assert violations(tracked, parsed, threshold=32) == []
    assert {path for _size, _oid, path in parsed} == {"big.bin"}


def test_repository_allowlist_matches_the_current_index():
    data = ALLOWLIST.read_bytes()
    assert hashlib.sha256(data).hexdigest() == PINNED_SHA256
    rows = parse_allowlist(data.decode("utf-8"), threshold=THRESHOLD_BYTES)
    tracked = list_tracked_blobs(REPO)
    assert violations(tracked, rows, threshold=THRESHOLD_BYTES) == []
    assert len(rows) == 235


def test_check_script_passes_on_this_repository():
    proc = subprocess.run(
        ["python3", "scripts/check_tracked_blob_budget.py"],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert proc.returncode == 0, proc.stderr
    assert "235 allowlisted blobs" in proc.stdout
