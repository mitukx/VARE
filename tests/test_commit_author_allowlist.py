from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from scripts.check_commit_authors import list_commit_identities, parse_allowlist, violations

REPO = Path(__file__).resolve().parents[1]


def test_repository_allowlist_parses():
    allowed = parse_allowlist((REPO / "protocols/commit_author_allowlist.txt").read_text(encoding="utf-8"))
    assert "249598142+mitukx@users.noreply.github.com" in allowed
    assert "noreply@github.com" in allowed


def test_parse_allowlist_rejects_malformed_and_empty():
    with pytest.raises(ValueError, match="single email"):
        parse_allowlist("not an email")
    with pytest.raises(ValueError, match="empty"):
        parse_allowlist("# only a comment\n")


def test_violations_flag_author_and_committer_case_insensitively():
    allowed = parse_allowlist("Ok@Example.com\n")
    rows = [("a" * 40, "ok@example.com", "OK@example.com"), ("b" * 40, "x@example.com", "ok@example.com")]
    problems = violations(rows, allowed)
    assert len(problems) == 1 and "author" in problems[0]
    rows = [("c" * 40, "ok@example.com", "y@example.com")]
    assert "committer" in violations(rows, allowed)[0]


def _commit(repo: Path, message: str, email: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": email,
    }
    subprocess.run(
        ["git", "commit", "--allow-empty", "-q", "-m", message], cwd=repo, env=env, check=True
    )
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    return out.stdout.strip()


def test_range_excludes_base_history(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    base = _commit(tmp_path, "old", "outside@example.com")
    head = _commit(tmp_path, "new", "ok@example.com")
    rows = list_commit_identities(tmp_path, base, head)
    assert [row[0] for row in rows] == [head]
    assert violations(rows, frozenset({"ok@example.com"})) == []
