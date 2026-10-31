#!/usr/bin/env python3
"""Fail if any commit in BASE..HEAD has an author or committer email that is
not listed in protocols/commit_author_allowlist.txt.

Usage: python scripts/check_commit_authors.py BASE HEAD
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ALLOWLIST_RELATIVE = Path("protocols/commit_author_allowlist.txt")


def parse_allowlist(text: str) -> frozenset[str]:
    """Return lowercased emails; blank lines and ``#`` comments are ignored."""
    emails: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if " " in stripped or stripped.count("@") != 1:
            raise ValueError(f"allowlist line {line_no} is not a single email")
        emails.add(stripped.lower())
    if not emails:
        raise ValueError("allowlist is empty")
    return frozenset(emails)


def list_commit_identities(repo: Path, base: str, head: str) -> list[tuple[str, str, str]]:
    """Return ``(sha, author_email, committer_email)`` for ``base..head``."""
    result = subprocess.run(
        ["git", "log", "--format=%H%x00%ae%x00%ce", f"{base}..{head}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git log failed: {result.stderr.strip()}")
    rows: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        sha, author, committer = line.split("\x00")
        rows.append((sha, author, committer))
    return rows


def violations(rows: list[tuple[str, str, str]], allowed: frozenset[str]) -> list[str]:
    problems: list[str] = []
    for sha, author, committer in rows:
        for role, email in (("author", author), ("committer", committer)):
            if email.lower() not in allowed:
                problems.append(f"{sha[:12]} {role} email {email!r} is not allowlisted")
    return problems



def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: check_commit_authors.py BASE HEAD", file=sys.stderr)
        return 2
    base, head = argv
    repo = Path(__file__).resolve().parents[1]
    try:
        allowed = parse_allowlist((repo / ALLOWLIST_RELATIVE).read_text(encoding="utf-8"))
        rows = list_commit_identities(repo, base, head)
        problems = violations(rows, allowed)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(
            f"fix the commit identities or update {ALLOWLIST_RELATIVE}",
            file=sys.stderr,
        )
        return 1
    print(f"commit identities ok: {len(rows)} commits in {base}..{head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
