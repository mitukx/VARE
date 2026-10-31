"""Check commit author and committer emails against an allowlist.

Pull requests should only carry identities that belong to this repository's
maintainers. The check reads ``base..head`` and fails closed on any author or
committer email that is not listed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


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
