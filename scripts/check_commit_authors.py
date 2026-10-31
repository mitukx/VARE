#!/usr/bin/env python3
"""Fail if any commit in BASE..HEAD has an author or committer email that is
not listed in protocols/commit_author_allowlist.txt.

Usage: PYTHONPATH=src python scripts/check_commit_authors.py BASE HEAD
"""

from __future__ import annotations

import sys
from pathlib import Path

from vare.commit_authors import list_commit_identities, parse_allowlist, violations

ALLOWLIST_RELATIVE = Path("protocols/commit_author_allowlist.txt")


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
