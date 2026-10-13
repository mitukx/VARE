#!/usr/bin/env python3
"""Fail if tracked blobs at or above 1 MiB differ from the frozen census.

Existing oversized objects remain in git. This check rejects a tree that adds,
replaces, or drops one of them. It does not rewrite published history.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from vare.blob_budget import THRESHOLD_BYTES, list_tracked_blobs, parse_allowlist, violations

# SHA-256 of protocols/tracked_large_blob_allowlist_v1.txt. Changing the
# census requires changing this pin in the same commit; that diff is the
# review point. Do not refresh it to admit a new large checkpoint.
ALLOWLIST_SHA256 = "946a940250e2610a3ccc0e9bb8507f2f523395dd1b774c283a0da58a0e3ffd34"
ALLOWLIST_RELATIVE = Path("protocols/tracked_large_blob_allowlist_v1.txt")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    allow_path = repo / ALLOWLIST_RELATIVE
    data = allow_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != ALLOWLIST_SHA256:
        print(
            f"allowlist SHA-256 {digest} does not match pinned {ALLOWLIST_SHA256}",
            file=sys.stderr,
        )
        return 1
    try:
        rows = parse_allowlist(data.decode("utf-8"), threshold=THRESHOLD_BYTES)
        tracked = list_tracked_blobs(repo)
        problems = violations(tracked, rows, threshold=THRESHOLD_BYTES)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"tracked blob budget ok: {len(rows)} allowlisted blobs >= {THRESHOLD_BYTES} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
