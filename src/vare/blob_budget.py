"""Compare tracked git blobs with a frozen large-blob allowlist.

The allowlist records objects that are already in the published tree. It does
not authorize new checkpoints. Files at or above the threshold that are not
listed, and listed objects that disappear or change, fail closed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

THRESHOLD_BYTES = 1_048_576


def parse_allowlist(text: str, *, threshold: int) -> list[tuple[int, str, str]]:
    """Return ``(size, oid, path)`` rows.

    Blank lines and ``#`` comments are ignored. Every data row must name a
    blob at or above ``threshold``.
    """
    rows: list[tuple[int, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(" ", 2)
        if len(parts) != 3:
            raise ValueError(f"allowlist line {line_no} is malformed")
        size_text, oid, path = parts
        if not size_text.isdigit():
            raise ValueError(f"allowlist line {line_no} has a non-integer size")
        size = int(size_text)
        if size < threshold:
            raise ValueError(f"allowlist line {line_no} is below the size threshold")
        if len(oid) != 40 or any(char not in "0123456789abcdef" for char in oid):
            raise ValueError(f"allowlist line {line_no} has a non-hex blob id")
        path_parts = Path(path).parts
        if not path or path.startswith("/") or ".." in path_parts or "\\" in path:
            raise ValueError(f"allowlist line {line_no} has an unsafe path")
        key = (oid, path)
        if key in seen:
            raise ValueError(f"allowlist line {line_no} repeats {oid} {path}")
        seen.add(key)
        rows.append((size, oid, path))
    return rows


def violations(
    tracked: list[tuple[str, str, int]],
    allowlist: list[tuple[int, str, str]],
    *,
    threshold: int,
) -> list[str]:
    """Return fail-closed messages for tree/allowlist disagreement.

    ``tracked`` entries are ``(path, oid, size)``.
    """
    present = {(path, oid): size for path, oid, size in tracked}
    allowed = {(path, oid): size for size, oid, path in allowlist}
    problems: list[str] = []
    for path, oid, size in sorted(tracked):
        if size < threshold:
            continue
        expected = allowed.get((path, oid))
        if expected is None:
            problems.append(
                f"tracked blob {size} {oid} {path} is at or above {threshold} bytes "
                "and is not allowlisted"
            )
        elif expected != size:
            problems.append(
                f"tracked blob {path} size {size} does not match allowlist size {expected}"
            )
    for size, oid, path in sorted(allowlist, key=lambda row: (row[2], row[1])):
        actual = present.get((path, oid))
        if actual is None:
            problems.append(f"allowlisted blob {size} {oid} {path} is missing from the tree")
        elif actual != size:
            problems.append(
                f"allowlisted blob {path} lists size {size} but the tree has {actual}"
            )
    return problems


def list_tracked_blobs(repo: Path) -> list[tuple[str, str, int]]:
    """Return ``(path, oid, size)`` for every staged blob in ``repo``."""
    listed = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        cwd=repo,
        check=False,
        capture_output=True,
    )
    if listed.returncode != 0:
        detail = listed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {detail or listed.returncode}")
    paths: list[str] = []
    oids: list[str] = []
    for record in listed.stdout.split(b"\0"):
        if not record:
            continue
        meta, path_bytes = record.split(b"\t", 1)
        _mode, oid, stage = meta.decode("ascii").split(" ")
        path = path_bytes.decode("utf-8")
        if stage != "0":
            raise RuntimeError(f"unmerged index entry: {path}")
        paths.append(path)
        oids.append(oid)
    if not oids:
        return []
    sized = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
        cwd=repo,
        input="".join(f"{oid}\n" for oid in oids).encode("ascii"),
        check=False,
        capture_output=True,
    )
    if sized.returncode != 0:
        detail = sized.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git cat-file failed: {detail or sized.returncode}")
    sizes: dict[str, int] = {}
    for line in sized.stdout.decode("ascii").splitlines():
        oid, size_text = line.split(" ", 1)
        sizes[oid] = int(size_text)
    missing = [oid for oid in oids if oid not in sizes]
    if missing:
        raise RuntimeError(f"git cat-file omitted {missing[0]}")
    return [(path, oid, sizes[oid]) for path, oid in zip(paths, oids, strict=True)]


def format_allowlist(tracked: list[tuple[str, str, int]], *, threshold: int) -> str:
    """Render the census header and one row per oversized blob."""
    header = (
        "# Frozen census of tracked git blobs at or above "
        f"{threshold} bytes.\n"
        "# Each row is: <size> <blob-oid> <path>\n"
        "# These objects stay in git because they are retained raw records.\n"
        "# Do not append a row to admit a new checkpoint. Store new large files\n"
        "# outside the repository and record their SHA-256 in the study manifest.\n"
        "# This file does not rewrite published history.\n"
    )
    rows = [
        f"{size} {oid} {path}\n"
        for path, oid, size in sorted(tracked)
        if size >= threshold
    ]
    return header + "".join(rows)
