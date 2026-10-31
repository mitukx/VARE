"""Deduplication against every task VARE has already used.

Three layers, all fail-closed:

1. Repository-level exclusion. Any upstream repository that hosts a VARE
   Train, Development, harness-debugging or Held-Out task is excluded from the
   curriculum entirely (e.g. vllm-project/vllm for #64830 and
   ml-explore/mlx-lm for #1904). This also covers held-out identities that are
   not spelled out in public files.
2. Identifier exclusion. ``owner/repo#N`` references found anywhere in the
   VARE tree are collected and recorded; a curriculum instance whose upstream
   repository appears there is rejected.
3. Content deduplication. Instances with identical normalized patches, or
   with an identical (target file, problem statement) pair, are collapsed
   before splitting, and Train/Held-Out are repository-disjoint, so no
   held-out bug shares a repository, file or patch with any training row.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

# Repositories that host VARE evaluation, Development, Train-diagnostic or
# held-out tasks. Kept explicit so the exclusion does not depend on scanning.
PROTECTED_REPOSITORIES = frozenset(
    {
        "vllm-project/vllm",  # Development: #64830, #64575, #64675, ...
        "ml-explore/mlx-lm",  # Train: #1904; historical held-out family
        "ml-explore/mlx",
        "huggingface/transformers",
        "huggingface/trl",
        "ray-project/ray",
        "urllib3/urllib3",
        "pytorch/pytorch",
        "huggingface/accelerate",
        "huggingface/peft",
    }
)

# Whole organizations excluded conservatively (repository-level overlap with
# VARE Development families, e.g. huggingface/transformers).
PROTECTED_OWNERS = frozenset({"huggingface", "vllm-project", "ml-explore"})

REQUIRED_IDENTIFIERS = ("vllm-project/vllm#64830", "ml-explore/mlx-lm#1904")

_REF = re.compile(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+) ?#(\d{2,7})\b")
_SCAN_SUFFIXES = {".md", ".json", ".py", ".txt", ".yml", ".yaml", ".toml"}
_SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache"}


def normalized_patch_sha256(patch: str) -> str:
    lines = [ln.rstrip() for ln in patch.replace("\r\n", "\n").split("\n")]
    lines = [ln for ln in lines if not ln.startswith("index ")]
    return hashlib.sha256("\n".join(lines).strip().encode()).hexdigest()


def scan_vare_tree(root: Path, *, skip: Iterable[str] = ()) -> dict[str, object]:
    """Collect task identifiers referenced by the VARE tree.

    ``skip`` lists relative path prefixes to ignore (the pipeline's own output
    directory). Held-out benchmark directories are scanned for identifiers only
    with a regex; their contents are not otherwise parsed or recorded.
    """
    skip_prefixes = tuple(skip)
    refs: set[str] = set()
    files = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in _SCAN_SUFFIXES:
            continue
        rel = path.relative_to(root).as_posix()
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if skip_prefixes and rel.startswith(skip_prefixes):
            continue
        try:
            if path.stat().st_size > 8_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        files += 1
        for owner_repo, num in _REF.findall(text):
            refs.add(f"{owner_repo.lower()}#{num}")
    repos = sorted({r.split("#", 1)[0] for r in refs})
    return {"files_scanned": files, "identifiers": sorted(refs), "repositories": repos}


def excluded_repositories(scan: dict[str, object]) -> frozenset[str]:
    """Protected repositories plus every repository that has a scanned
    identifier and looks like a GitHub ``owner/repo`` pair."""
    scanned = {r for r in scan.get("repositories", []) if "/" in r}  # type: ignore[union-attr]
    return frozenset({r.lower() for r in PROTECTED_REPOSITORIES} | scanned)


def is_excluded(upstream_repo: str, excluded: frozenset[str]) -> str | None:
    """Return the exclusion reason for ``owner/repo``, or None if admitted."""
    repo = upstream_repo.lower()
    if repo.split("/", 1)[0] in PROTECTED_OWNERS:
        return "protected_owner"
    if repo in excluded:
        return "repository_used_by_vare"
    return None


def check_required(scan: dict[str, object]) -> list[str]:
    """Return required identifiers that the scan failed to find (should be [])."""
    found = set(scan.get("identifiers", []))  # type: ignore[arg-type]
    return [ident for ident in REQUIRED_IDENTIFIERS if ident.lower() not in found]
