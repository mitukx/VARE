"""Canonical, shared prompt-ID fingerprint used by CPU and GPU gate paths."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def canonical_prompt_id_payload(prompt_ids: Iterable[str]) -> bytes:
    """UTF-8 of sorted lowercase SHA-256 IDs, one per LF line, final LF included."""
    values = list(prompt_ids)
    if not values or any(not isinstance(value, str) or not _SHA256_HEX.fullmatch(value) for value in values):
        raise ValueError("prompt IDs must be non-empty lowercase SHA-256 hex strings")
    if len(set(values)) != len(values):
        raise ValueError("prompt IDs must be unique")
    return ("\n".join(sorted(values)) + "\n").encode("utf-8")


def canonical_prompt_id_sha256(prompt_ids: Iterable[str]) -> str:
    return hashlib.sha256(canonical_prompt_id_payload(prompt_ids)).hexdigest()


def verify_prompt_id_sha256(prompt_ids: Iterable[str], expected_sha256: str, *, expected_count: int) -> str:
    values = list(prompt_ids)
    if len(values) != expected_count:
        raise ValueError(f"expected {expected_count} prompt IDs, found {len(values)}")
    observed = canonical_prompt_id_sha256(values)
    if observed != expected_sha256:
        raise ValueError(f"prompt-ID SHA-256 mismatch: expected {expected_sha256}, observed {observed}")
    return observed
