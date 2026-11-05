"""Canonical JSON helpers for ResearchGym execution records.

Hashes are content addresses of stored bytes. They do not authenticate an
untrusted host.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from vare.evidence import canonical_json, sha256_json


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_load(data: str | bytes) -> Any:
    """Parse JSON that rejects duplicates, NaN, and Infinity."""
    if isinstance(data, bytes):
        text = data.decode("utf-8")
    else:
        text = data

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def floating(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite JSON number")
        return number

    def constant(value: str) -> None:
        raise ValueError("nonfinite JSON constant")

    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_float=floating,
        parse_constant=constant,
    )


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    fd = os.open(str(temporary), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "atomic_write",
    "canonical_bytes",
    "canonical_json",
    "digest_bytes",
    "file_sha256",
    "sha256_json",
    "strict_load",
]
