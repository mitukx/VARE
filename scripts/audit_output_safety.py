"""Protect retained audit inputs from report output writes."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def preflight_new_output(path: Path, protected_roots: Iterable[Path]) -> Path:
    """Reject retained-input destinations and prepare a new output path."""
    output = Path(path)
    roots = tuple(Path(root).resolve() for root in protected_roots)

    def reject_protected_destination(candidate: Path) -> None:
        resolved = candidate.resolve()
        for root in roots:
            if resolved == root or root in resolved.parents:
                raise ValueError(
                    f"refusing audit output inside retained input: {root}"
                )

    reject_protected_destination(output)
    if output.exists() or output.is_symlink():
        raise ValueError("audit output already exists; refusing to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Recheck after creating parents in case a path appeared during preflight.
    reject_protected_destination(output)
    if output.exists() or output.is_symlink():
        raise ValueError("audit output appeared during preflight; refusing to overwrite")
    return output


def write_new_output(path: Path, content: str) -> None:
    """Write only if the destination is still absent after preflight."""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise ValueError(
            "audit output appeared after preflight; refusing to overwrite"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
