"""Protect retained audit inputs from report output writes."""
from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Iterable


class ExclusiveOutputTarget:
    """An output name anchored to an already validated directory descriptor."""

    def __init__(self, path: Path, name: str, parent_fd: int) -> None:
        self.path = path
        self.name = name
        self.parent_fd = parent_fd

    def __enter__(self) -> "ExclusiveOutputTarget":
        return self

    def __exit__(self, *_exc_info) -> None:
        os.close(self.parent_fd)


def _open_directory_without_symlinks(path: Path) -> int:
    """Open each canonical directory component without following symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags | nofollow)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(
                component, flags | nofollow, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def preflight_new_output(
    path: Path, protected_roots: Iterable[Path]
) -> ExclusiveOutputTarget:
    """Reject retained-input destinations and anchor a new output directory."""
    output = Path(path)
    roots = tuple(Path(root).resolve() for root in protected_roots)

    def reject_protected_destination(candidate: Path) -> Path:
        resolved = candidate.resolve(strict=False)
        for root in roots:
            if resolved == root or root in resolved.parents:
                raise ValueError(
                    f"refusing audit output inside retained input: {root}"
                )
        return resolved

    reject_protected_destination(output)
    if output.exists() or output.is_symlink():
        raise ValueError("audit output already exists; refusing to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Recheck after creating parents in case a path appeared during preflight.
    resolved = reject_protected_destination(output)
    if output.exists() or output.is_symlink():
        raise ValueError("audit output appeared during preflight; refusing to overwrite")
    parent_fd = _open_directory_without_symlinks(resolved.parent)
    try:
        opened_parent = os.fstat(parent_fd)
        named_parent = resolved.parent.stat()
        if not stat.S_ISDIR(opened_parent.st_mode) or (
            opened_parent.st_dev, opened_parent.st_ino
        ) != (named_parent.st_dev, named_parent.st_ino):
            raise ValueError("audit output parent changed during preflight")
        try:
            os.stat(resolved.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("audit output already exists; refusing to overwrite")
    except BaseException:
        os.close(parent_fd)
        raise
    return ExclusiveOutputTarget(resolved, resolved.name, parent_fd)


def write_new_output(target: ExclusiveOutputTarget, content: str) -> None:
    """Write only if the destination is still absent after preflight."""
    try:
        descriptor = os.open(
            target.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=target.parent_fd,
        )
    except FileExistsError as exc:
        raise ValueError(
            "audit output appeared after preflight; refusing to overwrite"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
