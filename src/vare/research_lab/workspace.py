"""Materialize a candidate workspace from a task pack."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from vare.evidence import sha256_file

from .paths import assert_regular_file, resolve_inside
from .spec import ResearchTask


class IntegrityError(ValueError):
    """The workspace failed a structural integrity check before execution."""


def materialize(task: ResearchTask, dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(dest)
    source = task.root / "workspace"
    dest.mkdir(parents=True)
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        if current.is_symlink():
            raise IntegrityError("workspace source contains a symlink directory")
        rel_dir = current.relative_to(source)
        target_dir = dest / rel_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        dirnames[:] = sorted(dirnames)
        for name in sorted(filenames):
            src = current / name
            if src.is_symlink() or not src.is_file():
                raise IntegrityError(f"workspace source is not a regular file: {name}")
            target = target_dir / name
            shutil.copyfile(src, target)
    for name in task.editable_files:
        assert_regular_file(resolve_inside(dest, name))


def overlay_control(task: ResearchTask, workspace: Path, control: str) -> None:
    """Replace editable files with a named calibration control."""
    control_root = task.control_dir(control)
    for name in task.editable_files:
        src = control_root / name
        if src.is_symlink() or not src.is_file():
            raise IntegrityError(f"control {control} missing regular file {name}")
        dest = resolve_inside(workspace, name)
        assert_regular_file(dest)
        shutil.copyfile(src, dest)


def source_hashes(workspace: Path, names: tuple[str, ...] | list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in names:
        path = resolve_inside(workspace, name)
        assert_regular_file(path)
        hashes[name] = sha256_file(path)
    return hashes


def stage_subject(workspace: Path, names: tuple[str, ...] | list[str], dest: Path) -> None:
    """Copy only editable regular files. Hidden graders are not staged."""
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir(parents=True)
    for name in names:
        src = resolve_inside(workspace, name)
        assert_regular_file(src)
        target = dest / name
        if target.exists() or target.is_symlink():
            raise IntegrityError("stage destination collided")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
