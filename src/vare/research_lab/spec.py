"""Load ResearchGym task packs. Hidden grader seeds are not part of a pack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vare.evidence import sha256_file, sha256_json

from .paths import PathViolation, safe_relative


class TaskSpecError(ValueError):
    """A task pack is missing a required field or has an illegal value."""


TASK_TYPES = frozenset({
    "training_correctness",
    "performance",
    "experiment_debug",
})

_REQUIRED = (
    "task_id",
    "task_type",
    "objective",
    "editable_files",
    "graded_functions",
    "max_iterations",
    "does_not_establish",
    "compute",
    "budget",
    "safety",
    "reproducibility",
)


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        catalog = parent / "benchmarks" / "research_lab_v1" / "catalog.json"
        if catalog.is_file():
            return parent
    raise TaskSpecError("benchmarks/research_lab_v1/catalog.json was not found")


def default_catalog_root() -> Path:
    return repo_root() / "benchmarks" / "research_lab_v1"


@dataclass(frozen=True, slots=True)
class ResearchTask:
    task_id: str
    task_type: str
    objective: str
    editable_files: tuple[str, ...]
    graded_functions: tuple[str, ...]
    max_iterations: int
    does_not_establish: str
    compute: dict[str, Any]
    budget: dict[str, Any]
    safety: dict[str, Any]
    reproducibility: dict[str, Any]
    root: Path
    spec_sha256: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "objective": self.objective,
            "editable_files": list(self.editable_files),
            "graded_functions": list(self.graded_functions),
            "max_iterations": self.max_iterations,
            "does_not_establish": self.does_not_establish,
            "compute": dict(self.compute),
            "budget": dict(self.budget),
            "safety": dict(self.safety),
            "reproducibility": dict(self.reproducibility),
            "spec_sha256": self.spec_sha256,
        }

    def control_dir(self, name: str) -> Path:
        safe_relative(name)
        path = self.root / "controls" / name
        if not path.is_dir() or path.is_symlink():
            raise TaskSpecError(f"missing control {name} for {self.task_id}")
        return path


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskSpecError(f"{label} must be an object")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x for x in value):
        raise TaskSpecError(f"{label} must be a non-empty list of strings")
    for item in value:
        safe_relative(item)
    return tuple(value)


def load_task(root: Path) -> ResearchTask:
    root = root.resolve()
    spec_path = root / "task.json"
    if not spec_path.is_file():
        raise TaskSpecError(f"missing task.json under {root.name}")
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskSpecError(f"{root.name} task.json is not JSON") from exc
    if not isinstance(raw, dict):
        raise TaskSpecError("task.json must be an object")
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise TaskSpecError(f"{root.name} missing {', '.join(missing)}")
    task_type = str(raw["task_type"])
    if task_type not in TASK_TYPES:
        raise TaskSpecError(f"unknown task_type {task_type}")
    try:
        max_iterations = int(raw["max_iterations"])
    except (TypeError, ValueError) as exc:
        raise TaskSpecError("max_iterations must be an integer") from exc
    if not 1 <= max_iterations <= 16:
        raise TaskSpecError("max_iterations must be between 1 and 16")
    editable = _strings(raw["editable_files"], "editable_files")
    graded = _strings(raw["graded_functions"], "graded_functions")
    compute = _object(raw["compute"], "compute")
    for key in ("wall_time_s", "cpu_time_s", "memory_mb", "max_output_bytes"):
        if key not in compute:
            raise TaskSpecError(f"compute.{key} is required")
    budget = _object(raw["budget"], "budget")
    safety = _object(raw["safety"], "safety")
    if safety.get("network") != "disallowed":
        raise TaskSpecError("safety.network must be disallowed")
    if safety.get("editable_only") is not True:
        raise TaskSpecError("safety.editable_only must be true")
    repro = _object(raw["reproducibility"], "reproducibility")
    if "public_seed" not in repro:
        raise TaskSpecError("reproducibility.public_seed is required")
    workspace = root / "workspace"
    if not workspace.is_dir():
        raise TaskSpecError(f"{raw['task_id']} has no workspace/")
    for name in editable:
        path = workspace / name
        if not path.is_file() or path.is_symlink():
            raise TaskSpecError(f"{raw['task_id']} workspace missing regular file {name}")
    return ResearchTask(
        task_id=str(raw["task_id"]),
        task_type=task_type,
        objective=str(raw["objective"]),
        editable_files=editable,
        graded_functions=graded,
        max_iterations=max_iterations,
        does_not_establish=str(raw["does_not_establish"]),
        compute=compute,
        budget=budget,
        safety=safety,
        reproducibility=repro,
        root=root,
        spec_sha256=sha256_file(spec_path),
    )


def load_catalog(catalog_root: Path | None = None) -> dict[str, ResearchTask]:
    root = (catalog_root or default_catalog_root()).resolve()
    catalog_path = root / "catalog.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskSpecError("catalog.json is unreadable") from exc
    if not isinstance(catalog, dict) or catalog.get("suite") != "research_lab_v1":
        raise TaskSpecError("catalog suite must be research_lab_v1")
    ids = catalog.get("tasks")
    if not isinstance(ids, list) or len(ids) < 3:
        raise TaskSpecError("catalog must list at least three tasks")
    tasks: dict[str, ResearchTask] = {}
    for task_id in ids:
        if not isinstance(task_id, str):
            raise TaskSpecError("catalog task ids must be strings")
        try:
            safe_relative(task_id)
        except PathViolation as exc:
            raise TaskSpecError("illegal task id") from exc
        task = load_task(root / "tasks" / task_id)
        if task.task_id != task_id:
            raise TaskSpecError(f"task id mismatch for {task_id}")
        if task_id in tasks:
            raise TaskSpecError(f"duplicate task id {task_id}")
        tasks[task_id] = task
    types = {task.task_type for task in tasks.values()}
    if types != TASK_TYPES:
        raise TaskSpecError(f"catalog must cover {sorted(TASK_TYPES)}")
    return tasks


def suite_manifest(catalog_root: Path | None = None) -> dict[str, Any]:
    root = (catalog_root or default_catalog_root()).resolve()
    files: list[dict[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        })
    payload = {"suite": "research_lab_v1", "files": files}
    payload["manifest_sha256"] = sha256_json(payload)
    return payload
