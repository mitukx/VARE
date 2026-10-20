"""Read either the historical agent task or an EngineeringTaskSpec.

The engineering form is the MLSys pack layout: editable files live in
``subject/``, and ``evaluator/`` is not part of the candidate allowlist.
This module does not open evaluator files.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(name: str) -> str:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not name or path.parts[:1] == ("evaluator",):
        raise ValueError(f"editable file escapes the subject workspace: {name}")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class AgentTaskContract:
    task_id: str
    schema: str
    allowed_files: tuple[str, ...]
    prompt_text: str
    base_revision: str | None
    calibration_revision: str | None
    split: str | None
    input_hashes: dict[str, str]

    @property
    def diff_revision(self) -> str:
        return self.base_revision or "HEAD"


def assert_train_only(contract: AgentTaskContract, split: dict) -> None:
    """Refuse held-out ids from a frozen pack split."""
    assert_split_member(contract, split, "train")


def assert_split_member(contract: AgentTaskContract, split: dict, split_name: str) -> None:
    """Require a task to match one explicitly selected frozen split."""
    if split_name not in {"train", "heldout"}:
        raise ValueError("evaluation split must be train or heldout")
    heldout = set(split.get("heldout", []))
    members = set(split.get(split_name, []))
    if split_name == "train" and (contract.task_id in heldout or contract.split == "heldout"):
        raise ValueError("held-out task must not be executed")
    if contract.split != split_name or contract.task_id not in members:
        raise ValueError(f"task is outside the frozen {split_name} split")


def load_agent_task(task_root: Path) -> AgentTaskContract:
    root = Path(task_root)
    task_path = root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    source = task.get("source")
    if not isinstance(source, dict):
        raise TypeError("task.json is missing source")
    hashes = {"task.json": _sha256(task_path)}
    brief = root / "TASK.md"
    if brief.is_file():
        hashes["TASK.md"] = _sha256(brief)
    lock = root / "protocol.lock.json"
    if lock.is_file():
        hashes["protocol.lock.json"] = _sha256(lock)
    historical_grader = root / "evaluator" / "grade.py"
    if "files" in source:
        if not brief.is_file() or not lock.is_file() or not historical_grader.is_file():
            raise ValueError("historical task is missing TASK.md, protocol.lock.json, or grader.py")
        hashes["grader.py"] = _sha256(historical_grader)
        files = [_safe_relative(name) for name in source["files"]]
        if not files:
            raise ValueError("historical task has no source files")
        return AgentTaskContract(
            task_id=str(task["task_id"]),
            schema="historical",
            allowed_files=tuple(files),
            prompt_text=brief.read_text(encoding="utf-8"),
            base_revision=source.get("base_revision"),
            calibration_revision=source.get("calibration_revision"),
            split=None,
            input_hashes=hashes,
        )
    metadata = task.get("metadata")
    if not isinstance(metadata, dict) or "editable_files" not in metadata:
        raise ValueError("unsupported task schema")
    files = [_safe_relative(name) for name in metadata["editable_files"]]
    if not files:
        raise ValueError("engineering task has no editable files")
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("engineering task prompt is empty")
    if brief.is_file():
        prompt_text = brief.read_text(encoding="utf-8")
    else:
        prompt_text = prompt
        hashes["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    task_id = task.get("task_id", task.get("id"))
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task id is missing")
    split = metadata.get("split")
    if split is not None and split not in {"train", "heldout"}:
        raise ValueError("task split must be train or heldout")
    return AgentTaskContract(
        task_id=task_id,
        schema="engineering_task_spec",
        allowed_files=tuple(files),
        prompt_text=prompt_text,
        base_revision=source.get("base_revision"),
        calibration_revision=source.get("calibration_revision"),
        split=split,
        input_hashes=hashes,
    )
