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
import subprocess


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
    source_hashes: dict[str, str]
    snapshot_sha256: str | None

    @property
    def diff_revision(self) -> str:
        return self.base_revision or "HEAD"


def assert_train_only(contract: AgentTaskContract, split: dict) -> None:
    """Refuse held-out ids from a frozen pack split."""
    assert_split_member(contract, split, "train")


def assert_split_member(contract: AgentTaskContract, split: dict, split_name: str) -> None:
    """Require a task to match one explicitly selected frozen split."""
    if split_name not in {"train", "development", "heldout"}:
        raise ValueError("evaluation split must be train, development, or heldout")
    heldout = set(split.get("heldout", []))
    members = set(split.get(split_name, []))
    if split_name == "train" and (contract.task_id in heldout or contract.split == "heldout"):
        raise ValueError("held-out task must not be executed")
    if split_name == "development" and contract.task_id in heldout:
        raise ValueError("held-out task must not be executed as development")
    if contract.split != split_name or contract.task_id not in members:
        raise ValueError(f"task is outside the frozen {split_name} split")


def verify_agent_workspace(workspace: Path, contract: AgentTaskContract) -> dict[str, str]:
    """Verify a task checkout identity and any frozen per-file source hashes."""
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("agent workspace is missing or unsafe")
    has_git = (root / ".git").exists()
    if contract.base_revision and has_git:
        try:
            revision = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("agent workspace is not a readable Git checkout") from exc
        if revision != contract.base_revision:
            raise ValueError("agent workspace is not at the pinned baseline revision")
        try:
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout
            index_flags = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-v"],
                check=True, capture_output=True, text=True, timeout=10,
            ).stdout.splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("agent workspace status could not be verified") from exc
        if status.strip():
            raise ValueError("agent workspace must be clean before candidate execution")
        if any(line[:1] in {"h", "S"} for line in index_flags):
            raise ValueError("agent workspace uses assume-unchanged or skip-worktree index flags")
    elif contract.base_revision and not contract.snapshot_sha256:
        raise ValueError("archive workspace requires a pinned full-snapshot SHA-256")
    actual: dict[str, str] = {}
    for name in contract.allowed_files:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"allowlisted task file is missing or unsafe: {name}")
        actual[name] = _sha256(path)
    if contract.source_hashes:
        if set(contract.source_hashes) != set(contract.allowed_files):
            raise ValueError("pinned source hash manifest must cover exactly the task allowlist")
        if actual != contract.source_hashes:
            raise ValueError("agent workspace source SHA-256 differs from the pinned baseline")
    elif contract.base_revision and has_git:
        # Historical tasks may predate file-level locks. Compare every file the
        # agent can edit to the pinned Git blob instead of treating HEAD as proof
        # that the worktree bytes are pristine.
        for name in contract.allowed_files:
            try:
                pinned = subprocess.run(
                    ["git", "-C", str(root), "show", f"{contract.base_revision}:{name}"],
                    check=True, capture_output=True, timeout=10,
                ).stdout
            except (OSError, subprocess.SubprocessError) as exc:
                raise ValueError(f"pinned baseline does not contain allowlisted file: {name}") from exc
            if (root / name).read_bytes() != pinned:
                raise ValueError("agent workspace source bytes differ from the pinned baseline")
    if contract.snapshot_sha256 and workspace_snapshot_sha256(root) != contract.snapshot_sha256:
        raise ValueError("agent workspace snapshot SHA-256 differs from the pinned baseline")
    return actual


def workspace_snapshot_sha256(root: Path) -> str:
    """Hash all regular workspace files without relying on Git history."""
    entries: list[bytes] = []
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise ValueError("agent workspace snapshot contains a symlink")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            entries.extend((name.encode("utf-8"), b"\0", bytes.fromhex(_sha256(path)), b"\n"))
    return hashlib.sha256(b"".join(entries)).hexdigest()


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
        source_hashes = source.get("file_sha256", {})
        if not isinstance(source_hashes, dict) or any(
            name not in files or not isinstance(digest, str) or len(digest) != 64
            for name, digest in source_hashes.items()
        ):
            raise ValueError("historical task source.file_sha256 is malformed")
        snapshot_sha256 = source.get("snapshot_sha256")
        if snapshot_sha256 is not None and (
            not isinstance(snapshot_sha256, str) or len(snapshot_sha256) != 64
        ):
            raise ValueError("historical task source.snapshot_sha256 is malformed")
        metadata = task.get("metadata", {})
        split = metadata.get("split") if isinstance(metadata, dict) else None
        if split is not None and split not in {"train", "development", "heldout"}:
            raise ValueError("task split must be train, development, or heldout")
        return AgentTaskContract(
            task_id=str(task["task_id"]),
            schema="historical",
            allowed_files=tuple(files),
            prompt_text=brief.read_text(encoding="utf-8"),
            base_revision=source.get("base_revision"),
            calibration_revision=source.get("calibration_revision"),
            split=split,
            input_hashes=hashes,
            source_hashes=dict(source_hashes),
            snapshot_sha256=snapshot_sha256,
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
    if split is not None and split not in {"train", "development", "heldout"}:
        raise ValueError("task split must be train, development, or heldout")
    return AgentTaskContract(
        task_id=task_id,
        schema="engineering_task_spec",
        allowed_files=tuple(files),
        prompt_text=prompt_text,
        base_revision=source.get("base_revision"),
        calibration_revision=source.get("calibration_revision"),
        split=split,
        input_hashes=hashes,
        source_hashes={},
        snapshot_sha256=None,
    )
