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
from pathlib import PurePosixPath
import re


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


_PROTECTED_TASK_DIRS = {
    ".git", "test", "tests", "grader", "graders", "evaluator", "evaluators",
    "known-fix", "known-fixes", "known_fixes", "fix", "fixes", "history",
    "historical", "reference", "references", "reference-data", "__tests__",
}
_PROTECTED_REFERENCE_BASENAMES = {
    "known_fix", "known_fix.py", "known_fix.patch", "known_fix.diff",
    "known-fix", "known-fix.py", "known-fix.patch", "known-fix.diff",
    "hidden_test", "hidden_tests", "hidden_test.py", "hidden_tests.py",
}
_TEST_OR_GRADER_BASENAME = re.compile(
    r"(?:^|[_.-])(?:test|tests|grader|graders|grade|grading|evaluator|evaluators)"
    r"(?:$|[_.-])"
)
_TEST_OR_GRADER_PREFIXES = ("test", "grader", "grade", "evaluator")
_TEST_OR_GRADER_SUFFIXES = ("test", "grader", "grade")


def _protected_test_or_grader_name(component: str) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", component).casefold()
    stem = normalized.split(".", 1)[0]
    return (
        normalized in _PROTECTED_REFERENCE_BASENAMES
        or _TEST_OR_GRADER_BASENAME.search(normalized) is not None
        or stem.startswith(_TEST_OR_GRADER_PREFIXES)
        or stem.endswith(tuple(f"_{suffix}" for suffix in _TEST_OR_GRADER_SUFFIXES))
    )


def candidate_visible_source_files(paths: object) -> tuple[str, ...]:
    """Return safe source paths, fail-closed on conventional test/grader names.

    The task's explicit root ``test_public.py`` remains available only to the
    dedicated sandboxed test runner. It is never a source read/edit/search
    capability. This policy is defense in depth for malformed task allowlists.
    """
    if not isinstance(paths, (tuple, list, set, frozenset)):
        return ()
    visible: set[str] = set()
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
            continue
        parsed = PurePosixPath(raw_path)
        if (parsed.is_absolute() or raw_path != parsed.as_posix()
                or any(part in {"", ".", ".."} for part in parsed.parts)):
            continue
        protected_dirs = False
        for raw_component in parsed.parts[:-1]:
            component = raw_component.casefold()
            if (component in _PROTECTED_TASK_DIRS
                    or _protected_test_or_grader_name(raw_component)):
                protected_dirs = True
                break
        if protected_dirs:
            continue
        # Split camel-case names before applying the same delimiter rule, so
        # TestPublic.py and implementation_test.cpp are both protected.
        if _protected_test_or_grader_name(parsed.name):
            continue
        visible.add(raw_path)
    return tuple(sorted(visible))


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
    def candidate_visible_source_files(self) -> tuple[str, ...]:
        """Candidate read/edit/search set after test/grader path filtering."""
        return candidate_visible_source_files(self.allowed_files)

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
    )
