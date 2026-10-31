#!/usr/bin/env python3
"""Fixed-controller, multi-turn tool episodes with durable step-boundary resume.

This module composes existing VARE pieces; it does not reimplement them:

* tools and their response bounds: ``scripts/run_local_agent.py`` helpers
  (``tools_for``, ``bounded_*``, ``attempt_configured_edit``,
  ``attempt_exact_text_replacement``, ``normalize_tool_call_format``,
  ``no_tool_retry_reminder``, ``bounded_tool_result``);
* policies: the ``vare.learning_pipeline.policy.Policy`` protocol, so the
  same object type that the single-turn pipeline samples from (``MLXPolicy``
  with an optional LoRA adapter) drives an episode here;
* persistence and recovery: ``vare.trajectory_store.TrajectoryStore``
  (SQLite, append-only hash chain, leases, ``audit_trajectory_store``);
* statistics: ``vare.eval_stats`` (paired bootstrap, exact McNemar).

The controller is the code between the policy and the tools: prompt
rendering, tool-call parsing, the optional assists inherited from
``run_local_agent`` (tool-call normalization, no-tool reminders, bare-edit
recovery, automatic public tests after an edit, the finish gate), and all
budgets. ``ControllerConfig`` fixes every one of them, and its digest is part
of each episode's enrollment, so a resumed or compared episode cannot run
under a different controller. ``CONTROLLER_PRESETS["minimal"]`` turns every
assist off with the same budgets; comparing ``full`` with ``minimal`` under
one policy measures what the controller, not the model, contributes.

Final task success comes only from a ``TrustedGrader`` whose result passes
``trusted_verdict``. A missing grader, a grader error, or a result without
attested isolation is recorded as not trusted and never counts as success.

The ``ScriptedPolicy`` stub replays fixed text; it is synthetic plumbing for
CPU tests, not model behavior. Nothing here establishes a capability claim.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import run_local_agent as rla  # noqa: E402
from vare.agent_edit_format import ParseFailure, parse_submission  # noqa: E402
from vare.eval_stats import mcnemar_exact, paired_bootstrap_interval  # noqa: E402
from vare.evidence import sha256_json  # noqa: E402
from vare.learning_pipeline.policy import Generation, Policy  # noqa: E402
from vare.trajectory_store import (  # noqa: E402
    AgentStep,
    PolicyCheckpoint,
    TrajectoryLease,
    TrajectoryStore,
    audit_trajectory_store,
)
from vare.types import Task, Verification  # noqa: E402

read_regular_source = rla.read_regular_source

SCHEMA = "vare_long_horizon_episode_v1"
EDIT_TOOLS = frozenset({"apply_patch", "replace_text", "recovered_edit"})
READ_TOOLS = frozenset({"list_files", "read_file", "search_text"})
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TRUSTED_GRADE_SCHEMAS = frozenset({"learning_pipeline_repo_env_grade_v2"})
TRUSTED_VERDICT_CHANNELS = frozenset({"hmac_frames_v1"})
TERMINAL_STORE_STATES = frozenset({"complete", "verified"})
_TEST_PATH = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


class ResumeMismatchError(RuntimeError):
    """Replaying committed edits did not reproduce the recorded workspace."""


# --------------------------------------------------------------------------
# Controller


@dataclass(frozen=True)
class ControllerConfig:
    """Everything between the policy and the tools. Identical across policy arms."""

    name: str
    max_turns: int = 16
    max_new_tokens_per_turn: int = 512
    max_total_new_tokens: int = 4096
    max_calls_per_turn: int = 4
    wall_seconds: float = 1800.0
    temperature: float = 0.0
    top_p: float = 1.0
    reveal_target: bool = True
    # Assists inherited from run_local_agent. "minimal" disables all of them.
    normalize_tool_calls: bool = True
    max_no_tool_retries: int = 2
    recover_bare_edits: bool = True
    auto_public_tests_after_edit: bool = True
    finish_gate: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_turns <= 4095:
            raise ValueError("max_turns must be in [1, 4095]")
        if self.max_new_tokens_per_turn < 1 or self.max_total_new_tokens < 1:
            raise ValueError("token budgets must be positive")
        if self.max_calls_per_turn < 1 or self.max_no_tool_retries < 0:
            raise ValueError("invalid call or retry budget")
        if not self.wall_seconds > 0:
            raise ValueError("wall budget must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        return sha256_json({"schema": "vare_long_horizon_controller_v1", **self.to_dict()})


CONTROLLER_PRESETS: dict[str, ControllerConfig] = {
    "full": ControllerConfig(name="full"),
    "minimal": ControllerConfig(
        name="minimal", normalize_tool_calls=False, max_no_tool_retries=0,
        recover_bare_edits=False, auto_public_tests_after_edit=False, finish_gate=False,
    ),
}


# --------------------------------------------------------------------------
# Tasks and workspaces


def is_test_path(path: str) -> bool:
    return _TEST_PATH.search(path) is not None


@dataclass(frozen=True)
class EpisodeTask:
    """One task as the controller sees it. No grader inputs live here.

    ``target_files`` are the navigation ground truth (for learning-pipeline
    tasks, the single ``target_file`` that the v1 prompt already names). They
    are shown to the policy only when ``ControllerConfig.reveal_target``.
    """

    task_id: str
    repository: str
    split: str
    issue: str
    readable_files: tuple[str, ...]
    editable_files: tuple[str, ...]
    target_files: tuple[str, ...]
    synthetic: bool = False

    def __post_init__(self) -> None:
        if self.split != "train":
            # Held-out execution is deliberately unavailable until a frozen
            # protocol supplies a disjoint, pre-registered task set.
            raise ValueError("only Train-split tasks may run through this harness")
        if not self.editable_files or not set(self.editable_files) <= set(self.readable_files):
            raise ValueError("editable files must be a non-empty subset of readable files")
        if not set(self.target_files) <= set(self.editable_files):
            raise ValueError("target files must be editable")
        leaked = [p for p in self.readable_files if is_test_path(p)]
        if leaked:
            raise ValueError(f"test files must not be policy-readable: {leaked[:3]}")

    def sha256(self) -> str:
        return sha256_json(dataclasses.asdict(self))


WorkspaceFactory = Callable[[Path], None]
"""Materialize the task's starting workspace into an empty directory (with .git)."""


def git_snapshot(workspace: Path) -> None:
    env = {"PATH": os.environ.get("PATH", os.defpath), "GIT_CONFIG_GLOBAL": os.devnull,
           "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(workspace)}
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=vare", "-c", "user.email=vare@localhost", "commit", "-q",
                  "--no-gpg-sign", "-m", "episode base"]):
        subprocess.run(["git", *args], cwd=workspace, env=env, check=True, capture_output=True)


def directory_workspace_factory(source: Path) -> WorkspaceFactory:
    """Copy a local tree (without its VCS metadata) and pin it in a fresh Git repo."""

    def build(target: Path) -> None:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=True)
        git_snapshot(target)

    return build


# --------------------------------------------------------------------------
# Tool runtime (dispatch mirrors scripts/run_local_agent.py main())


PublicTestRunner = Callable[[Path], Mapping[str, Any]]


def _validate_arguments(schema: Mapping[str, Any], arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return "arguments must be a JSON object"
    properties = schema.get("properties", {})
    missing = [name for name in schema.get("required", []) if name not in arguments]
    if missing:
        return f"missing required arguments: {missing}"
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            return f"unknown arguments: {unknown}"
    for name, value in arguments.items():
        expected = properties.get(name, {}).get("type")
        if expected == "string" and not isinstance(value, str):
            return f"argument {name} must be a string"
        if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return f"argument {name} must be an integer"
    return None


class ToolRuntime:
    """The existing bounded tool set over one workspace."""

    def __init__(self, workspace: Path, task: EpisodeTask, *, public_tests: PublicTestRunner | None,
                 base_hashes: Mapping[str, str]):
        self.workspace = workspace
        self.task = task
        self.readable = set(task.readable_files)
        self.editable = set(task.editable_files)
        self.public_tests = public_tests
        self.base_hashes = dict(base_hashes)
        self.tools = rla.tools_for("unified_diff", public_tests=public_tests is not None)
        self.schemas = {item["function"]["name"]: item["function"]["parameters"] for item in self.tools}

    @property
    def tool_names(self) -> set[str]:
        return set(self.schemas)

    def file_hashes(self) -> dict[str, str]:
        return {name: hashlib.sha256(read_regular_source(self.workspace, name)).hexdigest()
                for name in sorted(self.editable)}

    def run_public_tests(self) -> dict[str, Any]:
        if self.public_tests is None:
            return {"passed": False, "error": "public test runner is not configured"}
        try:
            return dict(self.public_tests(self.workspace))
        except Exception as exc:  # the runner's failure is an observation, not success
            return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    def edit(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if name == "apply_patch":
            return rla.attempt_configured_edit(self.workspace, str(arguments["patch"]), "unified_diff",
                                               self.editable)
        if name == "replace_text":
            return rla.attempt_exact_text_replacement(
                self.workspace, arguments["path"], arguments["old_text"], arguments["new_text"],
                self.editable)
        if name == "recovered_edit":
            return rla.attempt_configured_edit(self.workspace, str(arguments["text"]), "unified_diff",
                                               self.editable)
        raise ValueError(f"not an edit tool: {name}")

    def execute(self, name: str, arguments: Mapping[str, Any], *, public_passed: bool | None,
                finish_gate: bool) -> dict[str, Any]:
        result: dict[str, Any]
        if name == "list_files":
            result = rla.bounded_list_page(sorted(self.readable), offset=arguments.get("offset", 0))
        elif name == "read_file":
            path = rla.resolve_allowed_path(arguments["path"], self.readable)
            if path is None:
                result = {"error": "path is not allowed; use list_files to retrieve exact allowed paths"}
            else:
                source = read_regular_source(self.workspace, path)
                result = rla.bounded_source_page(
                    source.decode("utf-8"), start_line=arguments.get("start_line", 1),
                    end_line=arguments.get("end_line"), char_offset=arguments.get("char_offset", 0),
                    source_identity={"resolved_path": path,
                                     "source_sha256": hashlib.sha256(source).hexdigest()})
        elif name == "search_text":
            query = arguments["query"]
            if not query:
                result = {"error": "query must not be empty"}
            else:
                matches = []
                for path in sorted(self.readable):
                    text = read_regular_source(self.workspace, path).decode("utf-8")
                    for line_no, line in enumerate(text.splitlines(), 1):
                        if query in line:
                            matches.append({"path": path, "line": line_no, "text": line})
                result = rla.bounded_search_results(matches, offset=arguments.get("offset", 0))
        elif name in ("apply_patch", "replace_text"):
            result = self.edit(name, arguments)
        elif name == "run_public_tests":
            result = self.run_public_tests()
        elif name == "check_syntax":
            path = rla.resolve_allowed_path(arguments["path"], self.readable)
            if path is None:
                result = {"error": "path is not allowed", "allowed_paths": sorted(self.readable)}
            else:
                import ast
                try:
                    ast.parse(read_regular_source(self.workspace, path).decode("utf-8"), filename=path)
                    result = {"syntax_valid": True, "path": path}
                except SyntaxError as exc:
                    result = {"syntax_valid": False, "path": path, "error": str(exc)}
        elif name == "finish":
            unchanged = self.file_hashes() == self.base_hashes
            if finish_gate and unchanged:
                result = {"finished": False, "error": "source is unchanged; inspect and edit before finishing"}
            elif finish_gate and self.public_tests is not None and not public_passed:
                result = {"finished": False, "error": "public tests have not passed; run them and repair failures"}
            else:
                result = {"finished": True}
        else:
            raise ValueError(f"unsupported tool: {name}")
        return rla.bounded_tool_result(result, None)


def action_failed(name: str, result: Mapping[str, Any]) -> bool:
    if "error" in result:
        return True
    if name in EDIT_TOOLS:
        return result.get("applied") is not True
    if name == "run_public_tests":
        return result.get("passed") is not True
    if name == "check_syntax":
        return result.get("syntax_valid") is not True
    if name == "finish":
        return result.get("finished") is not True
    return False


# --------------------------------------------------------------------------
# Prompt rendering (role/content strings only, so any chat template works)


def system_prompt(task: EpisodeTask, runtime: ToolRuntime, controller: ControllerConfig) -> str:
    tools = json.dumps([item["function"] for item in runtime.tools], sort_keys=True)
    target = (f"The bug is in `{', '.join(task.target_files)}`. " if controller.reveal_target else "")
    public = ("After changing source, call run_public_tests and repair failures. "
              if runtime.public_tests is not None else "")
    return (
        "You are an offline code repair agent working in a prepared repository workspace. "
        "Each reply must contain one or more tool calls, each written exactly as "
        '<tool_call>{"name": <tool name>, "arguments": {...}}</tool_call>. '
        f"At most {controller.max_calls_per_turn} calls per reply are executed. "
        f"Tools (JSON schema): {tools}\n"
        "Readable paths are returned by list_files; only these are editable: "
        f"{', '.join(sorted(task.editable_files))}. {target}{public}"
        "Call finish when the fix is complete. A hidden evaluator judges the final workspace.\n\n"
        f"Repository: {task.repository}\nIssue:\n{task.issue.strip()}"
    )


def tool_results_message(calls: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    payload = [{"name": c.get("name"), "result": c.get("result")} for c in calls]
    return {"role": "user", "content": "TOOL RESULTS:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def messages_for_step(action: Mapping[str, Any]) -> list[dict[str, str]]:
    """Deterministic messages appended by one committed step."""
    if action.get("kind") == "observation":
        return []
    out = [{"role": "assistant", "content": str(action["model_output"])}]
    if action["calls"]:
        out.append(tool_results_message(action["calls"]))
    reminder = action["controller"].get("reminder_text")
    if reminder:
        out.append({"role": "user", "content": reminder})
    return out


# --------------------------------------------------------------------------
# Policies


class ScriptedPolicy:
    """Deterministic CPU stub: reply k is ``script[k]``; a SYNTHETIC record.

    The reply index is the number of assistant messages already in the
    history, so a resumed episode (history rebuilt from the store) continues
    with exactly the reply an uninterrupted run would produce.
    """

    synthetic = True

    def __init__(self, script: Sequence[str], policy_id: str = "scripted-stub"):
        self.script = tuple(script)
        self.policy_id = policy_id
        self.calls = 0

    def checkpoint_sha256(self) -> str:
        return sha256_json({"policy_id": self.policy_id, "script": list(self.script)})

    def generate(self, messages, *, max_tokens, temperature, top_p, seed, task=None) -> Generation:
        self.calls += 1
        index = sum(m["role"] == "assistant" for m in messages)
        text = self.script[index] if index < len(self.script) else "I have nothing further."
        words = text.split()
        if len(words) > max_tokens:
            text, words = " ".join(words[:max_tokens]), words[:max_tokens]
            finish = "length"
        else:
            finish = "stop"
        prompt_tokens = sum(len(str(m["content"]).split()) for m in messages)
        return Generation(text, prompt_tokens, len(words), 0.0, finish)

    def close(self) -> None:
        pass

    def peak_memory_bytes(self) -> int | None:
        return None


def mlx_policy(model: str, adapter_path: str | None = None, **kwargs: Any) -> Policy:
    """Lazy hook for the Mac: the same MLXPolicy the single-turn pipeline uses."""
    from vare.learning_pipeline.policy import MLXPolicy

    return MLXPolicy(model, adapter_path, **kwargs)


def directory_sha256(path: Path) -> str:
    """Content digest of a model/adapter directory (file names and bytes)."""
    digest = hashlib.sha256()
    for item in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode() + b"\0")
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


@dataclass
class PolicyArm:
    """A named policy under test; ``checkpoint`` binds every step to its identity."""

    name: str
    policy: Policy
    checkpoint: PolicyCheckpoint


# --------------------------------------------------------------------------
# Trusted grading (fail closed)


class TrustedGrader(Protocol):
    name: str
    version: int

    def config_sha256(self) -> str: ...

    def grade(self, *, task: EpisodeTask, files: Mapping[str, str]) -> Mapping[str, Any]: ...


def trusted_verdict(result: Any) -> dict[str, Any]:
    """Accept only an executed, isolation-attested v2 grade. Anything else fails closed."""
    if not isinstance(result, Mapping):
        return {"trusted": False, "passed": False, "reason": "grade_not_a_mapping"}
    isolation = result.get("isolation")
    checks = [
        (result.get("schema") in TRUSTED_GRADE_SCHEMAS, "untrusted_grade_schema"),
        (result.get("executed") is True, "grade_not_executed"),
        (isinstance(isolation, Mapping), "isolation_record_missing"),
        (isinstance(isolation, Mapping) and isolation.get("mode") == "required", "isolation_not_required"),
        (isinstance(isolation, Mapping) and isolation.get("attested") is True, "isolation_unverified"),
        (isinstance(isolation, Mapping) and isolation.get("verdict_channel") in TRUSTED_VERDICT_CHANNELS,
         "untrusted_verdict_channel"),
        (type(result.get("passed")) is bool, "passed_not_boolean"),
    ]
    for ok, reason in checks:
        if not ok:
            return {"trusted": False, "passed": False, "reason": reason,
                    "grader_reason": str(result.get("reason"))[:200]}
    return {"trusted": True, "passed": bool(result["passed"]), "reason": str(result.get("reason"))[:200]}


class UnavailableGrader:
    """Recorded when no trusted grader exists for a task: never a success."""

    name = "no_trusted_grader"
    version = 0

    def config_sha256(self) -> str:
        return sha256_json({"grader": None})

    def grade(self, *, task: EpisodeTask, files: Mapping[str, str]) -> Mapping[str, Any]:
        return {"schema": None, "passed": False, "reason": "grader_unavailable"}


class RepoEnvTrustedGrader:
    """Adapter for the sandboxed ``TrustedRepoEnv`` (open PR #170), duck-typed.

    The task record holds grader-only inputs (bug patch, test ids); they never
    reach the policy. Only ``isolation="required"`` is used.
    """

    name = "learning_pipeline_repo_env_grade_v2"
    version = 2

    def __init__(self, env: Any, task_record: Mapping[str, Any], *, timeout: float = 180):
        self.env = env
        self.task_record = task_record
        self.timeout = timeout

    def config_sha256(self) -> str:
        info = self.env.isolation_record("required")
        stable = {k: info.get(k) for k in ("mode", "backend", "config_digest", "verdict_channel",
                                            "grader", "runner_sha256", "static_scan")}
        return sha256_json(stable)

    def grade(self, *, task: EpisodeTask, files: Mapping[str, str]) -> Mapping[str, Any]:
        target = self.task_record["target_file"]
        if set(files) != {target}:
            return {"schema": None, "passed": False, "reason": "grader_grades_single_target_file"}
        return self.env.grade(
            bug_patch=self.task_record["_patch"], target_file=target, candidate_text=files[target],
            fail_to_pass=self.task_record["fail_to_pass"], pass_to_pass=self.task_record["pass_to_pass"],
            timeout=self.timeout, isolation="required")


# --------------------------------------------------------------------------
# Episode execution


def episode_id(arm: str, controller: ControllerConfig, task_id: str, seed: int) -> str:
    value = f"{arm}:{controller.name}:{task_id}:s{seed}"
    if len(value) > 160 or re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None:
        value = f"{arm}:{controller.name}:{hashlib.sha256(task_id.encode()).hexdigest()[:24]}:s{seed}"
    return value


@dataclass(frozen=True)
class EpisodeSpec:
    task: EpisodeTask
    seed: int

    def episode_id(self, arm: str, controller: ControllerConfig) -> str:
        return episode_id(arm, controller, self.task.task_id, self.seed)


CrashHook = Callable[[str, int], None]


@dataclass
class EpisodeHarness:
    """Runs episodes for ONE (policy arm, controller) pair through one store."""

    root: Path
    arm: PolicyArm
    controller: ControllerConfig
    factories: Mapping[str, WorkspaceFactory]
    public_tests: Mapping[str, PublicTestRunner] = field(default_factory=dict)
    grader_for: Callable[[EpisodeTask], TrustedGrader | None] | None = None
    lease_seconds: float = 600.0
    max_attempts: int = 5
    worker_id: str = "lh-worker"
    crash_hook: CrashHook | None = None
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        path = self.root / "trajectories.sqlite"
        self.store = (TrajectoryStore(path) if path.exists() else
                      TrajectoryStore.create(path, run_id=f"lh-{self.arm.name}-{self.controller.name}"[:160],
                                             record_bytes=512 * 1024))
        self.store.register_policy(self.arm.checkpoint)
        self.tasks: dict[str, EpisodeTask] = {}

    # ---- enrollment
    def enroll(self, specs: Iterable[EpisodeSpec]) -> list[str]:
        ids = []
        for spec in specs:
            ident = spec.episode_id(self.arm.name, self.controller)
            meta = {"schema": SCHEMA, "arm": self.arm.name, "seed": spec.seed,
                    "controller": self.controller.to_dict(), "controller_sha256": self.controller.sha256(),
                    "task_sha256": spec.task.sha256(), "repository": spec.task.repository,
                    "target_files": list(spec.task.target_files),
                    "policy": dataclasses.asdict(self.arm.checkpoint),
                    "synthetic_policy": bool(getattr(self.arm.policy, "synthetic", True)),
                    "synthetic_task": spec.task.synthetic,
                    "public_tests": spec.task.task_id in self.public_tests}
            task = Task(id=spec.task.task_id, prompt=spec.task.issue, family=spec.task.repository, metadata=meta)
            self.store.submit(ident, task, expected_steps=self.controller.max_turns, group_id=ident,
                              member=0, group_size=1, max_attempts=self.max_attempts,
                              lifetime_seconds=86400, now=self.clock())
            self.tasks[ident] = spec.task
            ids.append(ident)
        return ids

    # ---- workspace reconstruction
    def _workspace(self, ident: str) -> Path:
        return self.root / "workspaces" / ident.replace(":", "_")

    def rebuild(self, ident: str) -> tuple[ToolRuntime, list[dict[str, Any]]]:
        """Fresh workspace + replay of committed edits; fail closed on any drift."""
        task = self.tasks[ident]
        workspace = self._workspace(ident)
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        self.factories[task.task_id](workspace)
        base = {name: hashlib.sha256(read_regular_source(workspace, name)).hexdigest()
                for name in sorted(task.editable_files)}
        runtime = ToolRuntime(workspace, task, public_tests=self.public_tests.get(task.task_id), base_hashes=base)
        steps = list(self.store.iter_steps(ident))
        for step in steps:
            for call in step["action"].get("calls", []):
                if call.get("name") in EDIT_TOOLS and (call.get("result") or {}).get("applied") is True:
                    replayed = runtime.edit(call["name"], call["arguments"])
                    if replayed.get("applied") is not True:
                        raise ResumeMismatchError(f"committed edit did not replay at step {step['index']}")
            if step["state"].get("file_sha256") and runtime.file_hashes() != step["state"]["file_sha256"]:
                raise ResumeMismatchError(f"workspace hash drift after step {step['index']}")
        return runtime, steps

    # ---- execution
    def _claim(self, stage: str) -> TrajectoryLease | None:
        return self.store.claim(stage, self.worker_id, lease_seconds=self.lease_seconds, now=self.clock())

    def run(self, *, wait_for_expired: bool = True, poll_seconds: float = 0.05) -> None:
        """Drain execute then verify work; resumes any interrupted episode."""
        while True:
            lease = self._claim("execute")
            if lease is None:
                counts = self.store.counts()
                if wait_for_expired and counts.get("running", 0):
                    time.sleep(poll_seconds)
                    continue
                break
            self._execute(lease)
        if self.grader_for is not None:
            while (lease := self._claim("verify")) is not None:
                self._verify(lease)

    def _commit(self, lease: TrajectoryLease, step: AgentStep) -> None:
        self.store.heartbeat(lease, lease_seconds=self.lease_seconds, now=self.clock())
        self.store.append_step(lease, step, now=self.clock())

    def _execute(self, lease: TrajectoryLease) -> None:
        ident = lease.trajectory_id
        if ident not in self.tasks:
            raise KeyError(f"{ident} was enrolled but its task is not loaded in this process")
        task = self.tasks[ident]
        meta = json.loads(self.store.trajectory(ident)["task"])["task"]["metadata"]
        if (meta["controller_sha256"] != self.controller.sha256() or meta["arm"] != self.arm.name
                or meta["policy"] != dataclasses.asdict(self.arm.checkpoint)):
            raise ResumeMismatchError(f"{ident} was enrolled under a different controller or policy")
        runtime, steps = self.rebuild(ident)
        controller = self.controller
        policy = self.arm.policy
        seed = meta["seed"]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt(task, runtime, controller)},
            {"role": "user", "content": "Begin. First call list_files."},
        ]
        for step in steps:
            messages.extend(messages_for_step(step["action"]))
        state: dict[str, Any] = dict(steps[-1]["state"]) if steps else {
            "file_sha256": runtime.file_hashes(), "edit_applied": False, "public_passed": None,
            "tokens_total": 0, "no_tool_retries": 0, "stop_reason": None}
        wall_used = sum(float(s["action"].get("timing", {}).get("step_seconds", 0.0)) for s in steps)
        for index in range(len(steps), controller.max_turns):
            event = f"{ident}:{index}"
            if state["stop_reason"] is not None:
                self._commit(lease, AgentStep(index, event, self.arm.checkpoint,
                                              {"kind": "observation", "padding": True}, {}, dict(state)))
                continue
            started = time.perf_counter()
            remaining = controller.max_total_new_tokens - state["tokens_total"]
            stop = ("token_budget_exhausted" if remaining <= 0 else
                    "wall_time_exhausted" if wall_used >= controller.wall_seconds else None)
            if stop is not None:
                state = {**state, "stop_reason": stop}
                self._commit(lease, AgentStep(index, event, self.arm.checkpoint,
                                              {"kind": "observation", "terminal": True, "stop_reason": stop},
                                              {}, dict(state)))
                continue
            limit = min(controller.max_new_tokens_per_turn, remaining)
            gen_started = time.perf_counter()
            generation = policy.generate(messages, max_tokens=limit, temperature=controller.temperature,
                                         top_p=controller.top_p, seed=seed * 4096 + index)
            gen_seconds = time.perf_counter() - gen_started
            state, action = self._turn(runtime, state, generation, limit, ident, index)
            step_seconds = time.perf_counter() - started
            wall_used += step_seconds
            action["timing"] = {"generation_seconds": round(gen_seconds, 6),
                                "policy_reported_seconds": generation.seconds,
                                "step_seconds": round(step_seconds, 6)}
            if self.crash_hook is not None:
                self.crash_hook("after_tools", index)
            self._commit(lease, AgentStep(index, event, self.arm.checkpoint, action,
                                          {"results_sha256": sha256_json([{k: v for k, v in c.items() if k != "seconds"}
                                                                 for c in action["calls"]])}, dict(state)))
            messages.extend(messages_for_step(action))
            if self.crash_hook is not None:
                self.crash_hook("after_commit", index)
        self.store.finish(lease, now=self.clock())

    def _turn(self, runtime: ToolRuntime, state: dict[str, Any], generation: Generation, limit: int,
              ident: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        controller = self.controller
        raw = generation.text
        assists: dict[str, Any] = {"normalized_tool_call": False, "reminder_text": None,
                                   "recovered_bare_edit": False, "auto_public_tests": 0, "finish_refused": 0}
        state = {**state, "tokens_total": state["tokens_total"] + int(generation.completion_tokens)}
        calls: list[dict[str, Any]] = []
        stop_reason: str | None = None
        if generation.finish == "prompt_too_long":
            stop_reason = "context_budget_exhausted"
        parse_text = raw
        if stop_reason is None and controller.normalize_tool_calls:
            parse_text, assists["normalized_tool_call"] = rla.normalize_tool_call_format(raw, runtime.tool_names)
        matches = list(TOOL_CALL.finditer(parse_text)) if stop_reason is None else []
        public_passed = state["public_passed"]
        edit_applied = state["edit_applied"]

        def after_edit(result: dict[str, Any]) -> dict[str, Any]:
            nonlocal public_passed, edit_applied
            if result.get("applied") is True:
                edit_applied = True
                if controller.auto_public_tests_after_edit and runtime.public_tests is not None:
                    public = runtime.run_public_tests()
                    public_passed = public.get("passed") is True
                    assists["auto_public_tests"] += 1
                    result = {**result, "automatic_public_test": public}
            return result

        if stop_reason is None and not matches:
            recovered = None
            if controller.recover_bare_edits:
                parsed = parse_submission(raw, edit_format="unified_diff")
                if not isinstance(parsed, ParseFailure) or parsed.reason != "no_edit":
                    recovered = after_edit(runtime.edit("recovered_edit", {"text": raw}))
                    assists["recovered_bare_edit"] = True
                    calls.append({"name": "recovered_edit", "arguments": {"text": raw}, "valid": True,
                                  "failed": action_failed("recovered_edit", recovered),
                                  "result": rla.bounded_tool_result(recovered, None)})
            if recovered is None:
                changed = runtime.file_hashes() != runtime.base_hashes
                if not changed and state["no_tool_retries"] < controller.max_no_tool_retries:
                    state = {**state, "no_tool_retries": state["no_tool_retries"] + 1}
                    assists["reminder_text"] = rla.no_tool_retry_reminder(
                        raw, edit_format="unified_diff", allowed_paths=sorted(runtime.editable),
                        token_limit=limit)
                else:
                    stop_reason = "no_tool_call_after_edit" if changed else "no_tool_call_retries_exhausted"
            elif recovered.get("applied") is not True:
                stop_reason = "malformed_edit"
        for match in matches[:controller.max_calls_per_turn]:
            call: dict[str, Any] = {"raw_call": match.group(1)[:4000]}
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError as exc:
                calls.append({**call, "name": None, "valid": False, "failed": True,
                              "result": {"error": f"invalid JSON: {exc.msg}"}})
                continue
            name = payload.get("name") if isinstance(payload, dict) else None
            arguments = payload.get("arguments", {}) if isinstance(payload, dict) else None
            problem = ("unknown tool" if name not in runtime.tool_names
                       else _validate_arguments(runtime.schemas[name], arguments))
            if problem is not None or not isinstance(name, str) or not isinstance(arguments, dict):
                calls.append({"name": name if isinstance(name, str) else None, "arguments": arguments,
                              "valid": False, "failed": True, "result": {"error": problem}})
                continue
            tool_started = time.perf_counter()
            try:
                result = runtime.execute(name, arguments, public_passed=public_passed,
                                         finish_gate=controller.finish_gate)
            except Exception as exc:  # tool-level failures are observations
                result = rla.bounded_tool_result({"error": f"{type(exc).__name__}: {exc}"}, None)
            if name in EDIT_TOOLS:
                result = after_edit(result)
            if name == "run_public_tests":
                public_passed = result.get("passed") is True
            if name == "finish" and result.get("finished") is not True:
                assists["finish_refused"] += 1
            calls.append({"name": name, "arguments": arguments, "valid": True,
                          "failed": action_failed(name, result), "result": result,
                          "seconds": round(time.perf_counter() - tool_started, 6)})
            if name == "finish" and result.get("finished") is True:
                stop_reason = "agent_finished"
                break
        dropped = max(0, len(matches) - controller.max_calls_per_turn)
        state = {**state, "file_sha256": runtime.file_hashes(), "edit_applied": edit_applied,
                 "public_passed": public_passed, "stop_reason": stop_reason}
        action = {"kind": "tool" if calls else "inference", "turn": index, "model_output": raw,
                  "generation": {"prompt_tokens": int(generation.prompt_tokens),
                                 "completion_tokens": int(generation.completion_tokens),
                                 "finish": generation.finish, "max_tokens": limit},
                  "controller": assists, "calls": calls, "calls_dropped": dropped}
        if calls:
            action.update(operation_id=f"{ident}:op:{index}", semantics="at_least_once")
        return state, action

    def _verify(self, lease: TrajectoryLease) -> None:
        ident = lease.trajectory_id
        task = self.tasks[ident]
        head = self.store.trajectory(ident)["head"]
        assert self.grader_for is not None
        grader: TrustedGrader = self.grader_for(task) or UnavailableGrader()
        self.store.register_verifier(grader.name, grader.version, grader.config_sha256())
        runtime, _ = self.rebuild(ident)
        files = {name: read_regular_source(runtime.workspace, name).decode("utf-8")
                 for name in sorted(task.editable_files)
                 if runtime.file_hashes()[name] != runtime.base_hashes[name]}
        started = time.perf_counter()
        raw: Mapping[str, Any] = {}
        if isinstance(grader, UnavailableGrader):
            verdict = {"trusted": False, "passed": False, "reason": "grader_unavailable"}
        elif not files:
            verdict = {"trusted": False, "passed": False, "reason": "no_source_change_not_graded"}
        else:
            try:
                raw = grader.grade(task=task, files=files)
                verdict = trusted_verdict(raw)
            except Exception as exc:
                raw, verdict = {}, {"trusted": False, "passed": False,
                                    "reason": f"grader_error:{type(exc).__name__}"}
        seconds = time.perf_counter() - started
        summary = {k: raw.get(k) for k in ("schema", "reason", "executed", "f2p_passed", "f2p_total",
                                           "p2p_passed", "p2p_total", "candidate_sha256")
                   if isinstance(raw, Mapping) and k in raw}
        passed = bool(verdict["trusted"] and verdict["passed"])
        value = Verification(score=1.0 if passed else 0.0, passed=passed, confidence=1.0 if verdict["trusted"] else 0.0,
                             verifier_version=grader.version, verifier_name=grader.name,
                             trusted=bool(verdict["trusted"]),
                             metadata={"trajectory_sha256": head, "verifier_config_sha256": grader.config_sha256(),
                                       "verdict": verdict, "grade_summary": summary,
                                       "grade_sha256": _grade_digest(raw),
                                       "grading_seconds": round(seconds, 6),
                                       "graded_files": sorted(files)})
        self.store.record_verification(lease, value, now=self.clock())


def _grade_digest(raw: Any) -> str | None:
    try:
        return sha256_json(dict(raw))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Learning-pipeline Train tasks (real repositories; exercised on the Mac)


def load_train_manifest(path: Path) -> list[dict[str, Any]]:
    """Read a Train manifest. Any non-Train row stops the load (held-out stays closed)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [r.get("instance_id") for r in rows if r.get("split") != "train"]
    if bad:
        raise ValueError(f"manifest contains non-Train rows: {bad[:3]}")
    return rows


def repo_workspace_factory(clone: Path, commit: str, bug_patch: str) -> WorkspaceFactory:
    """Copy of a pinned clone at ``commit`` with the bug applied, re-pinned as the episode base."""

    def build(target: Path) -> None:
        env = {"PATH": os.environ.get("PATH", os.defpath), "GIT_CONFIG_GLOBAL": os.devnull,
               "GIT_CONFIG_NOSYSTEM": "1"}
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", "--no-checkout", str(clone), str(target)],
                       check=True, capture_output=True, env=env)
        subprocess.run(["git", "checkout", "-q", "--detach", commit], cwd=target, check=True,
                       capture_output=True, env=env)
        subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], cwd=target, input=bug_patch.encode(),
                       check=True, capture_output=True, env=env)
        shutil.rmtree(target / ".git")
        git_snapshot(target)

    return build


def readable_python_files(clone: Path, commit: str, *, limit: int = 400) -> tuple[str, ...]:
    """Tracked non-test ``.py`` files at ``commit`` (navigation scope)."""
    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit], cwd=clone, check=True,
                            capture_output=True, text=True).stdout.splitlines()
    files = sorted(p for p in listed if p.endswith(".py") and not is_test_path(p))
    if len(files) > limit:
        raise ValueError(f"navigation scope exceeds {limit} files")
    return tuple(files)


def learning_pipeline_task(record: Mapping[str, Any], readable: Sequence[str]) -> EpisodeTask:
    """Policy-visible view of a Train record: issue text and file scope only."""
    target = str(record["target_file"])
    files = tuple(sorted(set(readable) | {target}))
    return EpisodeTask(task_id=str(record["instance_id"]), repository=str(record["upstream_repo"]),
                       split=str(record["split"]), issue=str(record["_problem"]), readable_files=files,
                       editable_files=(target,), target_files=(target,))


# --------------------------------------------------------------------------
# Metrics (computed from the audited store only)


def _real_steps(steps: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [s for s in steps if s["action"].get("kind") in ("tool", "inference")]


def episode_metrics(store: TrajectoryStore, ident: str) -> dict[str, Any]:
    row = store.trajectory(ident)
    spec = json.loads(row["task"])
    meta = spec["task"]["metadata"]
    steps = list(store.iter_steps(ident))
    real = _real_steps(steps)
    calls = [c for s in real for c in s["action"]["calls"]]
    targets = set(spec["task"].get("metadata", {}).get("target_files", []))
    terminal = [s for s in steps if s["action"].get("terminal")]
    last_state = steps[-1]["state"] if steps else {}
    stop_reason = last_state.get("stop_reason") or ("turn_budget_exhausted" if len(steps) == row["expected_steps"]
                                                    else None)
    # navigation
    first_edit = next((i for i, c in enumerate(calls) if c.get("name") in EDIT_TOOLS
                       and (c.get("result") or {}).get("applied") is True), None)
    target_reads = [i for i, c in enumerate(calls) if c.get("name") == "read_file" and c.get("valid")
                    and (c.get("result") or {}).get("resolved_path") in targets]
    edited_paths = {p for c in calls if c.get("name") in EDIT_TOOLS
                    and (c.get("result") or {}).get("applied") is True for p in (c["result"].get("paths") or [])}
    nontarget_reads = sum(1 for c in calls if c.get("name") == "read_file"
                          and (c.get("result") or {}).get("resolved_path") not in targets
                          and (c.get("result") or {}).get("resolved_path") is not None)
    # recovery: a failed action followed later by a successful action of the same tool
    failed_idx = [i for i, c in enumerate(calls) if c.get("failed")]
    recovered = sum(1 for i in failed_idx if any(
        not d.get("failed") and _category(d.get("name")) == _category(calls[i].get("name"))
        for d in calls[i + 1:]))
    edits = [c for c in calls if c.get("name") in EDIT_TOOLS and c.get("valid")]
    public = [c for c in calls if c.get("name") == "run_public_tests" and c.get("valid")]
    auto_public = sum(s["action"]["controller"].get("auto_public_tests", 0) for s in real)
    verification = json.loads(row["verification"]) if row["verification"] else None
    trusted = bool(verification and verification["trusted"])
    attempts = _execute_claims(store, ident)
    return {
        "episode_id": ident, "task_id": spec["task"]["id"], "repository": meta["repository"],
        "arm": meta["arm"], "controller": meta["controller"]["name"], "controller_sha256": meta["controller_sha256"],
        "seed": meta["seed"], "synthetic_policy": meta["synthetic_policy"], "synthetic_task": meta["synthetic_task"],
        "store_state": row["state"], "completed": row["state"] in TERMINAL_STORE_STATES,
        "stop_reason": stop_reason, "turns": len(real), "padding_steps": len(steps) - len(real) - len(terminal),
        "execute_attempts": attempts, "resumed": attempts > 1,
        "tool_invocations": len(calls), "valid_tool_calls": sum(bool(c.get("valid")) for c in calls),
        "tool_call_validity": (sum(bool(c.get("valid")) for c in calls) / len(calls)) if calls else None,
        "no_tool_turns": sum(1 for s in real if not s["action"]["calls"]),
        "calls_dropped": sum(s["action"].get("calls_dropped", 0) for s in real),
        "navigation": {
            "target_read": bool(target_reads),
            "target_read_before_first_edit": bool(target_reads) and (first_edit is None or target_reads[0] < first_edit),
            "edits_only_targets": bool(edited_paths) and edited_paths <= targets,
            "correct": bool(target_reads) and first_edit is not None and target_reads[0] < first_edit
            and edited_paths <= targets,
            "calls_before_target_read": target_reads[0] if target_reads else None,
            "nontarget_reads": nontarget_reads,
        },
        "edits": {"attempted": len(edits), "applied": sum((c.get("result") or {}).get("applied") is True for c in edits),
                  "application_rate": (sum((c.get("result") or {}).get("applied") is True for c in edits) / len(edits))
                  if edits else None,
                  "final_source_changed": bool(last_state.get("edit_applied")) and bool(edited_paths)},
        "public_tests": {"available": bool(meta.get("public_tests")), "explicit_runs": len(public),
                         "automatic_runs": auto_public, "final_passed": last_state.get("public_passed"),
                         "finished_while_failing": stop_reason == "agent_finished" and bool(meta.get("public_tests"))
                         and last_state.get("public_passed") is not True},
        "recovery": {"failed_actions": len(failed_idx), "recovered": recovered,
                     "rate": recovered / len(failed_idx) if failed_idx else None},
        "controller_assists": {
            "normalized_tool_calls": sum(bool(s["action"]["controller"].get("normalized_tool_call")) for s in real),
            "reminders": sum(bool(s["action"]["controller"].get("reminder_text")) for s in real),
            "recovered_bare_edits": sum(bool(s["action"]["controller"].get("recovered_bare_edit")) for s in real),
            "automatic_public_tests": auto_public,
            "finish_refusals": sum(s["action"]["controller"].get("finish_refused", 0) for s in real)},
        "compute": {
            "generated_tokens": sum(s["action"]["generation"]["completion_tokens"] for s in real),
            "prompt_tokens": sum(s["action"]["generation"]["prompt_tokens"] for s in real),
            "generation_seconds": round(sum(s["action"]["timing"]["generation_seconds"] for s in real), 6),
            "tool_seconds": round(sum(c.get("seconds", 0.0) for c in calls), 6),
            "step_seconds": round(sum(s["action"]["timing"]["step_seconds"] for s in real), 6),
            "grading_seconds": (verification or {}).get("metadata", {}).get("grading_seconds"),
        },
        "grade_recorded": verification is not None,
        "graded": trusted,
        "trusted_grade": trusted,
        "grade_reason": (verification or {}).get("metadata", {}).get("verdict", {}).get("reason", "ungraded"),
        # Primary outcome. Ungraded or untrusted is never success.
        "final_success": bool(verification and verification["trusted"] and verification["passed"]),
        "trajectory_head": row["head"],
    }


def _category(name: Any) -> str:
    return "edit" if name in EDIT_TOOLS else str(name)


def _execute_claims(store: TrajectoryStore, ident: str) -> int:
    with store.connection() as db:
        return int(db.execute(
            "SELECT count(*) FROM events WHERE json_extract(event,'$.job_id')=? AND "
            "json_extract(event,'$.kind')='work_claimed' AND json_extract(event,'$.details.stage')='execute'",
            (ident,)).fetchone()[0])


def canonical_trajectory(store: TrajectoryStore, ident: str) -> list[dict[str, Any]]:
    """Step payloads without wall-clock fields, for determinism comparisons."""
    out = []
    for step in store.iter_steps(ident):
        action = {k: v for k, v in step["action"].items() if k != "timing"}
        action["calls"] = [{k: v for k, v in c.items() if k != "seconds"} for c in action.get("calls", [])]
        out.append({**step, "action": action})
    return out


def run_metrics(root: Path) -> dict[str, Any]:
    """Audit the store first; metrics from an unaudited store are refused."""
    path = Path(root) / "trajectories.sqlite"
    audit = audit_trajectory_store(path)
    store = TrajectoryStore(path, audit_only=True)
    with store.connection() as db:
        ids = [r[0] for r in db.execute("SELECT id FROM trajectories ORDER BY id")]
    episodes = [episode_metrics(store, ident) for ident in ids]
    return {"audit": audit, "episodes": episodes, "aggregate": aggregate(episodes)}


def _mean(values: Iterable[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(episodes)
    if n == 0:
        return {"episodes": 0}
    return {
        "episodes": n,
        "completion_rate": sum(e["completed"] for e in episodes) / n,
        "trusted_graded_rate": sum(e["trusted_grade"] for e in episodes) / n,
        "ungraded": sum(not e["grade_recorded"] for e in episodes),
        "untrusted_grades": sum(e["grade_recorded"] and not e["trusted_grade"] for e in episodes),
        # Denominator is every enrolled episode: ungraded counts as failure.
        "final_success_rate": sum(e["final_success"] for e in episodes) / n,
        "navigation_correct_rate": sum(e["navigation"]["correct"] for e in episodes) / n,
        "tool_call_validity_mean": _mean(e["tool_call_validity"] for e in episodes),
        "edit_application_rate_mean": _mean(e["edits"]["application_rate"] for e in episodes),
        "recovery_rate_mean": _mean(e["recovery"]["rate"] for e in episodes),
        "resumed_episodes": sum(e["resumed"] for e in episodes),
        "stop_reasons": {r: sum(e["stop_reason"] == r for e in episodes)
                         for r in sorted({str(e["stop_reason"]) for e in episodes})},
        "generated_tokens": sum(e["compute"]["generated_tokens"] for e in episodes),
        "prompt_tokens": sum(e["compute"]["prompt_tokens"] for e in episodes),
        "tool_invocations": sum(e["tool_invocations"] for e in episodes),
        "step_seconds": round(sum(e["compute"]["step_seconds"] for e in episodes), 6),
        "controller_assists": {k: sum(e["controller_assists"][k] for e in episodes)
                               for k in episodes[0]["controller_assists"]},
        "synthetic": any(e["synthetic_policy"] or e["synthetic_task"] for e in episodes),
    }


def paired_comparison(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *,
                      metric: str = "final_success", samples: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Paired by (task, seed); bootstrap resamples repositories (cluster means).

    Both sides must share one controller digest unless the comparison is a
    declared controller ablation (same arm, different controller).
    """
    def key(e: Mapping[str, Any]) -> tuple[str, int]:
        return (e["task_id"], e["seed"])

    lmap, rmap = {key(e): e for e in left}, {key(e): e for e in right}
    if set(lmap) != set(rmap):
        raise ValueError("paired comparison requires identical (task, seed) sets")
    same_controller = {e["controller_sha256"] for e in left} == {e["controller_sha256"] for e in right}
    same_arm = {e["arm"] for e in left} == {e["arm"] for e in right}
    if not same_controller and not same_arm:
        raise ValueError("policy comparison under different controllers is not attributable")
    keys = sorted(lmap)
    lv = [float(bool(lmap[k][metric])) for k in keys]
    rv = [float(bool(rmap[k][metric])) for k in keys]
    clusters: dict[str, list[tuple[float, float]]] = {}
    for k, lval, rval in zip(keys, lv, rv):
        clusters.setdefault(lmap[k]["repository"], []).append((lval, rval))
    names = sorted(clusters)
    cl = [sum(a for a, _ in clusters[c]) / len(clusters[c]) for c in names]
    cr = [sum(b for _, b in clusters[c]) / len(clusters[c]) for c in names]
    gained = sum(1 for x, y in zip(lv, rv) if x == 0 and y == 1)
    lost = sum(1 for x, y in zip(lv, rv) if x == 1 and y == 0)
    return {"metric": metric, "pairs": len(keys), "repositories": len(names),
            "kind": "policy" if same_controller else "controller_ablation",
            "left_rate": sum(lv) / len(lv), "right_rate": sum(rv) / len(rv),
            "repository_cluster_bootstrap": paired_bootstrap_interval(cl, cr, samples=samples, seed=seed),
            "task_level_mcnemar": mcnemar_exact(gained, lost)}


# --------------------------------------------------------------------------
# Synthetic CPU fixture (labelled synthetic; for smoke runs and tests only)


FIXTURE_SOURCE = {
    "calc/__init__.py": "from .ops import add, mul\n",
    "calc/ops.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
    "calc/util.py": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
    "tests/test_ops.py": "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
}


def write_fixture(target: Path) -> None:
    for name, text in FIXTURE_SOURCE.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git_snapshot(target)


def fixture_task(task_id: str = "fixture-calc-add") -> EpisodeTask:
    return EpisodeTask(task_id=task_id, repository="fixture/calc", split="train",
                       issue="add(2, 3) returns -1 instead of 5.",
                       readable_files=("calc/__init__.py", "calc/ops.py", "calc/util.py"),
                       editable_files=("calc/ops.py",), target_files=("calc/ops.py",), synthetic=True)


def fixture_public_tests(workspace: Path) -> dict[str, Any]:
    """Runs only ``tests/test_ops.py`` in a subprocess (not a security boundary)."""
    proc = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.');"
                           "from tests.test_ops import test_add; test_add()"],
                          cwd=workspace, capture_output=True, text=True, timeout=30)
    return {"passed": proc.returncode == 0, "returncode": proc.returncode, "stderr": proc.stderr[-1000:]}


def _call(name: str, **arguments: Any) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"


FIXTURE_SCRIPT = (
    _call("list_files"),
    _call("search_text", query="def add"),
    _call("read_file", path="calc/ops.py"),
    _call("replace_text", path="calc/ops.py", old_text="return a + b", new_text="return a - b"),
    "<tool_call>{not json}</tool_call>",
    # Bare JSON: only a controller with tool-call normalization executes it.
    json.dumps({"name": "replace_text", "arguments": {"path": "calc/ops.py", "old_text": "return a - b",
                                                       "new_text": "return a + b"}}),
    _call("run_public_tests"),
    _call("finish", summary="fixed add"),
)


# --------------------------------------------------------------------------
# CLI


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="CPU smoke: scripted stub policy on the synthetic fixture")
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--controller", choices=sorted(CONTROLLER_PRESETS), default="full")
    smoke.add_argument("--seeds", type=int, default=2)
    run = sub.add_parser("run", help="one policy arm on learning-pipeline Train tasks (needs MLX)")
    run.add_argument("--arm", required=True, help="arm name, e.g. base, rft, grpo")
    run.add_argument("--policy-version", type=int, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--controller", choices=sorted(CONTROLLER_PRESETS), default="full")
    run.add_argument("--manifest", type=Path, required=True, help="a Train manifest; non-Train rows are refused")
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--limit", type=int, required=True)
    run.add_argument("--seeds", type=int, default=1)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--grader-factory", help="module:function(task_record, env) -> TrustedGrader | None")
    report = sub.add_parser("report", help="audit a run directory and print metrics")
    report.add_argument("run", type=Path)
    compare = sub.add_parser("compare", help="paired, repository-clustered comparison of run directories")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("others", type=Path, nargs="+")
    return parser.parse_args(argv)


def _load_factory(spec: str | None) -> Callable[..., Any] | None:
    if not spec:
        return None
    import importlib

    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def run_learning_pipeline_arm(args: argparse.Namespace) -> Path:
    from vare.learning_pipeline import curriculum, repo_env

    records = load_train_manifest(args.manifest)[: args.limit]
    records = curriculum.attach_payloads(records, args.cache)
    grader_factory = _load_factory(args.grader_factory)
    factories: dict[str, WorkspaceFactory] = {}
    graders: dict[str, TrustedGrader | None] = {}
    tasks = []
    for record in records:
        env = repo_env.env_for(record, args.cache)
        if not (env.path / ".git").exists():
            env.materialize()
        task = learning_pipeline_task(record, readable_python_files(env.path, env.commit))
        factories[task.task_id] = repo_workspace_factory(env.path, env.commit, record["_patch"])
        graders[task.task_id] = grader_factory(record, env) if grader_factory else None
        tasks.append(task)
    digest = sha256_json({"model": args.model,
                          "adapter": directory_sha256(args.adapter) if args.adapter else None})
    policy = mlx_policy(args.model, str(args.adapter) if args.adapter else None)
    arm = PolicyArm(args.arm, policy, PolicyCheckpoint(args.arm, args.policy_version, digest))
    root = args.output / f"{args.arm}-{args.controller}"
    harness = EpisodeHarness(root, arm, CONTROLLER_PRESETS[args.controller], factories=factories,
                             grader_for=lambda t: graders.get(t.task_id))
    harness.enroll(EpisodeSpec(task, seed) for task in tasks for seed in range(args.seeds))
    harness.run()
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    if args.command == "report":
        print(json.dumps(run_metrics(args.run), indent=2, sort_keys=True))
        return 0
    if args.command == "compare":
        base = run_metrics(args.baseline)["episodes"]
        out = {str(o): paired_comparison(base, run_metrics(o)["episodes"]) for o in args.others}
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        root = run_learning_pipeline_arm(args)
        print(json.dumps(run_metrics(root)["aggregate"], indent=2, sort_keys=True))
        return 0
    root = args.output / f"stub-{args.controller}"
    policy = ScriptedPolicy(FIXTURE_SCRIPT)
    arm = PolicyArm("stub", policy, PolicyCheckpoint("stub", 0, policy.checkpoint_sha256()))
    task = fixture_task()
    harness = EpisodeHarness(root, arm, CONTROLLER_PRESETS[args.controller],
                             factories={task.task_id: write_fixture},
                             public_tests={task.task_id: fixture_public_tests},
                             grader_for=lambda _task: None)
    harness.enroll(EpisodeSpec(task, seed) for seed in range(args.seeds))
    harness.run()
    result = run_metrics(root)
    print(json.dumps({"aggregate": result["aggregate"], "audit": result["audit"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
