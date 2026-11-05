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

Final task success requires a passing ``TrustedGrader`` result and, when public
tests are configured for the episode, passing public tests. A missing grader,
a grader error, or a result without attested isolation is recorded as not
trusted and never counts as success.

The ``ScriptedPolicy`` stub replays fixed text; it is synthetic plumbing for
CPU tests, not model behavior. Nothing here establishes a capability claim.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
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
from vare.learning_pipeline.policy import Generation, MLXPolicy, Policy  # noqa: E402
from vare.trajectory_store import (  # noqa: E402
    AgentStep,
    PolicyCheckpoint,
    TrajectoryLease,
    TrajectoryStore,
    audit_trajectory_store,
)
from vare.types import Task, Verification  # noqa: E402

read_regular_source = rla.read_regular_source

SCHEMA = "vare_long_horizon_episode_v2"
EDIT_TOOLS = frozenset({"apply_patch", "replace_text", "recovered_edit"})
READ_TOOLS = frozenset({"list_files", "read_file", "search_text", "list_symbols"})
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TRUSTED_GRADE_SCHEMAS = frozenset({"vare_function_oracle_v1"})
TERMINAL_STORE_STATES = frozenset({"complete", "verified"})
_TEST_PATH = re.compile(r"(^|/)(tests?|testing)(/|$)|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


class ResumeMismatchError(RuntimeError):
    """Replaying committed edits did not reproduce the recorded workspace."""


@dataclass
class TraceGeneration(Generation):
    """Generation plus exact behavior tokens for the long-horizon GRPO bridge."""

    prompt_ids: list[int] | None = None
    completion_ids: list[int] | None = None
    sampling_temperature: float | None = None
    sampling_top_p: float | None = None


class TraceableMLXPolicy(MLXPolicy):
    """MLXPolicy variant that retains token IDs without changing frozen policy code."""

    def generate(self, messages, *, max_tokens, temperature, top_p, seed, task=None) -> TraceGeneration:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        ids = [int(token) for token in self.prompt_ids(messages)]
        if len(ids) > self.max_prompt_tokens:
            return TraceGeneration("", len(ids), 0, 0.0, "prompt_too_long",
                                   prompt_ids=ids, completion_ids=[],
                                   sampling_temperature=float(temperature), sampling_top_p=float(top_p))
        self._mx.random.seed(seed)
        sampler = make_sampler(temp=temperature, top_p=top_p if temperature > 0 else 0.0,
                               min_p=0.0, top_k=0)
        t0 = time.time()
        text, n, finish = "", 0, "length"
        completion_ids: list[int] = []
        for resp in stream_generate(self.model, self.tokenizer, ids, max_tokens=max_tokens, sampler=sampler):
            text += resp.text
            n = resp.generation_tokens
            completion_ids.append(int(resp.token))
            if resp.finish_reason:
                finish = "stop" if resp.finish_reason == "stop" else "length"
        return TraceGeneration(text, len(ids), n, round(time.time() - t0, 3), finish,
                               prompt_ids=ids, completion_ids=completion_ids,
                               sampling_temperature=float(temperature), sampling_top_p=float(top_p))


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
    source_revision: str | None = None
    workspace_patch_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.split not in {"train", "development"}:
            # Exploration is not an execution split. Development is accepted
            # here only so a separately frozen cohort can use the harness;
            # enrollment below requires its protocol digest.
            raise ValueError("episode tasks must use the Train or Development split")
        if not self.editable_files or not set(self.editable_files) <= set(self.readable_files):
            raise ValueError("editable files must be a non-empty subset of readable files")
        if not set(self.target_files) <= set(self.editable_files):
            raise ValueError("target files must be editable")
        leaked = [p for p in self.readable_files if is_test_path(p)]
        if leaked:
            raise ValueError(f"test files must not be policy-readable: {leaked[:3]}")
        if self.source_revision is not None and re.fullmatch(r"[0-9a-f]{40}", self.source_revision) is None:
            raise ValueError("source_revision must be a full commit SHA")
        if (self.workspace_patch_sha256 is not None
                and re.fullmatch(r"[0-9a-f]{64}", self.workspace_patch_sha256) is None):
            raise ValueError("workspace_patch_sha256 must be a SHA-256 digest")

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
        self.tools = rla.tools_for(
            "unified_diff", public_tests=public_tests is not None, include_navigation=True,
        )
        self.schemas = {item["function"]["name"]: item["function"]["parameters"] for item in self.tools}
        self.search_guard = rla.SearchLoopGuard(empty_search_limit=3)

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
                self.editable,
                expected_source_sha256=arguments.get("expected_source_sha256", ""))
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
                self.search_guard.reset()
        elif name == "search_text":
            query = arguments["query"]
            if not query:
                result = {"error": "query must not be empty"}
            elif not isinstance(query, str):
                result = {"error": "query must be a string"}
            else:
                raw_path = arguments.get("path")
                path = None
                if raw_path is not None:
                    path = rla.resolve_allowed_path(raw_path, self.readable)
                    if path is None:
                        result = {"error": "path is not an exact allowlisted source file"}
                redirect = (self.search_guard.intercept(query, path)
                            if raw_path is None or path is not None else None)
                if redirect is not None:
                    result = redirect
                elif raw_path is not None and path is None:
                    pass
                else:
                    matches = []
                    for source_path in ([path] if path is not None else sorted(self.readable)):
                        text = read_regular_source(self.workspace, source_path).decode("utf-8")
                        for line_no, line in enumerate(text.splitlines(), 1):
                            if query in line:
                                matches.append({"path": source_path, "line": line_no, "text": line})
                    result = rla.bounded_search_results(matches, offset=arguments.get("offset", 0))
                    result.update({"query": query, "match_mode": "case_sensitive_literal",
                                   "match_count": len(matches)})
                    redirect_hint = self.search_guard.observe(query, len(matches), path)
                    if redirect_hint is not None:
                        result["controller_redirect"] = redirect_hint
        elif name == "list_symbols":
            path = rla.resolve_allowed_path(arguments.get("path", ""), self.readable)
            query = arguments.get("query", "")
            if path is None:
                result = {"error": "path is not an exact allowlisted source file"}
            elif not isinstance(query, str):
                result = {"error": "query must be a string"}
            else:
                source = read_regular_source(self.workspace, path, max_bytes=1024 * 1024)
                result = rla.list_python_symbols(source.decode("utf-8"), query=query,
                                                 offset=arguments.get("offset", 0),
                                                 limit=arguments.get("limit", 20))
                result["path"] = path
                result["source_sha256"] = hashlib.sha256(source).hexdigest()
                self.search_guard.reset()
        elif name in ("apply_patch", "replace_text"):
            result = self.edit(name, arguments)
            if result.get("applied") is True:
                self.search_guard.reset()
        elif name == "run_public_tests":
            result = self.run_public_tests()
        elif name == "check_syntax":
            result = rla.check_syntax_source(
                self.workspace, arguments["path"], self.readable,
            )
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
        "search_text is case-sensitive literal search; after related empty searches, use list_symbols or read a bounded source range. "
        "replace_text requires the current source_sha256 returned by read_file. "
        "Call finish when the fix is complete. The operator may run an independent grader after the episode; "
        "you will only see grader feedback if it is configured as a tool.\n\n"
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
    """Lazy hook for the Mac: MLXPolicy with trajectory token tracing."""
    return TraceableMLXPolicy(model, adapter_path, **kwargs)


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
    """Accept only parent-owned cases run through the separate-process function oracle."""
    if not isinstance(result, Mapping):
        return {"trusted": False, "passed": False, "reason": "grade_not_a_mapping"}
    isolation = result.get("isolation")
    report = result.get("call_report")
    cases = report.get("cases") if isinstance(report, Mapping) else None
    if not isinstance(cases, list) or not cases or any(
        not isinstance(case, Mapping) or type(case.get("passed")) is not bool for case in cases
    ):
        return {"trusted": False, "passed": False, "reason": "case_results_invalid"}
    if not isinstance(report, Mapping) or type(report.get("passed")) is not bool:
        return {"trusted": False, "passed": False, "reason": "call_report_passed_invalid"}
    if report["passed"] is not all(case["passed"] for case in cases):
        return {"trusted": False, "passed": False, "reason": "call_report_inconsistent"}
    for name in ("candidate_sha256", "case_set_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(result.get(name, ""))) is None:
            return {"trusted": False, "passed": False, "reason": f"{name}_invalid"}
    checks = [
        (result.get("schema") in TRUSTED_GRADE_SCHEMAS, "untrusted_grade_schema"),
        (result.get("executed") is True, "grade_not_executed"),
        (isinstance(isolation, Mapping), "isolation_record_missing"),
        (isinstance(isolation, Mapping) and isolation.get("mode") == "docker_function_call",
         "isolation_not_function_call"),
        (isinstance(isolation, Mapping) and isolation.get("candidate_isolated") is True,
         "candidate_isolation_unverified"),
        (isinstance(isolation, Mapping) and isolation.get("expected_values_parent_owned") is True,
         "expected_values_not_parent_owned"),
        (isinstance(report, Mapping) and report.get("task_kind") == "correctness", "call_report_missing"),
        (type(result.get("passed")) is bool, "passed_not_boolean"),
        (isinstance(report, Mapping) and result.get("passed") is report.get("passed"),
         "grade_result_inconsistent"),
    ]
    for ok, reason in checks:
        if not ok:
            return {"trusted": False, "passed": False, "reason": reason,
                    "grader_reason": str(result.get("reason"))[:200]}
    return {"trusted": True, "passed": bool(result["passed"]), "reason": str(result.get("reason"))[:200]}


def trusted_final_success(
    *,
    store_state: str,
    expected_steps: int,
    committed_steps: int,
    verification: Mapping[str, Any] | None,
    expected_trajectory_sha256: str,
    expected_verifier_config_sha256: str | None,
    initial_source_hashes: Mapping[str, str],
    final_source_hashes: Mapping[str, str],
    public_tests_configured: bool,
    public_tests_passed: Any,
    expected_task_sha256: str | None = None,
    expected_source_revision: str | None = None,
    expected_workspace_patch_sha256: str | None = None,
    require_task_binding: bool = False,
) -> bool:
    """Require a complete, source-bound grade, an edit, and public success."""
    if store_state not in TERMINAL_STORE_STATES or committed_steps != expected_steps:
        return False
    if (not isinstance(initial_source_hashes, Mapping)
            or not isinstance(final_source_hashes, Mapping)
            or not initial_source_hashes
            or set(initial_source_hashes) != set(final_source_hashes)
            or not any(initial_source_hashes[path] != final_source_hashes[path]
                       for path in final_source_hashes)):
        return False
    if not isinstance(verification, Mapping) or verification.get("trusted") is not True:
        return False
    if verification.get("passed") is not True:
        return False
    metadata = verification.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    if metadata.get("trajectory_sha256") != expected_trajectory_sha256:
        return False
    if (not expected_verifier_config_sha256
            or metadata.get("verifier_config_sha256") != expected_verifier_config_sha256):
        return False
    raw_grade = metadata.get("raw_grade")
    graded_files = metadata.get("graded_files")
    if not isinstance(raw_grade, Mapping) or not isinstance(graded_files, list) or len(graded_files) != 1:
        return False
    graded_path = graded_files[0]
    if (not isinstance(graded_path, str)
            or raw_grade.get("candidate_sha256") != final_source_hashes.get(graded_path)):
        return False
    if require_task_binding and (
        metadata.get("task_sha256") != expected_task_sha256
        or metadata.get("source_revision") != expected_source_revision
        or metadata.get("workspace_patch_sha256") != expected_workspace_patch_sha256
    ):
        return False
    return not public_tests_configured or public_tests_passed is True


class UnavailableGrader:
    """Recorded when no trusted grader exists for a task: never a success."""

    name = "no_trusted_grader"
    version = 0

    def config_sha256(self) -> str:
        return sha256_json({"grader": None})

    def grade(self, *, task: EpisodeTask, files: Mapping[str, str]) -> Mapping[str, Any]:
        return {"schema": None, "passed": False, "reason": "grader_unavailable"}


class FunctionOracleGrader:
    """Parent-owned call/result oracle using the existing isolated candidate scorer.

    This supports a named Python function contract only. It does not execute a
    candidate-controlled test suite and never receives verdict logic from the
    candidate workspace. Optional support modules are parent-pinned source
    needed by a target module's imports; they are hashed with the oracle.
    """

    name = "vare_function_oracle"
    version = 1

    def __init__(self, *, task_id: str, target_file: str, calls: Sequence[Any], image: str,
                 timeout_s: float = 10, memory_mb: int = 512, rtol: float = 0, atol: float = 0,
                 support_files: Mapping[str, str] | None = None,
                 additional_source_hashes: Mapping[str, str] | None = None,
                 module: str | None = None, wrapper_source: str | None = None):
        from pathlib import PurePosixPath
        from vare.isolated_candidate import FunctionCall

        path = PurePosixPath(target_file)
        if path.is_absolute() or ".." in path.parts or not target_file.endswith(".py"):
            raise ValueError("target_file must be a relative Python source path")
        if not calls or any(not isinstance(call, FunctionCall) for call in calls):
            raise ValueError("function oracle requires parent-owned FunctionCall cases")
        if len({call.case_id for call in calls}) != len(calls):
            raise ValueError("function oracle case IDs must be unique")
        self.task_id = task_id
        self.target_file = target_file
        self.module = module or target_file[:-3].replace("/", ".")
        self.calls = tuple(calls)
        if any(call.module != self.module for call in self.calls):
            raise ValueError("every oracle call must target the declared source module")
        if len({call.function for call in self.calls}) != 1:
            raise ValueError("every oracle call must target one declared function")
        self.image = image
        self.timeout_s = timeout_s
        self.memory_mb = memory_mb
        self.rtol = rtol
        self.atol = atol
        self.support_files = dict(support_files or {})
        if target_file in self.support_files:
            raise ValueError("support files must not replace the candidate target")
        for name, source in self.support_files.items():
            if not isinstance(name, str):
                raise ValueError("support files must use safe relative Python source paths")
            relative = PurePosixPath(name)
            if ("\\" in name or relative.is_absolute()
                    or relative.as_posix() != name
                    or any(part in {"", ".", "..", ".git"} for part in name.split("/"))
                    or not name.endswith(".py")):
                raise ValueError("support files must use safe relative Python source paths")
            if not isinstance(source, str) or len(source.encode("utf-8")) > 128 * 1024:
                raise ValueError("support file source must be UTF-8 text no larger than 128 KiB")
        if wrapper_source is not None:
            if (not isinstance(module, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", module) is None
                    or not isinstance(wrapper_source, str)
                    or len(wrapper_source.encode("utf-8")) > 16 * 1024):
                raise ValueError("a wrapper requires a simple module name and bounded source text")
        self.wrapper_source = wrapper_source
        self.additional_source_hashes = dict(additional_source_hashes or {})
        if any(not isinstance(name, str) or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
               for name, digest in self.additional_source_hashes.items()):
            raise ValueError("additional source identities must map names to SHA-256 digests")
        repo_root = Path(__file__).resolve().parents[1]
        self.source_identity = {
            "grader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_harness_sha256": hashlib.sha256(
                (repo_root / "src/vare/isolated_candidate.py").read_bytes()).hexdigest(),
            "isolation_runner_sha256": hashlib.sha256(
                (repo_root / "src/vare/container_isolation.py").read_bytes()).hexdigest(),
            "support_files_sha256": {
                name: hashlib.sha256(source.encode("utf-8")).hexdigest()
                for name, source in sorted(self.support_files.items())
            },
            "wrapper_sha256": (hashlib.sha256(wrapper_source.encode("utf-8")).hexdigest()
                               if wrapper_source is not None else None),
            "additional_sources_sha256": dict(sorted(self.additional_source_hashes.items())),
        }
        self._config_sha256 = sha256_json({
            "schema": "vare_function_oracle_config_v1", "task_id": task_id,
            "target_file": target_file, "module": self.module,
            "image": image, "timeout_s": timeout_s,
            "memory_mb": memory_mb,
            "rtol": rtol, "atol": atol,
            "wrapper_sha256": self.source_identity["wrapper_sha256"],
            "calls": [dataclasses.asdict(call) for call in self.calls],
            "source_identity": self.source_identity,
        })
        # The store binds one verifier name/version to one immutable config.
        # A digest suffix lets a run safely grade multiple task-specific oracle
        # contracts without rebinding the same verifier identity.
        self.name = f"vare_function_oracle:{self._config_sha256}"

    def config_sha256(self) -> str:
        return self._config_sha256

    def grade(self, *, task: EpisodeTask, files: Mapping[str, str]) -> Mapping[str, Any]:
        from vare.isolated_candidate import grade_calls

        if task.task_id != self.task_id or set(files) != {self.target_file}:
            return {"schema": None, "passed": False, "reason": "function_oracle_scope_mismatch"}
        source = files[self.target_file]
        if len(source.encode("utf-8")) > 128 * 1024:
            return {"schema": None, "passed": False, "reason": "candidate_source_too_large"}
        with tempfile.TemporaryDirectory(prefix="vare-function-oracle-") as temp:
            subject = Path(temp)
            source_path = subject / self.target_file
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source, encoding="utf-8")
            for name, support_source in sorted(self.support_files.items()):
                support_path = subject / name
                support_path.parent.mkdir(parents=True, exist_ok=True)
                support_path.write_text(support_source, encoding="utf-8")
            if self.wrapper_source is not None:
                (subject / f"{self.module}.py").write_text(self.wrapper_source, encoding="utf-8")
            report = grade_calls(subject, self.calls, task_kind="correctness", image=self.image,
                                 timeout_s=self.timeout_s, memory_mb=self.memory_mb,
                                 rtol=self.rtol, atol=self.atol)
        raw = report.to_dict()
        return {
            "schema": "vare_function_oracle_v1", "executed": True,
            "passed": bool(report.passed), "reason": "all_cases_passed" if report.passed else "case_failed",
            "candidate_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "case_set_sha256": self._config_sha256, "call_report": raw,
            "source_identity": dict(self.source_identity),
            "isolation": {"mode": "docker_function_call", "candidate_isolated": True,
                          "expected_values_parent_owned": True},
        }


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
    protocol_sha256: str | None = None
    crash_hook: CrashHook | None = None
    fault_injection: Mapping[str, Any] = field(default_factory=dict)
    clock: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if (self.protocol_sha256 is not None
                and re.fullmatch(r"[0-9a-f]{64}", self.protocol_sha256) is None):
            raise ValueError("protocol_sha256 must be a SHA-256 digest")
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
            if spec.task.split == "development" and self.protocol_sha256 is None:
                raise ValueError("Development episodes require a validated frozen protocol")
            ident = spec.episode_id(self.arm.name, self.controller)
            meta = {"schema": SCHEMA, "arm": self.arm.name, "seed": spec.seed,
                    "controller": self.controller.to_dict(), "controller_sha256": self.controller.sha256(),
                    "task_sha256": spec.task.sha256(), "repository": spec.task.repository,
                    "task_split": spec.task.split,
                    "source_revision": spec.task.source_revision,
                    "workspace_patch_sha256": spec.task.workspace_patch_sha256,
                    "protocol_sha256": self.protocol_sha256,
                    "target_files": list(spec.task.target_files),
                    "policy": dataclasses.asdict(self.arm.checkpoint),
                    "fault_injection": dict(self.fault_injection),
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

    def enroll_grpo_group(self, specs: Sequence[EpisodeSpec], *, group_id: str) -> list[str]:
        """Enroll one complete, same-task group for first-turn GRPO.

        The shared task metadata contains the ordered member seeds, so the
        trajectory store can verify that every member has the same task while
        the worker still recovers the right per-member sampling seed.
        """
        from vare.trajectory_store import identity

        specs = tuple(specs)
        identity(group_id)
        if len(specs) < 2:
            raise ValueError("GRPO groups require at least two episodes")
        temperature, top_p = self.controller.temperature, self.controller.top_p
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature <= 0
                or isinstance(top_p, bool) or not isinstance(top_p, (int, float))
                or not math.isfinite(top_p) or top_p != 1.0):
            raise ValueError("GRPO rollouts require temperature > 0 and top_p == 1.0")
        if self.fault_injection:
            raise ValueError("GRPO groups cannot enroll with fault injection")
        first = specs[0]
        if first.task.split != "train":
            raise ValueError("GRPO rollouts may use only Train tasks")
        if any(spec.task.sha256() != first.task.sha256() for spec in specs):
            raise ValueError("GRPO group members must use one immutable task")
        seeds = [spec.seed for spec in specs]
        if any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("GRPO group seeds must be unique nonnegative integers")

        common_meta = {
            "schema": SCHEMA, "arm": self.arm.name, "group_seeds": seeds,
            "grpo_group_id": group_id,
            "controller": self.controller.to_dict(), "controller_sha256": self.controller.sha256(),
            "task_sha256": first.task.sha256(), "repository": first.task.repository,
            "task_split": first.task.split, "source_revision": first.task.source_revision,
            "workspace_patch_sha256": first.task.workspace_patch_sha256,
            "protocol_sha256": self.protocol_sha256,
            "target_files": list(first.task.target_files),
            "policy": dataclasses.asdict(self.arm.checkpoint), "fault_injection": {},
            "synthetic_policy": bool(getattr(self.arm.policy, "synthetic", True)),
            "synthetic_task": first.task.synthetic,
            "public_tests": first.task.task_id in self.public_tests,
        }
        task = Task(id=first.task.task_id, prompt=first.task.issue,
                    family=first.task.repository, metadata=common_meta)
        ids = []
        group_suffix = hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16]
        for member, spec in enumerate(specs):
            base = spec.episode_id(self.arm.name, self.controller)
            ident = f"{base[:140]}:g{group_suffix}"
            self.store.submit(ident, task, expected_steps=self.controller.max_turns,
                              group_id=group_id, member=member, group_size=len(specs),
                              max_attempts=self.max_attempts, lifetime_seconds=86400,
                              now=self.clock())
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
            while True:
                lease = self._claim("verify")
                if lease is None:
                    counts = self.store.work_counts("verify")
                    if wait_for_expired and (counts.get("queued", 0) or counts.get("running", 0)):
                        time.sleep(poll_seconds)
                        continue
                    break
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
                or meta["policy"] != dataclasses.asdict(self.arm.checkpoint)
                or meta.get("fault_injection", {}) != dict(self.fault_injection)):
            raise ResumeMismatchError(f"{ident} was enrolled under a different controller or policy")
        runtime, steps = self.rebuild(ident)
        controller = self.controller
        policy = self.arm.policy
        if "group_seeds" in meta:
            member = int(self.store.trajectory(ident)["member"])
            if member < 0 or member >= len(meta["group_seeds"]):
                raise ResumeMismatchError(f"{ident} has an invalid group member seed")
            seed = meta["group_seeds"][member]
        else:
            seed = meta["seed"]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt(task, runtime, controller)},
            {"role": "user", "content": "Begin. First call list_files."},
        ]
        for step in steps:
            messages.extend(messages_for_step(step["action"]))
        baseline_hashes = runtime.file_hashes()
        state: dict[str, Any] = dict(steps[-1]["state"]) if steps else {
            "file_sha256": baseline_hashes, "initial_file_sha256": baseline_hashes,
            "edit_applied": False, "public_passed": None,
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
            action["generation"]["policy_peak_memory_bytes"] = policy.peak_memory_bytes()
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
        trace_prompt_ids = getattr(generation, "prompt_ids", None)
        trace_completion_ids = getattr(generation, "completion_ids", None)
        if index == 0 and trace_prompt_ids is not None and trace_completion_ids is not None:
            prompt_ids = list(trace_prompt_ids)
            completion_ids = list(trace_completion_ids)
            if (prompt_ids and completion_ids
                    and all(type(token) is int and token >= 0 for token in prompt_ids)
                    and all(type(token) is int and token >= 0 for token in completion_ids)):
                temperature = getattr(generation, "sampling_temperature", None)
                top_p = getattr(generation, "sampling_top_p", None)
                distribution = (
                    "untruncated_softmax_temperature"
                    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
                    and temperature > 0 and top_p == 1.0
                    else "greedy" if temperature == 0
                    else "truncated_or_unverified"
                )
                action["behavior_trace"] = {
                    "schema": "vare_long_horizon_behavior_trace_v1",
                    "prompt_ids": prompt_ids, "completion_ids": completion_ids,
                    "prompt_ids_sha256": sha256_json(prompt_ids),
                    "completion_ids_sha256": sha256_json(completion_ids),
                    "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "temperature": temperature, "top_p": top_p,
                    "distribution": distribution,
                }
        if calls:
            action.update(operation_id=f"{ident}:op:{index}", semantics="at_least_once")
        return state, action

    def _verify(self, lease: TrajectoryLease) -> None:
        ident = lease.trajectory_id
        task = self.tasks[ident]
        trajectory = self.store.trajectory(ident)
        head = trajectory["head"]
        task_metadata = json.loads(trajectory["task"])["task"]["metadata"]
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
                                       "task_sha256": task_metadata["task_sha256"],
                                       "source_revision": task_metadata.get("source_revision"),
                                       "workspace_patch_sha256": task_metadata.get("workspace_patch_sha256"),
                                       "verdict": verdict, "grade_summary": summary,
                                       "raw_grade": dict(raw) if isinstance(raw, Mapping) else {},
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


def learning_pipeline_task(record: Mapping[str, Any], readable: Sequence[str], *,
                           source_revision: str | None = None) -> EpisodeTask:
    """Policy-visible view of a Train record: issue text and file scope only."""
    target = str(record["target_file"])
    files = tuple(sorted(set(readable) | {target}))
    bug_patch = record.get("_patch")
    patch_sha256 = (hashlib.sha256(bug_patch.encode("utf-8")).hexdigest()
                    if isinstance(bug_patch, str) else None)
    return EpisodeTask(task_id=str(record["instance_id"]), repository=str(record["upstream_repo"]),
                       split=str(record["split"]), issue=str(record["_problem"]), readable_files=files,
                       editable_files=(target,), target_files=(target,), source_revision=source_revision,
                       workspace_patch_sha256=patch_sha256)


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
    verifier_config_sha256 = None
    if verification is not None:
        with store.connection() as db:
            verifier_row = db.execute(
                "SELECT config FROM verifiers WHERE name=? AND version=?",
                (verification["verifier_name"], verification["verifier_version"]),
            ).fetchone()
        verifier_config_sha256 = verifier_row[0] if verifier_row is not None else None
    final_success = trusted_final_success(
        store_state=row["state"],
        expected_steps=row["expected_steps"],
        committed_steps=len(steps),
        verification=verification,
        expected_trajectory_sha256=row["head"],
        expected_verifier_config_sha256=verifier_config_sha256,
        initial_source_hashes=(steps[0]["state"].get("initial_file_sha256", {}) if steps else {}),
        final_source_hashes=last_state.get("file_sha256", {}),
        public_tests_configured=bool(meta.get("public_tests")),
        public_tests_passed=last_state.get("public_passed"),
        expected_task_sha256=meta.get("task_sha256"),
        expected_source_revision=meta.get("source_revision"),
        expected_workspace_patch_sha256=meta.get("workspace_patch_sha256"),
        require_task_binding=meta.get("schema") == SCHEMA,
    )
    attempts = _execute_claims(store, ident)
    member = int(row["member"])
    if isinstance(meta.get("group_seeds"), list):
        if member >= len(meta["group_seeds"]):
            raise ValueError("trajectory member has no enrolled seed")
        seed = meta["group_seeds"][member]
    else:
        seed = meta["seed"]
    return {
        "episode_id": ident, "task_id": spec["task"]["id"], "repository": meta["repository"],
        "arm": meta["arm"], "controller": meta["controller"]["name"], "controller_sha256": meta["controller_sha256"],
        "protocol_sha256": meta.get("protocol_sha256"),
        "seed": seed, "synthetic_policy": meta["synthetic_policy"], "synthetic_task": meta["synthetic_task"],
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
            "peak_policy_memory_bytes": max(
                (s["action"]["generation"].get("policy_peak_memory_bytes") for s in real
                 if isinstance(s["action"]["generation"].get("policy_peak_memory_bytes"), int)),
                default=None,
            ),
            "generation_seconds": round(sum(s["action"]["timing"]["generation_seconds"] for s in real), 6),
            "tool_seconds": round(sum(c.get("seconds", 0.0) for c in calls), 6),
            "step_seconds": round(sum(s["action"]["timing"]["step_seconds"] for s in real), 6),
            "grading_seconds": (verification or {}).get("metadata", {}).get("grading_seconds"),
        },
        "grade_recorded": verification is not None,
        "graded": trusted,
        "trusted_grade": trusted,
        "grade_reason": (verification or {}).get("metadata", {}).get("verdict", {}).get("reason", "ungraded"),
        "raw_grade": (verification or {}).get("metadata", {}).get("raw_grade"),
        # Primary outcome requires both the independent parent grade and the
        # public checks when the episode has them. The finish-tool gate cannot
        # enforce this on episodes that stop at a turn or token budget.
        "final_success": final_success,
        "trajectory_head": row["head"],
    }


def grpo_samples_from_verified_group(store: TrajectoryStore, group_id: str) -> list[Any]:
    """Convert one intact trusted Train group into first-turn GRPO samples.

    GRPO's current prompt-group contract requires identical input tokens. The
    first turn is the only shared state across independently executed episodes;
    later turns depend on each member's tool observations. Reward is the final
    trusted episode outcome, including required public checks. Behavior log
    probabilities are intentionally recomputed by the existing on-policy
    trainer from the exact token trace and checkpoint before the optimizer step.
    """
    from vare.grpo_mlx import Sample, group_samples

    audit = audit_trajectory_store(store.path)
    if audit.get("pass") is not True:
        raise ValueError("trajectory store integrity audit did not pass")
    rows = store.group(group_id)
    if not rows:
        raise ValueError("empty trajectory group")
    group_size = int(rows[0]["group_size"])
    if (len(rows) != group_size or [int(row["member"]) for row in rows] != list(range(group_size))
            or any(row["group_size"] != group_size or row["state"] != "verified" for row in rows)):
        raise ValueError("GRPO requires a complete verified trajectory group")

    samples = []
    shared_identity: tuple[Any, ...] | None = None
    shared_prompt_ids: list[int] | None = None
    for row in rows:
        task_record = json.loads(row["task"])
        task = task_record["task"]
        meta = task["metadata"]
        if (meta.get("schema") != SCHEMA or meta.get("task_split") != "train"
                or meta.get("synthetic_policy") is not False or meta.get("synthetic_task") is not False):
            raise ValueError("GRPO source must be a real-policy, real Train trajectory")
        if meta.get("grpo_group_id") != group_id:
            raise ValueError("trajectory group identity differs from its enrollment")
        if not isinstance(meta.get("group_seeds"), list) or len(meta["group_seeds"]) != group_size:
            raise ValueError("GRPO group seed manifest is missing or incomplete")
        if (any(type(seed) is not int or seed < 0 for seed in meta["group_seeds"])
                or len(set(meta["group_seeds"])) != group_size):
            raise ValueError("GRPO group seed manifest is invalid")
        if (not isinstance(meta.get("task_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", meta["task_sha256"]) is None
                or not isinstance(meta.get("source_revision"), str)
                or re.fullmatch(r"[0-9a-f]{40}", meta["source_revision"]) is None
                or not isinstance(meta.get("workspace_patch_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", meta["workspace_patch_sha256"]) is None):
            raise ValueError("GRPO task lacks immutable source and patch provenance")

        verification = json.loads(row["verification"] or "null")
        if (not isinstance(verification, dict) or verification.get("trusted") is not True
                or type(verification.get("passed")) is not bool
                or not isinstance(verification.get("metadata"), dict)
                or verification["metadata"].get("trajectory_sha256") != row["head"]):
            raise ValueError("GRPO group contains an untrusted or unbound verifier result")
        verifier_config = None
        with store.connection() as db:
            verifier_row = db.execute(
                "SELECT config FROM verifiers WHERE name=? AND version=?",
                (verification["verifier_name"], verification["verifier_version"]),
            ).fetchone()
        if verifier_row is not None:
            verifier_config = verifier_row[0]
        if verifier_config is None or verification["metadata"].get("verifier_config_sha256") != verifier_config:
            raise ValueError("GRPO verifier identity is not registered and source-bound")

        steps = list(store.iter_steps(row["id"]))
        if not steps or steps[0]["index"] != 0:
            raise ValueError("GRPO trajectory lacks its first committed action")
        initial_hashes = steps[0].get("state", {}).get("initial_file_sha256")
        if (not isinstance(initial_hashes, dict) or not initial_hashes
                or any(not isinstance(name, str) or not isinstance(digest, str)
                       or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                       for name, digest in initial_hashes.items())):
            raise ValueError("GRPO trajectory lacks valid initial source hashes")
        action = steps[0]["action"]
        trace = action.get("behavior_trace")
        if not isinstance(trace, dict) or trace.get("schema") != "vare_long_horizon_behavior_trace_v1":
            raise ValueError("GRPO trajectory lacks an exact first-turn behavior trace")
        prompt_ids, completion_ids = trace.get("prompt_ids"), trace.get("completion_ids")
        if (not isinstance(prompt_ids, list) or not prompt_ids
                or any(type(token) is not int or token < 0 for token in prompt_ids)
                or not isinstance(completion_ids, list) or not completion_ids
                or any(type(token) is not int or token < 0 for token in completion_ids)):
            raise ValueError("GRPO behavior trace token IDs are invalid")
        if (trace.get("prompt_ids_sha256") != sha256_json(prompt_ids)
                or trace.get("completion_ids_sha256") != sha256_json(completion_ids)
                or trace.get("response_sha256") != hashlib.sha256(
                    str(action.get("model_output", "")).encode("utf-8")).hexdigest()):
            raise ValueError("GRPO behavior trace content digest mismatch")
        if (action.get("generation", {}).get("prompt_tokens") != len(prompt_ids)
                or action.get("generation", {}).get("completion_tokens") != len(completion_ids)):
            raise ValueError("GRPO behavior trace token counts differ from the action record")
        controller = meta.get("controller", {})
        temperature, top_p = trace.get("temperature"), trace.get("top_p")
        if (trace.get("distribution") != "untruncated_softmax_temperature"
                or isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature <= 0
                or isinstance(top_p, bool) or not isinstance(top_p, (int, float))
                or not math.isfinite(top_p) or top_p != 1.0 or controller.get("temperature") != temperature
                or controller.get("top_p") != top_p):
            raise ValueError("GRPO behavior sampler is not the exact untruncated temperature policy")

        policy = steps[0].get("policy")
        if policy != meta.get("policy") or not isinstance(policy, dict):
            raise ValueError("GRPO behavior checkpoint differs from the enrolled policy")
        policy_id, policy_version, checkpoint_sha256 = (
            policy.get("policy_id"), policy.get("version"), policy.get("checkpoint_sha256"))
        if (not isinstance(policy_id, str) or type(policy_version) is not int
                or not isinstance(checkpoint_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None):
            raise ValueError("GRPO behavior checkpoint identity is invalid")
        initial_hashes_sha256 = sha256_json(initial_hashes)
        identity = (task["id"], meta["task_sha256"], meta["source_revision"],
                    meta["workspace_patch_sha256"], initial_hashes_sha256,
                    policy_id, policy_version, checkpoint_sha256, verification["verifier_name"],
                    verification["verifier_version"], verifier_config, temperature, top_p,
                    meta.get("controller_sha256"))
        if shared_identity is None:
            shared_identity = identity
            shared_prompt_ids = list(prompt_ids)
        elif identity != shared_identity or prompt_ids != shared_prompt_ids:
            raise ValueError("GRPO group mixes task, policy, verifier, sampler, controller, or prompt identity")

        metrics = episode_metrics(store, row["id"])
        samples.append(Sample(
            instance_id=group_id,
            behavior_checkpoint_id=f"{policy_id}:v{policy_version}:{checkpoint_sha256}",
            behavior_step=policy_version,
            prompt_ids=list(prompt_ids), completion_ids=list(completion_ids),
            reward=float(bool(metrics["final_success"])),
            verifier=f"{verification['verifier_name']}:v{verification['verifier_version']}:{verifier_config}",
            sample_index=int(row["member"]), behavior_logprobs=None,
            meta={"trajectory_id": row["id"], "trajectory_sha256": row["head"],
                  "task_id": task["id"], "task_sha256": meta["task_sha256"],
                  "source_revision": meta["source_revision"],
                  "initial_source_hashes_sha256": initial_hashes_sha256,
                  "policy_checkpoint_sha256": checkpoint_sha256,
                  "behavior_temperature": float(temperature), "behavior_top_p": float(top_p),
                  "final_success": bool(metrics["final_success"])},
        ))

    group_samples(samples, group_size=group_size)
    if len({sample.verifier for sample in samples}) != 1:
        raise ValueError("GRPO group mixes verifier identities")
    return samples


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
FIXTURE_OPS_SHA256 = hashlib.sha256(FIXTURE_SOURCE["calc/ops.py"].encode("utf-8")).hexdigest()


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
    _call("replace_text", path="calc/ops.py", old_text="return a + b", new_text="return a - b",
          expected_source_sha256=FIXTURE_OPS_SHA256),
    "<tool_call>{not json}</tool_call>",
    # Bare JSON: only a controller with tool-call normalization executes it.
    json.dumps({"name": "replace_text", "arguments": {"path": "calc/ops.py", "old_text": "return a - b",
                                                       "new_text": "return a + b",
                                                       "expected_source_sha256": FIXTURE_OPS_SHA256}}),
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
    run.add_argument("--revision", help="model revision (pin it; recorded in the checkpoint digest)")
    run.add_argument("--max-prompt-tokens", type=int, default=8192)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--controller", choices=sorted(CONTROLLER_PRESETS), default="full")
    run.add_argument("--manifest", type=Path, required=True, help="a Train manifest; non-Train rows are refused")
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--limit", type=int, required=True)
    run.add_argument("--seeds", type=int, default=1)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--protocol", type=Path,
                     help="copy and bind a preregistered protocol into each episode root")
    run.add_argument("--instances", help="comma-separated Train instance ids (default: first --limit rows)")
    run.add_argument("--lease-seconds", type=float, default=600.0,
                     help="must exceed the slowest turn or grade; a killed run resumes after it expires")
    run.add_argument("--crash-after-commit-index", type=int,
                     help="send SIGKILL after this zero-based step is durably committed; repeat the same flag to resume")
    run.add_argument("--min-available-memory-gib", type=float, default=8.0)
    run.add_argument("--min-free-disk-gib", type=float, default=4.0)
    run.add_argument("--grader-factory", help=("optional module:function(task_record, env) -> TrustedGrader; "
                                               "only vare_function_oracle_v1 results can be trusted"))
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


def validate_frozen_protocol(
    protocol: Mapping[str, Any], *, model_id: str, revision: str | None,
    controller: ControllerConfig, tasks: Sequence[EpisodeTask], graders: Mapping[str, TrustedGrader | None],
    arm: str, policy_version: int, adapter: Path | None, seeds: Sequence[int], crash_after_commit_index: int | None,
    lease_seconds: float, max_attempts: int, manifest_sha256: str,
    max_prompt_tokens: int, task_limit: int, grader_factory: str | None, source_root: Path,
    min_available_memory_gib: float, min_free_disk_gib: float,
) -> None:
    """Reject execution settings that do not match a frozen protocol exactly.

    This runs after Train task/grader construction but before policy loading,
    trajectory enrollment, or model inference. A protocol digest is meaningful
    only when the executed configuration and implementation identities match it.
    """
    from pathlib import PurePosixPath

    mismatches: list[str] = []

    def require(ok: bool, name: str) -> None:
        if not ok:
            mismatches.append(name)

    require(protocol.get("preregistered_before_model_inference") is True,
            "preregistered_before_model_inference")
    require(isinstance(protocol.get("protocol_id"), str) and bool(protocol.get("protocol_id")),
            "protocol_id")

    model = protocol.get("model", {})
    require(model.get("id") == model_id, "model.id")
    require(model.get("revision") == revision and bool(revision), "model.revision")
    if model.get("local_only") is True:
        require(os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                "model.local_only environment")

    frozen_controller = protocol.get("controller", {})
    require(frozen_controller.get("name") == controller.name, "controller.name")
    require(frozen_controller.get("config") == controller.to_dict(), "controller.config")
    require(frozen_controller.get("sha256") == controller.sha256(), "controller.sha256")

    budgets = protocol.get("budgets", {})
    controller_budget_names = (
        "max_calls_per_turn", "max_new_tokens_per_turn", "max_total_new_tokens",
        "max_turns", "temperature", "top_p", "wall_seconds",
    )
    for name in controller_budget_names:
        if name in budgets:
            require(budgets[name] == getattr(controller, name), f"budgets.{name}")
    if "lease_seconds" in budgets:
        require(budgets["lease_seconds"] == lease_seconds, "budgets.lease_seconds")
    if "max_attempts" in budgets:
        require(budgets["max_attempts"] == max_attempts, "budgets.max_attempts")
    require(budgets.get("max_prompt_tokens") == max_prompt_tokens, "budgets.max_prompt_tokens")
    require(budgets.get("task_limit") == task_limit, "budgets.task_limit")
    if "seed" in budgets:
        require(list(seeds) == [budgets["seed"]], "budgets.seed")

    resources = protocol.get("resources", {})
    if "min_available_memory_gib" in resources:
        require(resources["min_available_memory_gib"] == min_available_memory_gib,
                "resources.min_available_memory_gib")
    if "min_free_disk_gib" in resources:
        require(resources["min_free_disk_gib"] == min_free_disk_gib,
                "resources.min_free_disk_gib")

    task_spec = protocol.get("task", {})
    require(len(tasks) == int(protocol.get("execution", protocol.get("episodes", {})).get(
        "task_count", len(tasks))), "task_count")
    if len(tasks) == 1:
        task = tasks[0]
        expected_task_fields = {
            "task_id": task.task_id,
            "repository": task.repository,
            "split": task.split,
            "source_revision": task.source_revision,
            "episode_task_sha256": task.sha256(),
            "issue_sha256": hashlib.sha256(task.issue.encode("utf-8")).hexdigest(),
            "readable_file_count": len(task.readable_files),
        }
        for name, actual in expected_task_fields.items():
            if name in task_spec:
                require(task_spec[name] == actual, f"task.{name}")
        target_file = task_spec.get("target_file")
        if target_file is not None:
            require(task.target_files == (target_file,), "task.target_file")
        if "bug_patch_sha256" in task_spec:
            require(task.workspace_patch_sha256 == task_spec["bug_patch_sha256"],
                    "task.bug_patch_sha256")
    require(manifest_sha256 == task_spec.get("train_manifest_sha256"), "task.train_manifest_sha256")

    contract_root = protocol.get("execution", protocol.get("episodes", {}))
    expected_fault = None
    if crash_after_commit_index is not None:
        expected_fault = {"signal": "SIGKILL", "stage": "after_commit",
                          "zero_based_step_index": crash_after_commit_index}
    for seed in seeds:
        matching: list[Mapping[str, Any]] = []
        for candidate in contract_root.values():
            if not isinstance(candidate, Mapping) or candidate.get("arm") != arm or candidate.get("seed") != seed:
                continue
            fault = candidate.get("fault")
            if expected_fault is None:
                if fault is None:
                    matching.append(candidate)
            elif (isinstance(fault, Mapping)
                  and all(fault.get(key) == value for key, value in expected_fault.items())):
                matching.append(candidate)
        require(len(matching) == 1, f"execution scenario arm={arm},seed={seed},fault={expected_fault}")
        if len(matching) == 1 and "adapter" in matching[0]:
            expected_adapter = matching[0]["adapter"]
            actual_adapter = directory_sha256(adapter) if adapter is not None else None
            require(actual_adapter == expected_adapter, f"execution adapter for seed={seed}")
    if "effective_episode_seed" in contract_root:
        require(list(seeds) == [contract_root["effective_episode_seed"]], "effective_episode_seed")
    if "seeds_argument" in contract_root:
        require(len(seeds) == contract_root["seeds_argument"], "seeds_argument")
    if "task_count" in contract_root:
        require(len(tasks) == contract_root["task_count"], "execution.task_count")
    if arm == "base":
        require(len(seeds) == 1, "base arm requires one preregistered seed")
        require(policy_version == 0, "base arm policy_version")

    frozen_grader = protocol.get("grader", {})
    require(frozen_grader.get("factory") == grader_factory, "grader.factory")
    for task in tasks:
        grader = graders.get(task.task_id)
        if grader is None:
            require(frozen_grader.get("class") is None, f"grader missing for {task.task_id}")
            continue
        require(type(grader).__name__ == frozen_grader.get("class"), "grader.class")
        require(grader.config_sha256() == frozen_grader.get("config_sha256"), "grader.config_sha256")
        if "source_identity" in frozen_grader:
            require(getattr(grader, "source_identity", None) == frozen_grader["source_identity"],
                    "grader.source_identity")
        if "case_ids" in frozen_grader and hasattr(grader, "calls"):
            require([call.case_id for call in grader.calls] == frozen_grader["case_ids"], "grader.case_ids")

    source_hashes = protocol.get("source_hashes", {})
    if not isinstance(source_hashes, Mapping):
        mismatches.append("source_hashes")
    else:
        required_source_paths = {
            "scripts/long_horizon_eval.py", "scripts/funcy_tree_oracle.py",
            "scripts/run_local_agent.py", "src/vare/trajectory_store.py",
            "src/vare/isolated_candidate.py", "src/vare/container_isolation.py",
        }
        require(required_source_paths <= set(source_hashes), "source_hashes.required_sources")
        for name, expected in source_hashes.items():
            relative = PurePosixPath(name)
            if (relative.is_absolute() or relative.as_posix() != name
                    or any(part in {"", ".", ".."} for part in relative.parts)):
                mismatches.append(f"source_hashes.{name}: unsafe path")
                continue
            path = source_root / Path(*relative.parts)
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            require(actual == expected, f"source_hashes.{name}")

    if mismatches:
        raise ValueError("execution does not match frozen protocol: " + ", ".join(mismatches))


def run_learning_pipeline_arm(args: argparse.Namespace) -> Path:
    from vare.learning_pipeline import curriculum, repo_env, resources

    root = args.output / f"{args.arm}-{args.controller}"
    root.mkdir(parents=True, exist_ok=True)
    protocol_sha256 = None
    protocol_id = None
    protocol = None
    if args.protocol is not None:
        protocol_path = args.protocol.expanduser().resolve()
        protocol_bytes = protocol_path.read_bytes()
        protocol = json.loads(protocol_bytes)
        if (not isinstance(protocol, Mapping)
                or protocol.get("preregistered_before_model_inference") is not True
                or not isinstance(protocol.get("protocol_id"), str)):
            raise ValueError("--protocol must identify a frozen pre-inference protocol")
        protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
        protocol_id = protocol["protocol_id"]
    records = load_train_manifest(args.manifest)
    if args.instances:
        wanted = args.instances.split(",")
        by_id = {r["instance_id"]: r for r in records}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise ValueError(f"not in the Train manifest: {missing}")
        records = [by_id[i] for i in wanted]
    records = curriculum.attach_payloads(records[: args.limit], args.cache)
    grader_factory = _load_factory(args.grader_factory)
    factories: dict[str, WorkspaceFactory] = {}
    graders: dict[str, TrustedGrader | None] = {}
    tasks = []
    for record in records:
        env = repo_env.env_for(record, args.cache)
        if not (env.path / ".git").exists():
            env.materialize()
        task = learning_pipeline_task(record, readable_python_files(env.path, env.commit),
                                      source_revision=env.commit)
        factories[task.task_id] = repo_workspace_factory(env.path, env.commit, record["_patch"])
        graders[task.task_id] = grader_factory(record, env) if grader_factory else None
        tasks.append(task)
    controller = CONTROLLER_PRESETS[args.controller]
    if protocol is not None:
        validate_frozen_protocol(
            protocol, model_id=args.model, revision=args.revision, controller=controller,
            tasks=tasks, graders=graders, arm=args.arm, policy_version=args.policy_version,
            adapter=args.adapter, seeds=list(range(args.seeds)),
            crash_after_commit_index=args.crash_after_commit_index,
            lease_seconds=args.lease_seconds, max_attempts=5,
            manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            max_prompt_tokens=args.max_prompt_tokens, task_limit=args.limit,
            grader_factory=args.grader_factory, source_root=ROOT,
            min_available_memory_gib=args.min_available_memory_gib,
            min_free_disk_gib=args.min_free_disk_gib,
        )
        saved_protocol = root / "protocol.json"
        if saved_protocol.exists() and saved_protocol.read_bytes() != protocol_bytes:
            raise ValueError("episode root already contains a different frozen protocol")
        if not saved_protocol.exists():
            saved_protocol.write_bytes(protocol_bytes)
    preflight = {"memory_start": resources.check_memory(args.min_available_memory_gib, stage="lh_preflight"),
                 "free_disk_bytes": resources.check_disk(args.output, args.min_free_disk_gib, stage="lh_preflight"),
                 "argv": sys.argv[1:], "protocol_id": protocol_id,
                 "protocol_sha256": protocol_sha256}
    (root / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True, default=str) + "\n",
                                         encoding="utf-8")
    digest = sha256_json({"model": args.model, "revision": args.revision,
                          "adapter": directory_sha256(args.adapter) if args.adapter else None})
    policy = mlx_policy(args.model, str(args.adapter) if args.adapter else None, revision=args.revision,
                        max_prompt_tokens=args.max_prompt_tokens)
    arm = PolicyArm(args.arm, policy, PolicyCheckpoint(args.arm, args.policy_version, digest))
    crash_index = args.crash_after_commit_index
    if crash_index is not None and crash_index < 0:
        raise ValueError("crash-after-commit-index must be nonnegative")
    fault_injection = ({"signal": "SIGKILL", "stage": "after_commit", "step_index": crash_index}
                       if crash_index is not None else {})
    crash_hook = None
    if crash_index is not None:
        def crash_hook(stage: str, index: int) -> None:
            if stage == "after_commit" and index == crash_index:
                print(json.dumps({"fault_injection": fault_injection, "pid": os.getpid()}), flush=True)
                os.kill(os.getpid(), signal.SIGKILL)

    harness = EpisodeHarness(root, arm, controller, factories=factories,
                             grader_for=lambda t: graders.get(t.task_id), lease_seconds=args.lease_seconds,
                             protocol_sha256=protocol_sha256,
                             crash_hook=crash_hook, fault_injection=fault_injection)
    harness.enroll(EpisodeSpec(task, seed) for task in tasks for seed in range(args.seeds))
    harness.run()
    (root / "postflight.json").write_text(json.dumps({
        "memory_end": resources.memory_snapshot(),
        "peak_policy_memory_bytes": policy.peak_memory_bytes(),
        "policy_checkpoint": dataclasses.asdict(arm.checkpoint),
    }, indent=2,
                                                     sort_keys=True, default=str) + "\n", encoding="utf-8")
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
