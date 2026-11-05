"""Experiment execution API.

Teammates can call :class:`ExperimentRunner` from a controller or CLI. This
module does not grade research quality and does not import candidate code.
Candidate commands run in a child process, or in Docker when that isolation
mode is requested and actually starts. Docker never falls back to the
process group.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Callable, Mapping

from vare.container_isolation import IsolationError

from .classify import Judgment, judge, read_metrics
from .headroom import decide_accelerator_headroom, local_available_bytes, local_cmdlines
from .isolation import RESIDUAL_RISK, docker_argv, docker_execute
from .jsonutil import atomic_write, canonical_bytes, canonical_json, digest_bytes, file_sha256, sha256_json, strict_load
from .patching import PatchApplication, PatchError, apply_unified_diff
from .paths import PathPolicyError, resolve_inside
from .process_owner import OwnedCommand, process_identity, run_bounded_command
from .spec import ExperimentSpec, config_sha256, load_spec, spec_sha256, spec_to_dict
from .store import TERMINAL_STATES, FenceError, ResumeStore, StoreError

_BLOCKED_PROGRAMS = frozenset({"bash", "sh", "dash", "zsh", "curl", "wget", "nc", "netcat", "ssh", "scp"})


@dataclass(frozen=True, slots=True)
class RunView:
    """Read model of one run. ``duplicate`` is response metadata, not a new result."""

    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)

    @property
    def accepted(self) -> bool:
        return bool(self.data["accepted"])

    @property
    def state(self) -> str:
        return str(self.data["state"])

    @property
    def failure_class(self) -> str | None:
        value = self.data.get("failure_class")
        return None if value is None else str(value)

    @property
    def duplicate(self) -> bool:
        return bool(self.data.get("duplicate"))

    @property
    def candidate_incorrect(self) -> bool:
        return bool(self.data.get("candidate_incorrect"))


class ExperimentRunner:
    def __init__(
        self,
        store_path: str | Path,
        runs_root: str | Path,
        *,
        docker_execute: Callable[..., Any] | None = None,
        memory_probe: Callable[[], int | None] | None = None,
        process_probe: Callable[[], list[str]] | None = None,
    ) -> None:
        self.store = ResumeStore(store_path)
        self.runs_root = Path(runs_root)
        if self.runs_root.is_symlink():
            raise StoreError("runs root is a symlink")
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._docker_execute = docker_execute
        self._memory_probe = memory_probe or local_available_bytes
        self._process_probe = process_probe or local_cmdlines

    def prepare(self, spec: ExperimentSpec) -> RunView:
        document = spec_to_dict(spec)
        encoded = canonical_json(document)
        action = self.store.enroll(
            run_id=spec.run_id,
            task_id=spec.task_id,
            spec_json=encoded,
            spec_sha256=spec_sha256(spec),
            config_sha256=config_sha256(spec),
            source_fingerprint=_source_fingerprint(spec),
            protected_fingerprint=_protected_fingerprint(spec),
            step_names=[step.name for step in spec.steps],
        )
        directory = self._directory(spec.run_id)
        atomic_write(directory / "spec.json", canonical_bytes(document))
        self._sync_log()
        return self.status(spec.run_id, duplicate=action == "exists")

    def apply_patch(self, run_id: str, patch_text: str) -> PatchApplication:
        spec = self._spec(run_id)
        row = self.store.get_run(run_id)
        if row is None or row["state"] != "prepared":
            raise FenceError("patches are only admitted before execution")
        try:
            applied = apply_unified_diff(spec.workspace, patch_text)
        except PatchError as exc:
            digest = digest_bytes(patch_text.encode("utf-8"))
            self.store.reject_patch(run_id, exc.failure_class, digest)
            self._sync_log()
            return PatchApplication(False, exc.failure_class, (), digest)
        self.store.update_source_fingerprint(run_id, _source_fingerprint(spec), applied.patch_sha256)
        self._sync_log()
        return applied

    def execute(self, run_id: str, *, owner: str = "local") -> RunView:
        return self._execute(run_id, owner)

    def resume(self, run_id: str, *, owner: str = "local") -> RunView:
        """Continue a run whose owner died. Terminal runs are returned unchanged."""
        return self._execute(run_id, owner)

    def status(self, run_id: str, *, duplicate: bool = False, already_running: bool = False) -> RunView:
        row = self.store.get_run(run_id)
        if row is None:
            raise StoreError("unknown run")
        log_ok = self._log_matches()
        artifacts_ok = self._artifacts_match(run_id)
        if row["manifest_json"]:
            payload = strict_load(row["manifest_json"])
        else:
            payload = _open_payload(row, self.store.get_steps(run_id))
        payload = dict(payload)
        payload["duplicate"] = duplicate
        payload["owner_alive"] = _owner_alive(row)
        payload["already_running"] = already_running
        if row["state"] in TERMINAL_STATES and (not log_ok or not artifacts_ok):
            payload["accepted"] = False
            payload["candidate_incorrect"] = False
            payload["failure_class"] = "log_tamper" if not log_ok else "manifest_tamper"
            payload["state"] = "failed_closed"
        return RunView(payload)

    def cancel(self, run_id: str) -> RunView:
        row = self.store.get_run(run_id)
        if row is None:
            raise StoreError("unknown run")
        if row["state"] in TERMINAL_STATES:
            return self.status(run_id, duplicate=True)
        current = self.store.request_cancel(run_id)
        assert current is not None
        _signal_child(current.get("child_pid"), current.get("child_starttime"))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            current = self.store.get_run(run_id)
            assert current is not None
            if current["state"] in TERMINAL_STATES:
                self._sync_log()
                return self.status(run_id)
            if not _owner_alive(current):
                break
            time.sleep(0.02)
        claim = self.store.claim(run_id, "cancel")
        if claim == "terminal":
            return self.status(run_id, duplicate=True)
        if claim == "busy":
            return self.status(run_id, already_running=True)
        spec = self._spec(run_id)
        self._finish(spec, Judgment("cancelled", "cancelled", False, False), None, _notes(spec))
        return self.status(run_id)

    def _execute(self, run_id: str, owner: str) -> RunView:
        if not self._log_matches():
            claim = self.store.claim(run_id, owner)
            if claim == "busy":
                view = self.status(run_id, already_running=True)
                return view
            if claim != "terminal":
                spec = self._spec(run_id)
                self._finish(spec, Judgment("failed_closed", "log_tamper", False, False), None, _notes(spec))
            return self.status(run_id, duplicate=claim == "terminal")
        claim = self.store.claim(run_id, owner)
        self._sync_log()
        if claim == "terminal":
            return self.status(run_id, duplicate=True)
        if claim == "busy":
            return self.status(run_id, already_running=True)
        spec = self._spec(run_id)
        if not self._spec_file_matches(spec):
            self._finish(spec, Judgment("failed_closed", "spec_tamper", False, False), None, _notes(spec))
            return self.status(run_id)
        try:
            refusal = self._accelerator_refusal(spec)
            if refusal is not None:
                self._finish(spec, refusal[0], None, _notes(spec, refusal[1]))
                return self.status(run_id)
            self._run_steps(spec)
        except (PathPolicyError, StoreError, OSError, RuntimeError) as exc:
            failure = getattr(exc, "failure_class", "internal_error")
            if failure not in {
                "path_traversal", "symlink_escape", "executable_not_allowed", "internal_error", "log_tamper",
            }:
                failure = "internal_error"
            row = self.store.get_run(run_id)
            if row is not None and row["state"] not in TERMINAL_STATES:
                self._finish(spec, Judgment("failed_closed", failure, False, False), None, _notes(spec))
        return self.status(run_id)

    def _run_steps(self, spec: ExperimentSpec) -> None:
        timeout_flag = bool(spec.metric_contract and spec.metric_contract.timeout_counts_as_incorrect)
        for index, step in enumerate(spec.steps):
            if self.store.cancel_requested(spec.run_id):
                self._finish(spec, Judgment("cancelled", "cancelled", False, False), None, _notes(spec))
                return
            stored = self.store.get_steps(spec.run_id)[index]
            if stored["state"] == "committed":
                continue
            special, protected_ok, source_ok = self._fresh(spec)
            if special is not None or not protected_ok or not source_ok:
                self._finish(spec, _fresh_judgment(special, protected_ok, source_ok), None, _notes(spec))
                return
            try:
                argv = _resolve_argv(spec, step.argv)
                cwd = resolve_inside(spec.workspace, step.cwd, expect="dir")
            except PathPolicyError as exc:
                self._finish(spec, Judgment("failed_closed", exc.failure_class, False, False), None, _notes(spec))
                return
            if spec.isolation == "docker" and step.cwd != ".":
                self._finish(spec, Judgment("failed_closed", "invalid_spec", False, False), None, _notes(spec))
                return
            self.store.mark_step_running(spec.run_id, index)
            self._sync_log()
            result = self._launch(spec, argv, cwd)
            record = _step_record(step.name, argv, result)
            self.store.commit_step(spec.run_id, index, record, result.stdout, result.stderr)
            self._write_step(spec.run_id, index, record, result.stdout, result.stderr)
            self._sync_log()
            if result.failure_class != "completed":
                incorrect = result.failure_class == "wall_timeout" and timeout_flag
                state = "cancelled" if result.failure_class == "cancelled" else "failed_closed"
                self._finish(
                    spec,
                    Judgment(state, result.failure_class, False, incorrect),
                    None,
                    _notes(spec),
                )
                return
            special, protected_ok, source_ok = self._fresh(spec)
            if special is not None or not protected_ok or not source_ok:
                self._finish(spec, _fresh_judgment(special, protected_ok, source_ok), None, _notes(spec))
                return
        metrics = read_metrics(spec.workspace, spec.metric_contract)
        judgment = judge(
            step_failure="completed",
            metrics_status=metrics.status,
            protected_ok=True,
            source_ok=True,
            timeout_counts_as_incorrect=timeout_flag,
        )
        notes = list(_notes(spec))
        if metrics.self_reported:
            notes.append("self_reported_success_flag_ignored")
        self._finish(spec, judgment, metrics.values, notes)

    def _launch(self, spec: ExperimentSpec, argv: list[str], cwd: Path) -> OwnedCommand:
        if spec.isolation == "docker":
            return self._launch_docker(spec, argv)
        limits = _limit_payload(spec)

        def on_start(pid: int, starttime: int) -> None:
            self.store.set_child(spec.run_id, pid, starttime)

        return run_bounded_command(
            argv,
            cwd=cwd,
            limits=limits,
            extra_env=dict(spec.env),
            should_cancel=lambda: self.store.cancel_requested(spec.run_id),
            on_start=on_start,
        )

    def _launch_docker(self, spec: ExperimentSpec, argv: list[str]) -> OwnedCommand:
        try:
            limits, command, _name = docker_argv(
                image=str(spec.docker_image),
                workspace=spec.workspace,
                command=argv,
                wall_time_s=float(spec.limits.wall_time_s),
                memory_mb=spec.limits.memory_mb or 512,
                max_output_bytes=spec.limits.max_output_bytes,
            )
            if self._docker_execute is not None:
                result = self._docker_execute(command, limits)
            else:
                result = asyncio.run(docker_execute(command, limits))
        except (IsolationError, OSError, ValueError) as exc:
            return OwnedCommand(None, b"", str(exc).encode("utf-8"), 0.0, "isolation_unavailable", 0, None)
        stdout = result.stdout if isinstance(result.stdout, bytes) else str(result.stdout).encode("utf-8")
        stderr = result.stderr if isinstance(result.stderr, bytes) else str(result.stderr).encode("utf-8")
        failure = "completed" if result.returncode == 0 else "nonzero_exit"
        return OwnedCommand(result.returncode, stdout, stderr, 0.0, failure, 0, None)

    def _accelerator_refusal(self, spec: ExperimentSpec) -> tuple[Judgment, str] | None:
        if spec.backend != "mlx":
            return None
        decision = decide_accelerator_headroom(
            available_bytes=self._memory_probe(),
            min_available_bytes=spec.mlx_min_available_mb * 1024 * 1024,
            cmdlines=self._process_probe(),
        )
        failure = "mlx_refused_headroom" if not decision.allowed else "mlx_unavailable"
        return Judgment("failed_closed", failure, False, False), f"accelerator_headroom:{decision.reason}"

    def _fresh(self, spec: ExperimentSpec) -> tuple[str | None, bool, bool]:
        row = self.store.get_run(spec.run_id)
        assert row is not None
        try:
            source = _source_fingerprint(spec)
            protected = _protected_fingerprint(spec)
        except PathPolicyError as exc:
            if exc.failure_class == "symlink_escape":
                return "symlink_escape", False, False
            return "stale_source", False, True
        stored_source = strict_load(row["source_fingerprint"])
        stored_protected = strict_load(row["protected_fingerprint"])
        return None, protected == stored_protected, source == stored_source

    def _finish(self, spec: ExperimentSpec, judgment: Judgment, metrics: dict[str, float] | None, notes: list[str]) -> None:
        timeout_flag = bool(spec.metric_contract and spec.metric_contract.timeout_counts_as_incorrect)
        manifest, _duplicate = self.store.finalize(
            spec.run_id,
            state=judgment.state,
            failure_class=judgment.failure_class,
            accepted=judgment.accepted,
            candidate_incorrect=judgment.candidate_incorrect,
            metrics=metrics,
            notes=notes,
            timeout_counts_as_incorrect=timeout_flag,
        )
        self._publish(spec.run_id, manifest)
        self._sync_log()

    def _spec(self, run_id: str) -> ExperimentSpec:
        row = self.store.get_run(run_id)
        if row is None:
            raise StoreError("unknown run")
        return load_spec(row["spec_json"])

    def _directory(self, run_id: str) -> Path:
        path = self.runs_root / run_id
        if path.is_symlink():
            raise StoreError("run directory is a symlink")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _spec_file_matches(self, spec: ExperimentSpec) -> bool:
        path = self._directory(spec.run_id) / "spec.json"
        if not path.is_file() or path.is_symlink():
            return False
        try:
            return strict_load(path.read_bytes()) == spec_to_dict(spec)
        except (OSError, UnicodeError, ValueError):
            return False

    def _write_step(self, run_id: str, index: int, record: dict[str, Any], stdout: bytes, stderr: bytes) -> None:
        directory = self._directory(run_id) / "steps" / str(index)
        atomic_write(directory / "record.json", canonical_bytes(record))
        atomic_write(directory / "stdout.bin", stdout)
        atomic_write(directory / "stderr.bin", stderr)

    def _publish(self, run_id: str, manifest: Mapping[str, Any]) -> None:
        atomic_write(self._directory(run_id) / "manifest.json", canonical_bytes(dict(manifest)))

    def _artifacts_match(self, run_id: str) -> bool:
        row = self.store.get_run(run_id)
        if row is None or not row["manifest_json"]:
            return True
        path = self.runs_root / run_id / "manifest.json"
        if not path.exists():
            self._publish(run_id, strict_load(row["manifest_json"]))
            path = self.runs_root / run_id / "manifest.json"
        if path.is_symlink():
            return False
        try:
            if strict_load(path.read_bytes()) != strict_load(row["manifest_json"]):
                return False
        except (OSError, UnicodeError, ValueError):
            return False
        for step in self.store.get_steps(run_id):
            if step["state"] != "committed":
                continue
            directory = self.runs_root / run_id / "steps" / str(step["step_index"])
            record_path = directory / "record.json"
            if record_path.is_symlink() or not record_path.is_file():
                return False
            try:
                if file_sha256(directory / "stdout.bin") != strict_load(step["record_json"])["stdout_sha256"]:
                    return False
                if file_sha256(directory / "stderr.bin") != strict_load(step["record_json"])["stderr_sha256"]:
                    return False
            except (OSError, UnicodeError, ValueError, KeyError):
                return False
        return True

    def _log_path(self) -> Path:
        return self.runs_root / "events.jsonl"

    def _log_matches(self) -> bool:
        try:
            self._sync_log()
        except StoreError:
            return False
        return True

    def _sync_log(self) -> None:
        events = self.store.events()
        lines = []
        for event in events:
            body = {key: event[key] for key in ("sequence", "run_id", "kind", "payload", "prev_hash")}
            if sha256_json(body) != event["event_hash"]:
                raise StoreError("log_tamper")
            lines.append(canonical_json({**body, "event_hash": event["event_hash"]}))
        expected = ("".join(line + "\n" for line in lines)).encode("utf-8")
        path = self._log_path()
        if path.is_symlink():
            raise StoreError("log_tamper")
        if path.exists():
            current = path.read_bytes()
            if current == expected:
                return
            if expected.startswith(current) or current == b"":
                atomic_write(path, expected)
                return
            raise StoreError("log_tamper")
        atomic_write(path, expected)


def _notes(spec: ExperimentSpec, extra: str | None = None) -> list[str]:
    notes = [RESIDUAL_RISK] if spec.isolation == "process_group" else [
        "docker mode does not fall back to process_group",
    ]
    if extra:
        notes.append(extra)
    return notes


def _fresh_judgment(special: str | None, protected_ok: bool, source_ok: bool) -> Judgment:
    if special is not None:
        return Judgment("failed_closed", special, False, False)
    return judge(
        step_failure="completed",
        metrics_status="ok",
        protected_ok=protected_ok,
        source_ok=source_ok,
        timeout_counts_as_incorrect=False,
    )


def _source_fingerprint(spec: ExperimentSpec) -> dict[str, str]:
    return {
        relative: file_sha256(resolve_inside(spec.workspace, relative, expect="file"))
        for relative in spec.source_files
    }


def _protected_fingerprint(spec: ExperimentSpec) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in spec.protected_paths}


def _resolve_argv(spec: ExperimentSpec, argv: tuple[str, ...]) -> list[str]:
    program = argv[0]
    if program in _BLOCKED_PROGRAMS or program.startswith("-"):
        raise PathPolicyError("executable_not_allowed", "command program is blocked")
    if spec.isolation == "docker":
        if Path(program).is_absolute() or ".." in Path(program).parts:
            raise PathPolicyError("executable_not_allowed", "docker command must be a container-local program")
        return list(argv)
    if program == "python":
        program = sys.executable
    path = Path(program)
    if path.is_absolute():
        resolved = path.resolve()
        allowed = {Path(sys.executable).resolve(), *(Path(item).resolve() for item in spec.allowed_executables)}
        if resolved not in allowed or not resolved.is_file():
            raise PathPolicyError("executable_not_allowed", "executable is not allowed")
        return [str(resolved), *argv[1:]]
    leaf = resolve_inside(spec.workspace, program, expect="file")
    return [str(leaf), *argv[1:]]


def _limit_payload(spec: ExperimentSpec) -> dict[str, Any]:
    memory = spec.limits.memory_mb
    return {
        "wall_time_s": spec.limits.wall_time_s,
        "cpu_time_s": spec.limits.cpu_time_s,
        "memory_mb": memory,
        "max_output_bytes": spec.limits.max_output_bytes,
        "max_file_bytes": spec.limits.max_file_bytes,
        "max_child_processes": spec.limits.max_child_processes,
        "network": spec.network_policy,
        "address_space_mb": None if memory is None else max(memory * 8, 512),
    }


def _step_record(name: str, argv: list[str], result: OwnedCommand) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": name,
        "argv": argv,
        "returncode": result.returncode,
        "wall_s": round(result.wall_s, 6),
        "failure_class": result.failure_class,
        "stdout_sha256": digest_bytes(result.stdout),
        "stderr_sha256": digest_bytes(result.stderr),
        "peak_rss_bytes": result.peak_rss_bytes,
    }


def _owner_alive(row: Mapping[str, Any]) -> bool:
    pid = row.get("owner_pid")
    starttime = row.get("owner_starttime")
    if pid is None or starttime is None:
        return False
    return process_identity(int(pid)) == (int(pid), int(starttime))


def _signal_child(pid: Any, starttime: Any) -> None:
    if pid is None or starttime is None:
        return
    if process_identity(int(pid)) != (int(pid), int(starttime)):
        return
    try:
        os.killpg(int(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return


def _open_payload(row: Mapping[str, Any], steps: list[Mapping[str, Any]]) -> dict[str, Any]:
    spec = strict_load(row["spec_json"])
    return {
        "schema_version": 1,
        "run_id": row["run_id"],
        "task_id": row["task_id"],
        "state": row["state"],
        "failure_class": row.get("failure_class"),
        "accepted": False,
        "candidate_incorrect": False,
        "spec_sha256": row["spec_sha256"],
        "config_sha256": row["config_sha256"],
        "isolation": spec["isolation"],
        "backend": spec["backend"],
        "timeout_counts_as_incorrect": bool((spec.get("metric_contract") or {}).get("timeout_counts_as_incorrect", False)),
        "source_fingerprint": strict_load(row["source_fingerprint"]),
        "protected_fingerprint": strict_load(row["protected_fingerprint"]),
        "steps": [
            {
                "index": step["step_index"],
                "name": step["name"],
                "state": step["state"],
                "record_sha256": step["record_sha256"],
            }
            for step in steps
        ],
        "metrics": None,
        "notes": [],
    }
