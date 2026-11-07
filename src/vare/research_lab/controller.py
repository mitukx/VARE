"""Multi-step research controller.

Each action is committed to the journal before the next decision. Resume
reloads that journal and does not repeat a committed experiment id.
"""

from __future__ import annotations

import difflib
import math
from pathlib import Path
from typing import Any

from .edits import EditError, apply_edits, read_files, write_files
from .evaluation import grade, public_metrics
from .paths import PathViolation
from .proposer import HypothesisNotApplicable
from .spec import ResearchTask
from .store import RunStore, StoreError, _atomic_json
from .workspace import materialize

_COMMANDS = {
    "training_correctness": ["call accum.accumulated_gradient"],
    "performance": ["call project.score_matrix", "bench project.score_matrix"],
    "experiment_debug": ["call experiment.fit_affine", "call experiment.gradient_step"],
}
_ALLOWED_ACTIONS = frozenset({"run_public", "edit", "submit", "stop"})


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return math.inf
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf
    if not math.isfinite(number):
        return math.inf
    return number


def publicly_supported(task_type: str, metrics: dict[str, Any]) -> bool:
    if task_type == "training_correctness":
        return _metric(metrics, "max_abs_error") <= 1e-6
    if task_type == "performance":
        return _metric(metrics, "max_abs_error") <= 1e-8
    if task_type == "experiment_debug":
        return (
            _metric(metrics, "recomputed_mse") <= 1e-3
            and _metric(metrics, "gradient_step_abs_error") <= 1e-6
            and not bool(metrics.get("diverged"))
        )
    return False


def success_class(proposer_kind: str, verified: bool) -> str:
    if proposer_kind == "oracle_control":
        return "oracle_control"
    if proposer_kind == "single_attempt_control":
        return "single_attempt_control"
    if not verified:
        return "not_verified"
    if proposer_kind == "diagnostic_policy":
        return "diagnostic_policy"
    if proposer_kind == "external_agent":
        return "external_agent_assisted"
    return "unclassified"


class Controller:
    def __init__(self, run_dir: Path, task: ResearchTask, proposer: Any) -> None:
        self.run_dir = run_dir
        self.task = task
        self.proposer = proposer
        self.store = RunStore(run_dir)
        self.header = self.store.load_header()
        if self.header["task_id"] != task.task_id:
            raise StoreError("run task_id does not match the loaded task")
        self.restore_workspace()

    def restore_workspace(self) -> None:
        files = expected_files(self.header, self.store.read_journal())
        write_files(self.store.workspace, self.task.editable_files, files)

    def observation(self) -> dict[str, Any]:
        records = self.store.read_journal()
        memory = [_memory_row(record) for record in records]
        last_public = None
        for record in records:
            if isinstance(record.get("public"), dict):
                last_public = record["public"]
        return {
            "task_id": self.task.task_id,
            "task_type": self.task.task_type,
            "objective": self.task.objective,
            "does_not_establish": self.task.does_not_establish,
            "editable_files": list(self.task.editable_files),
            "baseline_files": dict(self.header["baseline_files"]),
            "current_files": read_files(self.store.workspace, self.task.editable_files),
            "memory": memory,
            "last_public": last_public,
            "steps_remaining": max(0, int(self.header["max_actions"]) - len(records)),
        }

    def run(self, *, max_actions: int | None = None) -> dict[str, Any]:
        budget = self.task.max_iterations if max_actions is None else max_actions
        performed = 0
        while performed < budget:
            if self.store.cancel_requested():
                self._commit_terminal("cancel", "cancel", "cancelled", "operator cancel file")
                break
            if self.store.write_state_cache()["done"]:
                break
            if self.proposer is None:
                break
            action = self.proposer.propose(self.observation())
            self.apply(action)
            performed += 1
        return self.summary()

    def apply(self, action: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise StoreError("action must be an object")
        action_type = str(action.get("type") or "")
        hypothesis_id = str(action.get("hypothesis_id") or "")
        if action_type not in _ALLOWED_ACTIONS or not hypothesis_id:
            return self._commit_rejected_action(action, "disallowed_action")
        experiment_id = f"{self.header['run_id']}:{hypothesis_id}"
        existing = self.store.find(experiment_id)
        if existing is not None:
            return existing
        if action_type == "stop":
            return self._commit_terminal("stop", hypothesis_id, "stopped", str(action.get("reason") or ""))
        if action_type == "run_public":
            return self._measure(experiment_id, hypothesis_id, "run_public", None)
        if action_type == "submit":
            return self._submit(experiment_id, hypothesis_id)
        return self._edit(experiment_id, hypothesis_id, action)

    def summary(self) -> dict[str, Any]:
        records = self.store.read_journal()
        state = self.store.write_state_cache()
        verified = bool(
            isinstance(state.get("verification"), dict) and state["verification"].get("passed") is True
        )
        kind = str(self.header["proposer_kind"])
        rejected = [
            record["hypothesis_id"]
            for record in records
            if record.get("status") in {"rejected", "edit_rejected"}
        ]
        return {
            "schema": "vare.research_lab.summary.v1",
            "run_id": self.header["run_id"],
            "task_id": self.task.task_id,
            "task_type": self.task.task_type,
            "proposer_name": self.header["proposer_name"],
            "proposer_kind": kind,
            "success_class": success_class(kind, verified),
            "counts_as_autonomous_model_success": False,
            "model_backed": False,
            "verified": verified and kind != "oracle_control",
            "oracle_calibration_passed": verified and kind == "oracle_control",
            "actions": len(records),
            "rejected_hypotheses": rejected,
            "status": state["status"],
            "journal_sha256": state["journal_sha256"],
            "does_not_establish": self.task.does_not_establish,
        }

    def _edit(self, experiment_id: str, hypothesis_id: str, action: dict[str, Any]) -> dict[str, Any]:
        before = read_files(self.store.workspace, self.task.editable_files)
        try:
            if isinstance(action.get("files"), dict):
                write_files(self.store.workspace, self.task.editable_files, action["files"])
            elif isinstance(action.get("edits"), list):
                apply_edits(self.store.workspace, self.task.editable_files, action["edits"])
            else:
                raise EditError("edit action needs files or edits")
        except (EditError, PathViolation, OSError, UnicodeError) as exc:
            write_files(self.store.workspace, self.task.editable_files, before)
            return self._commit({
                "experiment_id": experiment_id,
                "hypothesis_id": hypothesis_id,
                "action_type": "edit",
                "status": "edit_rejected",
                "failure_reason": exc.__class__.__name__,
                "metrics": {},
                "public": None,
                "files_after": None,
                "diff": "",
                "source_hashes": _hash_files(before),
                "commands": [],
                "wall_s": 0.0,
                "peak_rss_kb": None,
                "verification": None,
                "next_suggested_action": None,
            })
        if action.get("measure") is False:
            after = read_files(self.store.workspace, self.task.editable_files)
            return self._commit(self._record(
                experiment_id,
                hypothesis_id,
                "edit",
                "unmeasured",
                {},
                None,
                before,
                after,
                0.0,
                None,
                None,
            ))
        return self._measure(experiment_id, hypothesis_id, "edit", before)

    def _measure(
        self,
        experiment_id: str,
        hypothesis_id: str,
        action_type: str,
        before: dict[str, str] | None,
    ) -> dict[str, Any]:
        measured = public_metrics(self.task, self.store.workspace)
        metrics = _json_safe(measured.get("metrics") or {})
        after = read_files(self.store.workspace, self.task.editable_files)
        prior = before if before is not None else after
        status = "baseline" if hypothesis_id == "baseline" else (
            "supported" if measured.get("classification") == "public_measurement"
            and publicly_supported(self.task.task_type, metrics)
            else "rejected"
        )
        reason = ""
        if status == "rejected":
            reason = ",".join(measured.get("failures") or []) or "public acceptance criteria were not met"
        public = {
            "classification": measured.get("classification"),
            "metrics": metrics,
        }
        return self._commit(self._record(
            experiment_id,
            hypothesis_id,
            action_type,
            status,
            metrics,
            public,
            prior,
            after if action_type == "edit" else None,
            float(measured.get("wall_s") or 0.0),
            measured.get("peak_rss_kb"),
            None,
            failure_reason=reason,
        ))

    def _submit(self, experiment_id: str, hypothesis_id: str) -> dict[str, Any]:
        result = grade(self.task, self.store.workspace)
        verification = _json_safe(result)
        status = "accepted" if result.get("passed") is True else "rejected"
        after = read_files(self.store.workspace, self.task.editable_files)
        before = {name: str(text) for name, text in self.header["baseline_files"].items()}
        return self._commit(self._record(
            experiment_id,
            hypothesis_id,
            "submit",
            status,
            _json_safe(result.get("metrics") or {}),
            None,
            before,
            after,
            float(result.get("wall_s") or 0.0),
            result.get("peak_rss_kb"),
            verification,
            failure_reason="" if status == "accepted" else ",".join(result.get("failures") or []),
        ))

    def _record(
        self,
        experiment_id: str,
        hypothesis_id: str,
        action_type: str,
        status: str,
        metrics: dict[str, Any],
        public: dict[str, Any] | None,
        before: dict[str, str],
        after: dict[str, str] | None,
        wall_s: float,
        peak_rss_kb: int | None,
        verification: dict[str, Any] | None,
        failure_reason: str = "",
    ) -> dict[str, Any]:
        draft = {
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "action_type": action_type,
            "status": status,
            "failure_reason": failure_reason,
            "metrics": metrics,
            "public": public,
            "files_after": after,
            "diff": _diff(before, after or before),
            "source_hashes": _hash_files(after or before),
            "commands": list(_COMMANDS.get(self.task.task_type, [])),
            "wall_s": wall_s,
            "peak_rss_kb": peak_rss_kb,
            "verification": verification,
            "next_suggested_action": None,
        }
        draft["next_suggested_action"] = self._peek_next(draft)
        return draft

    def _peek_next(self, draft: dict[str, Any]) -> str | None:
        if self.proposer is None or draft["action_type"] in {"submit", "stop", "cancel"}:
            return None
        preview = self.observation()
        preview["memory"] = list(preview["memory"]) + [_memory_row(draft)]
        if isinstance(draft.get("public"), dict):
            preview["last_public"] = draft["public"]
        try:
            action = self.proposer.propose(preview)
        except (HypothesisNotApplicable, EditError, KeyError, TypeError, ValueError):
            return None
        if isinstance(action, dict):
            return str(action.get("hypothesis_id") or "") or None
        return None

    def _commit(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.store.commit(_json_safe(record))

    def _commit_terminal(self, action_type: str, hypothesis_id: str, status: str, reason: str) -> dict[str, Any]:
        experiment_id = f"{self.header['run_id']}:{hypothesis_id}"
        existing = self.store.find(experiment_id)
        if existing is not None:
            return existing
        files = read_files(self.store.workspace, self.task.editable_files)
        return self._commit({
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "action_type": action_type,
            "status": status,
            "failure_reason": reason,
            "metrics": {},
            "public": None,
            "files_after": None,
            "diff": "",
            "source_hashes": _hash_files(files),
            "commands": [],
            "wall_s": 0.0,
            "peak_rss_kb": None,
            "verification": None,
            "next_suggested_action": None,
        })

    def _commit_rejected_action(self, action: dict[str, Any], reason: str) -> dict[str, Any]:
        token = str(action.get("hypothesis_id") or action.get("type") or "rejected")
        return self._commit_terminal("stop", f"disallowed-{token}", "stopped", reason)


def expected_files(header: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, str]:
    """Replay committed file snapshots. This does not write the workspace."""
    files = {name: str(text) for name, text in header["baseline_files"].items()}
    for record in records:
        after = record.get("files_after")
        if isinstance(after, dict):
            for name, content in after.items():
                if name in files and isinstance(content, str):
                    files[name] = content
    return files


def _memory_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": record.get("experiment_id"),
        "hypothesis_id": record.get("hypothesis_id"),
        "action_type": record.get("action_type"),
        "status": record.get("status"),
        "failure_reason": record.get("failure_reason"),
        "metrics": record.get("metrics") or {},
        "wall_s": record.get("wall_s"),
        "peak_rss_kb": record.get("peak_rss_kb"),
        "source_hashes": record.get("source_hashes") or {},
        "commands": record.get("commands") or [],
        "next_suggested_action": record.get("next_suggested_action"),
    }


def _hash_files(files: dict[str, str]) -> dict[str, str]:
    import hashlib
    return {
        name: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for name, text in sorted(files.items())
    }


def _diff(before: dict[str, str], after: dict[str, str]) -> str:
    chunks: list[str] = []
    for name in sorted(set(before) | set(after)):
        old = before.get(name, "").splitlines(keepends=True)
        new = after.get(name, "").splitlines(keepends=True)
        if old == new:
            continue
        chunks.extend(difflib.unified_diff(old, new, fromfile=name, tofile=name))
    text = "".join(chunks)
    return text[:20000]


def create_run(
    run_dir: Path,
    task: ResearchTask,
    *,
    run_id: str,
    proposer_name: str,
    proposer_kind: str,
    suite_sha256: str,
) -> Controller:
    if run_dir.exists():
        raise StoreError("run directory already exists")
    run_dir.mkdir(parents=True)
    try:
        materialize(task, run_dir / "workspace")
        files = read_files(run_dir / "workspace", task.editable_files)
        header = {
            "schema": "vare.research_lab.run.v1",
            "run_id": run_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "spec_sha256": task.spec_sha256,
            "suite_sha256": suite_sha256,
            "proposer_name": proposer_name,
            "proposer_kind": proposer_kind,
            "max_actions": task.max_iterations,
            "baseline_files": files,
            "baseline_hashes": _hash_files(files),
            "model_backed": False,
            "counts_as_autonomous_model_success": False,
            "does_not_establish": task.does_not_establish,
        }
        _atomic_json(run_dir / "run.json", header)
        (run_dir / "journal.jsonl").write_text("", encoding="utf-8")
        RunStore(run_dir).write_state_cache()
    except Exception:
        import shutil
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    proposer = _load_proposer(proposer_name, task)
    return Controller(run_dir, task, proposer)


def open_run(run_dir: Path, task: ResearchTask) -> Controller:
    header = RunStore(run_dir).load_header()
    proposer = _load_proposer(str(header["proposer_name"]), task)
    return Controller(run_dir, task, proposer)


def _load_proposer(name: str, task: ResearchTask) -> Any:
    from .proposer import proposer_for
    if name == "external":
        return None
    return proposer_for(name, task)
