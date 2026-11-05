"""Append-only run journal.

The journal is the source of truth. ``state.json`` is a cache. A truncated
final line is ignored. Committing an existing ``experiment_id`` returns the
original record and does not execute anything again.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from vare.evidence import canonical_json, sha256_json


class StoreError(ValueError):
    """The run directory is not a ResearchGym journal."""


def _parse_journal_text(text: str) -> list[dict[str, Any]]:
    import json
    records: list[dict[str, Any]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise StoreError("journal contains a corrupt committed line") from None
        if not isinstance(item, dict) or "seq" not in item:
            raise StoreError("journal record is missing seq")
        records.append(item)
    expected = 1
    seen: set[str] = set()
    for record in records:
        if int(record["seq"]) != expected:
            raise StoreError("journal sequence is not contiguous")
        experiment_id = str(record["experiment_id"])
        if experiment_id in seen:
            raise StoreError("journal contains a duplicate experiment id")
        seen.add(experiment_id)
        expected += 1
    return records


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RunStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.header_path = run_dir / "run.json"
        self.journal_path = run_dir / "journal.jsonl"
        self.state_path = run_dir / "state.json"
        self.workspace = run_dir / "workspace"

    def load_header(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.header_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError("run.json is unreadable") from exc
        if not isinstance(payload, dict) or payload.get("schema") != "vare.research_lab.run.v1":
            raise StoreError("run.json schema mismatch")
        return payload

    def read_journal(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        return _parse_journal_text(self.journal_path.read_text(encoding="utf-8"))

    def find(self, experiment_id: str) -> dict[str, Any] | None:
        for record in self.read_journal():
            if record["experiment_id"] == experiment_id:
                return record
        return None

    def commit(self, record: dict[str, Any]) -> dict[str, Any]:
        import fcntl
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path.touch(exist_ok=True)
        experiment_id = str(record["experiment_id"])
        with self.journal_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            text = handle.read()
            records = _parse_journal_text(text)
            for existing in records:
                if existing["experiment_id"] == experiment_id:
                    return existing
            body = dict(record)
            body["seq"] = len(records) + 1
            body["record_sha256"] = sha256_json(
                {key: value for key, value in body.items() if key != "record_sha256"}
            )
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json(body) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(self.run_dir)
        self.write_state_cache()
        return body

    def write_state_cache(self) -> dict[str, Any]:
        records = self.read_journal()
        header = self.load_header()
        verification = None
        done = False
        status = "running"
        for record in records:
            if record.get("action_type") == "submit":
                verification = record.get("verification")
                done = True
                status = "verified" if record.get("status") == "accepted" else "failed"
            elif record.get("action_type") == "stop":
                done = True
                status = "stopped"
            elif record.get("action_type") == "cancel":
                done = True
                status = "cancelled"
        if self.cancel_requested() and not done:
            status = "cancel_requested"
        payload = {
            "schema": "vare.research_lab.state.v1",
            "run_id": header["run_id"],
            "task_id": header["task_id"],
            "seq": len(records),
            "done": done,
            "status": status,
            "verification": verification,
            "journal_sha256": sha256_json(records),
        }
        _atomic_json(self.state_path, payload)
        return payload

    def cancel_requested(self) -> bool:
        return (self.run_dir / "CANCEL").exists()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except OSError:
            pass
        raise


def init_run(run_dir: Path, header: dict[str, Any]) -> None:
    if run_dir.exists():
        raise StoreError(f"run directory already exists: {run_dir.name}")
    run_dir.mkdir(parents=True)
    _atomic_json(run_dir / "run.json", header)
    (run_dir / "journal.jsonl").write_text("", encoding="utf-8")
    RunStore(run_dir).write_state_cache()
