"""Export a verified ResearchGym run for later qualification review.

The export does not authorize training, does not assign a reward, and does
not read held-out protocol splits. Downstream GRPO code must treat
``training_authorized`` as false unless a separate, explicit qualification
step says otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import RunStore, StoreError

SCHEMA = "vare.research_lab.qualification_export.v1"
_DROPPED = frozenset({
    "reward",
    "advantage",
    "return",
    "heldout_ids",
    "training_authorized",
})


class ExportError(ValueError):
    """The run cannot be exported as an unqualified trajectory."""


def export_run(run_dir: Path) -> dict[str, Any]:
    store = RunStore(run_dir)
    try:
        header = store.load_header()
        records = store.read_journal()
    except StoreError as exc:
        raise ExportError(str(exc)) from exc
    task_id = str(header.get("task_id") or "")
    if task_id.startswith(("heldout", "run002", "coding_policy_learning")):
        raise ExportError("refusing to export a frozen protocol id")
    actions = []
    for record in records:
        actions.append({
            "seq": record.get("seq"),
            "experiment_id": record.get("experiment_id"),
            "hypothesis_id": record.get("hypothesis_id"),
            "action_type": record.get("action_type"),
            "status": record.get("status"),
            "failure_reason": record.get("failure_reason"),
            "metrics": record.get("metrics"),
            "source_hashes": record.get("source_hashes"),
            "commands": record.get("commands"),
            "wall_s": record.get("wall_s"),
            "peak_rss_kb": record.get("peak_rss_kb"),
            "verification_passed": _verification_passed(record.get("verification")),
            "verification_classification": _verification_class(record.get("verification")),
        })
    payload = {
        "schema": SCHEMA,
        "training_authorized": False,
        "reward_recomputed": False,
        "held_out_split_accessed": False,
        "task_id": task_id,
        "task_type": header.get("task_type"),
        "spec_sha256": header.get("spec_sha256"),
        "suite_sha256": header.get("suite_sha256"),
        "proposer_name": header.get("proposer_name"),
        "proposer_kind": header.get("proposer_kind"),
        "model_backed": False,
        "counts_as_autonomous_model_success": False,
        "baseline_hashes": header.get("baseline_hashes"),
        "actions": actions,
        "does_not_establish": header.get("does_not_establish"),
        "qualification_note": (
            "Unqualified ResearchGym trajectory. This record does not authorize "
            "a learner update, does not define a GRPO reward, and was not read "
            "from a held-out split."
        ),
    }
    _reject_forbidden(payload)
    if payload["training_authorized"] is not False:
        raise ExportError("training authorization failed closed")
    return payload


def _verification_passed(verification: Any) -> bool | None:
    if isinstance(verification, dict) and isinstance(verification.get("passed"), bool):
        return verification["passed"]
    return None


def _verification_class(verification: Any) -> str | None:
    if isinstance(verification, dict) and isinstance(verification.get("classification"), str):
        return verification["classification"]
    return None


def _reject_forbidden(payload: dict[str, Any]) -> None:
    found = _scan(payload)
    leaked = found & _DROPPED - {"training_authorized"}
    if leaked:
        raise ExportError(f"export contained forbidden keys: {sorted(leaked)}")
    if "reward" in payload or "advantage" in payload:
        raise ExportError("export contained a reward field")


def _scan(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            found.update(_scan(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_scan(item))
    return found
