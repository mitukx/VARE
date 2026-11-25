"""Operator CLI for ResearchGym. Existing ``vare`` commands are unchanged."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from vare.evidence import canonical_json

from .calibration import run_calibration
from .controller import create_run, expected_files, open_run
from .edits import read_files
from .evaluation import grade
from .export_adapter import ExportError, export_run
from .spec import load_catalog, suite_manifest
from .store import RunStore, StoreError

_KINDS = {
    "diagnostic": "diagnostic_policy",
    "first-hypothesis": "single_attempt_control",
    "oracle": "oracle_control",
    "external": "external_agent",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vare research")
    sub = parser.add_subparsers(dest="command", required=True)

    listed = sub.add_parser("list", help="list ResearchGym tasks")
    listed.set_defaults(func=_cmd_list)

    run = sub.add_parser("run", help="start or continue a research run")
    _add_run_args(run)
    run.add_argument("--max-actions", type=int)
    run.set_defaults(func=_cmd_run)

    resume = sub.add_parser("resume", help="continue a committed run without repeating it")
    _add_run_args(resume, include_proposer=False)
    resume.add_argument("--max-actions", type=int)
    resume.set_defaults(func=_cmd_resume)

    status = sub.add_parser("status", help="show journal status")
    _add_run_args(status, include_proposer=False)
    status.set_defaults(func=_cmd_status)

    evaluate = sub.add_parser("evaluate", help="re-grade the current workspace without rewriting the journal")
    _add_run_args(evaluate, include_proposer=False)
    evaluate.set_defaults(func=_cmd_evaluate)

    report = sub.add_parser("report", help="print the machine-readable run summary")
    _add_run_args(report, include_proposer=False)
    report.set_defaults(func=_cmd_report)

    act = sub.add_parser("act", help="apply one external-agent action")
    _add_run_args(act, include_proposer=False)
    act.add_argument("--action-json", required=True)
    act.set_defaults(func=_cmd_act)

    export = sub.add_parser("export", help="export an unqualified trajectory")
    _add_run_args(export, include_proposer=False)
    export.set_defaults(func=_cmd_export)

    calibrate = sub.add_parser("calibrate", help="run baseline, known-good, and incorrect controls")
    calibrate.add_argument("--work-root", required=True)
    calibrate.set_defaults(func=_cmd_calibrate)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (StoreError, ExportError, KeyError, OSError, ValueError) as exc:
        print(canonical_json({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2


def _add_run_args(parser: argparse.ArgumentParser, *, include_proposer: bool = True) -> None:
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", default="artifacts/research_lab")
    if include_proposer:
        parser.add_argument(
            "--proposer",
            default="diagnostic",
            choices=tuple(_KINDS),
        )
        parser.add_argument(
            "--oracle-control",
            action="store_true",
            help="required to run the known-good oracle control",
        )


def _task(task_id: str):
    catalog = load_catalog()
    if task_id not in catalog:
        raise StoreError(f"unknown task {task_id}")
    return catalog[task_id]


def _run_dir(args: argparse.Namespace) -> Path:
    if "/" in args.run_id or "\\" in args.run_id or args.run_id in {".", ".."}:
        raise StoreError("run-id must be a single path segment")
    return Path(args.run_root) / args.run_id


def _cmd_list(_args: argparse.Namespace) -> int:
    rows = []
    for task in load_catalog().values():
        rows.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "max_iterations": task.max_iterations,
            "spec_sha256": task.spec_sha256,
        })
    print(canonical_json({"tasks": rows}))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.proposer == "oracle" and not args.oracle_control:
        raise StoreError("oracle proposer is a grader control and needs --oracle-control")
    task = _task(args.task)
    run_dir = _run_dir(args)
    if run_dir.exists():
        controller = open_run(run_dir, task)
        if controller.header["proposer_name"] != args.proposer:
            raise StoreError("existing run proposer does not match --proposer")
    else:
        controller = create_run(
            run_dir,
            task,
            run_id=args.run_id,
            proposer_name=args.proposer,
            proposer_kind=_KINDS[args.proposer],
            suite_sha256=suite_manifest()["manifest_sha256"],
        )
    summary = controller.run(max_actions=args.max_actions)
    print(canonical_json(summary))
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    task = _task(args.task)
    controller = open_run(_run_dir(args), task)
    summary = controller.run(max_actions=args.max_actions)
    print(canonical_json(summary))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    store = RunStore(_run_dir(args))
    state = store.write_state_cache()
    records = store.read_journal()
    last = records[-1] if records else None
    print(canonical_json({
        "state": state,
        "last_hypothesis_id": None if last is None else last.get("hypothesis_id"),
        "last_status": None if last is None else last.get("status"),
    }))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    task = _task(args.task)
    run_dir = _run_dir(args)
    store = RunStore(run_dir)
    header = store.load_header()
    records = store.read_journal()
    current_files = read_files(store.workspace, task.editable_files)
    if current_files != expected_files(header, records):
        result = {
            "passed": False,
            "classification": "stale_source",
            "correctness_evaluated": False,
            "failures": ["workspace no longer matches the committed journal"],
        }
    else:
        result = grade(task, store.workspace)
        submitted = _submitted_hashes(records)
        current = result.get("source_hashes")
        if submitted is not None and current != submitted:
            result = dict(result)
            result["passed"] = False
            result["classification"] = "stale_source"
            result["correctness_evaluated"] = False
            result["failures"] = ["workspace no longer matches the submitted snapshot"]
    path = run_dir / "evaluations.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(result))
        handle.write("\n")
    print(canonical_json({
        "classification": result.get("classification"),
        "passed": result.get("passed"),
        "journal_unchanged": True,
    }))
    return 0 if result.get("passed") else 1


def _cmd_report(args: argparse.Namespace) -> int:
    task = _task(args.task)
    controller = open_run(_run_dir(args), task)
    print(canonical_json(controller.summary()))
    return 0


def _cmd_act(args: argparse.Namespace) -> int:
    task = _task(args.task)
    controller = open_run(_run_dir(args), task)
    if controller.header.get("proposer_kind") != "external_agent":
        raise StoreError("act is only available for external-agent runs")
    action = json.loads(Path(args.action_json).read_text(encoding="utf-8"))
    record = controller.apply(action)
    print(canonical_json({
        "experiment_id": record.get("experiment_id"),
        "hypothesis_id": record.get("hypothesis_id"),
        "status": record.get("status"),
        "metrics": record.get("metrics"),
        "failure_reason": record.get("failure_reason"),
        "verification_passed": (record.get("verification") or {}).get("passed")
        if isinstance(record.get("verification"), dict) else None,
    }))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    print(canonical_json(export_run(_run_dir(args))))
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    payload = run_calibration(Path(args.work_root))
    print(canonical_json({
        "expectations_met": payload["expectations_met"],
        "git_revision": payload["git_revision"],
        "suite_sha256": payload["suite_sha256"],
        "tasks": [
            {"task_id": row["task_id"], "matches_expectation": row["matches_expectation"]}
            for row in payload["tasks"]
        ],
    }))
    out = Path(args.work_root) / "calibration.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if payload["expectations_met"] else 1


def _submitted_hashes(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(records):
        if record.get("action_type") == "submit":
            hashes = record.get("source_hashes")
            return hashes if isinstance(hashes, dict) else None
    return None
