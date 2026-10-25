#!/usr/bin/env python3
"""Search retained VARE history and artifacts for prior use of candidate tasks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any


QUERIES = {
    "mlxlm-2004-jamba-moe-routing": [
        "#2004",
        "mlxlm-2004-jamba-moe-routing",
        "Fix Jamba MoE routing weights to match reference",
        "JambaSparseMoeBlock",
    ],
    "mlxlm-1996-nested-parameter-tags": [
        "#1996",
        "mlxlm-1996-nested-parameter-tags",
        "handle nested parameter tags in Qwen3 Coder parser",
        "_find_parameter_blocks",
    ],
}


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_NO_REPLACE_OBJECTS": "1"},
    )
    return result.stdout


def _history_matches(root: Path, literal: str) -> list[str]:
    output = _git(
        root,
        "log",
        "--all",
        "--format=%H",
        f"-S{literal}",
        "--",
        "docs",
        "protocols",
        "results",
        "benchmarks",
    )
    return sorted(set(output.splitlines()))


def _working_tree_matches(root: Path, literal: str) -> list[dict[str, Any]]:
    needle = literal.encode("utf-8")
    excluded = Path("benchmarks/mlsys_agent_development_v1/tasks")
    current_campaign = Path("results/agent-edit-interface-v1")
    matches = []
    for top in ("docs", "protocols", "results", "benchmarks"):
        for path in sorted((root / top).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if relative.is_relative_to(excluded) or relative.is_relative_to(current_campaign):
                continue
            if path.stat().st_size > 128 * 1024 * 1024:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if needle not in data:
                continue
            lines = data.decode("utf-8", errors="replace").splitlines()
            for number, line in enumerate(lines, 1):
                if literal in line:
                    matches.append({
                        "path": relative.as_posix(),
                        "line": number,
                        "excerpt": line[:240],
                    })
    return matches


def audit(root: Path) -> dict[str, Any]:
    tasks = {}
    for task_id, literals in QUERIES.items():
        query_results = []
        for literal in literals:
            query_results.append({
                "literal": literal,
                "history_commit_matches": _history_matches(root, literal),
                "working_tree_matches_before_this_task": _working_tree_matches(root, literal),
            })
        tasks[task_id] = {
            "exact_task_identity_prior_matches": sum(
                bool(item["history_commit_matches"] or item["working_tree_matches_before_this_task"])
                for item in query_results[:2]
            ),
            "queries": query_results,
        }
    return {
        "schema_version": 1,
        "audit_scope": "all local Git refs, plus current docs/protocols/results/benchmarks outside the new candidate task directory and this campaign's own evidence",
        "working_tree_exclusions": [
            "benchmarks/mlsys_agent_development_v1/tasks",
            "results/agent-edit-interface-v1",
        ],
        "git_replacement_refs_disabled": True,
        "external_model_pretraining_exposure": "unknown",
        "tasks": tasks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite exposure audit: {args.output}")
    record = audit(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        task_id: item["exact_task_identity_prior_matches"]
        for task_id, item in record["tasks"].items()
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
