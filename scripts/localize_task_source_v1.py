#!/usr/bin/env python3
"""Select one bounded source window using only candidate-visible prompt text.

This deterministic localizer does not read upstream fixes, tests, or graders.
It is a prospective context intervention, not a semantic repair oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vare.agent_task_contract import load_agent_task  # noqa: E402

WINDOW_LINES = 80
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
SPLIT_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|_+")
STOP_WORDS = frozenset({
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "because", "before", "but", "by", "can", "change", "do", "file", "fix",
    "for", "from", "in", "into", "it", "its", "make", "must", "not", "of", "on",
    "only", "or", "preserve", "should", "the", "then", "this", "to", "when", "where",
    "with", "without", "you", "your",
})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def terms(text: str) -> set[str]:
    result: set[str] = set()
    for match in TOKEN.findall(text):
        result.update(
            part.casefold() for part in SPLIT_BOUNDARY.split(match)
            if len(part) > 2 and part.casefold() not in STOP_WORDS
        )
    return result


def choose_window(prompt: str, source: str, path: str) -> dict[str, Any]:
    prompt_terms = terms(prompt)
    lines = source.splitlines(keepends=True)
    if not lines:
        raise ValueError(f"candidate source is empty: {path}")
    width = min(WINDOW_LINES, len(lines))
    line_scores = [len(prompt_terms & terms(line)) for line in lines]
    best_start = max(
        range(0, len(lines) - width + 1),
        key=lambda start: (sum(line_scores[start:start + width]),
                           sum(score > 0 for score in line_scores[start:start + width]), -start),
    )
    score = sum(line_scores[best_start:best_start + width])
    if score == 0:
        raise ValueError(f"task prompt has no lexical match in candidate source: {path}")
    selected = lines[best_start:best_start + width]
    return {
        "algorithm": "task_prompt_lexical_windows_v1",
        "path": path,
        "start_line": best_start + 1,
        "end_line": best_start + len(selected),
        "window_lines": len(selected),
        "lexical_overlap_score": score,
        "prompt_terms_sha256": sha256_bytes("\n".join(sorted(prompt_terms)).encode("utf-8")),
        "content": "".join(selected),
    }


def localize(task_root: Path, workspace: Path) -> dict[str, Any]:
    contract = load_agent_task(task_root)
    source_paths = [
        path for path in contract.candidate_visible_source_files
        if path.endswith(".py") and path != "test_public.py"
    ]
    if len(source_paths) != 1:
        raise ValueError("localization v1 requires exactly one candidate-visible Python source file")
    source_path = source_paths[0]
    source_bytes = (workspace / source_path).read_bytes()
    source = source_bytes.decode("utf-8", errors="strict")
    selected = choose_window(contract.prompt_text, source, source_path)
    prompt = contract.prompt_text.split("## Reproduce", 1)[0].rstrip()
    return {
        **selected,
        "schema_version": 1,
        "task_id": contract.task_id,
        "base_revision": contract.base_revision,
        "task_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "source_sha256": sha256_bytes(source_bytes),
        "content_sha256": sha256_bytes(selected["content"].encode("utf-8")),
        "selection_inputs": ["task_prompt", "candidate_source", "editable_path_from_task_metadata"],
        "known_fix_or_grader_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite localization record: {args.output}")
    record = localize(args.task_root, args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: record[key] for key in (
        "task_id", "path", "start_line", "end_line", "lexical_overlap_score", "content_sha256",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
