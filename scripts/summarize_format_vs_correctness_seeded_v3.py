"""Analyze completed v2 progress without training or overwriting old reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from scripts.run_format_vs_correctness_comparison_v2 import _index, _protocol  # noqa: E402
from vare.format_vs_correctness_v2 import confirm_eval_tasks, pack_eval_tasks  # noqa: E402
from vare.format_vs_correctness_seeded_summary_v3 import (  # noqa: E402
    ANALYSIS_SHA256, canonical_analysis_bytes, summarize_seeded_comparison, validate_analysis,
)
from vare.heldout_exact_integer import read_jsonl  # noqa: E402


def analyze(run: Path, output: Path, *, retrospective: bool) -> None:
    validate_analysis()
    protocol = _protocol()
    progress = run / "progress.jsonl"
    blob = progress.read_bytes()
    rows = read_jsonl(progress)
    if not rows:
        raise ValueError("completed progress is required")
    first = rows[0]
    backend = first.get("backend")
    if backend not in protocol["models"]:
        raise ValueError("progress backend is outside the frozen protocol")
    model_id = protocol["models"][backend]["repo_id"]
    if not retrospective:
        pin = run / "analysis-protocol.seeded-v3.json"
        if not pin.exists() or pin.read_bytes() != canonical_analysis_bytes():
            raise ValueError("run has no prospective analysis pin; pass --retrospective")
    _index(rows, backend, model_id, analysis_sha=None if retrospective else ANALYSIS_SHA256)
    confirm = confirm_eval_tasks()
    pack = pack_eval_tasks((ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl").read_text())
    result = summarize_seeded_comparison(confirm, pack, rows, prospective=not retrospective)
    result["analysis_file_pin_verified"] = not retrospective
    result["progress_sha256"] = hashlib.sha256(blob).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retrospective", action="store_true")
    args = parser.parse_args()
    try:
        analyze(args.run, args.output or args.run / "summary.seeded-v3.json",
                retrospective=args.retrospective)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
