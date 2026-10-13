"""Execute the actual legacy/seeded summaries on identical synthetic outputs."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, type=Path)
parser.add_argument("--implementation", choices=("legacy", "seeded"), required=True)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(args.source / "src"), str(ROOT)]

from vare import format_vs_correctness_v2 as legacy
from reproducers.seeded_summary_fixture_v1 import synthetic_grid

confirm, pack, rows = synthetic_grid()
if args.implementation == "legacy":
    result = legacy.summarize_comparison(confirm, pack, rows)
else:
    from vare.format_vs_correctness_seeded_summary_v3 import summarize_seeded_comparison
    result = summarize_seeded_comparison(confirm, pack, rows, prospective=False)
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
(args.output / "tasks.json").write_text(json.dumps({"confirm": confirm, "pack": pack}, indent=2) + "\n")
(args.output / "synthetic-progress.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
points = {row["arm"]: row for row in result["curves"]
          if row["split"] == "confirm" and row["checkpoint_step"] == 16}
assert abs(points["format_sft"]["pass_at_4"] - 0.2) < 1e-15, "pooled outputs do not estimate mean fitted-run pass@4"
assert abs(points["correctness_reward_rl"]["pass_at_4"] - 0.6) < 1e-15
