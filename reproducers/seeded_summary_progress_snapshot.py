"""Force a progress-file replacement between byte capture and analysis."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(args.source / "src"), str(args.source), str(ROOT)]

# Imports must follow selection of the implementation under test.
from reproducers.seeded_summary_fixture_v1 import synthetic_grid  # noqa: E402
from scripts import summarize_format_vs_correctness_seeded_v3 as cli  # noqa: E402
from vare import format_vs_correctness_v2 as legacy  # noqa: E402
from vare.format_vs_correctness_seeded_summary_v3 import summarize_seeded_comparison  # noqa: E402

confirm, pack, rows = synthetic_grid()
for row in rows:
    row.update(backend="mlx", model_id="synthetic-only-model", protocol_sha256=legacy.PROTOCOL_SHA256)
expected = summarize_seeded_comparison(confirm, pack, rows, prospective=False)
changed = copy.deepcopy(rows)
for row in changed:
    if row["kind"] == "eval" and row["arm"] == "format_sft":
        row["output"] = "no answer"
before = "".join(json.dumps(row) + "\n" for row in rows).encode()
after = "".join(json.dumps(row) + "\n" for row in changed).encode()
args.output.mkdir(parents=True, exist_ok=True)
progress = args.output / "progress.jsonl"
progress.write_bytes(before)
(args.output / "captured-progress.jsonl").write_bytes(before)
fake_root = args.output / "fixture-root"
pack_path = fake_root / "benchmarks/heldout/exact_integer_v1/tasks.jsonl"
pack_path.parent.mkdir(parents=True)
pack_path.write_text("hand-made fixture only")
real_read = Path.read_bytes


def replace_after_capture(path):
    blob = real_read(path)
    if path == progress:
        path.write_bytes(after)
    return blob


with patch.object(cli, "ROOT", fake_root), \
     patch.object(cli, "_protocol", lambda: {"models": {"mlx": {"repo_id": "synthetic-only-model"}}}), \
     patch.object(cli, "confirm_eval_tasks", lambda: confirm), \
     patch.object(cli, "pack_eval_tasks", lambda _: pack), \
     patch.object(Path, "read_bytes", replace_after_capture):
    cli.analyze(args.output, args.output / "summary.json", retrospective=True)
result = json.loads((args.output / "summary.json").read_text())
checks = {
    "digest_matches_captured_bytes": result["progress_sha256"] == hashlib.sha256(before).hexdigest(),
    "metrics_match_captured_bytes": result["paired_bootstrap"] == expected["paired_bootstrap"],
    "curves_match_captured_bytes": result["curves"] == expected["curves"],
    "writer_changes_retained": real_read(progress) == after,
}
(args.output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
assert all(checks.values()), "summary metrics and digest must refer to the same captured bytes"
