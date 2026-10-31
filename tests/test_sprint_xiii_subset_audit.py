from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "sprint_xiii" / "audit_completed_online_actor_subset.py"
spec = importlib.util.spec_from_file_location("sprint_xiii_subset_audit", SCRIPT)
subset_audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(subset_audit)


def test_expected_cases_are_seed_arm_pairs_in_frozen_order() -> None:
    protocol = {"execution_order": [[101, "A"], [202, "E"], [303, "C"]]}
    assert subset_audit.expected_case_ids(protocol) == [
        "seed-101-A",
        "seed-202-E",
        "seed-303-C",
    ]


def test_subset_audit_selects_every_completed_case_without_outcome_filtering(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases"
    (cases / "seed-1-A").mkdir(parents=True)
    (cases / "seed-1-A" / "case-result.json").write_text(json.dumps({"ok": True}))
    (cases / "seed-1-B").mkdir()
    (cases / "unrelated-file.txt").write_text("ignore")
    assert [p.name for p in subset_audit.completed_case_roots(tmp_path)] == [
        "seed-1-A"
    ]
