#!/usr/bin/env python3
"""Independently compare a proposed edit-format module with the frozen baseline.

Expected values are computed in the trusted parent. Candidate source runs only
inside the digest-pinned, network-disabled Colima container.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks" / "mlsys_edit_format_local_v1"))

from vare.container_isolation import DockerIsolationRunner  # noqa: E402
from vare.isolated_candidate import FunctionCall, grade_calls  # noqa: E402
from vare.learning_pipeline import edit_format as baseline  # noqa: E402
import workload  # noqa: E402

SOURCE = ROOT / "src" / "vare" / "learning_pipeline" / "edit_format.py"
ASSET = ROOT / "benchmarks" / "mlsys_edit_format_local_v1"
BASELINE_SHA256 = "4f0ee215d5769b42df3aa83128e947b7259234141678856fba74d98bdc7c1f59"
IMAGE = "python@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258"
RUNNER_SOURCE = (
    "import edit_format\n"
    "def run_case(original, completion):\n"
    "    o = edit_format.apply_blocks(original, edit_format.parse_blocks(completion))\n"
    "    return {'ok': o.ok, 'reason': o.reason, 'has_new_text': o.new_text is not None, "
    "'new_text': o.new_text if o.new_text is not None else ''}\n"
)
PUBLIC_CASES = [
    ("def add(a, b):\n    return a - b\n\n\nX = 1\n",
     "<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"),
    ("def add(a, b):\n    return a - b\n\n\nX = 1\n", "no blocks here"),
    ("def add(a, b):\n    return a - b\n\n\nX = 1\n",
     "<<<<<<< SEARCH\nnot in file\n=======\nx\n>>>>>>> REPLACE"),
    ("def add(a, b):\n    return a - b\n\n\nX = 1\n",
     "<<<<<<< SEARCH\n    return a - b\n=======\n    return a - b\n>>>>>>> REPLACE"),
    ("x = 1\nx = 1\n", "<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"),
    ("def add(a, b):\n    return a - b\n\n\nX = 1\n",
     "<think><<<<<<< SEARCH\nX = 1\n=======\nX = 9\n>>>>>>> REPLACE</think>"
     "<<<<<<< SEARCH\n    return a - b\n=======\n    return a + b\n>>>>>>> REPLACE"),
    ("a = 1   \nb = 2\n",
     "<<<<<<< SEARCH\na = 1\t\nb = 2\n=======\nc\n>>>>>>> REPLACE"),
]
DIFFERENTIAL_SEED = 780520261101


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_only_relaxed_replace_changed(baseline_text: str, candidate_text: str) -> None:
    baseline_tree = ast.parse(baseline_text)
    candidate_tree = ast.parse(candidate_text)
    target = "_relaxed_replace"
    baseline_matches = [i for i, node in enumerate(baseline_tree.body)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target]
    candidate_matches = [i for i, node in enumerate(candidate_tree.body)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target]
    if len(baseline_matches) != 1 or len(candidate_matches) != 1:
        raise ValueError("candidate must contain exactly one _relaxed_replace")
    normalized_baseline = copy.deepcopy(baseline_tree)
    normalized_candidate = copy.deepcopy(candidate_tree)
    normalized_baseline.body[baseline_matches[0]].body = [ast.Pass()]
    normalized_candidate.body[candidate_matches[0]].body = [ast.Pass()]
    if ast.dump(normalized_baseline, include_attributes=False) != ast.dump(normalized_candidate, include_attributes=False):
        raise ValueError("candidate changes code outside _relaxed_replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    if os.environ.get("DOCKER_CONTEXT") != "colima":
        raise SystemExit("set DOCKER_CONTEXT=colima")
    if sha256(SOURCE) != BASELINE_SHA256:
        raise SystemExit("frozen baseline source hash mismatch")
    candidate_text = args.candidate.read_text(encoding="utf-8")
    baseline_text = SOURCE.read_text(encoding="utf-8")
    assert_only_relaxed_replace_changed(baseline_text, candidate_text)
    candidate_sha = hashlib.sha256(candidate_text.encode()).hexdigest()
    cases = PUBLIC_CASES + workload.make_cases(DIFFERENTIAL_SEED, 24, profile="differential")
    calls = []
    for index, (file_text, completion) in enumerate(cases):
        expected = baseline.apply_blocks(file_text, baseline.parse_blocks(completion))
        calls.append(FunctionCall(
            f"case-{index:03d}", "candidate_runner", "run_case", [file_text, completion],
            {"ok": expected.ok, "reason": expected.reason,
             "has_new_text": expected.new_text is not None,
             "new_text": expected.new_text if expected.new_text is not None else ""},
        ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    work_root = ROOT / "work"
    work_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mlsys-edit-format-eval-", dir=work_root) as tmp:
        subject = Path(tmp)
        (subject / "edit_format.py").write_text(candidate_text, encoding="utf-8")
        (subject / "candidate_runner.py").write_text(RUNNER_SOURCE, encoding="utf-8")
        started = time.perf_counter()
        report = grade_calls(
            subject, calls, task_kind="correctness", image=IMAGE, timeout_s=90,
            runner=DockerIsolationRunner(), preserve_session=True, discard_candidate_stdout=True,
        )
        elapsed = time.perf_counter() - started
    passed = sum(case.passed for case in report.cases)
    record = {
        "schema": "vare.mlsys_edit_format_proposal_evaluation.v1",
        "attempt_id": args.attempt_id,
        "evaluator_source_sha256": sha256(Path(__file__).resolve()),
        "differential_seed_sha256": hashlib.sha256(str(DIFFERENTIAL_SEED).encode()).hexdigest(),
        "candidate_source_sha256": candidate_sha,
        "baseline_source_sha256": sha256(SOURCE),
        "workload_sha256": sha256(ASSET / "workload.py"),
        "case_count": len(cases),
        "public_case_count": len(PUBLIC_CASES),
        "differential_case_count": len(cases) - len(PUBLIC_CASES),
        "pass_count": passed,
        "correct": report.passed,
        "failures": [{"case_id": case.case_id, "failure": case.failure}
                     for case in report.cases if not case.passed],
        "evaluator_wall_seconds": round(elapsed, 6),
        "python": platform.python_version(),
        "container_image": IMAGE,
        "grader": "trusted parent computes unchanged baseline outputs; candidate runs in isolated Colima session",
    }
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({"attempt_id": args.attempt_id, "correct": record["correct"],
                      "pass_count": passed, "case_count": len(cases), "out": str(args.out)}, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
