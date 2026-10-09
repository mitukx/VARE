#!/usr/bin/env python3
"""Show and close one inherited bad_words_ids false-accept gap in the RVL grader."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4_ROOT = ROOT / "benchmarks/historical/rvl_behavior_policy_parity"
V4_EVIDENCE = ROOT / "results/rvl-hf-behavior-policy-parity-v1/cpu-calibration-v4/protocol_snapshot"
V5_ROOT = ROOT / "benchmarks/historical/rvl_bad_words_neutrality"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def git(args: list[str], cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                          capture_output=True).stdout.strip()


def checkout_fixed(task: dict, destination: Path) -> None:
    destination.mkdir()
    git(["init", "--quiet"], destination)
    git(["remote", "add", "origin", task["source"]["repository"]], destination)
    for revision in (task["source"]["base_revision"], task["source"]["calibration_revision"]):
        git(["fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", revision], destination)
    git(["sparse-checkout", "init", "--no-cone"], destination)
    (destination / ".git/info/sparse-checkout").write_text(
        "".join(f"/{path}\n" for path in task["source"]["files"]), encoding="utf-8")
    git(["checkout", "--quiet", "--detach", task["source"]["calibration_revision"]], destination)
    if git(["rev-parse", "HEAD"], destination) != task["source"]["calibration_revision"]:
        raise RuntimeError("fixed source revision mismatch")


def grade(evaluator: Path, task_root: Path, workspace: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, str(evaluator), "--workspace", str(workspace),
         "--task-root", str(task_root)], check=False, text=True, capture_output=True, timeout=45)
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"grader returned invalid JSON: {completed.stdout[:300]} {completed.stderr[:300]}") from exc
    if completed.returncode not in (0, 1, 2) or "grader_error" in result:
        raise RuntimeError(f"grader could not evaluate candidate: {result} {completed.stderr[:300]}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/rvl-hf-bad-words-neutrality-v1/mutation-audit-v1")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not V4_EVIDENCE.is_dir():
        parser.error(f"retained v4 grader snapshot missing: {V4_EVIDENCE}")
    v4_evaluator = V4_EVIDENCE / "evaluator/grade.py"
    v5_evaluator = V5_ROOT / "evaluator/grade.py"
    task = json.loads((V5_ROOT / "task.json").read_text(encoding="utf-8"))
    backend_rel = "src/rvl_systems/hf_backend.py"
    output.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="vare-rvl-bad-words-") as temp:
        workspace = Path(temp) / "fixed-source"
        checkout_fixed(task, workspace)
        v4_control = grade(v4_evaluator, V4_ROOT, workspace)
        v5_control = grade(v5_evaluator, V5_ROOT, workspace)
        source_path = workspace / backend_rel
        original = source_path.read_text(encoding="utf-8")
        needle = "            top_p=1.0,\n"
        if original.count(needle) != 1:
            raise RuntimeError("expected one neutral GenerationConfig top_p field")
        mutated = original.replace(needle, needle + "            bad_words_ids=[[2]],\n")
        source_path.write_text(mutated, encoding="utf-8")
        v4_mutation = grade(v4_evaluator, V4_ROOT, workspace)
        v5_mutation = grade(v5_evaluator, V5_ROOT, workspace)
        (output / "candidate-mutation.patch").write_text(
            "".join(__import__("difflib").unified_diff(
                original.splitlines(keepends=True), mutated.splitlines(keepends=True),
                fromfile=f"a/{backend_rel}", tofile=f"b/{backend_rel}")), encoding="utf-8")

    write_json(output / "fixed-v4.grade.json", v4_control)
    write_json(output / "fixed-v5.grade.json", v5_control)
    write_json(output / "bad-words-v4.grade.json", v4_mutation)
    write_json(output / "bad-words-v5.grade.json", v5_mutation)
    passed = (v4_control["passed"] and v5_control["passed"]
              and v4_mutation["passed"] and not v5_mutation["passed"])
    summary = {
        "schema_version": 1,
        "study_id": "rvl-bad-words-grader-mutation-v1",
        "source_repository": task["source"]["repository"],
        "source_revision": task["source"]["calibration_revision"],
        "protocols": {
            "v4": "retained RVL behavior-policy parity protocol v4",
            "v5": json.loads((V5_ROOT / "protocol.lock.json").read_text())["protocol_id"],
        },
        "mutation": "Set inherited bad_words_ids=[[2]] in the fixed generation configuration; token 2 is the first expected response token.",
        "expected": {"fixed_v4": True, "fixed_v5": True, "mutation_v4": True, "mutation_v5": False},
        "observed": {"fixed_v4": v4_control["passed"], "fixed_v5": v5_control["passed"],
                     "mutation_v4": v4_mutation["passed"], "mutation_v5": v5_mutation["passed"]},
        "acceptance_passed": passed,
        "claim_limit": "One deterministic mutation shows that v4 did not cover bad_words_ids and v5 rejects this single-token case. It does not estimate grader error rates or cover multi-token constraints, every GenerationConfig field, agent performance, or model capability.",
        "resources": {"model_weights_downloaded": False, "third_party_python_packages": 0,
                      "accelerator_hours": 0, "paid_api_calls": 0, "external_compute_usd": 0},
        "runtime": {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "python": sys.version, "platform": platform.platform()},
    }
    write_json(output / "summary.json", summary)
    inputs = {
        "script_sha256": sha(Path(__file__).resolve()),
        "v4_evaluator_sha256": sha(v4_evaluator),
        "v4_protocol_lock_sha256": sha(V4_ROOT / "protocol.lock.json"),
        "v5_task_json_sha256": sha(V5_ROOT / "task.json"),
        "v5_task_brief_sha256": sha(V5_ROOT / "TASK.md"),
        "v5_protocol_lock_sha256": sha(V5_ROOT / "protocol.lock.json"),
        "v5_evaluator_sha256": sha(v5_evaluator),
    }
    files = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
             if path.is_file() and path.name != "manifest.json"}
    write_json(output / "manifest.json", {"schema_version": 1, "inputs": inputs, "files": files})
    print(json.dumps({"acceptance_passed": passed, "output": str(output),
                      "observed": summary["observed"]}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
