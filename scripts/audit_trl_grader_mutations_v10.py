#!/usr/bin/env python3
"""Compare frozen TRL grader v9 with v10 on alias-rooted storage writes."""
from __future__ import annotations

import argparse
import ast
import json
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_trl_grader_mutations import (  # noqa: E402
    checkout,
    grade,
    load_grader,
    mutate_source,
    run,
    sha,
    write_json,
)

V9_ROOT = ROOT / "benchmarks/historical/trl_grpo_accumulation_scale_v9"
V10_ROOT = ROOT / "benchmarks/historical/trl_grpo_accumulation_scale_v10"


def mutate_tuple_alias_store(path: Path, write: str = "loss_alias[...] = 0") -> None:
    """Write through a tuple-unpacked alias of the returned loss tensor."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    method = next(node for node in trainer.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss")
    expected_numerator = ast.dump(
        ast.parse("(per_token_loss * mask).sum()").body[0].value,
        include_attributes=False,
    )
    targets = []
    for node in ast.walk(method):
        if not (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "loss"
                        for target in node.targets)
                and isinstance(node.value, ast.BinOp)
                and isinstance(node.value.op, ast.Div)
                and isinstance(node.value.right, ast.Name)
                and node.value.right.id == "normalizer"
                and ast.dump(node.value.left, include_attributes=False) == expected_numerator):
            continue
        targets.append(node)
    if len(targets) != 1:
        raise RuntimeError(f"expected one pinned loss assignment in {path}, found {len(targets)}")

    assignment = targets[0]
    lines = source.splitlines(keepends=True)
    line = lines[assignment.end_lineno - 1]
    indent = line[:len(line) - len(line.lstrip())]
    inserted = [
        f"{indent}loss_alias, = (loss,)  # mutation: tuple-unpacked Tensor alias\n",
        f"{indent}{write}  # mutation: alter returned policy loss\n",
    ]
    lines[assignment.end_lineno:assignment.end_lineno] = inserted
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="new output directory; defaults under results/")
    args = parser.parse_args()
    output = args.output or ROOT / "results/trl-grpo-accumulation-window-normalizer-v1/grader-mutation-v10"
    output = output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")

    task = json.loads((V10_ROOT / "task.json").read_text(encoding="utf-8"))
    v9 = load_grader(V9_ROOT / "evaluator/grade.py", "vare_trl_v9_grader")
    v10 = load_grader(V10_ROOT / "evaluator/grade.py", "vare_trl_v10_grader")
    output.mkdir(parents=True)
    (output / "mutations").mkdir()
    results = {}
    with tempfile.TemporaryDirectory(prefix="vare-trl-grader-mutation-v10-") as temporary:
        source = Path(temporary) / "candidate"
        checkout(task, source, task["source"]["calibration_revision"])
        cases = (
            ("control", None),
            ("early_return", "early_return"),
            ("zero_numerator", "zero_numerator"),
            ("tuple_alias_method", "tuple_alias_method"),
            ("tuple_alias_store", "tuple_alias_store"),
        )
        for label, mutation in cases:
            if mutation in {"tuple_alias_method", "tuple_alias_store"}:
                for relative in task["source"]["files"]:
                    mutate_tuple_alias_store(
                        source / relative,
                        "loss_alias.data.mul_(2)" if mutation == "tuple_alias_method"
                        else "loss_alias[...] = 0",
                    )
            elif mutation:
                for relative in task["source"]["files"]:
                    mutate_source(source / relative, mutation)
            old_result = grade(v9, source, V9_ROOT)
            new_result = grade(v10, source, V10_ROOT)
            results[label] = {
                "v9": {"passed": old_result["passed"], "failures": old_result["failures"],
                       "candidate_diff_sha256": old_result["candidate_diff_sha256"]},
                "v10": {"passed": new_result["passed"], "failures": new_result["failures"],
                       "candidate_diff_sha256": new_result["candidate_diff_sha256"]},
            }
            write_json(output / f"{label}.v9.grade.json", old_result)
            write_json(output / f"{label}.v10.grade.json", new_result)
            if mutation:
                patch = run(["git", "diff", "--binary", "-U0",
                             task["source"]["calibration_revision"],
                             "--", *task["source"]["files"]], source)
                (output / "mutations" / f"{label}.patch").write_text(patch, encoding="utf-8")
                run(["git", "checkout", "--quiet", "--", *task["source"]["files"]], source)

        expected = {
            "control": (True, True),
            "early_return": (False, False),
            "zero_numerator": (False, False),
            "tuple_alias_method": (False, False),
            "tuple_alias_store": (True, False),
        }
        observed_ok = all(
            (results[name]["v9"]["passed"], results[name]["v10"]["passed"]) == pair
            for name, pair in expected.items()
        )

    summary = {
        "schema_version": 1,
        "study_id": "trl-grader-alias-store-mutation-v10",
        "source_repository": task["source"]["repository"],
        "source_revision": task["source"]["calibration_revision"],
        "base_revision": task["source"]["base_revision"],
        "protocols": {"comparison": "trl-grpo-normalizer-v9", "candidate": "trl-grpo-normalizer-v10"},
        "expected_outcomes": {name: {"v9_passed": pair[0], "v10_passed": pair[1]}
                              for name, pair in expected.items()},
        "observed_outcomes": results,
        "acceptance_passed": observed_ok,
        "resources": {"model_weights_downloaded": False, "third_party_python_packages": 0,
                      "accelerator_hours": 0, "paid_api_calls": 0, "external_compute_usd": 0},
        "claim_limit": "One review-discovered tuple-alias storage write is accepted by v9 and rejected by v10; controls show only narrow source-level behavior and do not establish arbitrary-code soundness, broad precision/recall, or full trainer correctness.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    write_json(output / "summary.json", summary)
    manifest_files = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
                      if path.is_file() and path.name != "manifest.json"}
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "files": manifest_files,
        "v9_evaluator_sha256": sha(V9_ROOT / "evaluator/grade.py"),
        "v9_protocol_lock_sha256": sha(V9_ROOT / "protocol.lock.json"),
        "v10_evaluator_sha256": sha(V10_ROOT / "evaluator/grade.py"),
        "v10_protocol_lock_sha256": sha(V10_ROOT / "protocol.lock.json"),
        "v10_task_json_sha256": sha(V10_ROOT / "task.json"),
    })
    print(json.dumps({
        "acceptance_passed": observed_ok,
        "output": str(output),
        "outcomes": {name: {"v9": row["v9"]["passed"], "v10": row["v10"]["passed"]}
                     for name, row in results.items()},
    }, indent=2))
    return 0 if observed_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
