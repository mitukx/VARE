#!/usr/bin/env python3
"""Compare frozen TRL grader v10 with v11 on derived-alias loss writes."""
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

V10_ROOT = ROOT / "benchmarks/historical/trl_grpo_accumulation_scale_v10"
V11_ROOT = ROOT / "benchmarks/historical/trl_grpo_accumulation_scale_v11"


def _trainer_method(path: Path) -> tuple[str, list[str], ast.Assign]:
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
    return source, source.splitlines(keepends=True), targets[0]


def _insert(path: Path, line_number: int, statements: list[str],
            indentation_line: int | None = None) -> None:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    line = lines[(indentation_line or line_number) - 1]
    indent = line[:len(line) - len(line.lstrip())]
    inserted = [f"{indent}{statement}\n" for statement in statements]
    lines[line_number - 1:line_number - 1] = inserted
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def mutate_after_loss(path: Path, statements: list[str]) -> None:
    """Insert one controlled write after the checked loss division."""
    _, _, assignment = _trainer_method(path)
    _insert(path, assignment.end_lineno + 1, statements,
            indentation_line=assignment.lineno)


def mutate_entropy_adjustment(path: Path, statements: list[str]) -> None:
    """Insert a controlled alias write inside the otherwise permitted entropy update."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    matches = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.If)
                and isinstance(node.test, ast.Attribute)
                and node.test.attr == "_entropy_bonus_enabled"):
            continue
        for statement in node.body:
            if (isinstance(statement, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "loss"
                            for target in statement.targets)):
                matches.append(statement)
    if len(matches) != 1:
        raise RuntimeError(f"expected one entropy loss adjustment in {path}, found {len(matches)}")
    _insert(path, matches[0].lineno, statements)


def add_module_mutator_alias(path: Path) -> None:
    """Place a valid module-level operator alias before the trainer class."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    lines = source.splitlines(keepends=True)
    lines[trainer.lineno - 1:trainer.lineno - 1] = [
        "import operator\n",
        "put_item = operator.setitem\n",
    ]
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def add_conditional_module_mutator_alias(path: Path) -> None:
    """Place a module-scope mutator alias inside top-level control flow."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    lines = source.splitlines(keepends=True)
    lines[trainer.lineno - 1:trainer.lineno - 1] = [
        "import operator\n",
        "if True:\n",
        "    put_item = operator.setitem\n",
    ]
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def add_conditional_expression_mutator_alias(path: Path) -> None:
    """Place a conditional-expression module alias before the trainer class."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    lines = source.splitlines(keepends=True)
    lines[trainer.lineno - 1:trainer.lineno - 1] = [
        "import operator\n",
        "put_item = operator.setitem if True else operator.delitem\n",
    ]
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def add_boolean_expression_mutator_alias(path: Path) -> None:
    """Place a boolean-expression module alias before the trainer class."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    lines = source.splitlines(keepends=True)
    lines[trainer.lineno - 1:trainer.lineno - 1] = [
        "import operator\n",
        "put_item = operator.setitem or operator.delitem\n",
    ]
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def add_getattr_mutator_alias(path: Path) -> None:
    """Place a module-scope mutator alias resolved through getattr."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    lines = source.splitlines(keepends=True)
    lines[trainer.lineno - 1:trainer.lineno - 1] = [
        "import operator\n",
        'put_item = getattr(operator, "setitem")\n',
    ]
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def add_module_dict_mutator_alias(path: Path) -> None:
    """Place a module-scope mutator alias resolved from the API module dict."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    lines = source.splitlines(keepends=True)
    lines[trainer.lineno - 1:trainer.lineno - 1] = [
        "import operator\n",
        'put_item = operator.__dict__["setitem"]\n',
    ]
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def add_module_dict_get_mutator_alias(path: Path) -> None:
    """Place a module-scope mutator alias resolved with module-dict get."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    lines = source.splitlines(keepends=True)
    lines[trainer.lineno - 1:trainer.lineno - 1] = [
        "import operator\n",
        'put_item = operator.__dict__.get("setitem")\n',
    ]
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def add_qualified_getattr_mutator_alias(path: Path) -> None:
    """Place a module-scope mutator alias resolved through builtins.getattr."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    trainer = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer")
    lines = source.splitlines(keepends=True)
    lines[trainer.lineno - 1:trainer.lineno - 1] = [
        "import operator\n",
        "import builtins\n",
        'put_item = builtins.getattr(operator, "setitem")\n',
    ]
    mutated = "".join(lines)
    ast.parse(mutated, filename=str(path))
    path.write_text(mutated, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="new output directory; defaults under results/")
    args = parser.parse_args()
    output = args.output or ROOT / "results/trl-grpo-accumulation-window-normalizer-v1/grader-mutation-v11"
    output = output.expanduser().resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")

    task = json.loads((V11_ROOT / "task.json").read_text(encoding="utf-8"))
    v10 = load_grader(V10_ROOT / "evaluator/grade.py", "vare_trl_v10_grader")
    v11 = load_grader(V11_ROOT / "evaluator/grade.py", "vare_trl_v11_grader")
    output.mkdir(parents=True)
    (output / "mutations").mkdir()
    results = {}
    with tempfile.TemporaryDirectory(prefix="vare-trl-grader-mutation-v11-") as temporary:
        source = Path(temporary) / "candidate"
        checkout(task, source, task["source"]["calibration_revision"])
        cases = (
            ("control", None),
            ("early_return", "early_return"),
            ("zero_numerator", "zero_numerator"),
            ("tuple_alias_store", "tuple_alias_store"),
            ("starred_tuple_alias_store", "starred_tuple_alias_store"),
            ("pattern_capture_loss_store", "pattern_capture_loss_store"),
            ("pattern_capture_normalizer_store", "pattern_capture_normalizer_store"),
            ("generator_alias_store", "generator_alias_store"),
            ("list_comprehension_alias_store", "list_comprehension_alias_store"),
            ("dict_comprehension_alias_store", "dict_comprehension_alias_store"),
            ("normalizer_generator_alias_store", "normalizer_generator_alias_store"),
            ("derived_data_store", "derived_data_store"),
            ("derived_view_store", "derived_view_store"),
            ("derived_call_store", "derived_call_store"),
            ("dict_derived_store", "dict_derived_store"),
            ("for_target_store", "for_target_store"),
            ("with_bound_alias_store", "with_bound_alias_store"),
            ("normalizer_alias_store", "normalizer_alias_store"),
            ("entropy_normalizer_alias_store", "entropy_normalizer_alias_store"),
            ("entropy_adjustment_store", "entropy_adjustment_store"),
            ("operator_setitem", "operator_setitem"),
            ("imported_setitem_alias", "imported_setitem_alias"),
            ("unbound_tensor_setitem", "unbound_tensor_setitem"),
            ("module_setitem_alias", "module_setitem_alias"),
            ("conditional_module_setitem_alias", "conditional_module_setitem_alias"),
            ("conditional_expression_setitem_alias", "conditional_expression_setitem_alias"),
            ("boolean_expression_setitem_alias", "boolean_expression_setitem_alias"),
            ("getattr_setitem_alias", "getattr_setitem_alias"),
            ("module_dict_setitem_alias", "module_dict_setitem_alias"),
            ("module_dict_get_setitem_alias", "module_dict_get_setitem_alias"),
            ("qualified_getattr_setitem_alias", "qualified_getattr_setitem_alias"),
            ("tuple_out_argument", "tuple_out_argument"),
            ("call_result_out_argument", "call_result_out_argument"),
            ("entropy_operator_setitem", "entropy_operator_setitem"),
        )
        for label, mutation in cases:
            if mutation == "tuple_alias_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias, = (loss,)", "loss_alias[...] = 0"])
            elif mutation == "starred_tuple_alias_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias, *rest = (loss,)",
                                       "loss_alias.data.mul_(2)"])
            elif mutation == "pattern_capture_loss_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["match [loss]:", "    case [loss_alias]:",
                                       "        loss_alias[...] = 0"])
            elif mutation == "pattern_capture_normalizer_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["match [normalizer]:",
                                       "    case [normalizer_alias]:",
                                       "        normalizer_alias[...] = 0"])
            elif mutation == "generator_alias_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias = next(item for item in (loss,))",
                                       "loss_alias[...] = 0"])
            elif mutation == "list_comprehension_alias_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias = [item for item in (loss,)][0]",
                                       "loss_alias[...] = 0"])
            elif mutation == "dict_comprehension_alias_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ['loss_alias = {"root": item for item in (loss,)}["root"]',
                                       "loss_alias[...] = 0"])
            elif mutation == "normalizer_generator_alias_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["normalizer_alias = next(item for item in (normalizer,))",
                                       "normalizer_alias[...] = 0"])
            elif mutation == "derived_data_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias = loss.data", "loss_alias[...] = 0"])
            elif mutation == "derived_view_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias = loss.view(-1)", "loss_alias[...] = 0"])
            elif mutation == "derived_call_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias = loss.detach()", "loss_alias[...] = 0"])
            elif mutation == "dict_derived_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ['loss_alias = {"root": loss}["root"]',
                                       "loss_alias[...] = 0"])
            elif mutation == "for_target_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias, = (loss,)", "for loss_alias[...] in [0]:",
                                       "    pass"])
            elif mutation == "with_bound_alias_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["from contextlib import nullcontext",
                                       "with nullcontext(loss) as loss_alias:",
                                       "    loss_alias[...] = 0"])
            elif mutation == "normalizer_alias_store":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["normalizer_alias, = (normalizer,)",
                                       "normalizer_alias[...] = 0"])
            elif mutation == "entropy_adjustment_store":
                relative = "trl/trainer/grpo_trainer.py"
                mutate_entropy_adjustment(source / relative,
                                          ["loss_alias, = (loss,)", "loss_alias[...] = 0"])
            elif mutation == "entropy_normalizer_alias_store":
                relative = "trl/trainer/grpo_trainer.py"
                mutate_entropy_adjustment(source / relative,
                                          ["normalizer_alias, = (normalizer,)",
                                           "normalizer_alias[...] = 0"])
            elif mutation == "operator_setitem":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias, = (loss,)", "import operator",
                                       "operator.setitem(loss_alias, Ellipsis, 0)"])
            elif mutation == "imported_setitem_alias":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias, = (loss,)",
                                       "from operator import setitem as put_item",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "unbound_tensor_setitem":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias, = (loss,)",
                                       "torch.Tensor.__setitem__(loss_alias, Ellipsis, 0)"])
            elif mutation == "module_setitem_alias":
                for relative in task["source"]["files"]:
                    path = source / relative
                    add_module_mutator_alias(path)
                    mutate_after_loss(path,
                                      ["loss_alias = loss",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "conditional_module_setitem_alias":
                for relative in task["source"]["files"]:
                    path = source / relative
                    add_conditional_module_mutator_alias(path)
                    mutate_after_loss(path,
                                      ["loss_alias = loss",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "conditional_expression_setitem_alias":
                for relative in task["source"]["files"]:
                    path = source / relative
                    add_conditional_expression_mutator_alias(path)
                    mutate_after_loss(path,
                                      ["loss_alias = loss",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "boolean_expression_setitem_alias":
                for relative in task["source"]["files"]:
                    path = source / relative
                    add_boolean_expression_mutator_alias(path)
                    mutate_after_loss(path,
                                      ["loss_alias = loss",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "getattr_setitem_alias":
                for relative in task["source"]["files"]:
                    path = source / relative
                    add_getattr_mutator_alias(path)
                    mutate_after_loss(path,
                                      ["loss_alias = loss",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "module_dict_setitem_alias":
                for relative in task["source"]["files"]:
                    path = source / relative
                    add_module_dict_mutator_alias(path)
                    mutate_after_loss(path,
                                      ["loss_alias = loss",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "module_dict_get_setitem_alias":
                for relative in task["source"]["files"]:
                    path = source / relative
                    add_module_dict_get_mutator_alias(path)
                    mutate_after_loss(path,
                                      ["loss_alias = loss",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "qualified_getattr_setitem_alias":
                for relative in task["source"]["files"]:
                    path = source / relative
                    add_qualified_getattr_mutator_alias(path)
                    mutate_after_loss(path,
                                      ["loss_alias = loss",
                                       "put_item(loss_alias, Ellipsis, 0)"])
            elif mutation == "tuple_out_argument":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ["loss_alias = loss",
                                       "torch.frexp(torch.ones_like(loss), out=(loss_alias, torch.empty_like(loss, dtype=torch.int32)))"])
            elif mutation == "call_result_out_argument":
                for relative in task["source"]["files"]:
                    mutate_after_loss(source / relative,
                                      ['holder = {"loss": loss}',
                                       'torch.add(torch.ones_like(loss), 1.0, out=holder.get("loss"))'])
            elif mutation == "entropy_operator_setitem":
                relative = "trl/trainer/grpo_trainer.py"
                mutate_entropy_adjustment(source / relative,
                                          ["loss_alias, = (loss,)", "import operator",
                                           "operator.setitem(loss_alias, Ellipsis, 0)"])
            elif mutation:
                for relative in task["source"]["files"]:
                    mutate_source(source / relative, mutation)
            old_result = grade(v10, source, V10_ROOT)
            new_result = grade(v11, source, V11_ROOT)
            results[label] = {
                "v10": {"passed": old_result["passed"], "failures": old_result["failures"],
                       "candidate_diff_sha256": old_result["candidate_diff_sha256"]},
                "v11": {"passed": new_result["passed"], "failures": new_result["failures"],
                       "candidate_diff_sha256": new_result["candidate_diff_sha256"]},
            }
            write_json(output / f"{label}.v10.grade.json", old_result)
            write_json(output / f"{label}.v11.grade.json", new_result)
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
            "tuple_alias_store": (False, False),
            "starred_tuple_alias_store": (True, False),
            "pattern_capture_loss_store": (True, False),
            "pattern_capture_normalizer_store": (True, False),
            "generator_alias_store": (True, False),
            "list_comprehension_alias_store": (True, False),
            "dict_comprehension_alias_store": (True, False),
            "normalizer_generator_alias_store": (True, False),
            "derived_data_store": (True, False),
            "derived_view_store": (True, False),
            "derived_call_store": (True, False),
            "dict_derived_store": (True, False),
            "for_target_store": (True, False),
            "with_bound_alias_store": (True, False),
            "normalizer_alias_store": (True, False),
            "entropy_normalizer_alias_store": (True, False),
            "entropy_adjustment_store": (True, False),
            "operator_setitem": (True, False),
            "imported_setitem_alias": (True, False),
            "unbound_tensor_setitem": (True, False),
            "module_setitem_alias": (True, False),
            "conditional_module_setitem_alias": (True, False),
            "conditional_expression_setitem_alias": (True, False),
            "boolean_expression_setitem_alias": (True, False),
            "getattr_setitem_alias": (True, False),
            "module_dict_setitem_alias": (True, False),
            "module_dict_get_setitem_alias": (True, False),
            "qualified_getattr_setitem_alias": (True, False),
            "tuple_out_argument": (True, False),
            "call_result_out_argument": (True, False),
            "entropy_operator_setitem": (True, False),
        }
        observed_ok = all(
            (results[name]["v10"]["passed"], results[name]["v11"]["passed"]) == pair
            for name, pair in expected.items()
        )

    summary = {
        "schema_version": 1,
        "study_id": "trl-grader-derived-alias-mutation-v11",
        "source_repository": task["source"]["repository"],
        "source_revision": task["source"]["calibration_revision"],
        "base_revision": task["source"]["base_revision"],
        "protocols": {"comparison": "trl-grpo-normalizer-v10", "candidate": "trl-grpo-normalizer-v11"},
        "expected_outcomes": {name: {"v10_passed": pair[0], "v11_passed": pair[1]}
                              for name, pair in expected.items()},
        "observed_outcomes": results,
        "acceptance_passed": observed_ok,
        "resources": {"model_weights_downloaded": False, "third_party_python_packages": 0,
                      "accelerator_hours": 0, "paid_api_calls": 0, "external_compute_usd": 0},
        "claim_limit": "A bounded mutation set probes derived aliases, loop targets, and common mutating APIs on one pinned TRL task; v11 rejects the cases that v10 accepts. This does not establish arbitrary-code soundness, broad precision/recall, or full trainer correctness.",
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    write_json(output / "summary.json", summary)
    manifest_files = {str(path.relative_to(output)): sha(path) for path in sorted(output.rglob("*"))
                      if path.is_file() and path.name != "manifest.json"}
    write_json(output / "manifest.json", {
        "schema_version": 1,
        "files": manifest_files,
        "v10_evaluator_sha256": sha(V10_ROOT / "evaluator/grade.py"),
        "v10_protocol_lock_sha256": sha(V10_ROOT / "protocol.lock.json"),
        "v11_evaluator_sha256": sha(V11_ROOT / "evaluator/grade.py"),
        "v11_protocol_lock_sha256": sha(V11_ROOT / "protocol.lock.json"),
        "v11_task_json_sha256": sha(V11_ROOT / "task.json"),
    })
    print(json.dumps({
        "acceptance_passed": observed_ok,
        "output": str(output),
        "outcomes": {name: {"v10": row["v10"]["passed"], "v11": row["v11"]["passed"]}
                     for name, row in results.items()},
    }, indent=2))
    return 0 if observed_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
