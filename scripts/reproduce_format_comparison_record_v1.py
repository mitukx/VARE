"""Audit archived v2 records on CPU; never load a model or resume training."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from vare import format_vs_correctness_v2 as legacy  # noqa: E402
from vare.format_vs_correctness_seeded_summary_v3 import summarize_seeded_comparison  # noqa: E402

RECORD = ROOT / "results/format-comparison-record-v1"
PROGRESS_SHA = "10860ac27af371a7374bcb7ca38c6bd5171f6d467aef43c3a7c1e95468c73694"


def truth(expression: str) -> int:
    """Independent AST/Fraction oracle, not VARE's recursive-descent grader."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (1 if isinstance(node.op, ast.UAdd) else -1)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                value = left / right
                if value.denominator != 1:
                    raise ValueError("division is not exact")
                return value
        raise ValueError("unsupported expression")
    value = visit(ast.parse(expression, mode="eval").body)
    if value.denominator != 1:
        raise ValueError("noninteger result")
    return int(value)


def parsed(text: str) -> int | None:
    lines = text.rstrip().splitlines()
    final = lines[-1].strip() if lines else ""
    match = re.fullmatch(r"#### (-?\d+)", final)
    return int(match[1]) if match else None


def zero_b_tensors(blob: bytes) -> list[str]:
    """Inspect the two F32 LoRA B tensors without an accelerator runtime."""
    size = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + size])
    data = memoryview(blob)[8 + size:]
    names = []
    for name, spec in header.items():
        if not name.endswith(".lora_b"):
            continue
        if spec["dtype"] != "F32":
            raise ValueError("unexpected retained adapter dtype")
        start, stop = spec["data_offsets"]
        if not 0 <= start < stop <= len(data) or stop - start != math.prod(spec["shape"]) * 4:
            raise ValueError("invalid retained tensor bounds")
        if any(value[0] != 0.0 for value in struct.iter_unpack("<f", data[start:stop])):
            raise ValueError("nonzero retained LoRA B")
        names.append(name)
    if len(names) != 2:
        raise ValueError("expected q_proj and v_proj LoRA B tensors")
    return names


def zero_adam_moments(blob: bytes, step: int) -> None:
    size = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + size])
    data = memoryview(blob)[8 + size:]
    moments = 0
    for name, spec in header.items():
        if not name.endswith((".m", ".v")):
            continue
        start, stop = spec["data_offsets"]
        if spec["dtype"] != "F32" or not 0 <= start < stop <= len(data):
            raise ValueError("invalid retained moment tensor")
        if stop - start != math.prod(spec["shape"]) * 4:
            raise ValueError("invalid retained moment shape")
        if any(value[0] != 0.0 for value in struct.iter_unpack("<f", data[start:stop])):
            raise ValueError("nonzero retained Adam moment")
        moments += 1
    start, stop = header["step"]["data_offsets"]
    if moments != 8 or header["step"]["dtype"] != "U64" or struct.unpack("<Q", data[start:stop])[0] != step:
        raise ValueError("invalid retained optimizer step/moments")


def audit(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("output must be empty")
    manifest = json.loads((RECORD / "record-manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if hashlib.sha256((RECORD / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"record artifact changed: {name}")
    blob = gzip.decompress((RECORD / "progress.jsonl.gz").read_bytes())
    if hashlib.sha256(blob).hexdigest() != PROGRESS_SHA:
        raise ValueError("archived progress digest changed")
    rows = [json.loads(line) for line in blob.splitlines() if line.strip()]
    confirm = legacy.confirm_eval_tasks()
    pack = legacy.pack_eval_tasks((ROOT / "benchmarks/heldout/exact_integer_v1/tasks.jsonl").read_text())
    units = [*legacy.iter_train_units(), *legacy.iter_eval_units(confirm, pack)]
    expected = {legacy.unit_key(unit): unit for unit in units}
    for row in rows:
        unit = expected[legacy.row_key(row)]
        if any(type(row.get(k)) is not type(v) or row[k] != v for k, v in unit.items()):
            raise ValueError("row metadata differs from the frozen unit")
        if (row.get("backend"), row.get("model_id"), row.get("protocol_sha256")) != (
            "mlx", "mlx-community/Qwen2.5-3B-Instruct-4bit", legacy.PROTOCOL_SHA256
        ):
            raise ValueError("record model/protocol tags differ")
    result = summarize_seeded_comparison(confirm, pack, rows, prospective=False)
    result["progress_sha256"] = PROGRESS_SHA
    old = json.loads((RECORD / "original-summary.json").read_text())
    for field in ("paired_bootstrap", "decision", "decision_split", "by_seed_confirm_final", "promotes_weights"):
        if result[field] != old[field]:
            raise ValueError(f"legacy primary drift: {field}")
    expressions = {task["task_id"]: task["expression"] for task in [*confirm, *pack]}
    tasks = {task_id: truth(expression) for task_id, expression in expressions.items()}
    counts = defaultdict(list)
    evaluated = 0
    for row in rows:
        if row["kind"] != "eval":
            continue
        correct = parsed(row["output"]) == tasks[row["task_id"]]
        if correct != legacy.score_output(expressions[row["task_id"]], row["output"])["strict_correct"]:
            raise ValueError("independent strict grading differs on a retained output")
        counts[(row["arm"], row["checkpoint_step"], row["split"], row["run_seed"], row["task_id"])].append(correct)
        evaluated += 1
    for point in result["curves"]:
        split_tasks = confirm if point["split"] == "confirm" else pack
        groups = [counts[(point["arm"], point["checkpoint_step"], point["split"], seed, task["task_id"])]
                  for seed in legacy.RUN_SEEDS for task in split_tasks]
        for k in (1, 4):
            value = sum(Fraction(1) - Fraction(math.comb(4 - sum(group), k), math.comb(4, k))
                        for group in groups) / len(groups)
            if not math.isclose(float(value), point[f"pass_at_{k}"], abs_tol=1e-15, rel_tol=0):
                raise ValueError("independent AST/strict-output curve mismatch")
    examples = {example["seed"]: example for example in legacy.comparison_train_examples()}
    diagnostics = {}
    for arm in legacy.RL_ARMS:
        train = [row for row in rows if row["kind"] == "train" and row["arm"] == arm]
        by_seed = Counter()
        for row in train:
            texts = row["outputs"]
            if arm == "format_reward_rl":
                rewards = [float(parsed(text) is not None) for text in texts]
            elif arm == "correctness_reward_rl":
                answer = truth(examples[row["train_seed"]]["expression"])
                rewards = [float(parsed(text) == answer) for text in texts]
            else:
                # Random-label generation is checked against the frozen rule,
                # not used to validate either independent arithmetic oracle.
                rewards = legacy.random_group_rewards(row["run_seed"], row["step"])
            mean = sum(rewards) / 2
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / 2)
            advantages = [(r - mean) / (std + 1e-8) for r in rewards]
            if rewards != row["rewards"] or advantages != row["advantages"]:
                raise ValueError("retained training rewards/advantages mismatch")
            by_seed[row["run_seed"]] += len(set(rewards)) > 1
        diagnostics[arm] = {"groups": len(train), "mixed_reward_groups": sum(by_seed.values()),
            "mixed_by_run_seed": dict(by_seed), "zero_advantage_groups": sum(not any(r["advantages"]) for r in train),
            "nonzero_loss_groups": sum(r["loss"] != 0 for r in train),
            "distinct_outputs_in_mixed_groups": sum(len(set(r["rewards"])) > 1 and len(set(r["outputs"])) > 1 for r in train)}
    checkpoints = json.loads((RECORD / "checkpoint-manifest.json").read_text())
    optimizer_states = 0
    with tarfile.open(fileobj=io.BytesIO((RECORD / "format-zero-optimizers.tar.gz").read_bytes()), mode="r:gz") as archive:
        for seed in legacy.RUN_SEEDS:
            for step in range(1, 17):
                name = f"format_reward_rl/{seed}/step-{step:04d}/optimizer.safetensors"
                member = archive.extractfile(name)
                if member is None:
                    raise ValueError("missing retained optimizer")
                state = member.read()
                if hashlib.sha256(state).hexdigest() != checkpoints[name]["sha256"]:
                    raise ValueError("optimizer digest differs from checkpoint manifest")
                zero_adam_moments(state, step)
                optimizer_states += 1
    adapters = []
    for seed in legacy.RUN_SEEDS:
        relative = f"format_reward_rl/{seed}/step-0016/adapters.safetensors"
        saved = (RECORD / f"format-zero-adapters/{seed}.safetensors").read_bytes()
        digest = hashlib.sha256(saved).hexdigest()
        hashes = {checkpoints[f"format_reward_rl/{seed}/step-{step:04d}/adapters.safetensors"]["sha256"]
                  for step in range(1, 17)}
        if hashes != {digest} or checkpoints[relative]["sha256"] != digest:
            raise ValueError("format adapter manifest changed across steps")
        adapters.append({"run_seed": seed, "identical_step_1_through_16_hashes": True,
                         "final_adapter_sha256": digest, "zero_b_tensors": zero_b_tensors(saved)})
    report = {"progress_sha256": PROGRESS_SHA, "rows": len(rows), "independently_scored_eval_rows": evaluated,
        "primary_preserved": True, "all_curves_match_independent_oracle": True,
        "training_signal": diagnostics, "format_adapters": adapters,
        "format_zero_optimizer_states": optimizer_states,
        "recorded_train_seconds": sum(r["wall_seconds"] for r in rows if r["kind"] == "train"),
        "recorded_eval_seconds": sum(r["wall_seconds"] for r in rows if r["kind"] == "eval"),
        "original_execution_revision_verified": False, "original_execution_operator_verified": False,
        "new_model_training": False, "new_model_generation": False}
    (output / "summary.seeded-v3.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "eval_rows": evaluated, "decision": result["decision"],
                      "format_mixed_groups": diagnostics["format_reward_rl"]["mixed_reward_groups"],
                      "correctness_mixed_groups": diagnostics["correctness_reward_rl"]["mixed_reward_groups"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    audit(parser.parse_args().output)
