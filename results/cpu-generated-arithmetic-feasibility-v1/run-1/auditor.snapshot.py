#!/usr/bin/env python3
"""Audit the generated arithmetic feasibility pilot without model dependencies."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "protocols/cpu_generated_arithmetic_feasibility_v1.json"
LOCK_PATH = ROOT / "protocols/cpu_generated_arithmetic_feasibility_v1.lock.json"
DATA_PATH = ROOT / "data/cpu-arithmetic-feasibility-v1/pilot.json"
GENERATOR_PATH = ROOT / "scripts/generate_compositional_arithmetic_feasibility_pilot.py"
ANSWER_RE = re.compile(r"\s*FINAL:\s*(-?\d+)\s*", re.IGNORECASE)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def strict_parse(text: str):
    match = ANSWER_RE.fullmatch(text)
    return int(match.group(1)) if match else None


def safe_integer_expression(node):
    if isinstance(node, ast.Expression):
        return safe_integer_expression(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp):
        left, right = safe_integer_expression(node.left), safe_integer_expression(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    raise ValueError("expression contains a non-integer or unsupported operation")


def evaluate_oracle(expression: str) -> int:
    return safe_integer_expression(ast.parse(expression, mode="eval"))


def ensure_close(a, b, label):
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            raise ValueError("record mismatch: " + label)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(a - b) > 1e-12:
            raise ValueError("record mismatch: " + label)
    elif a != b:
        raise ValueError("record mismatch: " + label)


def audit(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    spec = load(bundle / "protocol.json")
    lock = load(bundle / "protocol.lock.json")
    locked = dict(lock)
    lock_hash = locked.pop("sha256", None)
    protocol_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if lock_hash != protocol_hash or canonical(locked) != canonical(spec):
        raise ValueError("bundle protocol does not match its lock")
    if spec["protocol_id"] != "vare-cpu-generated-arithmetic-feasibility-v1":
        raise ValueError("unexpected protocol id")
    if load(SPEC_PATH) != spec or load(LOCK_PATH) != lock:
        raise ValueError("bundle protocol differs from repository-frozen version")
    if sha256_file(DATA_PATH) != spec["data"]["pilot_rows_sha256"]:
        raise ValueError("repository pilot data hash differs from the protocol")
    if sha256_file(GENERATOR_PATH) != spec["data"]["generator_sha256"]:
        raise ValueError("repository pilot generator hash differs from the protocol")

    manifest = load(bundle / "manifest.json")
    files = manifest.get("files")
    if manifest.get("algorithm") != "sha256" or not isinstance(files, dict):
        raise ValueError("unsupported manifest")
    found = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
             if p.is_file() and p.name not in ("manifest.json", "audit.json")}
    if found != set(files):
        raise ValueError("bundle inventory differs from its manifest")
    for relative, digest in files.items():
        if sha256_file(bundle / relative) != digest:
            raise ValueError("bundle file hash mismatch: " + relative)

    pilot = load(bundle / "pilot.json")
    summary = load(bundle / "summary.json")
    expected_count = spec["data"]["pilot_rows"]
    records = summary.get("rows")
    if len(pilot) != expected_count or not isinstance(records, list) or len(records) != expected_count:
        raise ValueError("pilot row count differs from frozen cohort")
    compositions = {}
    for item in pilot:
        compositions[item["composition"]] = compositions.get(item["composition"], 0) + 1
        if evaluate_oracle(item["expression"]) != item["oracle_answer"]:
            raise ValueError("retained oracle answer disagrees with expression")
    if compositions != spec["data"]["composition_counts"]:
        raise ValueError("composition counts differ from protocol")

    exact_count = 0
    parsed_count = 0
    parsed_wrong = 0
    for index, (source, record) in enumerate(zip(pilot, records)):
        if record["pilot_index"] != index or source["pilot_index"] != index:
            raise ValueError("pilot rows are not in their frozen order")
        for key in ("composition", "expression", "oracle_answer"):
            ensure_close(record[key], source[key], "%s/%s" % (index, key))
        prompt_hash = hashlib.sha256(source["prompt"].encode("utf-8")).hexdigest()
        ensure_close(record["prompt_sha256"], prompt_hash, "%s/prompt_sha256" % index)
        parsed = strict_parse(record["generated_text"])
        ensure_close(record["parsed_answer"], parsed, "%s/parsed_answer" % index)
        ensure_close(record["parsed"], parsed is not None, "%s/parsed" % index)
        correct = parsed == source["oracle_answer"]
        ensure_close(record["exact_match"], correct, "%s/exact_match" % index)
        if record["input_tokens"] < 1 or record["input_tokens"] > spec["compute_limits"]["max_input_tokens"]:
            raise ValueError("input token limit violated")
        if record["generated_tokens"] < 0 or record["generated_tokens"] > spec["compute_limits"]["max_new_tokens"]:
            raise ValueError("generation token limit violated")
        if not math.isfinite(record["wall_seconds"]) or record["wall_seconds"] < 0:
            raise ValueError("invalid per-row runtime")
        exact_count += int(correct)
        parsed_count += int(parsed is not None)
        parsed_wrong += int(parsed is not None and not correct)

    parse_rate = parsed_count / expected_count
    max_rss = summary["peak_rss_bytes"]
    wall = summary["total_wall_seconds"]
    compute_ok = (summary["device"] == "cpu" and summary["cuda_initialized"] is False and
                  summary["mps_used"] is False and summary["network_disabled"] is True and
                  summary["paid_compute"] is False and max_rss <= spec["compute_limits"]["max_peak_rss_bytes"] and
                  wall <= spec["compute_limits"]["max_wall_seconds"])
    gates = spec["decision"]["pass_conditions"]
    passed = (exact_count >= gates["exact_match_count_minimum"] and
              exact_count <= gates["exact_match_count_maximum"] and
              parsed_wrong >= gates["parsed_wrong_answer_count_minimum"] and
              parse_rate >= gates["parse_rate_minimum"] and compute_ok)
    expected_metrics = {"pilot_rows": expected_count, "exact_match_count": exact_count,
                        "exact_match_rate": exact_count / expected_count,
                        "parsed_count": parsed_count, "parse_rate": parse_rate,
                        "parsed_wrong_answer_count": parsed_wrong, "unparsed_count": expected_count - parsed_count}
    for key, value in expected_metrics.items():
        ensure_close(summary["metrics"][key], value, "metrics/" + key)
    ensure_close(summary["decision"]["pass"], passed, "decision/pass")
    ensure_close(summary["decision"]["compute_limits_respected"], compute_ok, "decision/compute_limits")
    if summary.get("formal_update_performed") is not False or spec["study_boundary"]["formal_confirmation"]:
        raise ValueError("feasibility run crossed the training/confirmation boundary")
    if sha256_file(bundle / "pilot.json") != spec["data"]["pilot_rows_sha256"]:
        raise ValueError("retained pilot rows differ from their frozen hash")
    if (summary.get("model", {}).get("id") != spec["model"]["id"] or
            summary.get("model", {}).get("revision") != spec["model"]["revision"]):
        raise ValueError("summary model identity differs from protocol")
    for filename, expected in spec["model"]["files_sha256"].items():
        if summary["model"].get("files_sha256", {}).get(filename) != expected:
            raise ValueError("summary model asset hash differs from protocol: " + filename)
    if sha256_file(bundle / "generator.snapshot.py") != spec["data"]["generator_sha256"]:
        raise ValueError("bundle generator snapshot differs from protocol")
    if sha256_file(bundle / "runner.snapshot.py") != summary.get("runner_sha256"):
        raise ValueError("bundle runner snapshot does not match summary")
    if sha256_file(bundle / "auditor.snapshot.py") != summary.get("auditor_sha256"):
        raise ValueError("bundle auditor snapshot does not match summary")
    runtime = summary.get("runtime", {})
    if (runtime.get("python", "").split(" ", 1)[0] != spec["runtime"]["python"] or
            runtime.get("torch") != spec["runtime"]["torch"] or
            runtime.get("transformers") != spec["runtime"]["transformers"]):
        raise ValueError("runtime differs from the frozen protocol")
    return {"status": "pass", "audit_scope": "data, parser, oracle arithmetic, metrics, gates, hashes, and resource record",
            "model_inference_replayed": False, "formal_training_performed": False,
            "protocol_sha256": protocol_hash, "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
            "metrics": expected_metrics, "decision_pass": passed, "compute_limits_respected": compute_ok,
            "limitations": ["This auditor does not rerun model inference.",
                            "The result is a train-only feasibility pilot, not held-out evidence."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.bundle)
        output = args.bundle.resolve() / "audit.json"
        output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "fail", "error": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
