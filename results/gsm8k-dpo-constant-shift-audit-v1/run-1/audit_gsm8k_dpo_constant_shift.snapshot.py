#!/usr/bin/env python3
"""Read-only post hoc decomposition of the archived GSM8K DPO margin result."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/gsm8k_dpo_constant_shift_audit_v1.json"
LOCK = ROOT / "protocols/gsm8k_dpo_constant_shift_audit_v1.lock.json"
SOURCE = ROOT / "results/cpu-lm-gsm8k-dpo-confirmation-v1/run-1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def nll(margin: float, correct_a: bool) -> float:
    sign = 1.0 if correct_a else -1.0
    x = sign * margin
    return max(-x, 0.0) + math.log1p(math.exp(-abs(x)))


def mean_loss(base: list[float], addition: list[float], correct_a: list[bool]) -> float:
    return statistics.fmean(nll(b + d, label) for b, d, label in zip(base, addition, correct_a))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    locked_hash = lock.pop("sha256", None)
    protocol_hash = hashlib.sha256(canonical(spec)).hexdigest()
    if locked_hash != protocol_hash or canonical(lock) != canonical(spec):
        raise ValueError("analysis protocol differs from its lock")
    runner_hash = digest(Path(__file__))
    if runner_hash != spec["implementation"]["runner_sha256"]:
        raise ValueError("analysis source differs from frozen hash")

    expected_files = spec["inputs"]["files"]
    for name, expected in expected_files.items():
        if digest(SOURCE / name) != expected:
            raise ValueError(f"original frozen source hash mismatch: {name}")
    old_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in old_manifest["files"].items():
        if digest(SOURCE / name) != expected:
            raise ValueError(f"original source bundle manifest mismatch: {name}")
    data = json.loads((SOURCE / "examples.json").read_text(encoding="utf-8"))
    seeds = json.loads((SOURCE / "seed_records.json").read_text(encoding="utf-8"))
    train = data["train"]
    heldout = data["heldout"]
    train_labels = [row["correct_label"] == "A" for row in train]
    heldout_labels = [row["correct_label"] == "A" for row in heldout]
    result_by_seed = {}
    for seed in ("401", "503", "607"):
        record = seeds[seed]
        if len(record["train_base_margins"]) != len(train) or len(record["train_updated_margins"]) != len(train):
            raise ValueError(f"training margin count mismatch for seed {seed}")
        if len(record["heldout_base_margins"]) != len(heldout) or len(record["heldout_updated_margins"]) != len(heldout):
            raise ValueError(f"heldout margin count mismatch for seed {seed}")
        if [row["correct_label"] for row in train] != [row["correct_label"] for row in data["train"]]:
            raise ValueError("training labels changed during audit")

        train_delta = [u - b for b, u in zip(record["train_base_margins"], record["train_updated_margins"])]
        c = statistics.fmean(train_delta)
        base = [float(x) for x in record["heldout_base_margins"]]
        updated = [float(x) for x in record["heldout_updated_margins"]]
        actual_delta = [u - b for b, u in zip(base, updated)]
        residual = [delta - c for delta in actual_delta]
        zeros = [0.0] * len(base)
        constant = [c] * len(base)
        base_nll = mean_loss(base, zeros, heldout_labels)
        constant_nll = mean_loss(base, constant, heldout_labels)
        residual_nll = mean_loss(base, residual, heldout_labels)
        actual_nll = mean_loss(base, actual_delta, heldout_labels)
        phi_constant = ((constant_nll - base_nll) + (actual_nll - residual_nll)) / 2
        phi_residual = ((residual_nll - base_nll) + (actual_nll - constant_nll)) / 2
        if not math.isclose(phi_constant + phi_residual, actual_nll - base_nll, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Shapley attribution does not reconstruct the observed NLL change")
        archived_base = record["metrics"]["heldout_base"]["mean_conditional_preference_nll"]
        archived_updated = record["metrics"]["heldout_updated"]["mean_conditional_preference_nll"]
        if not math.isclose(base_nll, archived_base, rel_tol=0.0, abs_tol=1e-10) or not math.isclose(actual_nll, archived_updated, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"historical heldout NLL does not reconstruct for seed {seed}")
        count_a = sum(heldout_labels)
        result_by_seed[seed] = {
            "training_mean_logit_shift": c,
            "heldout_base_nll": base_nll,
            "heldout_constant_shift_nll": constant_nll,
            "heldout_residual_only_nll": residual_nll,
            "heldout_actual_dpo_nll": actual_nll,
            "heldout_actual_nll_change": actual_nll - base_nll,
            "shapley_constant_shift_nll_component": phi_constant,
            "shapley_nonconstant_residual_nll_component": phi_residual,
            "heldout_correct_a": count_a,
            "heldout_correct_b": len(heldout) - count_a,
            "heldout_mean_actual_logit_shift": statistics.fmean(actual_delta),
            "heldout_mean_nonconstant_residual_shift": statistics.fmean(residual),
        }
    fields = list(next(iter(result_by_seed.values())))
    equal_seed_means = {field: statistics.fmean(result_by_seed[s][field] for s in result_by_seed) for field in fields}
    analysis = {
        "protocol_id": spec["protocol_id"], "protocol_sha256": protocol_hash,
        "analysis_source_sha256": runner_hash,
        "status": "post_hoc_descriptive_mechanism_audit",
        "per_seed": result_by_seed,
        "equal_seed_means": equal_seed_means,
        "interpretation_limit": spec["claim_boundary"],
    }
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    (output / "manifest.json").write_text(json.dumps({"algorithm": "sha256", "files": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
