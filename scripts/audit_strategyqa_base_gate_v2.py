#!/usr/bin/env python3
"""Independent standard-library replay for the StrategyQA base-gate output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rank_ids(rows: list[dict], salt: str, count: int) -> list[str]:
    return [str(x["qid"]) for x in sorted(rows, key=lambda x: hashlib.sha256((salt + "|" + str(x["qid"])).encode()).hexdigest())[:count]]


def audit_parse(text: str) -> str | None:
    lowered = text.lower()
    marker_at = lowered.rfind("final answer:")
    if marker_at < 0:
        return None
    tail = lowered[marker_at + len("final answer:"):].strip()
    if not tail:
        return None
    first = tail.split()[0].strip(" .!?,;:*`'\"()[]")
    return first if first in ("yes", "no") else None


def wilson(k: int, n: int) -> float:
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    lock_path = ROOT / "protocols/cpu_strategyqa_grpo_shift_v2.lock.json"
    lock = json.loads(lock_path.read_text())
    data_path = Path(os.environ["VARE_STRATEGYQA_TRAIN_JSON"])
    summary_path = Path(args.run_dir) / "summary.json"
    summary = json.loads(summary_path.read_text())
    data = json.loads(data_path.read_text())
    rows = data if isinstance(data, list) else list(data.values())
    qids = {str(row["qid"]): row for row in rows}
    gate = rank_ids(rows, lock["selection"]["salt"], lock["selection"]["base_gate_n"])
    rest = [row for row in rows if str(row["qid"]) not in set(gate)]
    confirmation = rank_ids(rest, lock["selection"]["salt"] + "|confirmation", lock["selection"]["confirmation_n"])
    remain = [row for row in rest if str(row["qid"]) not in set(confirmation)]
    train = rank_ids(remain, lock["selection"]["salt"] + "|train", lock["selection"]["train_n"])
    checks = {
        "dataset_hash": digest(data_path) == lock["dataset"]["train_json_sha256"],
        "runner_hash": digest(ROOT / "scripts/run_strategyqa_base_gate_v2.py") == lock["frozen_sources"]["runner_sha256"],
        "frozen_gate_ids": gate == lock["selection"]["base_gate_ids"],
        "frozen_confirmation_ids": confirmation == lock["selection"]["confirmation_ids"],
        "frozen_train_ids": train == lock["selection"]["train_ids"],
        "summary_protocol": summary.get("protocol_id") == lock["protocol_id"],
        "summary_records": len(summary.get("records", [])) == len(gate),
        "summary_eval_selection": summary.get("selection_ids", {}).get("base_gate") == gate,
        "sets_disjoint": not (set(gate) & set(confirmation) or set(gate) & set(train) or set(confirmation) & set(train)),
        "confirmation_not_scored": not (set(confirmation) & {str(row.get("qid")) for row in summary.get("records", [])}),
    }
    records = summary["records"]
    recomputed = []
    for qid, row in zip(gate, records):
        src = qids[qid]
        parsed = audit_parse(row["completion"])
        gold = "yes" if bool(src["answer"]) else "no"
        recomputed.append({"qid_matches": row["qid"] == qid, "parsed_matches": row["parsed"] == parsed, "gold_matches": row["gold"] == gold, "correct_matches": row["correct"] == (parsed == gold), "parsed": parsed, "gold": gold})
    k = sum(x["parsed"] == x["gold"] for x in recomputed)
    n = len(recomputed)
    parse_n = sum(x["parsed"] is not None for x in recomputed)
    accuracy = k / n
    parse_rate = parse_n / n
    gate_status = accuracy >= lock["gate"]["minimum_accuracy"] and wilson(k, n) > lock["gate"]["null_accuracy"] and parse_rate >= lock["gate"]["minimum_parse_rate"]
    checks["all_row_replays"] = all(all(bool(v) for key, v in row.items() if key.endswith("matches")) for row in recomputed)
    checks["metric_replay"] = abs(accuracy - summary["accuracy"]) < 1e-12 and abs(parse_rate - summary["parse_rate"]) < 1e-12 and abs(wilson(k, n) - summary["wilson_95_lower"]) < 1e-12
    checks["gate_replay"] = ("pass" if gate_status else "non_pass") == summary["status"]
    result = {
        "audit_id": "strategyqa_base_gate_v2_independent_replay",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "recomputed": {"n": n, "correct": k, "accuracy": accuracy, "parsed": parse_n, "parse_rate": parse_rate, "wilson_95_lower": wilson(k, n), "gate_status": "pass" if gate_status else "non_pass"},
        "source_sha256": digest(summary_path),
        "auditor_sha256": digest(Path(__file__)),
        "limitation": "Same host and source data; verifies deterministic replay of retained outputs, not an external reproduction or semantic validity of the benchmark.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
